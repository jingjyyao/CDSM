import os
import sys
import copy
import json
import time
import pickle
import logging
import numpy as np
from collections import defaultdict

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import RobertaConfig, RobertaModel, BertTokenizerFast

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import utils
from models.Selector_Evaluator import CounterPartSelector
from models.configuration_tnlrv3 import TuringNLRv3Config
from data_handler import DatasetForSelecting, DataCollatorForSelecting, \
    SingleProcessDataLoader, MultiProcessDataLoader

def load_bert(args):
    config = TuringNLRv3Config.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        output_hidden_states=True)
    model = CounterPartSelector(config, args.model_name_or_path)
    return model

def train(local_rank, args, global_prefetch_step, end, load):
    utils.setuplogging()
    os.environ["RANK"] = str(local_rank)
    utils.setup(local_rank, args.world_size)
    device = torch.device("cuda", local_rank)
    if args.fp16:
        from torch.cuda.amp import autocast
        scaler = torch.cuda.amp.GradScaler()

    model = load_bert(args)
    logging.info('loading model: {}'.format(args.model_type))
    model = model.to(device)

    if load:
        model.load_evaluator(args.load_ckpt_name)
        logging.info('load ckpt:{}'.format(args.load_ckpt_name))

    if args.world_size > 1:
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        ddp_model = model

    # args.pretrain_lr = args.pretrain_lr*args.world_size
    if args.warmup_lr:
        optimizer = optim.Adam([{'params': ddp_model.parameters(), 'lr': args.pretrain_lr * utils.warmup_linear(args, 0)}])
    else:
        optimizer = optim.Adam([{'params': ddp_model.parameters(), 'lr': args.pretrain_lr}])

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    data_collator = DataCollatorForSelecting(tokenizer=tokenizer, block_size=args.block_size)
    loss = 0.0
    global_step = 0
    best_acc, best_count = 0.0, 0
    best_model = copy.deepcopy(model)
    for ep in range(args.epochs):
        start_time = time.time()
        ddp_model.train()
        dataset = DatasetForSelecting(tokenizer=tokenizer, file_path=args.train_data_path, neighbor_num=args.neighbor_num, 
            negative_num=args.negative_num, positive_num=args.positive_num, evaluator=args.evaluator_type)
        end.value = False
        dataloader = MultiProcessDataLoader(dataset,
                                           batch_size=args.train_batch_size,
                                           collate_fn=data_collator,
                                           local_rank=local_rank,
                                           world_size=args.world_size,
                                           prefetch_step=global_prefetch_step,
                                           global_end=end)
        
        for step, batch in enumerate(dataloader):

            if args.enable_gpu:
                for k, v in batch.items():
                    if v is not None:
                        batch[k] = v.cuda(non_blocking=True)

            input_ids = batch['input_ids']
            attention_masks = batch['attention_masks']
            sample_weights = batch['sample_weights']
            if args.fp16:
                with autocast():
                    batch_loss = ddp_model(input_ids, attention_masks, sample_weights, args.selector_task)
            else:
                batch_loss = ddp_model(input_ids, attention_masks, sample_weights, args.selector_task)
            loss += batch_loss.item()
            optimizer.zero_grad()
            if args.fp16:
                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                optimizer.step()

            global_step += 1
            if args.warmup_lr:
                optimizer.param_groups[0]['lr'] = args.pretrain_lr * utils.warmup_linear(args, global_step)

            if local_rank == 0 and global_step % args.log_steps == 0:
                logging.info(
                    '[{}] cost_time:{} step:{}, lr:{}, train_loss: {:.5f}'.format(
                        local_rank, time.time() - start_time, global_step, optimizer.param_groups[0]['lr'],
                                    loss / args.log_steps))
                loss = 0.0

            # save model minibatch
            if local_rank == 0 and global_step % args.save_steps == 0:
                ckpt_path = os.path.join(args.model_dir, f'{args.savename}-epoch-{ep}-{global_step}.pt')
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, ckpt_path)
                logging.info(f"Model saved to {ckpt_path}")

                logging.info("Star validation for epoch-{}".format(ep + 1))
                acc = test_single_process(model, args, "valid")
                logging.info("validation time:{}".format(time.time() - start_time))

            dist.barrier()

        logging.info("train time:{}".format(time.time() - start_time))

        # save model last of epoch
        if local_rank == 0 and best_count < 2:
            ckpt_path = os.path.join(args.model_dir, '{}-epoch-{}.pt'.format(args.savename, ep + 1))
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, ckpt_path)
            logging.info(f"Model saved to {ckpt_path}")

            logging.info("Star validation for epoch-{}".format(ep + 1))
            acc = test_single_process(model, args, "valid")
            logging.info("validation time:{}".format(time.time() - start_time))
            if acc > best_acc:
                best_model = copy.deepcopy(model)
                best_acc = acc
                best_count = 0
            else:
                best_count += 1
                if best_count >= 2:
                    start_time = time.time()
                    ckpt_path = os.path.join(args.model_dir, '{}-best.pt'.format(args.savename))
                    torch.save(
                        {
                            'model_state_dict': best_model.state_dict(),
                            'optimizer': optimizer.state_dict()
                        }, ckpt_path)
                    logging.info(f"Model saved to {ckpt_path}")
                    logging.info("Star testing for best")
                    acc = test_single_process(best_model, args, "test")
                    logging.info("test time:{}".format(time.time() - start_time))
                    break
                    # exit()
        dist.barrier()

    if local_rank == 0 and best_count < 2:
        start_time = time.time()
        ckpt_path = os.path.join(args.model_dir, '{}-best.pt'.format(args.savename))
        torch.save(
            {
                'model_state_dict': best_model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, ckpt_path)
        logging.info(f"Model saved to {ckpt_path}")
        logging.info("Star testing for best")
        acc = test_single_process(best_model, args, "test")
        logging.info("test time:{}".format(time.time() - start_time))

    utils.cleanup()


def test_single_process(model, args, mode):
    assert mode in {"valid", "test"}
    model.eval()
    with torch.no_grad():
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        data_collator = DataCollatorForSelecting(tokenizer=tokenizer, block_size=args.block_size)
        if mode == "valid":
            dataset = DatasetForSelecting(tokenizer=tokenizer, file_path=args.valid_data_path, neighbor_num=args.neighbor_num, 
                negative_num=args.negative_num, positive_num=args.positive_num, evaluator=args.evaluator_type)
            dataloader = SingleProcessDataLoader(dataset, batch_size=args.valid_batch_size,
                                                collate_fn=data_collator)
        elif mode == "test":
            dataset = DatasetForSelecting(tokenizer=tokenizer, file_path=args.test_data_path, neighbor_num=args.neighbor_num, 
                negative_num=args.negative_num, positive_num=args.positive_num, evaluator=args.evaluator_type)
            dataloader = SingleProcessDataLoader(dataset, batch_size=args.test_batch_size,
                                                collate_fn=data_collator)

        total_acc = [0, 0]
        for step, batch in enumerate(dataloader):
            if args.enable_gpu:
                for k, v in batch.items():
                    if v is not None:
                        batch[k] = v.cuda(non_blocking=True)

            input_ids = batch['input_ids']
            attention_masks = batch['attention_masks']

            all_nodes_num = input_ids.shape[0]
            batch_size = all_nodes_num // 4
            node_embeds = model.text_encoder(input_ids, attention_mask=attention_masks).view(batch_size, 4, -1) # B*4 D
            query_embeds = node_embeds[:, :1, :] # B 1 D
            key_embeds = node_embeds[:, 1:2, :] # B 1 D
            neighbor_embeds = node_embeds[:, 2: , :] # B 2 D
            if args.selector_task == 'matching':
                expand_query = query_embeds.expand(-1, 2, -1) # B 2 D
                query_neighbor_embeds = model.concat_qn(torch.cat([expand_query, neighbor_embeds], axis=2)) # B 2 D
                new_key_embed = model.concat_qn(torch.cat([key_embeds, key_embeds], axis=2)) # B 1 D
                scores = torch.matmul(new_key_embed, query_neighbor_embeds.transpose(1,2)).squeeze(1) # B 2

            elif args.selector_task == 'similarity':
                scores = torch.matmul(key_embeds, neighbor_embeds.transpose(1,2)).squeeze(1)  # B 2

            acc = torch.sum(scores[:,0] > scores[:,1], axis=0)
            total_acc[0] += acc
            total_acc[1] += scores.size(0)

            # if step == 2: break
            # if step % log_steps == 0 and step > 0:
            #     logging.info('step {} -- acc:{}'.format(step, total_acc[0] / total_acc[1]))

        logging.info('Final -- acc:{}'.format(total_acc[0] / total_acc[1]))
        return total_acc[0] / total_acc[1]


def test(local_rank, args, global_prefetch_step, end):
    utils.setuplogging()
    os.environ["RANK"] = str(local_rank)
    utils.setup(local_rank, args.world_size)

    device = torch.device("cuda", local_rank)

    model = load_bert(args)
    logging.info('loading model: {}'.format(args.model_type))
    model = model.to(device)

    if os.path.exists(args.load_ckpt_name):
        checkpoint = torch.load(args.load_ckpt_name,map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        logging.info('load ckpt:{}'.format(args.load_ckpt_name))

    if args.world_size > 1:
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        ddp_model = model

    ddp_model.eval()
    torch.set_grad_enabled(False)

    if local_rank == 0:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        data_collator = DataCollatorForSelecting(tokenizer=tokenizer, block_size=args.block_size)
        dataset = DatasetForSelecting(tokenizer=tokenizer, file_path=args.test_data_path, neighbor_num=args.neighbor_num, 
            negative_num=args.negative_num, positive_num=args.positive_num, evaluator=args.evaluator_type)
        end.value = False
        dataloader = MultiProcessDataLoader(dataset,
                                           batch_size=args.test_batch_size,
                                           collate_fn=data_collator,
                                           local_rank=local_rank,
                                           world_size=args.world_size,
                                           prefetch_step=global_prefetch_step,
                                           global_end=end)
        start_time = time.time()
        logging.info("Star testing for best")

        total_acc = [0, 0]
        for step, batch in enumerate(dataloader):
            if args.enable_gpu:
                for k, v in batch.items():
                    if v is not None:
                        batch[k] = v.cuda(non_blocking=True)

            input_ids = batch['input_ids']
            attention_masks = batch['attention_masks']

            all_nodes_num = input_ids.shape[0]
            batch_size = all_nodes_num // 4
            node_embeds = model.text_encoder(input_ids, attention_mask=attention_masks).view(batch_size, 4, -1) # B*4 D
            query_embeds = node_embeds[:, :1, :] # B 1 D
            key_embeds = node_embeds[:, 1:2, :] # B 1 D
            neighbor_embeds = node_embeds[:, 2: , :] # B 2 D
            if args.selector_task == 'matching':
                expand_query = query_embeds.expand(-1, 2, -1) # B 2 D
                query_neighbor_embeds = model.concat_qn(torch.cat([expand_query, neighbor_embeds], axis=2)) # B 2 D
                new_key_embed = model.concat_qn(torch.cat([key_embeds, key_embeds], axis=2)) # B 1 D
                scores = torch.matmul(new_key_embed, query_neighbor_embeds.transpose(1,2)).squeeze(1) # B 2

            elif args.selector_task == 'similarity':
                scores = torch.matmul(key_embeds, neighbor_embeds.transpose(1,2)).squeeze(1)  # B 2

            # print("score: ", scores[:10])
            acc = torch.sum(scores[:,0] > scores[:,1], axis=0)
            # print("acc: ", acc)
            total_acc[0] += acc
            total_acc[1] += scores.size(0)

            # if step == 2: break
            if step % args.log_steps == 0 and step > 0:
                logging.info('step {} -- acc:{}'.format(step, total_acc[0] / total_acc[1]))

        logging.info('Final -- acc:{}'.format(total_acc[0] / total_acc[1]))

        logging.info("test time:{}".format(time.time() - start_time))

    utils.cleanup()