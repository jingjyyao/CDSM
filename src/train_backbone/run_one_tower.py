import os
import sys
import copy
import time
import logging

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import BertTokenizerFast

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import utils
from models.GAT_modeling import OneTowerForNeighborPredict
from models.configuration_tnlrv3 import TuringNLRv3Config
from data_handler_4_graph_only_title import DatasetForMatching, DataCollatorForMatching, \
    SingleProcessDataLoaderForMatching, MultiProcessDataLoaderForMatching
from run_retrive import compute_acc, compute_retrive_acc, setup, cleanup, warmup_linear, compute_metrics


def load_bert(args):
    config = TuringNLRv3Config.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path)
    model = OneTowerForNeighborPredict.from_pretrained(args.model_name_or_path, config=config)
    return model

def train(local_rank, args, global_prefetch_step, end, load):
    utils.setuplogging()
    os.environ["RANK"] = str(local_rank)
    setup(local_rank, args.world_size)
    device = torch.device("cuda", local_rank)
    if args.fp16:
        from torch.cuda.amp import autocast
        scaler = torch.cuda.amp.GradScaler()

    model = load_bert(args)
    logging.info('loading model: {}'.format(args.model_type))
    model = model.to(device)

    if load:
        checkpoint = torch.load(args.load_ckpt_name, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info('load ckpt:{}'.format(args.load_ckpt_name))

    if args.world_size > 1:
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        ddp_model = model

    # args.pretrain_lr = args.pretrain_lr*args.world_size
    if args.warmup_lr:
        optimizer = optim.Adam([{'params': ddp_model.parameters(), 'lr': args.pretrain_lr * warmup_linear(args, 0)}])
    else:
        optimizer = optim.Adam([{'params': ddp_model.parameters(), 'lr': args.pretrain_lr}])

    datacollator = DataCollatorForMatching(args.block_size)

    loss = 0.0
    global_step = 0
    best_acc, best_count = 0.0, 0
    best_model = copy.deepcopy(model)
    for ep in range(args.epochs):
        start_time = time.time()
        ddp_model.train()
        dataset=DatasetForMatching(file_path=args.train_data_path)
        if args.world_size>1:
            end.value=False
            dataloader = MultiProcessDataLoader(dataset,train_batch_size,datacollator,local_rank,args.world_size,end)
        else:
            dataloader = SingleProcessDataLoader(dataset,train_batch_size,datacollator)
        
        for step, batch in enumerate(dataloader):
            if args.enable_gpu:
                for k, v in batch.items():
                    if v is not None:
                        batch[k] = v.cuda(non_blocking=True)

            input_id_query = batch['input_id_query']
            attention_masks_query = batch['attention_masks_query']
            input_id_key = batch['input_id_key']
            attention_masks_key = batch['attention_masks_key']

            if args.fp16:
                with autocast():
                    batch_loss = ddp_model(input_id_query,
                                        attention_masks_query,
                                        input_id_key,
                                        attention_masks_key,
                                        neighbor_num=args.neighbor_num)
            else:
                batch_loss = ddp_model(input_id_query,
                                    attention_masks_query,
                                    input_id_key,
                                    attention_masks_key,
                                    neighbor_num=args.neighbor_num)

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
                optimizer.param_groups[0]['lr'] = args.pretrain_lr * warmup_linear(args, global_step)

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

            dist.barrier()

        logging.info("train time:{}".format(time.time() - start_time))

        # save model last of epoch
        if local_rank == 0:
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
                    exit()
        dist.barrier()

    if local_rank == 0:
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

    cleanup()


def test_single_process(model, args, mode):
    assert mode in {"valid", "test"}
    model.eval()
    with torch.no_grad():
        datacollator = DataCollatorForMatching(args.block_size)

        if mode == "valid":
            dataset = DatasetForMatching(args.valid_data_path)
            dataloader = SingleProcessDataLoader(dataset,args.valid_batch_size,datacollator)
        elif mode == "test":
            dataset = DatasetForMatching(args.test_data_path)
            dataloader = SingleProcessDataLoader(dataset, args.test_batch_size, datacollator)

        retrive_acc = [0, 0]
        for step, batch in enumerate(dataloader):
            if args.enable_gpu:
                for k, v in batch.items():
                    if v is not None:
                        batch[k] = v.cuda(non_blocking=True)

            input_id_query = batch['input_id_query']
            attention_masks_query = batch['attention_masks_query']
            input_id_key = batch['input_id_key']
            attention_masks_key = batch['attention_masks_key']

            query, key = model.test(input_id_query,
                                    attention_masks_query,
                                    input_id_key,
                                    attention_masks_key,
                                    neighbor_num=args.neighbor_num)
            hit_num, all_num = compute_retrive_acc(query, key)
            retrive_acc[0] += hit_num.item()
            retrive_acc[1] += all_num.item()

            logging.info('Final-- qk_acc:{}'.format(retrive_acc[0] / retrive_acc[1]))
        return retrive_acc[0] / retrive_acc[1]


def test(local_rank, args, global_prefetch_step, end):
    utils.setuplogging()
    os.environ["RANK"] = str(local_rank)
    setup(local_rank, args.world_size)
    device = torch.device("cuda", local_rank)

    model = load_bert(args)
    logging.info('loading model: {}'.format(args.model_type))
    model = model.to(device)

    checkpoint = torch.load(args.load_ckpt_name,map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    logging.info('load ckpt:{}'.format(args.load_ckpt_name))

    if args.world_size > 1:
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        ddp_model = model

    ddp_model.eval()
    torch.set_grad_enabled(False)

    datacollator = DataCollatorForMatching(args.block_size)
    dataset=DatasetForMatching(file_path=args.test_data_path)
    if args.world_size>1:
        end.value=False
        dataloader = MultiProcessDataLoader(dataset,args.test_batch_size,datacollator,local_rank,args.world_size,end)
    else:
        dataloader = SingleProcessDataLoader(dataset,args.test_batch_size,datacollator)

    retrive_acc = [0 for i in range(8)]
    for step, batch in enumerate(dataloader):
        if args.enable_gpu:
            for k, v in batch.items():
                if v is not None:
                    batch[k] = v.cuda(non_blocking=True)

        input_id_query = batch['input_id_query']
        attention_masks_query = batch['attention_masks_query']
        input_id_key = batch['input_id_key']
        attention_masks_key = batch['attention_masks_key']

        query, key = model.test(input_id_query,
                                attention_masks_query,
                                input_id_key,
                                attention_masks_key,
                                neighbor_num=args.neighbor_num)

        results= compute_metrics(query, key)
        for i,x in enumerate(results):
            retrive_acc[i]+=x
        if step % args.log_steps == 0:
            logging.info('[{}] step:{}, qk_acc:{}, auc:{}, mrr:{}, ndcg:{}, ndcg10:{}'.format(local_rank, step, (retrive_acc[0] / retrive_acc[1]).data,retrive_acc[3]/retrive_acc[2],retrive_acc[4]/retrive_acc[2],retrive_acc[5]/retrive_acc[2],retrive_acc[7]/retrive_acc[2]))

    logging.info('Final-- [{}] , qk_acc:{}, auc:{}, mrr:{}, ndcg:{}, ndcg10:{}'.format(local_rank,
                                                             (retrive_acc[0] / retrive_acc[1]).data
                                                             ,retrive_acc[3]/retrive_acc[2],retrive_acc[4]/retrive_acc[2],retrive_acc[5]/retrive_acc[2],retrive_acc[7]/retrive_acc[2]))

    cleanup()
