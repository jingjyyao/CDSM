import os
import time
import logging
from pathlib import Path
import numpy as np
import random
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn.functional as F
from models.retrieve_modeling import TuringNLRv3ForRetrieveLoss
from models.configuration_tnlrv3 import TuringNLRv3Config
from data_handler_4_product_bert import DatasetForSentencePairPrediction, \
    DataCollatorForSentencePairPrediction,DataLoaderForSentencePairPrediction
from transformers import BertTokenizerFast
from parameters import parse_args
import utils

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size,)
    torch.cuda.set_device(rank)
    # Explicitly setting seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

def cleanup():
    dist.destroy_process_group()


def warmup_linear(args,step):
    if step <= args.warmup_step:
        return max(step,1)/args.warmup_step
    return max(1e-4,(args.schedule_step-step)/(args.schedule_step-args.warmup_step))


def train(local_rank, args, global_prefetch_step, end):

    utils.setuplogging()
    os.environ["RANK"] = str(local_rank)
    setup(local_rank, args.world_size)
    device = torch.device("cuda", local_rank)
    if args.fp16:
        from torch.cuda.amp import autocast
        scaler = torch.cuda.amp.GradScaler()

    config = TuringNLRv3Config.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        output_hidden_states=True)
    model = TuringNLRv3ForRetrieveLoss.from_pretrained(args.model_name_or_path,
                                             from_tf=bool('.ckpt' in args.model_name_or_path),
                                             config=config)
    logging.info('loading model: {}'.format(args.model_type))
    model = model.to(device)
    if args.world_size > 1:
        ddp_model = DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
    else:
        ddp_model = model

    # args.pretrain_lr = args.pretrain_lr*args.world_size
    if args.warmup_lr:
        optimizer = optim.Adam([{'params': ddp_model.parameters(), 'lr': args.pretrain_lr*warmup_linear(args,0)}])
    else:
        optimizer = optim.Adam([{'params': ddp_model.parameters(), 'lr': args.pretrain_lr}])


    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    data_collator = DataCollatorForSentencePairPrediction(tokenizer=tokenizer)
    loss = 0.0
    global_step = 0
    for ep in range(args.epochs):
        start_time = time.time()
        dataset = DatasetForSentencePairPrediction(tokenizer=tokenizer, file_path=args.train_data_path)
        dataloader = DataLoaderForSentencePairPrediction(dataset,
                                                         batch_size=args.batch_size,
                                                         collate_fn=data_collator,
                                                         local_rank=local_rank,
                                                         world_size=args.world_size,
                                                         prefetch_step=global_prefetch_step,
                                                         end=end)
        for step, batch in enumerate(dataloader):
            if args.enable_gpu:
                for k,v in batch.items():
                    batch[k] = v.cuda(non_blocking=True)

            input_ids = batch['input_ids']
            attention_mask = batch['attention_masks']
            masked_lm_labels = batch['masked_lm_labels']
            input_id_title = batch['input_ids_title']
            attention_mask_title = batch['attention_masks_title']
            input_id_front = batch['input_ids_a']
            attention_mask_front = batch['attention_masks_a']
            input_id_back = batch['input_ids_b']
            attention_mask_back = batch['attention_masks_b']
            masks_title = batch['masks_title']
            masks_a = batch['masks_a']
            masks_b = batch['masks_b']

            if args.fp16:
                with autocast():
                    batch_loss = ddp_model(
                        input_ids, attention_mask, masked_lm_labels,
                        input_id_title,attention_mask_title,
                        input_id_front, attention_mask_front,
                        input_id_back, attention_mask_back,
                        masks_title,masks_a, masks_b,
                        mlm_loss=args.mlm_loss)
            else:
                batch_loss = ddp_model(
                    input_ids, attention_mask, masked_lm_labels,
                    input_id_title, attention_mask_title,
                    input_id_front, attention_mask_front,
                    input_id_back, attention_mask_back,
                    masks_title, masks_a, masks_b,
                    mlm_loss=args.mlm_loss)
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
                optimizer.param_groups[0]['lr'] = args.pretrain_lr*warmup_linear(args,global_step)

            if local_rank==0 and global_step % args.log_steps == 0:
                logging.info(
                    '[{}] cost_time:{} step:{}, train_loss: {:.5f}'.format(
                        local_rank, time.time()-start_time, global_step, loss / args.log_steps))
                loss=0.0
            # save model minibatch
            if local_rank == 0 and global_step % args.save_steps == 0:
                ckpt_path = os.path.join(args.model_dir, f'{args.savename}-epoch-{ep + 1}-{global_step}.pt')
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, ckpt_path)
                logging.info(f"Model saved to {ckpt_path}")

            dist.barrier()

        # loss /= (step+1)
        # logging.info('epoch:{}, loss:{}, time:{}'.format(ep+1, loss,time.time()-start_time))

        # save model last of epoch
        if local_rank == 0:
            ckpt_path = os.path.join(args.model_dir, '{}-epoch-{}.pt'.format(args.savename,ep+1))
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, ckpt_path)
            logging.info(f"Model saved to {ckpt_path}")
        logging.info("time:{}".format(time.time()-start_time))

    cleanup()




def test(local_rank, args, global_prefetch_step, end):

    utils.setuplogging()
    os.environ["RANK"] = str(local_rank)
    setup(local_rank, args.world_size)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    device = torch.device("cuda", local_rank)

    config = TuringNLRv3Config.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        output_hidden_states=True)
    model = TuringNLRv3ForRetrieveLoss.from_pretrained(args.model_name_or_path,
                                             from_tf=bool('.ckpt' in args.model_name_or_path),
                                             config=config)
    logging.info('loading model: {}'.format(args.model_type))
    model = model.to(device)

    checkpoint = torch.load(args.load_ckpt_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info('load ckpt:{}'.format(args.load_ckpt_name))

    if args.world_size > 1:
        ddp_model = DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
    else:
        ddp_model = model

    ddp_model.eval()
    torch.set_grad_enabled(False)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = DatasetForSentencePairPrediction(tokenizer=tokenizer, file_path=args.train_data_path)
    data_collator = DataCollatorForSentencePairPrediction(tokenizer=tokenizer)
    dataloader = DataLoaderForSentencePairPrediction(dataset,
                                             batch_size=args.batch_size,
                                             collate_fn=data_collator,
                                             local_rank=local_rank,
                                             world_size=args.world_size,
                                             prefetch_step=global_prefetch_step,
                                             end=end)

    mlm_acc = [0,0]
    retrive_acc_ta = [0,0]
    retrive_acc_tb = [0,0]
    retrive_acc_ab = [0,0]
    for step, batch in enumerate(dataloader):
        if args.enable_gpu:
            for k,v in batch.items():
                batch[k] = v.cuda(non_blocking=True)

        input_ids = batch['input_ids']
        attention_mask = batch['attention_masks']
        masked_lm_labels = batch['masked_lm_labels']
        input_id_title = batch['input_ids_title']
        attention_mask_title = batch['attention_masks_title']
        input_id_front = batch['input_ids_a']
        attention_mask_front = batch['attention_masks_a']
        input_id_back = batch['input_ids_b']
        attention_mask_back = batch['attention_masks_b']
        masks_title = batch['masks_title']
        masks_a = batch['masks_a']
        masks_b = batch['masks_b']

        sequence_output = ddp_model.bert(input_ids, attention_mask=attention_mask)[0]
        mlm_scores = ddp_model.cls.predictions(sequence_output,
                                  ddp_model.bert.embeddings.word_embeddings.weight) #N L V
        # print(mlm_scores[0])
        # print(masked_lm_labels[0])
        # print('-----------------')
        # input()
        hit_num, all_num = compute_acc(mlm_scores,masked_lm_labels)
        mlm_acc[0] += hit_num.data
        mlm_acc[1] += all_num.data

        outputs_title = ddp_model.bert(
            input_id_title,
            attention_mask=attention_mask_title,
        )[0][:, 0, :]
        outputs_front = ddp_model.bert(
            input_id_front,
            attention_mask=attention_mask_front,
        )[0][:, 0, :]
        outputs_back = ddp_model.bert(
            input_id_back,
            attention_mask=attention_mask_back,
        )[0][:, 0, :]

        hit_num, all_num = compute_retrive_acc(outputs_title, outputs_front, masks_title, masks_a)
        retrive_acc_ta[0] += hit_num.data
        retrive_acc_ta[1] += all_num.data

        hit_num, all_num = compute_retrive_acc(outputs_title, outputs_back, masks_title, masks_b)
        retrive_acc_tb[0] += hit_num.data
        retrive_acc_tb[1] += all_num.data

        hit_num, all_num = compute_retrive_acc(outputs_front, outputs_back, masks_a, masks_b)
        retrive_acc_ab[0] += hit_num.data
        retrive_acc_ab[1] += all_num.data

        if step%args.log_steps == 0:
            logging.info('[{}] step:{}, mlm_acc:{}, ta_acc:{}, tb_acc:{}, ab_acc:{}'.format(local_rank,step,
                (mlm_acc[0]/mlm_acc[1]).data, (retrive_acc_ta[0]/retrive_acc_ta[1]).data,
                (retrive_acc_tb[0] / retrive_acc_tb[1]).data, (retrive_acc_ab[0]/retrive_acc_ab[1]).data
            ))

    logging.info('Final-- [{}]  mlm_acc:{}, ta_acc:{}, tb_acc:{}, ab_acc:{}'.format(local_rank,
        (mlm_acc[0] / mlm_acc[1]).data, (retrive_acc_ta[0] / retrive_acc_ta[1]).data,
        (retrive_acc_tb[0] / retrive_acc_tb[1]).data, (retrive_acc_ab[0] / retrive_acc_ab[1]).data
    ))
    cleanup()


def test_qk(local_rank, args, global_prefetch_step, end):
    from data_handler_4_graph_only_title import DatasetForSentencePairPrediction, DataCollatorForSentencePairPrediction, \
        DataLoaderForSentencePairPrediction

    utils.setuplogging()
    os.environ["RANK"] = str(local_rank)
    setup(local_rank, args.world_size)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    device = torch.device("cuda", local_rank)

    config = TuringNLRv3Config.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        output_hidden_states=True)
    model = TuringNLRv3ForRetrieveLoss.from_pretrained(args.model_name_or_path,
                                             from_tf=bool('.ckpt' in args.model_name_or_path),
                                             config=config)
    logging.info('loading model: {}'.format(args.model_type))
    model = model.to(device)

    checkpoint = torch.load(args.load_ckpt_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info('load ckpt:{}'.format(args.load_ckpt_name))

    if args.world_size > 1:
        ddp_model = DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
    else:
        ddp_model = model

    ddp_model.eval()
    torch.set_grad_enabled(False)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = DatasetForSentencePairPrediction(tokenizer=tokenizer, file_path=args.train_data_path,
                                               neighbor_num=args.neighbor_num)
    data_collator = DataCollatorForSentencePairPrediction(tokenizer=tokenizer, block_size=args.block_size,mlm=True)
    dataloader = DataLoaderForSentencePairPrediction(dataset,
                                                     batch_size=args.batch_size,
                                                     collate_fn=data_collator,
                                                     local_rank=local_rank,
                                                     world_size=args.world_size,
                                                     prefetch_step=global_prefetch_step,
                                                     end=end)

    mlm_acc = [0,0]
    retrive_qk = [0,0]
    for step, batch in enumerate(dataloader):
        if args.enable_gpu:
            for k, v in batch.items():
                if v is not None:
                    batch[k] = v.cuda(non_blocking=True)
        input_id_query = batch['input_id_query'] #S L
        attention_masks_query = batch['attention_masks_query'] #S L
        masked_lm_labels_query = batch['masked_lm_labels_query'] #B L
        mask_query = batch['mask_query']
        input_id_query = input_id_query[::(args.neighbor_num + 1)] #B L
        attention_masks_query = attention_masks_query[::(args.neighbor_num + 1)] #B L

        input_id_key = batch['input_id_key']
        attention_masks_key = batch['attention_masks_key']
        masked_lm_labels_key = batch['masked_lm_labels_key']
        mask_key = batch['mask_key']
        input_id_key = input_id_key[::(args.neighbor_num + 1)]  # B L
        attention_masks_key = attention_masks_key[::(args.neighbor_num + 1)]  # B L

        query = ddp_model.bert(input_id_query, attention_mask=attention_masks_query)[0] #B L D
        mlm_scores = ddp_model.cls.predictions(query,
                                  ddp_model.bert.embeddings.word_embeddings.weight) #N L V
        hit_num, all_num = compute_acc(mlm_scores,masked_lm_labels_query)

        # print(mlm_scores[0])
        # print(input_id_query[0])
        # print(masked_lm_labels_query[0])
        # print('-----------------')
        # input()
        mlm_acc[0] += hit_num.data
        mlm_acc[1] += all_num.data
        #
        key = ddp_model.bert(input_id_key, attention_mask=attention_masks_key)[0]
        mlm_scores = ddp_model.cls.predictions(key,
                                  ddp_model.bert.embeddings.word_embeddings.weight) #N L V
        hit_num, all_num = compute_acc(mlm_scores,masked_lm_labels_key)
        mlm_acc[0] += hit_num.data
        mlm_acc[1] += all_num.data

        mask_query = mask_query[::(args.neighbor_num + 1)]
        mask_key = mask_key[::(args.neighbor_num + 1)]
        hit_num, all_num = compute_retrive_acc(query[:,0], key[:,0], mask_q=mask_query, mask_k=mask_key)
        retrive_qk[0] += hit_num.data
        retrive_qk[1] += all_num.data

        if step%args.log_steps == 0:
            logging.info('[{}], step:{}  mlm_acc:{}, qk_acc:{}'.format(local_rank, step,
                (mlm_acc[0]/mlm_acc[1]).data, (retrive_qk[0]/retrive_qk[1]).data
            ))
            # logging.info('[{}], step:{}  qk_acc:{}'.format(local_rank, step,
            #      (retrive_qk[0]/retrive_qk[1]).data
            # ))

        if step==100:break

    logging.info('Final-- [{}]  mlm_acc:{}, qk_acc:{}'.format(local_rank,
        (mlm_acc[0] / mlm_acc[1]).data, (retrive_qk[0] / retrive_qk[1]).data
    ))
    cleanup()


def compute_acc(scores,labels):
    #hit num
    prediction = torch.argmax(scores, dim=-1)  # N L
    hit = (prediction == labels).float()  # N　L
    hit = torch.sum(hit)

    #all num
    labels = labels.masked_fill(labels >= 0, 1)
    labels = labels.masked_fill(labels < 0, 0)
    labels = torch.sum(labels)

    return hit, labels


def compute_retrive_acc(q,k,mask_q=None,mask_k=None,Q=None,K=None):
    score = torch.matmul(q, k.transpose(0, 1)) #N N
    labels = torch.arange(start=0, end=score.shape[0],
                          dtype=torch.long, device=score.device) #N


    if mask_q is not None and mask_k is not None:
        mask = mask_q * mask_k
    elif mask_q is not None:
        mask = mask_q
    elif mask_k is not None:
        mask = mask_k
    else:
        mask = None

    if mask is not None:
        score = score.masked_fill(mask.unsqueeze(0) == 0, float("-inf")) #N N
        labels = labels.masked_fill(mask == 0, -100)

    # See some cases
    # ans=torch.argmax(score,dim=-1)
    # for i in range(labels.shape[0]):
    #     if labels[i]==ans[i]:
    #         print()
    #         continue
    #     _,ind=torch.sort(score[i],descending=True)
    #     rank=torch.argmax((ind==i).int())
    #     print("num: {}".format(i))
    #     print("rank: {}".format((rank+1).item()))
    #
    #     print("label score: {}".format(score[i][labels[i]].item()))
    #     print("ans score: {}".format(score[i][ans[i]].item()))
    #     print(Q[i])
    #     print(K[labels[i]])
    #     print(K[ans[i]])
    #     input()

    return compute_acc(score,labels)


def compute_metrics(q,k,mask_q=None,mask_k=None,Q=None,K=None):
    score = torch.matmul(q, k.transpose(0, 1)) #N N
    labels = torch.arange(start=0, end=score.shape[0],
                          dtype=torch.long, device=score.device) #N


    if mask_q is not None and mask_k is not None:
        mask = mask_q * mask_k
    elif mask_q is not None:
        mask = mask_q
    elif mask_k is not None:
        mask = mask_k
    else:
        mask = None

    if mask is not None:
        score = score.masked_fill(mask.unsqueeze(0) == 0, float("-inf")) #N N
        labels = labels.masked_fill(mask == 0, -100)

    # See some cases
    # ans=torch.argmax(score,dim=-1)
    # for i in range(labels.shape[0]):
    #     if labels[i]==ans[i]:
    #         print()
    #         continue
    #     _,ind=torch.sort(score[i],descending=True)
    #     rank=torch.argmax((ind==i).int())
    #     print("num: {}".format(i))
    #     print("rank: {}".format((rank+1).item()))
    #
    #     print("label score: {}".format(score[i][labels[i]].item()))
    #     print("ans score: {}".format(score[i][ans[i]].item()))
    #     print(Q[i])
    #     print(K[labels[i]])
    #     print(K[ans[i]])
    #     input()

    hit,all_num=compute_acc(score, labels)

    score=score.cpu().numpy()
    labels=F.one_hot(labels)
    labels = labels.cpu().numpy()
    # print(score.shape)
    # print(labels.shape)
    auc_all = [utils.roc_auc_score(labels[i], score[i]) for i in range(labels.shape[0])]
    auc=np.mean(auc_all)
    mrr_all=[utils.mrr_score(labels[i],score[i]) for i in range(labels.shape[0])]
    mrr=np.mean(mrr_all)
    ndcg5_all=[utils.ndcg_score(labels[i],score[i],5) for i in range(labels.shape[0])]
    ndcg5=np.mean(ndcg5_all)
    ndcg10_all = [utils.ndcg_score(labels[i], score[i],10) for i in range(labels.shape[0])]
    ndcg10=np.mean(ndcg10_all)
    ndcg_all=[utils.ndcg_score(labels[i],score[i],labels.shape[1]) for i in range(labels.shape[0])]
    ndcg=np.mean(ndcg_all)
    return hit,all_num, 1,auc,mrr,ndcg,ndcg5,ndcg10



