import os
import sys
import copy
import json
import time
import pickle
import logging
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Callable

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import utils
from models.Selector_Evaluator import SelectorPredictor
from models.configuration_tnlrv3 import TuringNLRv3Config
from data_handler import DatasetForMatching, DataCollatorForMatching, \
    SingleProcessDataLoader, MultiProcessDataLoader

def load_bert(args):
    # config = TuringNLRv3Config.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, output_hidden_states=True, num_hidden_layers=args.num_hidden_layers, hidden_size=args.hidden_size)
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else "roberta-base", num_hidden_layers=args.num_hidden_layers, hidden_size=args.hidden_size, output_hidden_states=True)
    config.neighbor_type = 0
    config.mapping_graph = 0
    config.graph_transform = 0
    model = SelectorPredictor(config, args, selector_ckpt=args.selector_ckpt, predictor_ckpt=args.predictor_ckpt, predictor_type=args.predictor_type)
    return model

def predict(local_rank, args, global_prefetch_step, end, mode='test'):
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

    if args.world_size > 1:
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        ddp_model = model

    ddp_model.eval()
    torch.set_grad_enabled(False)

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    data_collator = DataCollatorForMatching(tokenizer=tokenizer, mlm=args.mlm_loss, neighbor_num=args.neighbor_num, neighbor_mask=args.neighbor_mask, block_size=args.block_size)
    if mode == 'train':
        dataset = DatasetForMatching(tokenizer=tokenizer, file_path=args.train_data_path, total_neighbor_num=args.total_neighbor_num)
    elif mode == 'valid':
        dataset = DatasetForMatching(tokenizer=tokenizer, file_path=args.valid_data_path, total_neighbor_num=args.total_neighbor_num)
    else:
        dataset = DatasetForMatching(tokenizer=tokenizer, file_path=args.test_data_path, total_neighbor_num=args.total_neighbor_num)

    end.value = False
    dataloader = MultiProcessDataLoader(dataset, batch_size=args.test_batch_size,collate_fn=data_collator, local_rank=local_rank, world_size=args.world_size, global_end=end)
    if local_rank == 0:
        retrive_acc = [0 for i in range(8)]
        for step, batch in enumerate(dataloader):
            if args.enable_gpu:
                for k, v in batch.items():
                    if v is not None:
                        batch[k] = v.cuda(non_blocking=True)

            input_ids_query = batch['input_id_query']
            attention_masks_query = batch['attention_masks_query']
            mask_query = batch['mask_query']
            input_ids_key = batch['input_id_key']
            attention_masks_key = batch['attention_masks_key']
            mask_key = batch['mask_key']

            query, key = ddp_model(input_ids_query,
                                attention_masks_query,
                                mask_query,
                                input_ids_key,
                                attention_masks_key,
                                mask_key,
                                neighbor_num=args.neighbor_num,
                                select_num=args.select_num,
                                selector_task=args.selector_task,
                                basic=args.basic,
                                attention_type=args.attention_type,
                                predictor_type=args.predictor_type,
                                aggregation=args.aggregation,
                                stop_condition=args.stop_condition,
                                args=args)

            mask_query = mask_query[::(args.neighbor_num + 1)]
            mask_key = mask_key[::(args.neighbor_num + 1)]

            results= compute_metrics(query, key, mask_q=mask_query, mask_k=mask_key)
            for i,x in enumerate(results):
                retrive_acc[i]+=x

        logging.info('Final-- qk_acc:{}, auc:{}, mrr:{}, ndcg:{}, ndcg10:{}'.format(
                                                             (retrive_acc[0] / retrive_acc[1]).data,retrive_acc[3]/retrive_acc[2],retrive_acc[4]/retrive_acc[2],retrive_acc[5]/retrive_acc[2],retrive_acc[6]/retrive_acc[2]))
    utils.cleanup()

def predict_single_process(args, mode):
    assert mode in {"valid", "test"}
    utils.setuplogging()
    device = torch.device("cuda", 0)
    if args.fp16:
        from torch.cuda.amp import autocast
        scaler = torch.cuda.amp.GradScaler()

    model = load_bert(args)
    logging.info('loading model: {}'.format(args.model_type))
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        data_collator = DataCollatorForMatching(tokenizer=tokenizer, mlm=args.mlm_loss, neighbor_num=args.neighbor_num,neighbor_mask=args.neighbor_mask, block_size=args.block_size)

        if mode == 'train':
            dataset = DatasetForMatching(tokenizer=tokenizer, file_path=args.train_data_path, total_neighbor_num=args.total_neighbor_num)
        elif mode == 'valid':
            dataset = DatasetForMatching(tokenizer=tokenizer, file_path=args.valid_data_path, total_neighbor_num=args.total_neighbor_num)
        else:
            dataset = DatasetForMatching(tokenizer=tokenizer, file_path=args.test_data_path, total_neighbor_num=args.total_neighbor_num)

        dataloader = SingleProcessDataLoader(dataset, batch_size=args.test_batch_size, collate_fn=data_collator)

        retrive_acc = [0 for i in range(8)]
        for step, batch in enumerate(dataloader):
            if args.enable_gpu:
                for k, v in batch.items():
                    if v is not None:
                        batch[k] = v.cuda(non_blocking=True)

            input_ids_query = batch['input_id_query']
            attention_masks_query = batch['attention_masks_query']
            mask_query = batch['mask_query']
            input_ids_key = batch['input_id_key']
            attention_masks_key = batch['attention_masks_key']
            mask_key = batch['mask_key']

            query, key = model(input_ids_query,
                            attention_masks_query,
                            mask_query,
                            input_ids_key,
                            attention_masks_key,
                            mask_key,
                            neighbor_num=args.neighbor_num,
                            select_num=args.select_num,
                            selector_task=args.selector_task,
                            basic=args.basic,
                            attention_type=args.attention_type,
                            predictor_type=args.predictor_type,
                            aggregation=args.aggregation,
                            stop_condition=args.stop_condition,
                            args=args)

            mask_query = mask_query[::(args.neighbor_num + 1)]
            mask_key = mask_key[::(args.neighbor_num + 1)]

            results= compute_metrics(query, key, mask_q=mask_query, mask_k=mask_key)
            for i,x in enumerate(results):
                retrive_acc[i]+=x
            if step % args.log_steps == 0:
                logging.info('step:{}, qk_acc:{}, auc:{}, mrr:{}, ndcg:{}, ndcg10:{}'.format(step, (retrive_acc[0] / retrive_acc[1]).data, retrive_acc[3]/retrive_acc[2], retrive_acc[4]/retrive_acc[2], retrive_acc[5]/retrive_acc[2], retrive_acc[6]/retrive_acc[2]))

        logging.info('Final-- qk_acc:{}, auc:{}, mrr:{}, ndcg:{}, ndcg10:{}'.format((retrive_acc[0] / retrive_acc[1]).data,retrive_acc[3]/retrive_acc[2],retrive_acc[4]/retrive_acc[2],retrive_acc[5]/retrive_acc[2],retrive_acc[6]/retrive_acc[2]))

def compute_metrics(q, k, mask_q=None, mask_k=None, Q=None, K=None):
    if len(q.size()) == 3:
        score = torch.sum(torch.mul(q, k.transpose(0,1)), dim=2) # B B
    elif len(q.size()) == 2:
        score = torch.matmul(q, k.transpose(0,1)) # B B
    labels = torch.arange(start=0, end=score.shape[0], dtype=torch.long, device=score.device) #N

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
        masked_labels = labels.masked_fill(mask == 0, -100)

    hit, all_num=utils.compute_acc(score, masked_labels)

    score=score.cpu().numpy()
    labels=F.one_hot(labels)
    labels = labels.cpu().numpy()
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
    return hit, all_num, 1, auc, mrr, ndcg, ndcg5, ndcg10