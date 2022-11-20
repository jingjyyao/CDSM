import os
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

from transformers import BertTokenizerFast, RobertaTokenizerFast

import utils
from models.Selector_Evaluator import Evaluator
from models.configuration_tnlrv3 import TuringNLRv3Config
from data_handler import DatasetForMatching, DataCollatorForMatching, \
    SingleProcessDataLoader, MultiProcessDataLoader

def load_bert(args):
    config = TuringNLRv3Config.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, output_hidden_states=True)
    config.neighbor_type = args.neighbor_type
    config.mapping_graph = 1 if args.mapping_graph else 0
    config.graph_transform = 1 if args.return_last_station_emb else 0
    model = Evaluator(evaluator_type=args.evaluator_type, config=config, evaluator_ckpt=args.evaluator_ckpt)
    return model

def evaluate(local_rank, args, global_prefetch_step, end, mode='train'):
    utils.setuplogging()
    os.environ["RANK"] = str(local_rank)
    utils.setup(local_rank, args.world_size)
    device = torch.device("cuda", local_rank)
    if args.fp16:
        from torch.cuda.amp import autocast
        scaler = torch.cuda.amp.GradScaler()

    model = load_bert(args)
    logging.info('loading model: {}'.format(args.evaluator_type))
    model = model.to(device)

    if args.world_size > 1:
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        ddp_model = model

    ddp_model.eval()
    torch.set_grad_enabled(False)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    data_collator = DataCollatorForMatching(tokenizer=tokenizer, mlm=args.mlm_loss, neighbor_num=args.neighbor_num,
                                                neighbor_mask=args.neighbor_mask, block_size=args.block_size)
    if mode == 'train':
        dataset = DatasetForMatching(tokenizer=tokenizer, file_path=args.train_data_path, neighbor_num=args.neighbor_num)
    elif mode == 'valid':
        dataset = DatasetForMatching(tokenizer=tokenizer, file_path=args.valid_data_path, neighbor_num=args.neighbor_num)
    else:
        dataset = DatasetForMatching(tokenizer=tokenizer, file_path=args.test_data_path, neighbor_num=args.neighbor_num)

    end.value = False
    dataloader = MultiProcessDataLoader(dataset,
                                       batch_size=args.train_batch_size,
                                       collate_fn=data_collator,
                                       local_rank=local_rank,
                                       world_size=args.world_size,
                                       prefetch_step=global_prefetch_step,
                                       global_end=end)
    if local_rank == 0:
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

            query_neighbor_scores, key_neighbor_scores, k2q_attn, q2k_attn = model.evaluate(input_ids_query,
                                                                                            attention_masks_query,
                                                                                            mask_query,
                                                                                            input_ids_key,
                                                                                            attention_masks_key,
                                                                                            mask_key,
                                                                                            neighbor_num=args.neighbor_num,
                                                                                            aggregation=args.aggregation)


    with open(os.path.join(args.model_dir, '{}-evaluate.pt'.format(args.savename))) as fw:
        json.dump(neighbor_score, fw)

def evaluate_single_process(args, mode):
    assert mode in {"train", "valid", "test"}
    directory, filename = os.path.split(args.train_data_path)
    if args.evaluator_type == 'graphsage':
        fout = open(os.path.join(directory, "onestep_evaluation_{}_{}_{}.tsv".format(args.evaluator_type, args.aggregation, mode)), "w", encoding="utf-8")
    else:
        fout = open(os.path.join(directory, "onestep_evaluation_{}_{}.tsv".format(args.evaluator_type, mode)), "w", encoding="utf-8")

    utils.setuplogging()
    device = torch.device("cuda", 0)
    if args.fp16:
        from torch.cuda.amp import autocast
        scaler = torch.cuda.amp.GradScaler()

    model = load_bert(args)
    logging.info('loading model: {}'.format(args.evaluator_type))
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        data_collator = DataCollatorForMatching(tokenizer=tokenizer, mlm=args.mlm_loss, neighbor_num=args.neighbor_num,
                                                neighbor_mask=args.neighbor_mask, block_size=args.block_size)

        if mode == 'train':
            dataset = DatasetForMatching(tokenizer=tokenizer, file_path=args.train_data_path, neighbor_num=args.neighbor_num)
        elif mode == 'valid':
            dataset = DatasetForMatching(tokenizer=tokenizer, file_path=args.valid_data_path, neighbor_num=args.neighbor_num)
        else:
            dataset = DatasetForMatching(tokenizer=tokenizer, file_path=args.test_data_path, neighbor_num=args.neighbor_num)

        dataloader = SingleProcessDataLoader(dataset, batch_size=args.test_batch_size, collate_fn=data_collator)

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

            query_neighbor_scores, key_neighbor_scores, k2q_attns, q2k_attns = model.evaluate(input_ids_query,
                                                                                            attention_masks_query,
                                                                                            mask_query,
                                                                                            input_ids_key,
                                                                                            attention_masks_key,
                                                                                            mask_key,
                                                                                            neighbor_num=args.neighbor_num,
                                                                                            aggregation=args.aggregation)
            
            

            query_neighbor_scores = query_neighbor_scores.cpu().numpy()
            key_neighbor_scores = key_neighbor_scores.cpu().numpy()
            
            # for k2q_attn, q2k_attn in zip(k2q_attns, q2k_attns):
            #     max_query_neighbor = int(np.argmax(k2q_attn))
            #     max_key_neighbor = int(np.argmax(q2k_attn))

            #     query_neighbor_sort = np.argsort(k2q_attn).astype(int).tolist()
            #     key_neighbor_sort = np.argsort(q2k_attn).astype(int).tolist()

            #     query_neighbor_score = (k2q_attn > k2q_attn[0]).astype(int).tolist()
            #     key_neighbor_score = (q2k_attn > q2k_attn[0]).astype(int).tolist()
            #     fout.write(json.dumps([max_query_neighbor, max_key_neighbor, query_neighbor_sort, key_neighbor_sort, query_neighbor_score, key_neighbor_score]) + '\n')


            # sample 1 positive and 5 negative
            for query_neighbor_score, key_neighbor_score in zip(query_neighbor_scores, key_neighbor_scores):
                max_query_neighbor = int(np.argmax(query_neighbor_score))
                max_key_neighbor = int(np.argmax(key_neighbor_score))

                query_neighbor_sort = np.argsort(query_neighbor_score).astype(int).tolist()
                key_neighbor_sort = np.argsort(key_neighbor_score).astype(int).tolist()

                query_neighbor_score = (query_neighbor_score > query_neighbor_score[0]).astype(int).tolist()
                key_neighbor_score = (key_neighbor_score > key_neighbor_score[0]).astype(int).tolist()
                fout.write(json.dumps([max_query_neighbor, max_key_neighbor, query_neighbor_sort, key_neighbor_sort, query_neighbor_score, key_neighbor_score]) + '\n')

        fout.close()
