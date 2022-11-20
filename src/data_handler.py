import os
import json
import random
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import IterableDataset

from queue import Queue
from filelock import FileLock

import transformers
from transformers.utils import logging
from transformers import BertTokenizerFast
from concurrent.futures import ThreadPoolExecutor

logger = logging.get_logger(__name__)


class DatasetForMatching(IterableDataset):
    def __init__(
            self,
            tokenizer: BertTokenizerFast,
            file_path: str,
            neighbor_num:int,
            overwrite_cache=False,
            tokenizing_batch_size=32768
    ):
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            "cached_{}_{}".format(
                tokenizer.__class__.__name__,
                filename,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"

        # Input file format:
        # One title and one description per line, split by '\t'.
        #
        # Example:
        # US-12509721$$17017982 |*| alberta ferretti girl sweatshirt dark blue size 14 100% cotton  |'|  |'|  |'|  |'|  |'|      US-12520123$$17017982 |*| alberta ferretti girl sweatshirt black size 12 100% paper  |'|  |'|  |'|  |'|  |'|     1
        # catch105168$$6298 |*| karcher g3200xk 3200 psi (gas-cold water) pressure washer w/ kohler engine  |'|  |'|  |'|  |'|  |'|      catchCMFSBMPN-105168$$6298 |*| karcher 1.107-388.0  |'|  |'|  |'|  |'|  |'|      1

        with FileLock(lock_path):
            if os.path.exists(cached_features_file + ".finish") and not overwrite_cache:
                self.data_file = open(cached_features_file, "r", encoding="utf-8")
            else:
                logger.info(f"Creating features from dataset file at {directory}")
                batch_query, batch_key = [], []
                with open(file_path, encoding="utf-8") as f, open(cached_features_file, "w", encoding="utf-8")as fout:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        query_and_nn,key_and_nn=line.strip('\n').split('\t')[:2]
                        for query in query_and_nn.split("|\'|"):
                            query=query.strip()
                            if not query:
                                batch_query.append("")
                            else:
                                # batch_query.append(query.split("|*|")[1].strip())
                                batch_query.append(query)

                        for key in key_and_nn.split("|\'|"):
                            key=key.strip()
                            if not key:
                                batch_key.append("")
                            else:
                                # batch_key.append(key.split("|*|")[1].strip())
                                batch_key.append(key)
                        if len(batch_query) >= tokenizing_batch_size:
                            tokenized_result_query = tokenizer.batch_encode_plus(batch_query,add_special_tokens=False)
                            tokenized_result_key = tokenizer.batch_encode_plus(batch_key,add_special_tokens=False)
                            samples=[[],[]]
                            for j,(tokens_query, tokens_key) in enumerate(zip(tokenized_result_query['input_ids'],
                                                                           tokenized_result_key['input_ids'])):
                                samples[0].append(tokens_query)
                                samples[1].append(tokens_key)
                                if j%(neighbor_num+1)==neighbor_num:
                                    fout.write(json.dumps(samples)+'\n')
                                    samples=[[],[]]
                            batch_query, batch_key = [], []

                    if len(batch_query) > 0:
                        tokenized_result_query = tokenizer.batch_encode_plus(batch_query, add_special_tokens=False)
                        tokenized_result_key = tokenizer.batch_encode_plus(batch_key, add_special_tokens=False)
                        samples = [[], []]
                        for j, (tokens_query, tokens_key) in enumerate(zip(tokenized_result_query['input_ids'],
                                                                           tokenized_result_key['input_ids'])):
                            samples[0].append(tokens_query)
                            samples[1].append(tokens_key)
                            if j % (neighbor_num + 1) == neighbor_num:
                                fout.write(json.dumps(samples) + '\n')
                                samples = [[], []]
                        batch_query, batch_key = [], []
                    logger.info(f"Finish creating")
                with open(cached_features_file + ".finish", "w", encoding="utf-8"):
                    pass
                self.data_file = open(cached_features_file, "r", encoding="utf-8")

    def __iter__(self):
        for line in self.data_file:
            tokens_title = json.loads(line)
            yield tokens_title


@dataclass
class DataCollatorForMatching:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: BertTokenizerFast
    mlm: bool
    neighbor_num:int
    neighbor_mask:bool
    block_size: int
    mlm_probability: float = 0.15

    def __call__(self, samples: List[List[List[List[int]]]]) -> Dict[str, torch.Tensor]:
        input_id_queries=[]
        attention_mask_queries=[]
        mask_queries=[]
        input_id_keys=[]
        attention_mask_keys=[]
        mask_keys=[]
        for i, sample in (enumerate(samples)):
            input_id_queries_and_nn,attention_mask_queries_and_nn,mask_query,input_id_keys_and_nn,attention_mask_keys_and_nn,mask_key = self.create_training_sample(sample)
            input_id_queries.extend(input_id_queries_and_nn)
            attention_mask_queries.extend(attention_mask_queries_and_nn)
            mask_queries.extend(mask_query)
            input_id_keys.extend(input_id_keys_and_nn)
            attention_mask_keys.extend(attention_mask_keys_and_nn)
            mask_keys.extend(mask_key)
        if self.mlm:
            input_id_queries, mlm_labels_queries = self.mask_tokens(self._tensorize_batch(input_id_queries, self.tokenizer.pad_token_id), self.tokenizer.mask_token_id)
            input_id_keys, mlm_labels_keys = self.mask_tokens(self._tensorize_batch(input_id_keys, self.tokenizer.pad_token_id), self.tokenizer.mask_token_id)
        else:
            input_id_queries = self._tensorize_batch(input_id_queries, self.tokenizer.pad_token_id)
            input_id_keys = self._tensorize_batch(input_id_keys, self.tokenizer.pad_token_id)
        mask_queries=torch.tensor(mask_queries)
        mask_keys=torch.tensor(mask_keys)
        return {
            "input_id_query": input_id_queries,
            "attention_masks_query": self._tensorize_batch(attention_mask_queries, 0),
            "masked_lm_labels_query": mlm_labels_queries if self.mlm else None,
            "mask_query":mask_queries,
            "input_id_key": input_id_keys,
            "attention_masks_key": self._tensorize_batch(attention_mask_keys, 0),
            "masked_lm_labels_key": mlm_labels_keys if self.mlm else None,
            "mask_key":mask_keys,
        }

    def _tensorize_batch(self, examples: List[torch.Tensor], padding_value) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            return pad_sequence(examples, batch_first=True, padding_value=padding_value)

    def create_training_sample(self, sample: List[List[List[int]]]):
        """Creates a training sample from the tokens of a title."""

        max_num_tokens = self.block_size - self.tokenizer.num_special_tokens_to_add(pair=False)

        token_queries,token_keys = sample

        mask_queries, mask_keys = [], []
        query_neighbor_list=[]
        key_neighbor_list=[]
        for i, (token_query, token_key) in enumerate(zip(token_queries, token_keys)):
            if len(token_query)==0:
                mask_queries.append(torch.tensor(0))
            else:
                if i!=0: query_neighbor_list.append(i)
                mask_queries.append(torch.tensor(1))
            if len(token_key)==0:
                mask_keys.append(torch.tensor(0))
            else:
                if i!=0: key_neighbor_list.append(i)
                mask_keys.append(torch.tensor(1))

        if self.neighbor_mask:
            if np.random.random() < 0.5:
                mask_query_neighbor_num = min(np.random.randint(1, self.neighbor_num),len(query_neighbor_list))
            else:
                mask_query_neighbor_num = 0
            if np.random.random() < 0.5:
                mask_key_neighbor_num = min(np.random.randint(1, self.neighbor_num),len(key_neighbor_list))
            else:
                mask_key_neighbor_num = 0

            mask_query_set = set(
                np.random.choice(query_neighbor_list, mask_query_neighbor_num, replace=False))
            mask_key_set = set(
                np.random.choice(key_neighbor_list, mask_key_neighbor_num, replace=False))

        input_id_queries,input_id_keys,attention_mask_queries,attention_mask_keys=[],[],[],[]
        for i,(token_query,token_key) in enumerate(zip(token_queries,token_keys)):
            input_id_queries.append(torch.tensor(self.tokenizer.build_inputs_with_special_tokens(token_query[:max_num_tokens])))
            input_id_keys.append(torch.tensor(self.tokenizer.build_inputs_with_special_tokens(token_key[:max_num_tokens])))
            attention_mask_queries.append(torch.tensor([1]*len(input_id_queries[-1])))
            attention_mask_keys.append(torch.tensor([1]*len(input_id_keys[-1])))
            if self.neighbor_mask:
                if i in mask_query_set:
                    mask_queries[i]=torch.tensor(0)
                if i in mask_key_set:
                    mask_keys[i]=torch.tensor(0)

        return input_id_queries,attention_mask_queries,mask_queries,input_id_keys,attention_mask_keys,mask_keys

    def mask_tokens(self, inputs_origin: torch.Tensor, mask_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        """
        inputs = inputs_origin.clone()
        labels = torch.zeros((inputs.shape[0]//(self.neighbor_num+1),inputs.shape[1]),dtype=torch.long)-100
        num=0
        for i, input_origin in enumerate(inputs_origin):
            if i%(self.neighbor_num+1)!=0:continue
            mask_num, valid_length = 0, 0
            start_indexes=[]
            for index, x in enumerate(input_origin):
                if int(x) not in self.tokenizer.all_special_ids:
                    valid_length += 1
                    start_indexes.append(index)
                    labels[num][index] = -99
            random.shuffle(start_indexes)
            if valid_length>0:
                while mask_num / valid_length < self.mlm_probability:
                    start_index = start_indexes.pop()
                    span_length = 1e9
                    while span_length > 10: span_length = np.random.geometric(0.2)
                    for j in range(start_index, min(start_index+span_length,len(input_origin))):
                        if labels[num][j] != -99: continue
                        labels[num][j] = input_origin[j].clone()
                        rand=np.random.random()
                        if rand<0.8:
                            inputs[i][j] = mask_id
                        elif rand<0.9:
                            inputs[i][j]=np.random.randint(0,self.tokenizer.vocab_size-1)
                        mask_num += 1
                        if mask_num / valid_length >= self.mlm_probability:
                            break
            labels[num] = torch.masked_fill(labels[num], labels[num] < 0, -100)
            num+=1
        return inputs, labels


class DatasetForSelecting(IterableDataset):
    def __init__(
        self,
        tokenizer: BertTokenizerFast,
        file_path: str,
        neighbor_num:int,
        negative_num:int,
        positive_num: int,
        evaluator:str,
        overwrite_cache=False,
        tokenizing_batch_size=32768):
        
        self.negative_num = negative_num
        self.positive_num = positive_num
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            "cached_{}_{}".format(
                tokenizer.__class__.__name__,
                filename,
            ),
        )
        cached_evaluate_file = os.path.join(
            directory, "onestep_evaluation_{}_{}".format(evaluator, filename))

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"

        # Input file format:
        # One title and one description per line, split by '\t'.
        #
        # Example:
        # US-12509721$$17017982 |*| alberta ferretti girl sweatshirt dark blue size 14 100% cotton  |'|  |'|  |'|  |'|  |'|       US-12520123$$17017982 |*| alberta ferretti girl sweatshirt black size 12 100% paper  |'|  |'|  |'|  |'|  |'|        1
        # catch105168$$6298 |*| karcher g3200xk 3200 psi (gas-cold water) pressure washer w/ kohler engine  |'|  |'|  |'|  |'|  |'|       catchCMFSBMPN-105168$$6298 |*| karcher 1.107-388.0  |'|  |'|  |'|  |'|  |'|        1

        with FileLock(lock_path):
            if os.path.exists(cached_features_file + ".finish") and not overwrite_cache:
                self.data_file = open(cached_features_file, "r", encoding="utf-8")
            else:
                logger.info(f"Creating features from dataset file at {directory}")
                batch_query, batch_key = [], []
                with open(file_path, encoding="utf-8") as f, open(cached_features_file, "w", encoding="utf-8")as fout:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        query_and_nn,key_and_nn=line.strip('\n').split('\t')[:2]

                        for query in query_and_nn.split("|\'|")[:neighbor_num + 1]:
                            query=query.strip()
                            if not query:
                                batch_query.append("")
                            else:
                                # batch_query.append(query.split("|*|")[1].strip())
                                batch_query.append(query)

                        for key in key_and_nn.split("|\'|")[:neighbor_num + 1]:
                            key=key.strip()
                            if not key:
                                batch_key.append("")
                            else:
                                # batch_key.append(key.split("|*|")[1].strip())
                                batch_key.append(key)
                        if len(batch_query) >= tokenizing_batch_size:
                            tokenized_result_query = tokenizer.batch_encode_plus(batch_query,add_special_tokens=False)
                            tokenized_result_key = tokenizer.batch_encode_plus(batch_key,add_special_tokens=False)
                            samples=[[],[]]
                            for j,(tokens_query, tokens_key) in enumerate(zip(tokenized_result_query['input_ids'],
                                                                           tokenized_result_key['input_ids'])):
                                samples[0].append(tokens_query)
                                samples[1].append(tokens_key)
                                if j%(neighbor_num+1)==neighbor_num:
                                    fout.write(json.dumps(samples)+'\n')
                                    samples=[[],[]]
                            batch_query, batch_key = [], []

                    if len(batch_query) > 0:
                        tokenized_result_query = tokenizer.batch_encode_plus(batch_query, add_special_tokens=False)
                        tokenized_result_key = tokenizer.batch_encode_plus(batch_key, add_special_tokens=False)
                        samples = [[], []]
                        for j, (tokens_query, tokens_key) in enumerate(zip(tokenized_result_query['input_ids'],
                                                                           tokenized_result_key['input_ids'])):
                            samples[0].append(tokens_query)
                            samples[1].append(tokens_key)
                            if j % (neighbor_num + 1) == neighbor_num:
                                fout.write(json.dumps(samples) + '\n')
                                samples = [[], []]
                        batch_query, batch_key = [], []
                    logger.info(f"Finish creating")
                with open(cached_features_file + ".finish", "w", encoding="utf-8"):
                    pass
                self.data_file = open(cached_features_file, "r", encoding="utf-8")

        self.eval_file = open(cached_evaluate_file, "r", encoding="utf-8")

    def process(self, tokens_title, labels):
        token_queries, token_keys = tokens_title
        max_query_neighbor, max_key_neighbor, query_neighbor_sort, key_neighbor_sort, label_query_neighbors, label_key_neighbors = labels
        samples = []
        query, key = token_queries[0], token_keys[0]
        # pos_weights = F.softmax(torch.arange(start=1, end=self.positive_num+1, dtype=float)/4, dim=-1)
        # neg_weights = F.softmax(torch.arange(start=1, end=self.negative_num+1, dtype=float)/4, dim=-1)

        # #### create document pairs: first 5 positive sample with query; query with last 5 negative sample
        # for pid, index in enumerate(key_neighbor_sort[-self.positive_num:][::-1]):
        #     if index == 0: break
        #     if token_keys[index] == []: continue
        #     samples.append([query, token_keys[index], key, pos_weights[-(pid+1)].item()])

        # for nid, index in enumerate(key_neighbor_sort[:self.negative_num]):
        #     if index == 0: break
        #     if token_keys[index] == []: continue
        #     samples.append([query, key, token_keys[index], neg_weights[-(nid+1)].item()])


        # for pid, index in enumerate(query_neighbor_sort[-self.positive_num:][::-1]):
        #     if index == 0: break
        #     if token_queries[index] == []: continue
        #     samples.append([key, token_queries[index], query, pos_weights[-(pid+1)].item()])

        # for nid, index in enumerate(query_neighbor_sort[:self.negative_num]):
        #     if index == 0: break
        #     if token_queries[index] == []: continue
        #     samples.append([key, query, token_queries[index], neg_weights[-(nid+1)].item()])

        #### create document pairs with the first positive sample and the last 5 negative sample
        # print("max_key_neighbor: ", max_key_neighbor, label_key_neighbors[max_key_neighbor])
        if label_key_neighbors[max_key_neighbor]:
            count = 0
            for index in key_neighbor_sort:
                if index == 0 or count >= self.negative_num: break
                # if count >= self.negative_num or index == max_key_neighbor: break
                if token_keys[index] == []:
                    continue
                # print(max_key_neighbor, index)
                samples.append([key, query, token_keys[max_key_neighbor], token_keys[index], 1.0])
                count += 1

        # print("max_query_neighbor: ", max_query_neighbor, label_query_neighbors[max_query_neighbor])
        if label_query_neighbors[max_query_neighbor]:
            count = 0
            for index in query_neighbor_sort:
                if index == 0 or count >= self.negative_num: break
                # if count >= self.negative_num or index == max_query_neighbor: break
                if token_queries[index] == []:
                    continue
                samples.append([query, key, token_queries[max_query_neighbor], token_queries[index], 1.0])
                count += 1

        return samples

    def __iter__(self):
        # print("start processing...")
        for line, label in zip(self.data_file, self.eval_file):
            tokens_title = json.loads(line)
            labels = json.loads(label)
            for sample in self.process(tokens_title, labels):
                yield sample


@dataclass
class DataCollatorForSelecting:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: BertTokenizerFast
    block_size: int

    def __call__(self, samples: List[List[List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = []
        attention_masks = []
        sample_weights = []

        for i, sample in (enumerate(samples)):
            input_id_queries_and_nn, attention_mask_queries, weight = self.create_training_sample(sample)
            input_ids.extend(input_id_queries_and_nn)
            attention_masks.extend(attention_mask_queries)
            sample_weights.append(weight)

        input_ids = self._tensorize_batch(input_ids, self.tokenizer.pad_token_id)
        attention_masks = self._tensorize_batch(attention_masks, 0)
        sample_weights = torch.tensor(sample_weights)
        return {
            "input_ids": input_ids,
            "attention_masks": attention_masks,
            "sample_weights": sample_weights,
        }

    def _tensorize_batch(self, examples: List[torch.Tensor], padding_value) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            return pad_sequence(examples, batch_first=True, padding_value=padding_value)

    def create_training_sample(self, sample: List[List[int]]):
        """Creates a training sample from the tokens of a title."""

        max_num_tokens = self.block_size - self.tokenizer.num_special_tokens_to_add(pair=False) # return the number of added tokens whrn encoding with special tokens
        input_ids, attention_masks, weight = [], [], sample[-1]

        for i, token in enumerate(sample[:-1]):
            input_ids.append(torch.tensor(self.tokenizer.build_inputs_with_special_tokens(token[:max_num_tokens])))
            attention_masks.append(torch.tensor([1]*len(input_ids[-1])))
        return input_ids, attention_masks, weight


@dataclass
class MultiProcessDataLoader:
    dataset: IterableDataset
    batch_size: int
    collate_fn: Callable
    local_rank: int
    world_size: int
    prefetch_step: Any
    global_end: Any
    blocking: bool=False
    drop_last: bool = True

    def _start(self):
        self.local_end=False
        self.aval_count = 0
        self.outputs = Queue(10)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def sync(self):
        while sum(self.prefetch_step) != self.prefetch_step[self.local_rank] * self.world_size:
            if self.end.value: break

    def _produce(self):
        for batch in self._generate_batch():
            self.outputs.put(batch)
            self.aval_count += 1
            # print("produce one batch...")
        self.pool.shutdown(wait=False)
        raise

    def _generate_batch(self):
        batch = []
        for i, sample in enumerate(self.dataset):
            if i % self.world_size != self.local_rank: continue
            batch.append(sample)
            if len(batch)>=self.batch_size:
                # print("generate one batch...")
                yield self.collate_fn(batch[:self.batch_size])
                batch = batch[self.batch_size:]
        else:
            if len(batch) > 0 and not self.drop_last:
                yield self.collate_fn(batch)
                batch = []
        self.local_end=True

    def __iter__(self):
        if self.blocking:
            return self._generate_batch()
        self._start()
        return self

    def __next__(self):
        dist.barrier()
        while self.aval_count == 0:
            if self.local_end or self.global_end.value:
                self.global_end.value=True
                break
        dist.barrier()
        if self.global_end.value:
            raise StopIteration
        next_batch = self.outputs.get()
        self.aval_count -= 1
        return next_batch

@dataclass
class SingleProcessDataLoader:
    dataset: IterableDataset
    batch_size: int
    collate_fn: Callable
    blocking: bool=False
    drop_last: bool = True

    def _start(self):
        self.local_end = False
        self.aval_count = 0
        self.outputs = Queue(10)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def _produce(self):
        for batch in self._generate_batch():
            self.outputs.put(batch)
            self.aval_count += 1
        self.pool.shutdown(wait=False)
        raise

    def _generate_batch(self):
        batch = []
        for i, sample in enumerate(self.dataset):
            batch.append(sample)
            if len(batch)>=self.batch_size:
                yield self.collate_fn(batch[:self.batch_size])
                batch = batch[self.batch_size:]
        else:
            if len(batch) > 0 and not self.drop_last:
                yield self.collate_fn(batch)
                batch = []
        self.local_end=True

    def __iter__(self):
        if self.blocking:
            return self._generate_batch()
        self._start()
        return self

    def __next__(self):
        while self.aval_count==0:
            if self.local_end:raise StopIteration
        next_batch = self.outputs.get()
        self.aval_count -= 1
        return next_batch