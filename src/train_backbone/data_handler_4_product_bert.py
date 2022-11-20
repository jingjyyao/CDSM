import random
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union
from torch.utils.data.dataset import IterableDataset
import torch
from torch.nn.utils.rnn import pad_sequence
import json
import os
from queue import Queue
from filelock import FileLock

from transformers.utils import logging
from transformers import BertTokenizerFast
from concurrent.futures import ThreadPoolExecutor

logger = logging.get_logger(__name__)


class DatasetForSentencePairPrediction(IterableDataset):
    def __init__(
            self,
            tokenizer: BertTokenizerFast,
            file_path: str,
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
        # textured & polished sterling silver woven band ring - size 10	textured and polished sterling silver woven band ring. measures approximately 0.65"w and is not sizeable.
        # pin & sleeve plug 100 amp	male plug 100a 125/250vac 3 pole 4 wire ip67 watertight

        with FileLock(lock_path):
            if os.path.exists(cached_features_file + ".finish") and not overwrite_cache:
                self.data_file = open(cached_features_file, "r", encoding="utf-8")
            else:
                logger.info(f"Creating features from dataset file at {directory}")
                batch_title, batch_description = [], []
                with open(file_path, encoding="utf-8") as f, open(cached_features_file, "w", encoding="utf-8")as fout:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        id,title,description=line.strip('\n').split('\t')
                        batch_title.append(title.strip())
                        batch_description.append(description.strip())
                        if len(batch_title) >= tokenizing_batch_size:
                            tokenized_result_title = tokenizer.batch_encode_plus(batch_title,add_special_tokens=False)
                            tokenized_result_description = tokenizer.batch_encode_plus(batch_description,add_special_tokens=False)
                            for tokens_title, tokens_description in zip(tokenized_result_title['input_ids'],
                                                                           tokenized_result_description['input_ids']):
                                if len(tokens_title) == 0 or len(tokens_description) == 0: continue
                                fout.write(json.dumps([tokens_title,tokens_description])+'\n')
                            batch_title, batch_description = [], []

                    if len(batch_title) > 0:
                        tokenized_result_title = tokenizer.batch_encode_plus(batch_title, add_special_tokens=False)
                        tokenized_result_description = tokenizer.batch_encode_plus(batch_description,
                                                                                   add_special_tokens=False)
                        for tokens_title, tokens_description in zip(tokenized_result_title['input_ids'],
                                                                    tokenized_result_description['input_ids']):
                            if len(tokens_title) == 0 or len(tokens_description) == 0: continue
                            fout.write(json.dumps([tokens_title, tokens_description]) + '\n')
                        batch_title, batch_description = [], []
                    logger.info(f"Finish creating")
                with open(cached_features_file + ".finish", "w", encoding="utf-8"):
                    pass
                self.data_file = open(cached_features_file, "r", encoding="utf-8")

    def __iter__(self):
        for line in self.data_file:
            tokens = json.loads(line)
            yield tokens


@dataclass
class DataCollatorForSentencePairPrediction:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: BertTokenizerFast
    mlm: bool = True
    duplicate_size = 8 #
    title_size: int = 32 # >94%
    description_size:int=128 # >82%
    text_size:int=128 # >69%
    mlm_probability: float = 0.15

    def __call__(self, samples: List[Union[List[List[int]]]]) -> Dict[str, torch.Tensor]:
        input_ids = []
        input_ids_title=[]
        input_ids_front = []
        input_ids_back = []
        attention_masks = []
        attention_masks_title=[]
        attention_masks_front = []
        attention_masks_back = []
        masks_title=[]
        masks_front=[]
        masks_back=[]

        for i, sample in (enumerate(samples)):
            input_id, attention_mask, input_id_title,attention_mask_title,mask_title,input_id_front, attention_mask_front,mask_front, input_id_back, attention_mask_back,mask_back = self.create_training_sample(
                sample)
            input_ids.append(input_id)
            input_ids_title.append(input_id_title)
            input_ids_front.append(input_id_front)
            input_ids_back.append(input_id_back)
            attention_masks.append(attention_mask)
            attention_masks_title.append(attention_mask_title)
            attention_masks_front.append(attention_mask_front)
            attention_masks_back.append(attention_mask_back)
            masks_title.append(mask_title)
            masks_front.append(mask_front)
            masks_back.append(mask_back)
        if self.mlm:
            input_ids, mlm_labels = self.mask_tokens(self._tensorize_batch(input_ids, self.tokenizer.pad_token_id), self.tokenizer.mask_token_id)
        else:
            input_ids = self._tensorize_batch(input_ids, self.tokenizer.pad_token_id)
        input_ids_title=self._tensorize_batch(input_ids_title,self.tokenizer.pad_token_id)
        input_ids_front = self._tensorize_batch(input_ids_front, self.tokenizer.pad_token_id)
        input_ids_back = self._tensorize_batch(input_ids_back, self.tokenizer.pad_token_id)
        masks_title=torch.tensor(masks_title)
        masks_front=torch.tensor(masks_front)
        masks_back=torch.tensor(masks_back)
        return {
            "input_ids": input_ids,
            "attention_masks": self._tensorize_batch(attention_masks, 0),
            "masked_lm_labels": mlm_labels if self.mlm else None,
            "input_ids_title":input_ids_title,
            "attention_masks_title":self._tensorize_batch(attention_masks_title,0),
            "masks_title":masks_title,
            "input_ids_a": input_ids_front,
            "attention_masks_a": self._tensorize_batch(attention_masks_front, 0),
            "masks_a":masks_front,
            "input_ids_b": input_ids_back,
            "attention_masks_b": self._tensorize_batch(attention_masks_back, 0),
            "masks_b":masks_back,
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

        max_title_size = self.title_size - self.tokenizer.num_special_tokens_to_add(pair=False)
        max_description_size = self.description_size - self.tokenizer.num_special_tokens_to_add(pair=False)
        max_text_size = self.text_size - self.tokenizer.num_special_tokens_to_add(pair=False)

        tokens_title, tokens_description = sample
        if tokens_title[:min(self.duplicate_size,len(tokens_title))]==tokens_description[:min(self.duplicate_size,len(tokens_title))]:
            tokens=tokens_description
        else:
            tokens = tokens_title+tokens_description

        mask_title=1
        mask_front=1
        if len(tokens_description)<max_title_size*2:
            tokens_description_front=tokens_description
            tokens_description_back=[]
            max_description_front_size = max_description_size
            max_description_back_size=0
            mask_back=0
        else:
            tokens_description_front=tokens_description[:max_title_size]
            tokens_description_back=tokens_description[max_title_size:]
            max_description_front_size=max_title_size
            max_description_back_size=max_description_size
            mask_back=1



        input_id_title=self.tokenizer.build_inputs_with_special_tokens(tokens_title[:max_title_size])
        input_id_front = self.tokenizer.build_inputs_with_special_tokens(tokens_description_front[:max_description_front_size])
        input_id_back = self.tokenizer.build_inputs_with_special_tokens(tokens_description_back[:max_description_back_size])
        input_id = self.tokenizer.build_inputs_with_special_tokens(tokens[:max_text_size])

        attention_mask_title=[1]*len(input_id_title)
        attention_mask_front = [1] * len(input_id_front)
        attention_mask_back = [1] * len(input_id_back)
        attention_mask = [1] * len(input_id)

        # pad
        # while len(input_id_a) < self.block_size:
        #     input_id_a.append(0)
        #     attention_mask_a.append(0)
        # while len(input_id_b) < self.block_size:
        #     input_id_b.append(0)
        #     attention_mask_b.append(0)
        # while len(input_id) < self.block_size*2:
        #     input_id.append(0)
        #     attention_mask.append(0)

        input_id_title=torch.tensor(input_id_title)
        input_id_front = torch.tensor(input_id_front)
        input_id_back = torch.tensor(input_id_back)
        input_id = torch.tensor(input_id)
        attention_mask_title=torch.tensor(attention_mask_title)
        attention_mask_front = torch.tensor(attention_mask_front)
        attention_mask_back = torch.tensor(attention_mask_back)
        attention_mask = torch.tensor(attention_mask)
        mask_title=torch.tensor(mask_title)
        mask_front=torch.tensor(mask_front)
        mask_back=torch.tensor(mask_back)


        return input_id, attention_mask, input_id_title,attention_mask_title,mask_title,input_id_front, attention_mask_front,mask_front, input_id_back, attention_mask_back,mask_back

    def mask_tokens(self, inputs_origin: torch.Tensor, mask_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        """
        inputs = inputs_origin.clone()
        labels = torch.full_like(inputs_origin, -100)
        for num, input_origin in enumerate(inputs_origin):

            mask_num, valid_length = 0, 0
            start_indexes=[]
            for index, x in enumerate(input_origin):
                if int(x) not in self.tokenizer.all_special_ids:
                    valid_length += 1
                    start_indexes.append(index)
                    labels[num][index] = -99
            random.shuffle(start_indexes)
            while mask_num / valid_length < self.mlm_probability:
                start_index = start_indexes.pop()
                span_length = 1e9
                while span_length > 10: span_length = np.random.geometric(0.2)
                for i in range(start_index, min(start_index+span_length,len(input_origin))):
                    if labels[num][i] != -99: continue
                    labels[num][i] = input_origin[i].clone()
                    rand=np.random.random()
                    if rand<0.8:
                        inputs[num][i] = mask_id
                    elif rand<0.9:
                        inputs[num][i]=np.random.randint(0,self.tokenizer.vocab_size-1)
                    mask_num += 1
                    if mask_num / valid_length >= self.mlm_probability:
                        break
            labels[num] = torch.masked_fill(labels[num], labels[num] < 0, -100)
        return inputs, labels


@dataclass
class DataLoaderForSentencePairPrediction:
    dataset: IterableDataset
    batch_size: int
    collate_fn: Any
    local_rank: int
    world_size: int
    prefetch_step: Any
    end: Any
    drop_last: bool = True

    def _start(self):
        self.end.value = False
        self.aval_count = 0
        self.outputs = Queue(10)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def sync(self):
        while sum(self.prefetch_step) != self.prefetch_step[self.local_rank] * self.world_size:
            if self.end.value: break

    def _produce(self):
        for batch in self._generate_batch():
            self.prefetch_step[self.local_rank] += 1
            self.sync()
            if self.end.value:
                break
            self.outputs.put(batch)
            self.aval_count += 1
            # print(self.local_rank, self.prefetch_step[self.local_rank])
        self.pool.shutdown(wait=False)
        raise

    def _generate_batch(self):
        batch = []
        for i, sample in enumerate(self.dataset):
            if i % self.world_size != self.local_rank: continue
            batch.append(sample)
            if len(batch)>=self.batch_size:
                yield self.collate_fn(batch[:self.batch_size])
                batch = batch[self.batch_size:]
        else:
            if len(batch) > 0 and not self.drop_last:
                yield self.collate_fn(batch)
                batch = []
        self.end.value = True

    def __iter__(self):
        self._start()
        return self

    def __next__(self):
        if self.aval_count == 0 and self.end.value:
            raise StopIteration
        next_batch = self.outputs.get()
        self.outputs.task_done()
        self.aval_count -= 1
        return next_batch