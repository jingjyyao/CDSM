import utils
import numpy as np
import torch
import torch.nn.functional as F
from models.retrieve_modeling import TuringNLRv3PreTrainedModel, TuringNLRv3Model

class OneTowerForNeighborPredict(TuringNLRv3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = TuringNLRv3Model(config)
        self.init_weights()

    def infer_embedding(self,input_ids,attention_mask):
        vec=self.bert(input_ids,attention_mask)[0][:,0,:]
        return vec

    def retrieve_loss(self,q,k):
        score = torch.matmul(q, k.transpose(0, 1))
        loss = F.cross_entropy(score,torch.arange(start=0, end=score.shape[0],
                                                  dtype=torch.long, device=score.device))
        return loss

    def test(self,input_ids_query,attention_mask_query,input_ids_key,attention_mask_key,neighbor_num):
        all_nodes_num = input_ids_query.shape[0]
        batchsize = all_nodes_num // (neighbor_num+1)
        input_ids_query = input_ids_query.reshape(batchsize, -1)
        attention_mask_query = attention_mask_query.reshape(batchsize, -1)
        input_ids_key = input_ids_key.reshape(batchsize, -1)
        attention_mask_key = attention_mask_key.reshape(batchsize, -1)

        query_emb=self.infer_embedding(input_ids_query,attention_mask_query)
        key_emb=self.infer_embedding(input_ids_key,attention_mask_key)
        return query_emb, key_emb

    def forward(self,input_ids_query,attention_mask_query,input_ids_key,attention_mask_key,neighbor_num):
        """
            input_ids_query: batchsize*(neighbor_num+1), seq_length
            attention_mask_query: batchsize*(neighbor_num+1), seq_length
        """
        all_nodes_num = input_ids_query.shape[0]
        batchsize = all_nodes_num // (neighbor_num+1)
        input_ids_query = input_ids_query.reshape(batchsize, -1)
        attention_mask_query = attention_mask_query.reshape(batchsize, -1)
        input_ids_key = input_ids_key.reshape(batchsize, -1)
        attention_mask_key = attention_mask_key.reshape(batchsize, -1)

        query_emb=self.infer_embedding(input_ids_query,attention_mask_query)
        key_emb=self.infer_embedding(input_ids_key,attention_mask_key)
        neighbor_predict_loss = self.retrieve_loss(query_emb, key_emb)
        return neighbor_predict_loss

    def onestep_evaluate(self, input_ids_query, attention_masks_query, mask_query, input_ids_key, 
        attention_masks_key, mask_key, neighbor_num, mask_self_in_graph=False):
        """
            input_ids: Tensor(batch_size*(neighbor_num+1),seq_length)
            attention_masks: Tensor(batch_size*(neighbor_num+1), seq_length)
            mask_query/key: Tensor(batch_size*(neighbor_num+1))
        """
        all_nodes_num = mask_query.shape[0]
        batch_size = all_nodes_num//(neighbor_num+1)

        query_ids = input_ids_query.view(batch_size, neighbor_num+1, -1)[:,:1,:] # B 1 L
        query_ids = query_ids.expand(-1, neighbor_num+1, -1).reshape(batch_size*(neighbor_num+1), -1) # B*(N+1) L
        query_neighbor_ids = torch.cat([query_ids, input_ids_query], axis=1) # B*(N+1) 2*L

        neighbor_mask_query = mask_query.view(batch_size, 1+neighbor_num) # B 1+N

        query_attention = attention_masks_query.view(batch_size, neighbor_num+1, -1)[:,:1,:] # B 1 L
        query_attention = query_attention.expand(-1, neighbor_num+1, -1).reshape(batch_size*(neighbor_num+1), -1) # B*(N+1) L
        query_neighbor_attention = torch.cat([query_attention, attention_masks_query], axis=1) # B*(N+1) 2*L

        key_ids = input_ids_key.view(batch_size, neighbor_num+1, -1)[:,:1,:] # B 1 L
        key_ids = key_ids.expand(-1, neighbor_num+1, -1).reshape(batch_size*(neighbor_num+1), -1) # B*(N+1) L
        key_neighbor_ids = torch.cat([key_ids, input_ids_key], axis=1) # B*(N+1) 2*L

        key_attention = attention_masks_key.view(batch_size, neighbor_num+1, -1)[:,:1,:] # B 1 L
        key_attention = key_attention.expand(-1, neighbor_num+1, -1).reshape(batch_size*(neighbor_num+1), -1) # B*(N+1) L
        key_neighbor_attention = torch.cat([key_attention, attention_masks_key], axis=1) # B*(N+1) 2*L

        neighbor_mask_key = mask_key.view(batch_size, 1+neighbor_num) # B 1+N

        new_query_embed = self.infer_embedding(query_neighbor_ids, query_neighbor_attention).reshape(batch_size, neighbor_num+1, -1) # B*(N+1) D
        new_key_embed = self.infer_embedding(key_neighbor_ids, key_neighbor_attention).reshape(batch_size, neighbor_num+1, -1) # B*(N+1) D

        query_neighbor_scores = torch.matmul(new_query_embed, new_key_embed[:,:1,:].transpose(1,2)).squeeze(2)  # B 1+N
        query_neighbor_scores = query_neighbor_scores.masked_fill(neighbor_mask_query == 0, float(-1e4))
        key_neighbor_scores = torch.matmul(new_key_embed, new_query_embed[:,:1,:].transpose(1,2)).squeeze(2)  # B 1+N
        key_neighbor_scores = key_neighbor_scores.masked_fill(neighbor_mask_key == 0, float(-1e4))

        return query_neighbor_scores, key_neighbor_scores, None, None