import os
import torch.nn.functional as F
from models.retrieve_modeling import *
from models.GAT_modeling import GATForNeighborPredict, GraphSageForNeighborPredict
from models.topo_modeling import TopoGramForNeighborPredict
from models.one_tower_modeling import OneTowerForNeighborPredict
from models.counterpart_modeling import CounterPartForNeighborPredict

class Evaluator(nn.Module):
    def __init__(self, evaluator_type, config, evaluator_ckpt):
        super(Evaluator,self).__init__()

        if evaluator_type == 'gat':
            self.evaluator = GATForNeighborPredict(config)
        elif evaluator_type == 'graphsage':
            self.evaluator = GraphSageForNeighborPredict(config)
        elif evaluator_type == 'topogram':
            self.evaluator = TopoGramForNeighborPredict(config)
        elif evaluator_type == 'onetower':
            self.evaluator = OneTowerForNeighborPredict(config)
        elif evaluator_type == 'counterpart':
            self.evaluator = CounterPartForNeighborPredict(config)

        checkpoint = torch.load(evaluator_ckpt)
        source_dict = checkpoint['model_state_dict']
        target_dict = self.evaluator.state_dict()
        for k, v in target_dict.items():
            if k in source_dict:
                target_dict[k] = source_dict[k]
        self.evaluator.load_state_dict(target_dict)
        # self.evaluator.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("loading evaluator ckpt from {}.".format(evaluator_ckpt))

    def evaluate(self, input_ids_query, attention_masks_query, mask_query, 
                input_ids_key, attention_masks_key, mask_key, neighbor_num, aggregation='max'):
        """
        Args:
            input_ids: Tensor(batch_size*(neighbor_num+1),seq_length)
            attention_masks: Tensor(batch_size*(neighbor_num+1), seq_length)
            mask_query/key: Tensor(batch_size*(neighbor_num+1))
        """
        query_neighbor_scores, key_neighbor_scores, k2q_attn, q2k_attn = self.evaluator.onestep_evaluate(input_ids_query, attention_masks_query, mask_query, 
                input_ids_key, attention_masks_key, mask_key, neighbor_num, aggregation)
        return query_neighbor_scores, key_neighbor_scores, k2q_attn, q2k_attn


# build a CNN Model to fit the matching signals
class CounterPartSelector(TuringNLRv3PreTrainedModel):
    def __init__(self, config, model_name_or_path):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.conv = nn.Conv1d(in_channels=config.hidden_size, out_channels=config.hidden_size, kernel_size=3, padding=1)
        self.word_query = nn.parameter.Parameter(nn.init.normal_(torch.FloatTensor(1, config.hidden_size)))
        self.wordPreProject = nn.Linear(config.hidden_size, config.hidden_size)
        self.concat_qn = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.init_weights()
        self.from_pretrianed(model_name_or_path)

    def from_pretrianed(self, pretrained_model_name_or_path):
        pretrain_state_dict = torch.load(pretrained_model_name_or_path)
        model_state_dict = self.embeddings.state_dict()
        
        for k, v in model_state_dict.items():  # shape of embedding layers are the same
            if 'bert.embeddings.' + k in pretrain_state_dict:
                print("loading ", k)
                model_state_dict[k] = pretrain_state_dict['bert.embeddings.' + k]
        self.embeddings.load_state_dict(model_state_dict)

    def load_evaluator(self, load_ckpt_name):  # load the pretrained word embedding from the matching model / original pretrained model
        checkpoint = torch.load(load_ckpt_name)
        check_state_dict = checkpoint['model_state_dict']

        model_state_dict = self.embeddings.state_dict()
        
        for k, v in model_state_dict.items():  # shape of embedding layers are the same
            if 'bert.embeddings.' + k in check_state_dict:
                print("loading ", k)
                model_state_dict[k] = check_state_dict['bert.embeddings.' + k]
        self.embeddings.load_state_dict(model_state_dict)
 
    def text_encoder(self, input_ids, attention_mask):
        """
        Args:
            input_ids: Tensor(batch_size*(neighbor_num+1),seq_length)
            attention_masks: Tensor(batch_size*(neighbor_num+1), seq_length)
        """
        text_embed = self.embeddings(input_ids)[0] # B*(N+1) L D
        conv_text_embed = F.relu(self.conv(text_embed.permute(0,2,1))) # B*(N+1) D L
        word_query = torch.tanh(self.wordPreProject(self.word_query)).unsqueeze(0) # 1 1 D
        word_weight = torch.matmul(word_query, conv_text_embed) # B*(N+1) 1 L
        word_weight = word_weight.masked_fill(attention_mask.unsqueeze(1)==0, -1e4)
        word_attn = F.softmax(word_weight, dim=2).permute(0, 2, 1) # B*(N+1) L 1
        text_repr = torch.matmul(conv_text_embed, word_attn).squeeze(2) # B*(N+1) D
        return text_repr
 
    def forward(self, input_ids, attention_masks, sample_weights, task):
        """
            Args:
                input_ids: Tensor(batch_size*4, seq_length)
                attention_masks: Tensor(batch_size*4, seq_length)
                sample_weights: Tensor(batch_size)

        """
        all_nodes_num = input_ids.shape[0]
        batch_size = all_nodes_num // 4
        node_embeds = self.text_encoder(input_ids, attention_mask=attention_masks).view(batch_size, 4, -1) # B 4 D
        query_embeds = node_embeds[:, :1, :] # B 1 D
        key_embeds = node_embeds[:, 1:2, :] # B 1 D
        neighbor_embeds = node_embeds[:, 2:, :] # B 2 D

        #### compute the final matching score for ranking
        if task == 'matching':
            expand_query = query_embeds.expand(-1, 2, -1) # B 2 D
            query_neighbor_embeds = self.concat_qn(torch.cat([expand_query, neighbor_embeds], axis=2)) # B 2 D
            new_key_embed = self.concat_qn(torch.cat([key_embeds, key_embeds], axis=2)) # B 1 D
            scores = torch.matmul(new_key_embed, query_neighbor_embeds.transpose(1,2)).squeeze(1) # B 2

        elif task == 'similarity':
            scores = torch.matmul(key_embeds, neighbor_embeds.transpose(1,2)).squeeze(1)  # B 2
        
        logits = F.softmax(scores, dim=1)[:,0]  # B
        # loss = -torch.mean(torch.log(logits) * sample_weights)
        loss = F.cross_entropy(scores, torch.zeros(batch_size, dtype=torch.long, device=scores.device))
        return loss


class SelectorPredictor(TuringNLRv3PreTrainedModel):
    def __init__(self, config, selector_ckpt, predictor_ckpt, predictor_type='counterpart'):
        super().__init__(config)
        self.selector = CounterPartSelector(config, selector_ckpt)

        if predictor_type == 'gat':
            self.predictor = GATForNeighborPredict(config)
        elif predictor_type == 'graphsage':
            self.predictor = GraphSageForNeighborPredict(config)
        elif predictor_type == 'topogram':
            self.predictor = TopoGramForNeighborPredict(config)
        elif predictor_type == 'onetower':
            self.predictor = OneTowerForNeighborPredict(config)
        elif predictor_type == 'counterpart':
            self.predictor = CounterPartForNeighborPredict(config)

        selector_checkpoint = torch.load(selector_ckpt)
        self.selector.load_state_dict(selector_checkpoint['model_state_dict'])
        print("loading selector ckpt from {}.".format(selector_ckpt))
        
        predictor_checkpoint = torch.load(predictor_ckpt)
        source_dict = predictor_checkpoint['model_state_dict']
        target_dict = self.predictor.state_dict()
        for k, v in target_dict.items():
            if k in source_dict:
                print("loading ", k)
                target_dict[k] = source_dict[k]
        self.predictor.load_state_dict(target_dict)
        print("loading predictor ckpt from {}.".format(predictor_ckpt))

    ## select effective neighbors with selector --- predict matching probability
    def forward(self, input_ids_query, attention_masks_query, mask_query, input_ids_key, attention_masks_key, mask_key, 
        neighbor_num, select_num, selector_task, attention_type='dot_product',
        predictor_type='counterpart', aggregation='mean', stop_condition=None, args=None):
        '''
        Args:
            input_ids: Tensor(batch_size*(neighbor_num+1),seq_length)
            attention_masks: Tensor(batch_size*(neighbor_num+1), seq_length)
            mask_query/key: Tensor(batch_size*(neighbor_num+1))
        '''
        all_nodes_num = input_ids_query.shape[0]
        batch_size = all_nodes_num // (neighbor_num+1)
        neighbor_mask_query = mask_query.view(batch_size,(neighbor_num+1))
        neighbor_mask_key = mask_key.view(batch_size,(neighbor_num+1))

        query_neighbor_mask = neighbor_mask_query[:, 1:] # B N
        key_neighbor_mask = neighbor_mask_key[:, 1:] # B N        
        expand_query_neighbor_mask = query_neighbor_mask.unsqueeze(1).expand(-1, batch_size, -1).clone()  # B B N
        expand_key_neighbor_mask = key_neighbor_mask.unsqueeze(0).expand(batch_size, -1, -1).clone()  # B B N


        ### First, select effective neighbors with the selector
        if stop_condition != None:
            ## reshape query input
            query_node_embeds = self.selector.text_encoder(input_ids_query, attention_mask=attention_masks_query) # B*(N+1) D
            query_node_embeds = query_node_embeds.view(batch_size, neighbor_num+1, -1) # B N+1 D
            query_embeds = query_node_embeds[:, :1, :]  # B 1 D
            query_neighbor_embeds = query_node_embeds[:, 1:, :] # B N D

            ## reshape key input
            key_node_embeds = self.selector.text_encoder(input_ids_key, attention_mask=attention_masks_key)
            key_node_embeds = key_node_embeds.view(batch_size, neighbor_num+1, -1) # B N+1 D
            key_embeds = key_node_embeds[:, :1, :]  # B 1 D
            key_neighbor_embeds = key_node_embeds[:, 1:, :] # B N D


            qk_match = torch.matmul(query_embeds.squeeze(1), key_embeds.squeeze(1).transpose(1,0)) # B B
            expand_key_embeds = key_embeds.unsqueeze(0).expand(batch_size, -1, -1, -1)  # B B 1 D
            expand_query_embeds = query_embeds.unsqueeze(1).expand(-1, batch_size, -1, -1)  # B B 1 D

            expand_query_neighbor = query_neighbor_embeds.unsqueeze(1).expand(-1, batch_size, -1, -1)  # B B N D
            expand_key_neighbor = key_neighbor_embeds.unsqueeze(0).expand(batch_size, -1, -1, -1) # B B N D


            ## select query neighbors according to different keys
            if selector_task == 'matching':
                temp_query_embeds = query_embeds.unsqueeze(2).expand(-1, batch_size, expand_query_neighbor.size(2), -1) # B B N D
                concat_query_neighbors = torch.cat([temp_query_embeds, expand_query_neighbor], axis=-1) # B B N 2*D
                concat_query_neighbors = self.selector.concat_qn(concat_query_neighbors) # B B N D
                key_embed_for_match = self.selector.concat_qn(torch.cat([expand_key_embeds, expand_key_embeds], axis=-1)) # B B 1 D
                k2q_attn = torch.matmul(key_embed_for_match, concat_query_neighbors.transpose(-2,-1)).squeeze(-2)

            elif selector_task == 'similarity':
                k2q_attn = torch.matmul(expand_key_embeds, expand_query_neighbor.transpose(-2, -1)).squeeze(-2) / math.sqrt(query_embeds.size(-1))  # B B N            

            k2q_attn = k2q_attn.masked_fill(expand_query_neighbor_mask==0, float(-1e4))  # B B N
            if stop_condition == 'query':
                threshold = torch.matmul(query_embeds.squeeze(1), key_embeds.squeeze(1).transpose(0,1)) / math.sqrt(query_embeds.size(-1))  # B B             
                threshold = threshold.unsqueeze(-1) # B B 1
                max_exceed_num = torch.max(torch.sum(k2q_attn>threshold, axis=-1), axis=1)[0]  # B
            elif stop_condition == 'matching_threshold':
                max_exceed_num = torch.max(torch.sum(k2q_attn>threshold, axis=-1), axis=1)[0]  # B
            elif stop_condition == 'softmax_threshold':
                norm_attn = F.softmax(k2q_attn, axis=-1)  # B B N
                max_exceed_num = torch.max(torch.sum(norm_attn>threshold, axis=-1), axis=1)[0]  # B
            elif stop_condition == 'count_threshold':
                attn_sort = torch.sort(k2q_attn.view(k2q_attn.size(0), -1), dim=1, descending=True)[0]  # B B*N
                threshold = attn_sort[:, args.count_threshold].unsqueeze(1).unsqueeze(2) # B 1 1
                max_exceed_num = torch.max(torch.sum(k2q_attn>threshold, axis=-1), axis=1)[0]  # B
            elif stop_condition == 'fixed_num':
                max_exceed_num = [select_num] * query_embeds.size(0)

            k2q_attn_sort = torch.argsort(k2q_attn, axis=-1, descending=True) # B B N
            sample_id = 0
            index = torch.arange(0, query_embeds.size(0)) # B
            for select_num in max_exceed_num:
                select_num = max(1, select_num)
                attn_mask = k2q_attn_sort[sample_id,:,select_num:]
                index_dim2 = index.unsqueeze(1).expand(-1, attn_mask.size(1)) # B
                expand_query_neighbor_mask[sample_id, index_dim2, attn_mask] = 0
                sample_id += 1

            sl_query_neighbor_mask = expand_query_neighbor_mask


            ## select key neighbors according to different querys
            if selector_task == 'matching':
                temp_key_embeds = key_embeds.unsqueeze(2).expand(-1, batch_size, expand_key_neighbor.size(2), -1) # B B N D
                concat_key_neighbors = torch.cat([temp_key_embeds, expand_key_neighbor], axis=-1) # B B N 2*D
                concat_key_neighbors = self.selector.concat_qn(concat_key_neighbors) # B B N D
                query_embed_for_match = self.selector.concat_qn(torch.cat([expand_query_embeds, expand_query_embeds], axis=-1)) # B B 1 D
                q2k_attn = torch.matmul(query_embed_for_match, concat_key_neighbors.transpose(-2,-1)).squeeze(-2)

            elif selector_task == 'similarity':
                q2k_attn = torch.matmul(expand_query_embeds, expand_key_neighbor.transpose(-2, -1)).squeeze(-2) / math.sqrt(query_embeds.size(-1))  # B B N
            
            q2k_attn = q2k_attn.masked_fill(expand_key_neighbor_mask==0, float(-1e4))  # B B N
            if stop_condition == 'query':
                threshold = torch.matmul(query_embeds.squeeze(1), key_embeds.squeeze(1).transpose(0,1)) / math.sqrt(query_embeds.size(-1))  # B B             
                threshold = threshold.unsqueeze(-1) # B B 1
                max_exceed_num = torch.max(torch.sum(q2k_attn>threshold, axis=-1), axis=1)[0]  # B
            elif stop_condition == 'matching_threshold':
                max_exceed_num = torch.max(torch.sum(q2k_attn>threshold, axis=-1), axis=1)[0]  # B
            elif stop_condition == 'softmax_threshold':
                norm_attn = F.softmax(q2k_attn, axis=-1)  # B B N
                max_exceed_num = torch.max(torch.sum(norm_attn>threshold, axis=-1), axis=1)[0]  # B
            elif stop_condition == 'count_threshold':
                attn_sort = torch.sort(q2k_attn.view(q2k_attn.size(0), -1), dim=1, descending=True)[0]  # B B*N
                threshold = attn_sort[:, args.count_threshold].unsqueeze(1).unsqueeze(2) # B 1 1
                max_exceed_num = torch.max(torch.sum(q2k_attn>threshold, axis=-1), axis=1)[0]  # B
            elif stop_condition == 'fixed_num':
                max_exceed_num = [select_num] * query_embeds.size(0)

            q2k_attn_sort = torch.argsort(q2k_attn, axis=-1, descending=True) # B B N
            sample_id = 0
            index = torch.arange(0, query_embeds.size(0)) # B
            for select_num in max_exceed_num:
                select_num = max(1, select_num)
                attn_mask = q2k_attn_sort[sample_id,:,select_num:]
                index_dim2 = index.unsqueeze(1).expand(-1, attn_mask.size(1)) # 
                expand_key_neighbor_mask[sample_id, index_dim2, attn_mask] = 0
                sample_id += 1

            sl_key_neighbor_mask = expand_key_neighbor_mask.transpose(0,1)

        else:
            sl_query_neighbor_mask = expand_query_neighbor_mask
            # sl_bert_query_neighbor_embed = bert_query_neighbor_embed

            sl_key_neighbor_mask = expand_key_neighbor_mask.transpose(0,1)
            # sl_bert_key_neighbor_embed = bert_key_neighbor_embed.transpose(0,1)


        new_query_embed, new_key_embed = self.predictor.inference(input_ids_query, attention_masks_query, sl_query_neighbor_mask,
                                        input_ids_key, attention_masks_key, sl_key_neighbor_mask, neighbor_num, aggregation=aggregation)

        return new_query_embed, new_key_embed   


    def counterpart(self, query_neighbor_embed, query_neighbor_mask, key_embed, attention_type='dot_product'):
        """
            neighbor_embed: Tensor(batch_size, batch_size, neigbor_num, embed_dim)
            neighbor_mask: Tensor(batch_size, batch_size, neighbor_num)
            key_embed: Tensor(batch_size, 1, embed_dim)
        """
        if attention_type == 'dot_product':
            center_key = key_embed.unsqueeze(0).expand(key_embed.size(0), -1,-1, -1) # B B 1 D   change to dot product attention
        elif attention_type == 'additive':
            center_key = key_embed.unsqueeze(0).expand(key_embed.size(0), -1, query_neighbor_embed.size(2), -1) # B B N D

        if attention_type == 'additive':
            cat_key_query = torch.cat([center_key, query_neighbor_embed], dim=-1) # B B N 2*D
            k2q_attn = self.predictor.cp.leakyrelu(self.predictor.cp.a(cat_key_query).squeeze(-1)) # B B N
            
        elif attention_type == 'dot_product':
            """dot product attention"""
            k2q_attn = torch.matmul(center_key, query_neighbor_embed.transpose(-2, -1)).squeeze(-2) / math.sqrt(query_neighbor_embed.size(-1))  # B B N normaization

        k2q_attn = k2q_attn.masked_fill(query_neighbor_mask==0,float(-1e4))  # B B N
        k2q_attn = F.softmax(k2q_attn, dim=2).unsqueeze(2) # B B 1 N
        query_neighbor_embed = query_neighbor_embed.masked_fill(query_neighbor_mask.unsqueeze(-1)==0,0) # B B N D
        key_attn_query = torch.matmul(k2q_attn, query_neighbor_embed).squeeze(2)  # B B D
        return key_attn_query  


    def gat_aggregate(self, node_embed, node_mask):
        """
            node_embed: B B N+1 D
            node_mask: B B N+1
        """
        node_embed = self.predictor.gat.W(node_embed)
        center_node = node_embed[:,:,:1] # B B 1 D
        center_node = center_node.expand(-1, -1, node_embed.size(-2), -1) # B B N+1 D
        cat_embed = torch.cat([center_node, node_embed], dim=-1) # B B N+1 2*D
        e = self.predictor.gat.leakyrelu(self.predictor.gat.a(cat_embed).squeeze(-1)) # B B N+1
        e = e.masked_fill(node_mask==0, float(-1e4))
        attention = F.softmax(e, dim=2) # B B N+1
        node_embed = node_embed.masked_fill(node_mask.unsqueeze(-1)==0, 0) # B B N+1 D
        new_center_node = torch.matmul(attention.unsqueeze(2), node_embed).squeeze(2) # B B D
        return new_center_node 


    def sage_aggregate(self, neighbor_embed, neighbor_mask, center_embed=None, aggregation='mean'):
        """
            neighbor_embed: B B N D
            neighbor_mask: B B N
            center_embed: B B D
        """
        assert aggregation in('mean','max','gat')
        if aggregation == 'mean':
            neighbor_embed = neighbor_embed.masked_fill(neighbor_mask.unsqueeze(3)==0,0) # B B N D
            return torch.sum(neighbor_embed,dim=-2)/(torch.sum(neighbor_mask,dim=-1).unsqueeze(2).to(neighbor_embed.dtype)+1e-6) # B B D
        elif aggregation == 'max':
            neighbor_embed = F.relu(self.predictor.pooling_transform(neighbor_embed))
            neighbor_embed = neighbor_embed.masked_fill(neighbor_mask.unsqueeze(3)==0,0)
            return torch.max(neighbor_embed,dim=-2)[0] # B B D
        elif aggregation == 'gat':
            assert center_embed is not None
            node_embed = torch.cat([center_embed.unsqueeze(2),neighbor_embed],dim=2) #B B 1+N D
            center_mask = torch.zeros(neighbor_mask.size(0),neighbor_mask.size(0),1,dtype=neighbor_mask.dtype,device=neighbor_mask.device) # B B 1
            node_mask = torch.cat([center_mask,neighbor_mask],dim=-1) # B B 1+N
            neighbor_embed = self.gat_aggregate(node_embed, node_mask) # B B D
            neighbor_mask = torch.sum(neighbor_mask,dim=-1).unsqueeze(-1).expand(-1,-1,neighbor_embed.size(-1)) #B B D
            neighbor_embed = neighbor_embed.masked_fill(neighbor_mask==0,0) # B B D
            return neighbor_embed    