from models.retrieve_modeling import *

class CPLayer(nn.Module):
    def __init__(self,config):
        super(CPLayer,self).__init__()

        self.W = nn.Linear(config.hidden_size, config.hidden_size,bias=False)
        self.a = nn.Linear(config.hidden_size * 2, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.Concat_W = nn.Linear(config.hidden_size * 2, config.hidden_size,bias=False)
    
    def counterpart(self, query_neighbor_embed, query_neighbor_mask, key_embed, select_num, attention_type='dot_product'):
        """
            query_neighbor_embed: Tensor(batch_size, neighbor_num, embed_dim)
            query_neighbor_mask: Tensor(batch_size, neighbor_num)
            key_embed: Tensor(batch_size, 1, embed_dim)
        """
        expand_query_neighbor = query_neighbor_embed.unsqueeze(1).expand(-1, query_neighbor_embed.size(0), -1, -1) # B B N D
        expand_neighbor_mask = query_neighbor_mask.unsqueeze(1).expand(-1, query_neighbor_embed.size(0), -1).clone() # B B N
        if attention_type == 'dot_product':
            center_key = key_embed.unsqueeze(0).expand(key_embed.size(0), -1,-1, -1) # B B 1 D   change to dot product attention
            k2q_attn = torch.matmul(center_key, expand_query_neighbor.transpose(-2, -1)).squeeze(-2) / math.sqrt(query_neighbor_embed.size(-1))  # B B N normaization
        elif attention_type == 'additive':
            center_key = key_embed.unsqueeze(0).expand(key_embed.size(0), -1, query_neighbor_embed.size(1), -1) # B B N D
            cat_key_query = torch.cat([center_key, expand_query_neighbor], dim=-1) # B B N 2*D
            k2q_attn = self.leakyrelu(self.a(cat_key_query).squeeze(-1)) # B B N

        if select_num < expand_neighbor_mask.size(-1):
            k2q_attn = k2q_attn.masked_fill(expand_neighbor_mask==0,float(-1e4))  # B B N  mask empty neighbor at first, then compare all valid neighbors
            attn_mask = torch.argsort(k2q_attn, axis=-1, descending=True)[:,:,select_num:]  # B B N-select_num
            index = torch.arange(0, query_neighbor_embed.size(0))
            index_dim1 = index.unsqueeze(1).unsqueeze(2).expand(-1, attn_mask.size(1), attn_mask.size(2)) # B B start_index
            index_dim2 = index.unsqueeze(0).unsqueeze(2).expand(attn_mask.size(0), -1, attn_mask.size(2)) # B B start_index
            expand_neighbor_mask[index_dim1, index_dim2, attn_mask] = 0

        k2q_attn = k2q_attn.masked_fill(expand_neighbor_mask==0,float(-1e4))  # B B N
        k2q_attn = F.softmax(k2q_attn, dim=2).unsqueeze(2) # B B 1 N
        expand_query_neighbor = expand_query_neighbor.masked_fill(expand_neighbor_mask.unsqueeze(-1)==0,0) # B B N D
        key_attn_query = torch.matmul(k2q_attn, expand_query_neighbor).squeeze(2)  # B B D

        return key_attn_query

    def forward(self, query_node_embed, query_node_mask, key_node_embed, key_node_mask, select_num, attention_type='dot_product'):
        """
            query/key_node_embed: Tensor(batch_size, neighbor_num+1, embed_dim)
            query/key_neighbor_mask: Tensor(batch_size, neighbor_num+1)
        """
        query_node_embed = self.W(query_node_embed)  # B N+1 D
        key_node_embed = self.W(key_node_embed) # B N+1 D

        query_embed = query_node_embed[:,:1] # B 1 D
        query_neighbor_embed = query_node_embed[:,1:] # B N D
        query_neighbor_mask = query_node_mask[:,1:] # B N
        
        key_embed = key_node_embed[:,:1] # B 1 D
        key_neighbor_embed = key_node_embed[:,1:] # B N D
        key_neighbor_mask = key_node_mask[:,1:] # B N

        counterpart_query = self.counterpart(query_neighbor_embed, query_neighbor_mask, key_embed, select_num, attention_type) # B B D
        counterpart_key = self.counterpart(key_neighbor_embed, key_neighbor_mask, query_embed, select_num, attention_type) # B B D

        expand_query = query_embed.expand(-1, query_node_embed.size(0), -1) # B B D
        expand_key = key_embed.expand(-1, key_node_embed.size(0), -1) # B B D

        new_query_embed = self.Concat_W(torch.cat([expand_query, counterpart_query], axis=-1)) # B B D
        new_key_embed = self.Concat_W(torch.cat([expand_key, counterpart_key], axis=-1)) # B B D
        return new_query_embed, new_key_embed


class CounterPartForNeighborPredict(TuringNLRv3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = TuringNLRv3Model(config)
        self.cls = BertLMLoss(config)
        self.cp = CPLayer(config)

        self.init_weights()

    def retrieve_loss(self,q,k):
        score = torch.sum(torch.mul(q, k.transpose(0,1)), dim=2) # B B
        loss = F.cross_entropy(score,torch.arange(start=0, end=score.shape[0],
                                                  dtype=torch.long, device=score.device))
        return loss

    def forward(self,
        input_ids_query,
        attention_masks_query,
        masked_lm_labels_query,
        mask_query,
        input_ids_key,
        attention_masks_key,
        masked_lm_labels_key,
        mask_key,
        neighbor_num,
        mlm_loss=True,
        select_num=5,
        attention_type='dot_product'):
        '''
        Args:
            input_ids: Tensor(batch_size*(neighbor_num+1),seq_length)
            attention_masks: Tensor(batch_size*(neighbor_num+1), seq_length)
            masked_lm_labels: Tensor(batch_size, seq_len)
            mask_query/key: Tensor(batch_size*(neighbor_num+1))
        '''

        all_nodes_num = mask_query.shape[0]
        batch_size = all_nodes_num//(neighbor_num+1)
        neighbor_mask_query = mask_query.view(batch_size,(neighbor_num+1))
        neighbor_mask_key = mask_key.view(batch_size,(neighbor_num+1))

        last_hidden_states_query = self.bert(input_ids_query, attention_mask=attention_masks_query)[0]
        last_hidden_states_key = self.bert(input_ids_key, attention_mask=attention_masks_key)[0]

        #hidden_state:(N,L,D)->(B,L,D)
        query = last_hidden_states_query[::(neighbor_num+1)]
        key = last_hidden_states_key[::(neighbor_num+1)]

        masked_lm_loss = 0
        if mlm_loss:
            masked_lm_loss = self.cls(query,
                                      self.bert.embeddings.word_embeddings.weight,
                                      masked_lm_labels_query)
            masked_lm_loss += self.cls(key,
                                      self.bert.embeddings.word_embeddings.weight,
                                      masked_lm_labels_key)

        node_embed_query = last_hidden_states_query[:,0].view(batch_size,1+neighbor_num,-1) # B N+1 D
        node_embed_key = last_hidden_states_key[:,0].view(batch_size,1+neighbor_num,-1) # B N+1 D

        query, key = self.cp(node_embed_query, neighbor_mask_query, node_embed_key, neighbor_mask_key, select_num, attention_type=attention_type)

        neighbor_predict_loss = self.retrieve_loss(query, key)

        return masked_lm_loss+neighbor_predict_loss

    def onestep_evaluate(self, input_ids_query, attention_masks_query, mask_query,
        input_ids_key, attention_masks_key, mask_key, neighbor_num, aggregation='mean'):
        '''
        Args:
            input_ids: Tensor(batch_size*(neighbor_num+1),seq_length)
            attention_masks: Tensor(batch_size*(neighbor_num+1), seq_length)
            mask_query/key: Tensor(batch_size*(neighbor_num+1))
        '''

        all_nodes_num = mask_query.shape[0]
        batch_size = all_nodes_num//(neighbor_num+1)
        neighbor_mask_query = mask_query.view(batch_size,(neighbor_num+1))
        neighbor_mask_key = mask_key.view(batch_size,(neighbor_num+1))

        last_hidden_states_query = self.bert(input_ids_query, attention_mask=attention_masks_query)[0]
        last_hidden_states_key = self.bert(input_ids_key, attention_mask=attention_masks_key)[0]

        #hidden_state:(N,L,D)->(B,L,D)
        node_embed_query = last_hidden_states_query[:,0].view(batch_size,1+neighbor_num,-1) # B N+1 D
        node_embed_key = last_hidden_states_key[:,0].view(batch_size,1+neighbor_num,-1) # B N+1 D

        node_embed_query = self.cp.W(node_embed_query) # B N+1 D
        node_embed_key = self.cp.W(node_embed_key) # B N+1 D

        query_embed = node_embed_query[:, :1].expand(-1, neighbor_num+1, -1) # B N+1 D
        query_neighbor_embed = node_embed_query.masked_fill(neighbor_mask_query.unsqueeze(-1)==0,0) # B N+1 D
        key_embed = node_embed_key[:, :1].expand(-1, neighbor_num+1, -1) # B N+1 D
        key_neighbor_embed = node_embed_key.masked_fill(neighbor_mask_key.unsqueeze(-1)==0,0) # B N+1 D
        
        k2q_attn = torch.matmul(key_embed[:,:1,:], query_neighbor_embed.transpose(1,2)).squeeze()/math.sqrt(query_neighbor_embed.size(-1)) # B N+1
        q2k_attn = torch.matmul(query_embed[:,:1,:], key_neighbor_embed.transpose(1,2)).squeeze()/math.sqrt(key_neighbor_embed.size(-1)) # B N+1
        k2q_attn = k2q_attn.masked_fill(neighbor_mask_query==0, float(-1e4)) # B N+1
        q2k_attn = q2k_attn.masked_fill(neighbor_mask_key==0, float(-1e4)) # B N+1

        new_query_embed = self.cp.Concat_W(torch.cat([query_embed, query_neighbor_embed], axis=-1)) # B N+1 D
        new_key_embed = self.cp.Concat_W(torch.cat([key_embed, key_neighbor_embed], axis=-1)) # B N+1 D

        query_neighbor_scores = torch.matmul(new_query_embed, new_key_embed[:,:1,:].transpose(1,2)).squeeze(2)  # B N+1
        query_neighbor_scores = query_neighbor_scores.masked_fill(neighbor_mask_query == 0, float(-1e4))
        key_neighbor_scores = torch.matmul(new_key_embed, new_query_embed[:,:1,:].transpose(1,2)).squeeze(2)
        key_neighbor_scores = key_neighbor_scores.masked_fill(neighbor_mask_key == 0, float(-1e4))  # B N+1

        return query_neighbor_scores, key_neighbor_scores, k2q_attn, q2k_attn


    def inference(self, input_ids_query, attention_masks_query, mask_query,
        input_ids_key, attention_masks_key, mask_key, neighbor_num, aggregation='mean'):
        '''
        Args:
            input_ids: Tensor(batch_size*(neighbor_num+1),seq_length)
            attention_masks: Tensor(batch_size*(neighbor_num+1), seq_length)
            masked_lm_labels: Tensor(batch_size, seq_len)
            mask_query/key: Tensor(batch_size*(neighbor_num+1))
        '''

        all_nodes_num = mask_query.shape[0]
        batch_size = all_nodes_num//(neighbor_num+1)
        neighbor_mask_query = mask_query.view(batch_size,(neighbor_num+1))
        neighbor_mask_key = mask_key.view(batch_size,(neighbor_num+1))

        last_hidden_states_query = self.bert(input_ids_query, attention_mask=attention_masks_query)[0]
        last_hidden_states_key = self.bert(input_ids_key, attention_mask=attention_masks_key)[0]

        node_embed_query = last_hidden_states_query[:,0].view(batch_size,1+neighbor_num,-1) # B N+1 D
        node_embed_key = last_hidden_states_key[:,0].view(batch_size,1+neighbor_num,-1) # B N+1 D

        query, key = self.cp(node_embed_query, neighbor_mask_query, node_embed_key, neighbor_mask_key, select_num, attention_type=attention_type)

        return query, key