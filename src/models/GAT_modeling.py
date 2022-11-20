from models.retrieve_modeling import *

class GATLayer(nn.Module):
    def __init__(self,config):
        super(GATLayer,self).__init__()

        self.W = nn.Linear(config.hidden_size, config.hidden_size,bias=False)
        self.a = nn.Linear(config.hidden_size*2, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, node_embed, node_mask):
        '''
            node_embed: Tensor(batch_size, 1+neighbor_num, embed_dim)
        '''
        node_embed = self.W(node_embed) #B 1+N D
        center_node = node_embed[:,:1] #B 1 D
        center_node = center_node.expand(-1,node_embed.size(1),-1) #B 1+N D
        cat_embed = torch.cat([center_node,node_embed],dim=-1)
        e = self.leakyrelu(self.a(cat_embed).squeeze(2)) #B 1+N
        e = e.masked_fill(node_mask==0,float(-1e4))
        attention = F.softmax(e, dim=1) #B 1+N
        new_center_node = torch.matmul(attention.unsqueeze(1),node_embed).squeeze(1) #B D
        return new_center_node


class GATForNeighborPredict(TuringNLRv3PreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.bert = TuringNLRv3Model(config)
        self.cls = BertLMLoss(config)
        self.gat = GATLayer(config)

        self.init_weights()

    def retrieve_loss(self,q,k):
        score = torch.matmul(q, k.transpose(0, 1))
        loss = F.cross_entropy(score, torch.arange(start=0, end=score.shape[0],
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
        mlm_loss=True):
        '''
        Args:
            input_ids: Tensor(batch_size*(neighbor_num+1),seq_length)
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

        node_embed_query = last_hidden_states_query[:,0].view(batch_size,1+neighbor_num,-1)
        query = self.gat(node_embed_query,neighbor_mask_query)

        node_embed_key = last_hidden_states_key[:,0].view(batch_size,1+neighbor_num,-1)
        key = self.gat(node_embed_key,neighbor_mask_key)

        neighbor_predict_loss = self.retrieve_loss(query, key)

        return masked_lm_loss+neighbor_predict_loss

    def onestep_evaluate(self, input_ids_query, attention_masks_query, mask_query, input_ids_key, attention_masks_key, mask_key, neighbor_num, aggregation='mean'):
        """
            function -- evaluate the score of each neighbor in one-step selection
            
            input_ids: Tensor(batch_size*(neighbor_num+1),seq_length)
            attention_masks: Tensor(batch_size*(neighbor_num+1), seq_length)
            mask_query/key: Tensor(batch_size*(neighbor_num+1))
        """

        all_nodes_num = mask_query.shape[0]
        batch_size = all_nodes_num//(neighbor_num+1)
        neighbor_mask_query = mask_query.view(batch_size,(neighbor_num+1)) # B N+1
        neighbor_mask_key = mask_key.view(batch_size,(neighbor_num+1)) # B N+1

        last_hidden_states_query = self.bert(input_ids_query, attention_mask=attention_masks_query)[0] # B*(N+1) L D
        last_hidden_states_key = self.bert(input_ids_key, attention_mask=attention_masks_key)[0] # B*(N+1) L D

        node_embed_query = last_hidden_states_query[:,0].view(batch_size,1+neighbor_num,-1) # B N+1 D
        node_embed_key = last_hidden_states_key[:,0].view(batch_size,1+neighbor_num,-1) # B N+1 D

        query_embed = node_embed_query[:,:1].expand(-1, neighbor_num+1, -1).reshape(batch_size*(1+neighbor_num), 1, -1) # B*N+1 1 D
        query_neighbor_embed = torch.cat([query_embed, last_hidden_states_query[:,:1]], axis=-2) # B*(N+1) 2 D
        query_mask = neighbor_mask_query[:,:1].expand(-1, neighbor_num+1).reshape(batch_size*(1+neighbor_num), 1)
        query_neighbor_mask = torch.cat([query_mask, mask_query.unsqueeze(-1)], axis=-1) # B*(N+1) 2 

        key_embed = node_embed_key[:,:1].expand(-1, neighbor_num+1, -1).reshape(batch_size*(1+neighbor_num), 1, -1) # B*N+1 1 D
        key_neighbor_embed = torch.cat([key_embed, last_hidden_states_key[:,:1]], axis=-2) # B*(N+1) 2 D
        key_mask = neighbor_mask_key[:,:1].expand(-1, neighbor_num+1).reshape(batch_size*(1+neighbor_num), 1)
        key_neighbor_mask = torch.cat([key_mask, mask_key.unsqueeze(-1)], axis=-1) # B*(N+1) 2

        new_query_embed = self.gat(query_neighbor_embed, query_neighbor_mask).view(batch_size, 1+neighbor_num, -1) # B 1+N D
        new_key_embed = self.gat(key_neighbor_embed, key_neighbor_mask).view(batch_size, 1+neighbor_num, -1) # B 1+N D

        query_neighbor_scores = torch.matmul(new_query_embed, new_key_embed[:,:1,:].transpose(1,2)).squeeze(2)  # B 1+N
        query_neighbor_scores = query_neighbor_scores.masked_fill(neighbor_mask_query == 0, float(-1e4))
        key_neighbor_scores = torch.matmul(new_key_embed, new_query_embed[:,:1,:].transpose(1,2)).squeeze(2)  # B 1+N
        key_neighbor_scores = key_neighbor_scores.masked_fill(neighbor_mask_key == 0, float(-1e4))

        return query_neighbor_scores, key_neighbor_scores, None, None

    def inference(self, input_ids_query, attention_masks_query, mask_query,
        input_ids_key, attention_masks_key, mask_key, neighbor_num, aggregation='mean'):
        """
            inference the query and key representations aggregated with neighbors. 
        """

        all_nodes_num = mask_query.shape[0]
        batch_size = all_nodes_num//(neighbor_num+1)
        neighbor_mask_query = mask_query.view(batch_size,(neighbor_num+1))
        neighbor_mask_key = mask_key.view(batch_size,(neighbor_num+1))

        last_hidden_states_query = self.bert(input_ids_query, attention_mask=attention_masks_query)[0]
        last_hidden_states_key = self.bert(input_ids_key, attention_mask=attention_masks_key)[0]

        node_embed_query = last_hidden_states_query[:,0].view(batch_size,1+neighbor_num,-1)
        query = self.gat(node_embed_query,neighbor_mask_query)

        node_embed_key = last_hidden_states_key[:,0].view(batch_size,1+neighbor_num,-1)
        key = self.gat(node_embed_key,neighbor_mask_key)

        return query, key


class GraphSageForNeighborPredict(TuringNLRv3PreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.bert = TuringNLRv3Model(config)
        self.cls = BertLMLoss(config)
        self.graph_transform = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.pooling_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.gat = GATLayer(config)

        self.init_weights()

    def retrieve_loss(self,q,k):
        score = torch.matmul(q, k.transpose(0, 1))
        loss = F.cross_entropy(score,torch.arange(start=0, end=score.shape[0],
                                                  dtype=torch.long, device=score.device))
        return loss

    def aggregation(self, neighbor_embed, neighbor_mask, center_embed=None, aggragation='mean'):
        assert aggragation in('mean','max','gat')
        if aggragation == 'mean':
            neighbor_embed = neighbor_embed.masked_fill(neighbor_mask.unsqueeze(2)==0,0)
            return torch.sum(neighbor_embed,dim=-2)/(torch.sum(neighbor_mask,dim=-1).unsqueeze(1).to(neighbor_embed.dtype)+1e-6)
        elif aggragation == 'max':
            neighbor_embed = F.relu(self.pooling_transform(neighbor_embed))
            neighbor_embed = neighbor_embed.masked_fill(neighbor_mask.unsqueeze(2)==0,0)
            return torch.max(neighbor_embed,dim=-2)[0]
        elif aggragation == 'gat':
            assert center_embed is not None
            node_embed = torch.cat([center_embed.unsqueeze(1),neighbor_embed],dim=1) #B 1+N D
            center_mask = torch.zeros(neighbor_mask.size(0),1,dtype=neighbor_mask.dtype,device=neighbor_mask.device) # B 1
            node_mask = torch.cat([center_mask,neighbor_mask],dim=-1) # B 1+N
            neighbor_embed = self.gat(node_embed, node_mask)
            neighbor_mask = torch.sum(neighbor_mask,dim=-1).unsqueeze(1).expand(-1,neighbor_embed.size(-1)) #B D
            neighbor_embed = neighbor_embed.masked_fill(neighbor_mask==0,0)
            return neighbor_embed

    def graphsage(self,node_embed, node_mask, aggregation):
        neighbor_embed = node_embed[:, 1:]  # B N D
        neighbor_mask = node_mask[:,1:] # B N
        center_embed = node_embed[:,0] #B D
        neighbor_embed = self.aggregation(neighbor_embed, neighbor_mask, center_embed, aggregation)  # B D
        main_embed = torch.cat([center_embed, neighbor_embed], dim=-1)  # B 2D
        main_embed = self.graph_transform(main_embed)
        main_embed = F.relu(main_embed)
        return main_embed

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
        mlm_loss=True,aggregation='mean'):
        '''
        Args:
            input_ids: Tensor(batch_size*(neighbor_num+1),seq_length)
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

        node_embed_query = last_hidden_states_query[:,0].view(batch_size,1+neighbor_num,-1) #B 1+N D
        query = self.graphsage(node_embed_query, neighbor_mask_query, aggregation)

        node_embed_key = last_hidden_states_key[:,0].view(batch_size,1+neighbor_num,-1)
        key = self.graphsage(node_embed_key, neighbor_mask_key, aggregation)
        neighbor_predict_loss = self.retrieve_loss(query, key)
        return masked_lm_loss+neighbor_predict_loss

    def onestep_evaluate(self, input_ids_query, attention_masks_query, mask_query,
        input_ids_key, attention_masks_key, mask_key, neighbor_num, aggregation='mean'):

        all_nodes_num = mask_query.shape[0]
        batch_size = all_nodes_num//(neighbor_num+1)
        neighbor_mask_query = mask_query.view(batch_size,(neighbor_num+1)) # B N+1
        neighbor_mask_key = mask_key.view(batch_size,(neighbor_num+1)) # B N+1

        last_hidden_states_query = self.bert(input_ids_query, attention_mask=attention_masks_query)[0] # B*(N+1) L D
        last_hidden_states_key = self.bert(input_ids_key, attention_mask=attention_masks_key)[0] # B*(N+1) L D

        node_embed_query = last_hidden_states_query[:,0].view(batch_size,1+neighbor_num,-1) # B N+1 D
        node_embed_key = last_hidden_states_key[:,0].view(batch_size,1+neighbor_num,-1) # B N+1 D

        query_embed = node_embed_query[:,:1].expand(-1, neighbor_num+1, -1).reshape(batch_size*(1+neighbor_num), 1, -1) # B*N+1 1 D
        query_neighbor_embed = torch.cat([query_embed, last_hidden_states_query[:,:1]], axis=-2) # B*(N+1) 2 D
        query_mask = neighbor_mask_query[:,:1].expand(-1, neighbor_num+1).reshape(batch_size*(1+neighbor_num), 1)
        query_neighbor_mask = torch.cat([query_mask, mask_query.unsqueeze(-1)], axis=-1) # B*(N+1) 2 

        key_embed = node_embed_key[:,:1].expand(-1, neighbor_num+1, -1).reshape(batch_size*(1+neighbor_num), 1, -1) # B*N+1 1 D
        key_neighbor_embed = torch.cat([key_embed, last_hidden_states_key[:,:1]], axis=-2) # B*(N+1) 2 D
        key_mask = neighbor_mask_key[:,:1].expand(-1, neighbor_num+1).reshape(batch_size*(1+neighbor_num), 1)
        key_neighbor_mask = torch.cat([key_mask, mask_key.unsqueeze(-1)], axis=-1) # B*(N+1) 2

        new_query_embed = self.graphsage(query_neighbor_embed, query_neighbor_mask, aggregation).view(batch_size, 1+neighbor_num, -1) # B 1+N D
        new_key_embed = self.graphsage(key_neighbor_embed, key_neighbor_mask, aggregation).view(batch_size, 1+neighbor_num, -1) # B 1+N D

        query_neighbor_scores = torch.matmul(new_query_embed, new_key_embed[:,:1,:].transpose(1,2)).squeeze(2)  # B 1+N
        query_neighbor_scores = query_neighbor_scores.masked_fill(neighbor_mask_query == 0, float(-1e4))
        key_neighbor_scores = torch.matmul(new_key_embed, new_query_embed[:,:1,:].transpose(1,2)).squeeze(2)  # B 1+N
        key_neighbor_scores = key_neighbor_scores.masked_fill(neighbor_mask_key == 0, float(-1e4))

        return query_neighbor_scores, key_neighbor_scores, None, None

    def inference(self, input_ids_query, attention_masks_query, mask_query,
        input_ids_key, attention_masks_key, mask_key, neighbor_num, aggregation='mean'):

        all_nodes_num = mask_query.shape[0]
        batch_size = all_nodes_num//(neighbor_num+1)
        neighbor_mask_query = mask_query.view(batch_size,(neighbor_num+1))
        neighbor_mask_key = mask_key.view(batch_size,(neighbor_num+1))

        last_hidden_states_query = self.bert(input_ids_query, attention_mask=attention_masks_query)[0]
        last_hidden_states_key = self.bert(input_ids_key, attention_mask=attention_masks_key)[0]

        node_embed_query = last_hidden_states_query[:,0].view(batch_size,1+neighbor_num,-1) #B 1+N D
        query = self.graphsage(node_embed_query, neighbor_mask_query, aggregation)

        node_embed_key = last_hidden_states_key[:,0].view(batch_size,1+neighbor_num,-1)
        key = self.graphsage(node_embed_key, neighbor_mask_key, aggregation)
        
        return query, key