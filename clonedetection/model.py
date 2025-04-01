import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        #self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x,fin_bsfc_len):
        packed_sequence = pack_padded_sequence(x,fin_bsfc_len, batch_first=True, enforce_sorted=False)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # 
        out, _ = self.gru(packed_sequence, h0.detach())
        output, _ = pad_packed_sequence(out, batch_first=True)

        # 
        output = output[:, -1, :]
        return output


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args,all_len,bsf_len,hid_size,pad_id,bsf_context_length):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
        self.all_len = all_len
        self.bsf_len = bsf_len
        self.hid_size = hid_size
        self.pad_id = pad_id
        self.bsf_context_length = bsf_context_length
        self.alpa = 0.2
        self.grumodel = GRUNet(input_size=self.hid_size, hidden_size=self.hid_size, num_layers=1, output_size=self.hid_size)
        self.num_attention_heads = 4
        self.attention_head_sizess = int(self.hid_size/self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_sizess
        self.Q = nn.Linear(self.hid_size, self.hid_size)
        self.K = nn.Linear(self.hid_size, self.hid_size)
        self.V = nn.Linear(self.hid_size, self.hid_size)
        self.ff = nn.Linear(self.hid_size, self.hid_size)
        self.dropout = nn.Dropout(0.2)

    def att(self,input):
        input_Q = self.Q(input)
        input_k = self.K(input)
        input_v = self.V(input)
        head_size = input_Q.shape[-1]
        
        query_layer = self.transpose_for_scores(input_Q)
        key_layer = self.transpose_for_scores(input_k)
        value_layer = self.transpose_for_scores(input_v)
        
        attention_scores = torch.matmul(query_layer,key_layer.transpose(-1, -2))
        attention_scores = (attention_scores / math.sqrt(self.attention_head_sizess))
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)   
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_sizess)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
                        
    def forward(self, inputs_ids_1,position_idx_1,attn_mask_1,New_DFG_ids_1,inputs_ids_2,position_idx_2,attn_mask_2,New_DFG_ids_2,labels=None): 
        bs,l=inputs_ids_1.size()

        inputs_ids=torch.cat((inputs_ids_1.unsqueeze(1),inputs_ids_2.unsqueeze(1)),1).view(bs*2,l)
        position_idx=torch.cat((position_idx_1.unsqueeze(1),position_idx_2.unsqueeze(1)),1).view(bs*2,l)
        attn_mask=torch.cat((attn_mask_1.unsqueeze(1),attn_mask_2.unsqueeze(1)),1).view(bs*2,l,l)
        New_DFG_ids = torch.cat((New_DFG_ids_1,New_DFG_ids_2),0)
        inputs_embeddings,position_idx,attn_mask = self.process_sg(inputs_ids,position_idx,attn_mask,New_DFG_ids,bs*2)


        #print("outputs***************sucess***************")
        #print("inputs_embeddings",inputs_embeddings.shape)
        #print("attention_mask",attn_mask.shape)
        #print("position_ids",position_idx.shape)
        #print("token_type_ids",(position_idx.eq(-1).long()).shape)
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask =attn_mask ,position_ids=position_idx,token_type_ids=position_idx.eq(-1).long())[0]
        #print("outputs***************sucess2222222***************")
        logits=self.classifier(outputs)
        # shape: [batch_size, num_classes]
        prob=F.softmax(logits, dim=-1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob
      
        
    def process_sg(self, code_inputs=None, position_idx=None,attn_mask=None,New_DFG_ids = None,bs = None): 
        #embedding
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)    
        DFG_mask= nodes_mask.int()
        DFG_index = torch.sum(position_idx>=2, dim=1)
        DFG_len = torch.sum(position_idx ==0, dim=1)
        DFG_max_len = torch.max(DFG_len)
        fin_DFG_ids = []
        fin_DFG_mask = []
        fin_POS_emb = []
        for  loc,index in enumerate(DFG_index):
        	tmp = (New_DFG_ids[loc][index:][:DFG_max_len]).tolist()
        	if len(tmp)<DFG_max_len:
        		tmp.extend([[self.pad_id]*self.bsf_context_length]*(DFG_max_len - len(tmp)))
        	fin_DFG_ids.append(tmp)
        	fin_POS_emb.append([index for index in range(DFG_max_len)])
        fin_DFG_ids = torch.tensor(fin_DFG_ids).cuda()
        fin_POS_emb = torch.tensor(fin_POS_emb).cuda()
        bsfc_len = torch.sum((fin_DFG_ids.reshape(-1,self.bsf_context_length))!=1, dim=-1).to(torch.int64)
        fin_bsfc_len = torch.where(bsfc_len == 0, bsfc_len + 1, bsfc_len)
        fin_bsfc_len = fin_bsfc_len.cpu()
        DFG_embeddings=self.encoder.roberta.embeddings.word_embeddings(fin_DFG_ids).reshape(-1,self.bsf_context_length,self.hid_size)
        DFG_pos_embeddings = self.encoder.roberta.embeddings.position_embeddings(fin_POS_emb) 
        DFGout= self.grumodel(DFG_embeddings,fin_bsfc_len)
        DFGout = DFGout.unsqueeze(0)
        DFGout = DFGout.reshape(bs,-1,self.hid_size)+DFG_pos_embeddings
        		        
        inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(code_inputs)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        for index,i in enumerate(DFG_index):
        	avg_embeddings[index][i:i+DFG_len[index]] =(1-self.alpa)*avg_embeddings[index][i:i+DFG_len[index]]+self.alpa*DFGout[index][:DFG_len[index]]
        avg_embeddings = self.att(avg_embeddings)
        avg_embeddings = self.dropout(self.ff(avg_embeddings))                		
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
        return inputs_embeddings,position_idx,attn_mask