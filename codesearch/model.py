import torch.nn as nn
import torch
import math

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        #self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        self.gru.flatten_parameters()
        # 初始化隐状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播，返回输出和最终的隐状态
        out, _ = self.gru(x, h0.detach())

        # 选择序列中的最后一个时间步的输出作为预测
        out = out[:, -1, :]
        return out
    
class Model(nn.Module):   
    def __init__(self, encoder,all_len,bsf_len,hid_size,pad_id,bsf_context_length):
        super(Model, self).__init__()
        self.encoder = encoder
        self.all_len = all_len
        self.bsf_len = bsf_len
        self.hid_size = hid_size
        self.pad_id = pad_id
        self.bsf_context_length = bsf_context_length
        self.alpa = 0.6
        self.grumodel = GRUNet(input_size=self.hid_size, hidden_size=self.hid_size, num_layers=1, output_size=self.hid_size)
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
        attention_scores = torch.matmul(input_Q,input_k.transpose(-1, -2))
        attention_scores = (attention_scores / math.sqrt(head_size))
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, input_v)
        return context_layer
        
    
    def forward(self, code_inputs=None, attn_mask=None,position_idx=None, nl_inputs=None,New_DFG_ids = None): 
        if code_inputs is not None:
            bs = code_inputs.shape[0]
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
                   #DFG_att_tmp = ((attn_mask[loc][index:,index:][:DFG_max_len,:DFG_max_len]).int())
                   #pad_tensor = torch.zeros(len(DFG_att_tmp),DFG_max_len - len(DFG_att_tmp))
                   #DFG_att_tmp = torch.cat((DFG_att_tmp,pad_tensor),dim=-1).tolist()
                   tmp.extend([[self.pad_id]*self.bsf_context_length]*(DFG_max_len - len(tmp)))
                   #DFG_att_tmp.extend([[0]*DFG_max_len]*(DFG_max_len - len(DFG_att_tmp)))
                #else:
                  # DFG_att_tmp = ((attn_mask[loc][index:,index:][:DFG_max_len,:DFG_max_len]).int()).tolist()        
                fin_DFG_ids.append(tmp)
                #fin_DFG_mask.append(DFG_att_tmp)
                fin_POS_emb.append([index for index in range(DFG_max_len)])
            fin_DFG_ids = torch.tensor(fin_DFG_ids).cuda()
            #fin_DFG_mask = torch.tensor(fin_DFG_mask).cuda()
            fin_POS_emb = torch.tensor(fin_POS_emb).cuda()
            DFG_embeddings=self.encoder.embeddings.word_embeddings(fin_DFG_ids).reshape(-1,self.bsf_len,self.hid_size)
            DFG_pos_embeddings = self.encoder.embeddings.position_embeddings(fin_POS_emb)
            DFGout= self.grumodel(DFG_embeddings)
            DFGout = DFGout.unsqueeze(0)
            DFGout = DFGout.reshape(bs,-1,self.hid_size)+DFG_pos_embeddings
            DFGout = self.att(DFGout)
            DFGout = self.dropout(self.ff(DFGout))
            
            inputs_embeddings=self.encoder.embeddings.word_embeddings(code_inputs)
            nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
            nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
            avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
            for index,i in enumerate(DFG_index):
                avg_embeddings[index][i:i+DFG_len[index]] =(1-self.alpa)*avg_embeddings[index][i:i+DFG_len[index]]+self.alpa*DFGout[index][:DFG_len[index]]
                #avg_embeddings[index][i:i+DFG_len[index]] =DFGout[index][:DFG_len[index]]
            inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]
            return self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[1]
        else:
            return self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[1]

      
        
 
