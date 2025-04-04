# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tqdm import tqdm, trange
import multiprocessing
from model import Model

logger = logging.getLogger(__name__)
from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}

#load parsers
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('./parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser
    



 ################################################
def traverse(node):
    if node.is_named:
        code_stas[(node.start_point,node.end_point)]=str(node.type)
        for child in node.children:
            traverse(child)
    return code_stas
def get_named_sibling_nodes(node,index_to_code):
    if not node.parent:
        return None

    # 
    siblings = node.parent.children
    siblings_new = []
    #  is_named 
    for sibling in siblings:
      if (sibling.start_point,sibling.end_point) in index_to_code and sibling.is_named:
        siblings_new.append((sibling.start_point,sibling.end_point))
      else:
        pass
    #siblings = [(sibling.start_point,sibling.end_point) if (sibling.start_point,sibling.end_point) in index_to_code else pass for sibling in siblings if sibling.is_named]
    return siblings_new
def find_node_by_position(node,parent, start_point, end_point,index_to_code,tokenizer):
    if node.start_point == start_point and node.end_point == end_point:
        sibling_nodes = get_named_sibling_nodes(node,index_to_code)
        try:
          result =  (tokenizer.tokenize('@ '+node.type)[1:],(sibling_nodes,parent.type,parent.parent.type))
        except:
          result =  (tokenizer.tokenize('@ '+node.type)[1:],(sibling_nodes,parent.type,))
        # 
        return result
    for child in node.children:
        found_node = find_node_by_position(child,node, start_point, end_point,index_to_code,tokenizer)
        if found_node:
            return found_node
    return None
        
        
########################################      


    
#remove comments, tokenize code and extract dataflow                                        
def extract_dataflow(code, parser,lang,tokenizer,args):
        #remove comments
        try:
            code=remove_comments_and_docstrings(code,lang)
        except:
            pass    
        #obtain dataflow
        if lang=="php":
            code="<?php"+code+"?>"    
    #try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        #print(code)
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]
        #code_stas = traverse(root_node)
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        
        DFG=sorted(DFG,key=lambda x:x[1])
        
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        
        index_to_code_dc = index_to_code
        items = index_to_code_dc.items()
        index_to_code_dcs = list(items)
        for index, s in enumerate(new_DFG):
            point =index_to_code_dcs[s[1]][0]
            start = point[0]
            end = point[1]
            slb_node_bsf= find_node_by_position(root_node,None,start,end,index_to_code,tokenizer)
            slb_nodes= slb_node_bsf[1][0]
            sn = [index_to_code[slb][1] for slb in slb_nodes]+list(slb_node_bsf[1][1:])
            sn_tmp = []
            for x in sn:
                sn_tmp.extend(tokenizer.tokenize('@ '+x)[1:])
            new_s = s[:2] + (slb_node_bsf[0],slb_node_bsf[0]+sn_tmp,) + s[2:]
            new_DFG[index] =new_s
        dfg=new_DFG
        return code_tokens,dfg

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
             input_tokens_1,
             input_ids_1,
             position_idx_1,
             dfg_to_code_1,
             dfg_to_dfg_1,
             New_DFG_ids_1,
             dfg_to_bsf_1,
       
             input_tokens_2,
             input_ids_2,
             position_idx_2,
             dfg_to_code_2,
             dfg_to_dfg_2,
             New_DFG_ids_2,
             dfg_to_bsf_2,             

             label,
             url1,
             url2

    ):
        #The first code function
        self.input_tokens_1 = input_tokens_1
        self.input_ids_1 = input_ids_1
        self.position_idx_1=position_idx_1
        self.dfg_to_code_1=dfg_to_code_1
        self.dfg_to_dfg_1=dfg_to_dfg_1
        self.New_DFG_ids_1 = New_DFG_ids_1
        self.dfg_to_bsf_1 = dfg_to_bsf_1      

        
        #The second code function
        self.input_tokens_2 = input_tokens_2
        self.input_ids_2 = input_ids_2
        self.position_idx_2=position_idx_2
        self.dfg_to_code_2=dfg_to_code_2
        self.dfg_to_dfg_2=dfg_to_dfg_2
        self.New_DFG_ids_2 = New_DFG_ids_2
        self.dfg_to_bsf_2 = dfg_to_bsf_2
        
        #label
        self.label=label
        self.url1=url1
        self.url2=url2
        

def convert_examples_to_features(item):
    #source
    url1,url2,label,tokenizer, args,cache,url_to_code=item
    parser=parsers['java']
    
    for url in [url1,url2]:
        if url not in cache:
            func=url_to_code[url]
            
            #extract data flow
            code_tokens,dfg=extract_dataflow(func,parser,'java',tokenizer,args)
            code_stas_set = set()
            for i in dfg:
            	code_stas_set.add(i[2][0])
            code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]	             
            ori2cur_pos={}
            ori2cur_pos[-1]=(0,0)
            for i in range(len(code_tokens)):
                ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
            code_tokens=[y for x in code_tokens for y in x]  
            
            #truncating
            code_tokens=code_tokens[:args.code_length+args.data_flow_length-2-min(len(dfg),args.data_flow_length)][:510-2]
            source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
            source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
            position_idx = [i+tokenizer.pad_token_id + 2 for i in range(len(source_tokens))]
            
            ###########################################

            for d in dfg:
            	w_l= d[3:4][0]
            	padding = [tokenizer.pad_token for _ in range(args.bsf_context_length - len(w_l))]
            	w_l.extend(padding)
            	d = d[:3]+(w_l[:args.bsf_context_length],)+d[3:]
            code_stas_List= list(code_stas_set)
            code_stas_List = code_stas_List[:args.bsf_length]
            code_stas_ids = tokenizer.convert_tokens_to_ids(code_stas_List)
            source_tokens +=code_stas_List
            source_ids += code_stas_ids
            position_idx+=[2 for x in code_stas_ids]
            bsf_code_tokens_len= len(source_tokens) 
            dfg=dfg[:args.code_length+args.data_flow_length+args.bsf_length-bsf_code_tokens_len]
            ##########################################
            
            source_tokens+=[x[0] for x in dfg]
            position_idx+=[0 for x in dfg]
            source_ids+=[tokenizer.unk_token_id for x in dfg]
            padding_length=args.code_length+args.data_flow_length+args.bsf_length-len(source_ids)
            position_idx+=[tokenizer.pad_token_id]*padding_length
            source_ids+=[tokenizer.pad_token_id]*padding_length      
            
            #reindex
            reverse_index={}
            for idx,x in enumerate(dfg):
                reverse_index[x[1]]=idx
            ##########################################
            
            bsf_point = len(ori2cur_pos)-2
            bsf_pos_point_start = ori2cur_pos[bsf_point]
            ori2cur_pos_bsf  = {}
            bsf_idx_cict = {} 
            for idx,i in enumerate(code_stas_List):
                ori2cur_pos_bsf[i] = (bsf_pos_point_start[1],bsf_pos_point_start[1]+len([i]))
                bsf_idx_cict[i] = idx
                bsf_pos_point_start = (bsf_pos_point_start[1],bsf_pos_point_start[1]+len([i]))           
            ##########################################

            ##########################################           
            for idx,x in enumerate(dfg):
                bsf_p =x[2][0]
                tmp= x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)
                dfg[idx]=tmp[:2]+((bsf_idx_cict[bsf_p]),)+tmp[2:]
            ##########################################    
            dfg_to_dfg=[x[-1] for x in dfg]
            dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
            length=len([tokenizer.cls_token])
            
            ########################################## 
            dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]  
            dfg_to_bsf=[ori2cur_pos_bsf[x[3][0]] for x in dfg]
            #SEP
            dfg_to_bsf=[(x[0]+length+1,x[1]+length+1) for x in dfg_to_bsf]       
            DFG_ids = []
            for d in dfg:
                new_d = tokenizer.convert_tokens_to_ids(d[4][:args.bsf_context_length])
                DFG_ids.append(new_d)
            New_DFG_ids = [[tokenizer.pad_token_id]*args.bsf_context_length]*bsf_code_tokens_len
            New_DFG_ids.extend(DFG_ids)
            New_DFG_pad = [[tokenizer.pad_token_id]*args.bsf_context_length]*(args.code_length+args.data_flow_length+args.bsf_length-len(New_DFG_ids))
            New_DFG_ids.extend(New_DFG_pad)
            New_DFG_ids = New_DFG_ids[:args.code_length+args.data_flow_length+args.bsf_length]
            ########################################## 
          
            cache[url]=source_tokens,source_ids,position_idx,dfg_to_code,dfg_to_dfg,New_DFG_ids,dfg_to_bsf

        
    source_tokens_1,source_ids_1,position_idx_1,dfg_to_code_1,dfg_to_dfg_1,New_DFG_ids_1,dfg_to_bsf_1=cache[url1]   
    source_tokens_2,source_ids_2,position_idx_2,dfg_to_code_2,dfg_to_dfg_2,New_DFG_ids_2,dfg_to_bsf_2=cache[url2]   
    return InputFeatures(source_tokens_1,source_ids_1,position_idx_1,dfg_to_code_1,dfg_to_dfg_1,New_DFG_ids_1,dfg_to_bsf_1,
                         source_tokens_2,source_ids_2,position_idx_2,dfg_to_code_2,dfg_to_dfg_2,New_DFG_ids_2,dfg_to_bsf_2,label,url1,url2)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train'):
        self.examples = []
        self.args=args
        index_filename=file_path
        
        #load index
        logger.info("Creating features from index file at %s ", index_filename)
        url_to_code={}
        with open('/'.join(index_filename.split('/')[:-1])+'/data.jsonl') as f:
            for line in f:
                line=line.strip()
                js=json.loads(line)
                url_to_code[js['idx']]=js['func']
                
        #load code function according to index
        data=[]
        cache={}
        f=open(index_filename)
        with open(index_filename) as f:
            for line in f:
                line=line.strip()
                url1,url2,label=line.split('\t')
                if url1 not in url_to_code or url2 not in url_to_code:
                    continue
                if label=='0':
                    label=0
                else:
                    label=1
                data.append((url1,url2,label,tokenizer, args,cache,url_to_code))
                
        #only use 10% valid data to keep best model        
        if 'valid' in file_path:
            data=random.sample(data,int(len(data)*0.1))
            
        #convert example to input features    
        self.examples=[convert_examples_to_features(x) for x in tqdm(data,total=len(data))]
        
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens_1: {}".format([x.replace('\u0120','_') for x in example.input_tokens_1]))
                logger.info("input_ids_1: {}".format(' '.join(map(str, example.input_ids_1))))       
                logger.info("position_idx_1: {}".format(example.position_idx_1))
                logger.info("dfg_to_code_1: {}".format(' '.join(map(str, example.dfg_to_code_1))))
                logger.info("dfg_to_dfg_1: {}".format(' '.join(map(str, example.dfg_to_dfg_1))))
                
                logger.info("input_tokens_2: {}".format([x.replace('\u0120','_') for x in example.input_tokens_2]))
                logger.info("input_ids_2: {}".format(' '.join(map(str, example.input_ids_2))))       
                logger.info("position_idx_2: {}".format(example.position_idx_2))
                logger.info("dfg_to_code_2: {}".format(' '.join(map(str, example.dfg_to_code_2))))
                logger.info("dfg_to_dfg_2: {}".format(' '.join(map(str, example.dfg_to_dfg_2))))


    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        #calculate graph-guided masked function
        attn_mask_1= np.zeros((self.args.code_length+self.args.data_flow_length+self.args.bsf_length,
                        self.args.code_length+self.args.data_flow_length+self.args.bsf_length),dtype=np.bool_)
        #calculate begin index of node and max length of input
        node_index_1=sum([i>2 for i in self.examples[item].position_idx_1])
        max_length_1=sum([i!=1 for i in self.examples[item].position_idx_1])
        
        bsf_len_1 = sum([i==2 for i in self.examples[item].position_idx_1])
        
        
        #sequence can attend to sequence
        attn_mask_1[:node_index_1,:node_index_1]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].input_ids_1):
            if i in [0,2]:
                attn_mask_1[idx,:max_length_1]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code_1):
            if a<node_index_1 and b<node_index_1:
                attn_mask_1[idx+bsf_len_1+node_index_1,a:b]=True
                attn_mask_1[a:b,idx+bsf_len_1+node_index_1]=True
        #nodes attend to adjacent nodes 
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg_1):
            for a in nodes:
                if a+node_index_1+bsf_len_1<len(self.examples[item].position_idx_1):
                        attn_mask_1[idx+bsf_len_1+node_index_1,a+bsf_len_1+node_index_1]=True
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_bsf_1):
            if a<node_index_1+bsf_len_1 and b<node_index_1+bsf_len_1:
                attn_mask_1[idx+bsf_len_1+node_index_1,a:b]=True
                attn_mask_1[a:b,idx+bsf_len_1+node_index_1]=True    
                    
        #calculate graph-guided masked function
        attn_mask_2= np.zeros((self.args.code_length+self.args.data_flow_length+self.args.bsf_length,
                        self.args.code_length+self.args.data_flow_length+self.args.bsf_length),dtype=np.bool_)
        #calculate begin index of node and max length of input
        node_index_2=sum([i>2 for i in self.examples[item].position_idx_2])
        max_length_2=sum([i!=1 for i in self.examples[item].position_idx_2])

        bsf_len_2 = sum([i==2 for i in self.examples[item].position_idx_2])
        
        #sequence can attend to sequence
        attn_mask_2[:node_index_2,:node_index_2]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].input_ids_2):
            if i in [0,2]:
                attn_mask_2[idx,:max_length_2]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code_2):
            if a<node_index_2 and b<node_index_2:
                attn_mask_2[idx+bsf_len_2+node_index_2,a:b]=True
                attn_mask_2[a:b,idx+bsf_len_2+node_index_2]=True
        #nodes attend to adjacent nodes 
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg_2):
            for a in nodes:
                if a+bsf_len_2+node_index_2<len(self.examples[item].position_idx_2):
                        attn_mask_2[idx+bsf_len_2+node_index_2,a+bsf_len_2+node_index_2]=True

        for idx,(a,b) in enumerate(self.examples[item].dfg_to_bsf_2):
            if a<node_index_2+bsf_len_2 and b<node_index_2+bsf_len_2:
                attn_mask_2[idx+bsf_len_2+node_index_2,a:b]=True
                attn_mask_2[a:b,idx+bsf_len_2+node_index_2]=True  

                    
        return (torch.tensor(self.examples[item].input_ids_1),
                torch.tensor(self.examples[item].position_idx_1),
                torch.tensor(attn_mask_1),
                torch.tensor(self.examples[item].New_DFG_ids_1),
                
                torch.tensor(self.examples[item].input_ids_2),
                torch.tensor(self.examples[item].position_idx_2),
                torch.tensor(attn_mask_2),
                torch.tensor(self.examples[item].New_DFG_ids_2), 
                
                torch.tensor(self.examples[item].label))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    
    #build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    args.max_steps=args.epochs*len( train_dataloader)
    args.save_steps=len( train_dataloader)//10
    args.warmup_steps=args.max_steps//5
    model.to(args.device)
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step=0
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_f1=0

    model.zero_grad()
 
    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
            (inputs_ids_1,position_idx_1,attn_mask_1,New_DFG_ids_1,
             inputs_ids_2,position_idx_2,attn_mask_2,New_DFG_ids_2,
            labels)=[x.to(args.device)  for x in batch]
            model.train()
            loss,logits = model(inputs_ids_1,position_idx_1,attn_mask_1,New_DFG_ids_1,inputs_ids_2,position_idx_2,attn_mask_2,New_DFG_ids_2,labels)

            if args.n_gpu > 1:
                loss = loss.mean()
                
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
                
            avg_loss=round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
              
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer, eval_when_training=True)    
                    
                    # Save model checkpoint
                    if results['eval_f1']>best_f1:
                        best_f1=results['eval_f1']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best f1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        
def evaluate(args, model, tokenizer, eval_when_training=False):
    #build dataloader
    eval_dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in eval_dataloader:
        (inputs_ids_1,position_idx_1,attn_mask_1,New_DFG_ids_1,inputs_ids_2,position_idx_2,attn_mask_2,New_DFG_ids_2,labels)=[x.to(args.device)  for x in batch]
        with torch.no_grad():
            lm_loss,logit = model(inputs_ids_1,position_idx_1,attn_mask_1,New_DFG_ids_1,inputs_ids_2,position_idx_2,attn_mask_2,New_DFG_ids_2,labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    
    #calculate scores
    logits=np.concatenate(logits,0)
    y_trues=np.concatenate(y_trues,0)
    best_threshold=0.5
    best_f1=0

    y_preds=logits[:,1]>best_threshold
    from sklearn.metrics import recall_score
    recall=recall_score(y_trues, y_preds)
    from sklearn.metrics import precision_score
    precision=precision_score(y_trues, y_preds)   
    from sklearn.metrics import f1_score
    f1=f1_score(y_trues, y_preds)             
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold":best_threshold,
        
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    return result

def test(args, model, tokenizer, best_threshold=0):
    #build dataloader
    eval_dataset = TextDataset(tokenizer, args, file_path=args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in eval_dataloader:
        (inputs_ids_1,position_idx_1,attn_mask_1,New_DFG_ids_1,inputs_ids_2,position_idx_2,attn_mask_2,New_DFG_ids_2,labels)=[x.to(args.device)  for x in batch]
        with torch.no_grad():
            lm_loss,logit = model(inputs_ids_1,position_idx_1,attn_mask_1,New_DFG_ids_1,inputs_ids_2,position_idx_2,attn_mask_2,New_DFG_ids_2,labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    
    #output result
    logits=np.concatenate(logits,0)
    y_preds=logits[:,1]>best_threshold
    with open(os.path.join(args.output_dir,"predictions.txt"),'w') as f:
        for example,pred in zip(eval_dataset.examples,y_preds):
            if pred:
                f.write(example.url1+'\t'+example.url2+'\t'+'1'+'\n')
            else:
                f.write(example.url1+'\t'+example.url2+'\t'+'0'+'\n')
                                                
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.") 
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--bsf_length", default=20, type=int,
                        help="Total number of training epochs to perform.")
                        
    parser.add_argument("--bsf_context_length", default=10, type=int,
                        help="Total number of training epochs to perform.")
    
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")

    args = parser.parse_args()

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu,)


    # Set seed
    set_seed(args)
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels=1
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path,config=config)
    model=Model(model,config,tokenizer,args,args.data_flow_length+args.code_length+args.bsf_length,args.bsf_length,768,tokenizer.pad_token_id,args.bsf_context_length)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, file_path=args.train_data_file)
        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result=evaluate(args, model, tokenizer)
        
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test(args, model, tokenizer,best_threshold=0.5)

    return results


if __name__ == "__main__":
    main()