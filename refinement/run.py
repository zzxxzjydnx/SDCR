# coding=utf-8
from __future__ import absolute_import
import os
import sys
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from bleu import _bleu
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
logger = logging.getLogger(__name__)
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
    LANGUAGE = Language('parser/my-languages.so', lang)
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



class Example(object):
    """A single training/test example."""
    def __init__(self,
                 source,
                 target,
                 ):
        self.source = source
        self.target = target

def read_examples(filename):
    """Read examples from filename."""
    examples=[]
    source,target=filename.split(',')
    with open(source,encoding="utf-8") as f1,open(target,encoding="utf-8") as f2:
        for line1,line2 in zip(f1,f2):
            line1=line1.strip()
            line2=line2.strip()
            examples.append(
                Example(
                    source=line1,
                    target=line2,
                        ) 
            )

    return examples


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,                 
                 target_ids,
                 source_mask,
                 target_mask,
                 New_DFG_ids,
                 dfg_to_bsf,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        self.dfg_to_bsf = dfg_to_bsf
        self.New_DFG_ids = New_DFG_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask       
        

def convert_examples_to_features(examples, tokenizer, args,stage=None):
    features = []
    for example_index, example in enumerate(tqdm(examples,total=len(examples))):
        ##extract data flow
        code_tokens,dfg=extract_dataflow(example.source,parsers['java'],'java',tokenizer,args)
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
        code_tokens=code_tokens[:args.max_source_length-2]
        source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        position_idx = [i+tokenizer.pad_token_id + 2 for i in range(len(source_tokens))]

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
        dfg=dfg[:args.max_source_length+args.bsf_length-bsf_code_tokens_len]
        ##########################################
                    		

        source_tokens+=[x[0] for x in dfg]
        position_idx+=[0 for x in dfg]
        source_ids+=[tokenizer.unk_token_id for x in dfg]
        padding_length=args.max_source_length+args.bsf_length-len(source_ids)
        position_idx+=[tokenizer.pad_token_id]*padding_length
        source_ids+=[tokenizer.pad_token_id]*padding_length


        source_mask = [1] * (len(source_tokens))
        source_mask+=[0]*padding_length        
        
        #reindex
        reverse_index={}
        for idx,x in enumerate(dfg):
            reverse_index[x[1]]=idx

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
        New_DFG_pad = [[tokenizer.pad_token_id]*args.bsf_context_length]*(args.max_source_length+args.bsf_length-len(New_DFG_ids))
        New_DFG_ids.extend(New_DFG_pad)
        New_DFG_ids = New_DFG_ids[:args.max_source_length+args.bsf_length]
        	
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   
   
        if example_index < 5:
            if stage=='train':
                logger.info("*** Example ***")
                logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))
                logger.info("position_idx: {}".format(position_idx))
                logger.info("dfg_to_code: {}".format(' '.join(map(str, dfg_to_code))))
                logger.info("dfg_to_dfg: {}".format(' '.join(map(str, dfg_to_dfg))))
                
                logger.info("target_tokens: {}".format([x.replace('\u0120','_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))
       
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,
                 target_ids,
                 source_mask,
                 target_mask,
                 New_DFG_ids,
                 dfg_to_bsf,
            )
        )
    return features    

class TextDataset(Dataset):
    def __init__(self, examples, args):
        self.examples = examples
        self.args=args  
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        #calculate graph-guided masked function
        attn_mask=np.zeros((self.args.max_source_length+self.args.bsf_length,self.args.max_source_length+self.args.bsf_length),dtype=np.bool_)
        #calculate begin index of node and max length of input

        node_index=sum([i>2 for i in self.examples[item].position_idx])
        max_length=sum([i!=1 for i in self.examples[item].position_idx])
        bsf_len = sum([i==2 for i in self.examples[item].position_idx])
                
        #sequence can attend to sequence
        attn_mask[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].source_ids):
            if i in [0,2]:
                attn_mask[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
            if a<node_index and b<node_index:
                attn_mask[idx+bsf_len+node_index,a:b]=True
                attn_mask[a:b,idx+bsf_len+node_index]=True
        #nodes attend to adjacent nodes 
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a+node_index+bsf_len<len(self.examples[item].position_idx):
                        attn_mask[idx+bsf_len+node_index,a+bsf_len+node_index]=True
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_bsf):
            if a<node_index+bsf_len and b<node_index+bsf_len:
                attn_mask[idx+bsf_len+node_index,a:b]=True
                attn_mask[a:b,idx+bsf_len+node_index]=True   
                    
        return (torch.tensor(self.examples[item].source_ids),
                torch.tensor(self.examples[item].source_mask),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(self.examples[item].New_DFG_ids),
                torch.tensor(attn_mask), 
                torch.tensor(self.examples[item].target_ids),
                torch.tensor(self.examples[item].target_mask),)
    
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name") 
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--bsf_context_length', type=int, default=20,
                        help="random seed for initialization")
    parser.add_argument('--bsf_length', type=int, default=20,
                        help="random seed for initialization")
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    # Set seed
    set_seed(args.seed)
    
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
        
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name )
    
    #budild model
    encoder = model_class.from_pretrained(args.model_name_or_path,config=config)    
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  bsf_len = args.bsf_length,hid_size= 768,pad_id = tokenizer.pad_token_id,bsf_context_length = args.bsf_context_length,beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        
    model.to(device)
    if args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer,args,stage='train')
        train_data = TextDataset(train_features,args)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps,num_workers=4)

        num_train_optimization_steps =  args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*args.num_train_epochs*0.1,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)
        
        model.train()
        dev_dataset={}
        nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0, 0,0,0,0,1e6 
        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader,total=len(train_dataloader))
            for batch in bar:
                batch = tuple(t.to(device) for t in batch)
                source_ids,source_mask,position_idx,New_DFG_ids,att_mask,target_ids,target_mask = batch
                loss,_,_ = model(source_ids,source_mask,position_idx,New_DFG_ids,att_mask,target_ids,target_mask)

                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    
                tr_loss += loss.item()
                train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
                bar.set_description("epoch {} loss {}".format(epoch,train_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    #Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            if args.do_eval and epoch in [ int(args.num_train_epochs*(i+1)//20) for i in range(20)]:
                #Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0                     
                eval_flag=False    
                if 'dev_loss' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_loss']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='dev')
                    eval_data = TextDataset(eval_features,args)
                    dev_dataset['dev_loss']=eval_examples,eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                #Start Evaling model
                model.eval()
                eval_loss,tokens_num = 0,0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)               
                    source_ids,source_mask,position_idx,New_DFG_ids,att_mask,target_ids,target_mask = batch
                    with torch.no_grad():
                        _,loss,num = model(source_ids,source_mask,position_idx,New_DFG_ids,att_mask,target_ids,target_mask)        
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                #Pring loss of dev dataset    
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {'eval_ppl': round(np.exp(eval_loss),5),
                          'global_step': global_step+1,
                          'train_loss': round(train_loss,5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  "+"*"*20)   

                #save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)                    
                if eval_loss<best_loss:
                    logger.info("  Best ppl:%s",round(np.exp(eval_loss),5))
                    logger.info("  "+"*"*20)
                    best_loss=eval_loss
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)  


                #Calculate bleu  
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_examples = random.sample(eval_examples,min(1000,len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
                    eval_data = TextDataset(eval_features,args)
                    dev_dataset['dev_bleu']=eval_examples,eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4)
                model.eval() 
                p=[]
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,source_mask,position_idx,New_DFG_ids,att_mask,target_ids,target_mask = batch                 
                    with torch.no_grad():
                        preds = model(source_ids,source_mask,position_idx,New_DFG_ids,att_mask)  
                        for pred in preds:
                            t=pred[0].cpu().numpy()
                            t=list(t)
                            if 0 in t:
                                t=t[:t.index(0)]
                            text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                            p.append(text)
                model.train()
                predictions=[]
                accs=[]
                with open(os.path.join(args.output_dir,"dev.output"),'w') as f, open(os.path.join(args.output_dir,"dev.gold"),'w') as f1:
                    for ref,gold in zip(p,eval_examples):
                        predictions.append(ref)
                        f.write(ref+'\n')
                        f1.write(gold.target+'\n')     
                        accs.append(ref==gold.target)

                dev_bleu=round(_bleu(os.path.join(args.output_dir, "dev.gold"), os.path.join(args.output_dir, "dev.output")),2)
                xmatch=round(np.mean(accs)*100,4)
                logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                logger.info("  %s = %s "%("xMatch",str(round(np.mean(accs)*100,4))))
                logger.info("  "+"*"*20)    
                if dev_bleu+xmatch>best_bleu:
                    logger.info("  Best BLEU+xMatch:%s",dev_bleu+xmatch)
                    logger.info("  "+"*"*20)
                    best_bleu=dev_bleu+xmatch
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
               
    if args.do_test:
        files=[]
        if args.dev_filename is not None:
            files.append(args.dev_filename)
        if args.test_filename is not None:
            files.append(args.test_filename)
        for idx,file in enumerate(files):   
            logger.info("Test file: {}".format(file))
            eval_examples = read_examples(file)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
            eval_data = TextDataset(eval_features,args) 

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4)

            model.eval() 
            p=[]
            for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                source_ids,source_mask,position_idx,New_DFG_ids,att_mask,target_ids,target_mask = batch                    
                with torch.no_grad():
                    preds = model(source_ids,source_mask,position_idx,New_DFG_ids,att_mask)  
                    for pred in preds:
                        t=pred[0].cpu().numpy()
                        t=list(t)
                        if 0 in t:
                            t=t[:t.index(0)]
                        text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                        p.append(text)
            model.train()
            predictions=[]
            accs=[]
            with open(os.path.join(args.output_dir,"test_{}.output".format(str(idx))),'w') as f, open(os.path.join(args.output_dir,"test_{}.gold".format(str(idx))),'w') as f1:
                for ref,gold in zip(p,eval_examples):
                    predictions.append(ref)
                    f.write(ref+'\n')
                    f1.write(gold.target+'\n')    
                    accs.append(ref==gold.target)
            dev_bleu=round(_bleu(os.path.join(args.output_dir, "test_{}.gold".format(str(idx))).format(file), 
                                 os.path.join(args.output_dir, "test_{}.output".format(str(idx))).format(file)),2)
            logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
            logger.info("  %s = %s "%("xMatch",str(round(np.mean(accs)*100,4))))
            logger.info("  "+"*"*20)   
            
if __name__ == "__main__":
    main()


