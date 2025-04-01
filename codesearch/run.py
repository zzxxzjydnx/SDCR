import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
from model import Model
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)

logger = logging.getLogger(__name__)

from tqdm import tqdm, trange
import multiprocessing
cpu_cont = 16

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
        # 节点的位置范围包含了给定的起始和结束位置
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
                 code_tokens,
                 code_ids,
                 New_DFG_ids,
                 dfg_to_bsf,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,                 
                 nl_tokens,
                 nl_ids,
                 url,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.New_DFG_ids = New_DFG_ids
        self.dfg_to_bsf = dfg_to_bsf
        self.position_idx=position_idx
        self.dfg_to_code=dfg_to_code
        self.dfg_to_dfg=dfg_to_dfg        
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url=url
        
          
def convert_examples_to_features(item):
    js,tokenizer,args=item
    #code
    parser=parsers[args.lang]
    #extract data flow
    code_tokens,dfg=extract_dataflow(js['original_string'],parser,args.lang,tokenizer,args)
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
    code_tokens=code_tokens[:args.code_length+args.data_flow_length+args.bsf_length-2-min(len(dfg),args.data_flow_length)]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    position_idx = [i+tokenizer.pad_token_id + 2 for i in range(len(code_tokens))]
    
    for d in dfg:
	    w_l= d[3:4][0]
	    padding = [tokenizer.pad_token for _ in range(args.bsf_context_length - len(w_l))]
	    w_l.extend(padding)
	    d = d[:3]+(w_l[:args.bsf_context_length],)+d[3:]    
    
    
    #only_code_tokens_len= len(code_tokens)
    code_stas_List= list(code_stas_set)
    code_stas_List = code_stas_List[:10]
    code_stas_ids = tokenizer.convert_tokens_to_ids(code_stas_List)
    code_tokens +=code_stas_List
    code_ids += code_stas_ids
    position_idx+=[2 for x in code_stas_ids]
    bsf_code_tokens_len= len(code_tokens)    
    
    
    dfg=dfg[:args.code_length+args.data_flow_length+args.bsf_length-bsf_code_tokens_len]
    code_tokens+=[x[0] for x in dfg]
    position_idx+=[0 for x in dfg]
    #print("dfg",position_idx)
    code_ids+=[tokenizer.unk_token_id for x in dfg]
    padding_length=args.code_length+args.data_flow_length+args.bsf_length-len(code_ids)
    position_idx+=[tokenizer.pad_token_id]*padding_length
    code_ids+=[tokenizer.pad_token_id]*padding_length    
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
	    #bsf_idx_cict[(bsf_pos_point_start[1],bsf_pos_point_start[1]+len([i]))] = idx
	    bsf_pos_point_start = (bsf_pos_point_start[1],bsf_pos_point_start[1]+len([i]))
        
    for idx,x in enumerate(dfg):
        bsf_p =x[2][0]
        tmp= x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)
        dfg[idx]=tmp[:2]+((bsf_idx_cict[bsf_p]),)+tmp[2:]
        	  
    dfg_to_dfg=[x[-1] for x in dfg]
    dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
    length=len([tokenizer.cls_token])
    dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]
    
    dfg_to_bsf=[ori2cur_pos_bsf[x[3][0]] for x in dfg]
    #还需要考虑SEP
    dfg_to_bsf=[(x[0]+length+1,x[1]+length+1) for x in dfg_to_bsf]    
            
    #nl
    nl=' '.join(js['docstring_tokens'])
    nl_tokens=tokenizer.tokenize(nl)[:args.nl_length-2]
    nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids+=[tokenizer.pad_token_id]*padding_length    


    DFG_ids = []
    for d in dfg:
	    new_d = tokenizer.convert_tokens_to_ids(d[4][:args.bsf_context_length])
	    DFG_ids.append(new_d)
    New_DFG_ids = [[tokenizer.pad_token_id]*args.bsf_context_length]*bsf_code_tokens_len
    New_DFG_ids.extend(DFG_ids)
    New_DFG_pad = [[tokenizer.pad_token_id]*args.bsf_context_length]*(args.code_length+args.data_flow_length+args.bsf_length-len(New_DFG_ids))
    New_DFG_ids.extend(New_DFG_pad)
    New_DFG_ids = New_DFG_ids[:args.code_length+args.data_flow_length+args.bsf_length]
    
    return InputFeatures(code_tokens,code_ids,New_DFG_ids,dfg_to_bsf,position_idx,dfg_to_code,dfg_to_dfg,nl_tokens,nl_ids,js['url'])

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None,pool=None):
        self.args=args
        prefix=file_path.split('/')[-1][:-6]
        cache_file=args.output_dir+'/'+prefix+'.pkl'
        if os.path.exists(cache_file):
            print("abc")
            self.examples=pickle.load(open(cache_file,'rb'))
        else:
            self.examples = []
            data=[]
            with open(file_path) as f:
                for line in f:
                    line=line.strip()
                    js=json.loads(line)
                    data.append((js,tokenizer,args))
            data =data
            self.examples=pool.map(convert_examples_to_features, tqdm(data,total=len(data)))
            pickle.dump(self.examples,open(cache_file,'wb'))
            
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("position_idx: {}".format(example.position_idx))
                logger.info("dfg_to_code: {}".format(' '.join(map(str, example.dfg_to_code))))
                logger.info("dfg_to_dfg: {}".format(' '.join(map(str, example.dfg_to_dfg))))                
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))          
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item): 
        #calculate graph-guided masked function
        attn_mask=np.zeros((self.args.code_length+self.args.data_flow_length+self.args.bsf_length,
                            self.args.code_length+self.args.data_flow_length+self.args.bsf_length),dtype=np.bool_)
        #calculate begin index of node and max length of input
        node_index=sum([i>2 for i in self.examples[item].position_idx])
        max_length=sum([i!=1 for i in self.examples[item].position_idx])
        bsf_len = sum([i==2 for i in self.examples[item].position_idx])
        #sequence can attend to sequence
        attn_mask[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].code_ids):
            if i in [0,2]:
                attn_mask[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
            if a<node_index and b<node_index:
                attn_mask[idx+bsf_len+node_index,a:b]=True
                attn_mask[a:b,idx+node_index+bsf_len]=True
        #nodes attend to adjacent nodes 
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a+node_index+bsf_len<len(self.examples[item].position_idx):
                    attn_mask[idx+node_index+bsf_len,a+node_index+bsf_len]=True  
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_bsf):
            if a<node_index+bsf_len and b<node_index+bsf_len:
                attn_mask[idx+bsf_len+node_index,a:b]=True
                attn_mask[a:b,idx+bsf_len+node_index]=True                    
        return (torch.tensor(self.examples[item].code_ids),torch.tensor(attn_mask),
              torch.tensor(self.examples[item].position_idx), 
              torch.tensor(self.examples[item].nl_ids),
              torch.tensor(self.examples[item].New_DFG_ids))
            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.train_data_file, pool)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    
    model.train()
    tr_num,tr_loss,best_mrr=0,0,0 
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            #get inputs
            code_inputs = batch[0].to(args.device)  
            attn_mask = batch[1].to(args.device)
            position_idx = batch[2].to(args.device)
            nl_inputs = batch[3].to(args.device)
            New_DFG_ids =batch[4].to(args.device)
            #get code and nl vectors
            code_vec= model(code_inputs=code_inputs, attn_mask=attn_mask,position_idx=position_idx,New_DFG_ids =New_DFG_ids)
            nl_vec = model(nl_inputs=nl_inputs)
            
            #calculate scores and loss
            scores=torch.einsum("ab,cb->ac",nl_vec,code_vec)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(scores, torch.arange(code_inputs.size(0), device=scores.device))
            
            #report loss
            tr_loss += loss.item()
            tr_num+=1
            if (step+1)% 100==0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss=0
                tr_num=0
            
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
            
        #evaluate    
        print("train is over %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        results = evaluate(args, model, tokenizer,args.eval_data_file, pool, eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    
            
        #save best model
        if results['eval_mrr']>best_mrr:
            best_mrr=results['eval_mrr']
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s",round(best_mrr,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer,file_name,pool, eval_when_training=False):
    query_dataset = TextDataset(tokenizer, args, file_name, pool)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
    
    code_dataset = TextDataset(tokenizer, args, args.codebase_file, pool)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=4)    

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    code_vecs=[] 
    nl_vecs=[]
    for batch in query_dataloader:  
        nl_inputs = batch[3].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs) 
            nl_vecs.append(nl_vec.cpu().numpy()) 

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)    
        attn_mask = batch[1].to(args.device)
        position_idx =batch[2].to(args.device)
        New_DFG_ids =batch[4].to(args.device)
        with torch.no_grad():
            code_vec= model(code_inputs=code_inputs, attn_mask=attn_mask,position_idx=position_idx,New_DFG_ids =New_DFG_ids)
            code_vecs.append(code_vec.cpu().numpy())  
    model.train()    
    code_vecs=np.concatenate(code_vecs,0)
    nl_vecs=np.concatenate(nl_vecs,0)

    scores=np.matmul(nl_vecs,code_vecs.T)
    
    sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    
    nl_urls=[]
    code_urls=[]
    for example in query_dataset.examples:
        nl_urls.append(example.url)
        
    for example in code_dataset.examples:
        code_urls.append(example.url)
        
    ranks=[]
    for url, sort_id in zip(nl_urls,sort_ids):
        rank=0
        find=False
        for idx in sort_id[:1000]:
            if find is False:
                rank+=1
            if code_urls[idx]==url:
                find=True
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)
    
    result = {
        "eval_mrr":float(np.mean(ranks))
    }

    return result

                        
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    
    parser.add_argument("--lang", default=None, type=str,
                        help="language.")  
    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.") 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
                        
    parser.add_argument("--bsf_length", default=10, type=int,
                        help="Total number of training epochs to perform.")
                        
    parser.add_argument("--bsf_context_length", default=10, type=int,
                        help="Total number of training epochs to perform.")
                        
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    pool = multiprocessing.Pool(cpu_cont)
    
    #print arguments
    args = parser.parse_args()
    
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    #build model
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaModel.from_pretrained(args.model_name_or_path)    
    model=Model(model,args.data_flow_length+args.code_length+args.bsf_length,args.bsf_length,768,tokenizer.pad_token_id,args.bsf_context_length)
    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)
    
    # Training
    if args.do_train:
        train(args, model, tokenizer, pool)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir),strict=False)      
        model.to(args.device)
        result=evaluate(args, model, tokenizer,args.eval_data_file, pool)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
            
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir),strict=False)      
        model.to(args.device)
        result=evaluate(args, model, tokenizer,args.test_data_file, pool)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))

    return results


if __name__ == "__main__":
    main()
