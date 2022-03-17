import random
import time
from typing import List

import jsonlines
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer
import yaml
import codecs
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logger.add('./log.txt')
f=codecs.open('/home/chezhonghao/projects/competition/ranking/competition_search/config.yml','r')

config=yaml.load(f)
# 基本参数
EPOCHS = 5
BATCH_SIZE = 50
INFERENCE_BATCH_SIZE=100
LR = 1e-5
MAXLEN = config['model']['model_config']['max_position_size']
POOLING = 'first-last-avg'   # choose in ['cls', 'pooler', 'last-avg', 'first-last-avg']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model_base_path=config['model']['model_path']['model_saved_base_path']
embedding_model_dir=os.path.join(model_base_path,config['model']['model_path']['embedding_model_dir'])
unsup_model=os.path.join(model_base_path,config['model']['model_path']['unsup_corpus_model_dir'])
embedding_model_path=os.path.join(embedding_model_dir,'model.pt')
unsup_model_path=os.path.join(embedding_model_dir,'model.pt')
SAVE_PATH=embedding_model_path

# 预训练模型目录
BERT = 'pretrained_model/bert_pytorch'
BERT_WWM_EXT = 'pretrained_model/bert_wwm_ext_pytorch'
ROBERTA = '/home/chezhonghao/projects/competition/ranking/pretrain_model/unsup_roberta_base/simcse-chinese-roberta-wwm-ext'
model_path = ROBERTA
def load_data(name:str,path:str)->List:
    '''
    根据任务的不同加载数据集
    '''
    def sup_embedding(path):
        '''
        '''
        res=[]
        with codecs.open(path,'r') as f:
            line=f.readline()
            while line:
                array=line.replace('\n','').split('\t')
                res.append(array)
                line=f.readline()
            return res
    assert name in ["sup_embedding","dev_mrr"] 
    if name == "sup_embedding":
        return    sup_embedding(path)
        

class TrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法
    """
    def __init__(self, data: List):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        return tokenizer([text[0], text[1], text[2]], max_length=MAXLEN, 
                         truncation=True, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index: int):
        return self.text_2_id(self.data[index])
    
    
class TestDataset(Dataset):
    """测试数据集, 重写__getitem__和__len__方法
    """
    def __init__(self, data: List):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        return tokenizer(text, max_length=MAXLEN, truncation=True, 
                         padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index):
        line = self.data[index]
        return self.text_2_id([line[1]])
    
    
class SimcseModel(nn.Module):
    """Simcse有监督模型定义"""
    def __init__(self, pretrained_model: str, pooling: str):
        super(SimcseModel, self).__init__()
        # config = BertConfig.from_pretrained(pretrained_model)   # 有监督不需要修改dropout
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.pooling = pooling
        self.liner_layer=torch.nn.Linear(768,128)
    def forward(self, input_ids, attention_mask, token_type_ids):
        
        # out = self.bert(input_ids, attention_mask, token_type_ids)
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]
        
        if self.pooling == 'pooler':
            return out.pooler_output            # [batch, 768]
        
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)    # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)       # [batch, 768]
        
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)    # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)    # [batch, 768, seqlen]                   
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1) # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)   # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)     # [batch, 2, 768]
            avg_first_last=torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)     # [batch, 768]
            return self.liner_layer(avg_first_last)
                  
            
def simcse_sup_loss(y_pred: 'tensor') -> 'tensor':
    """有监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 3, 768]
    
    """
    # 得到y_pred对应的label, 每第三句没有label, 跳过, label= [1, 0, 4, 3, ...]
    y_true = torch.arange(y_pred.shape[0], device=DEVICE)
    use_row = torch.where((y_true + 1) % 3 != 0)[0]
    y_true = (use_row - use_row % 3 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
    # 选取有效的行
    sim = torch.index_select(sim, 0, use_row)
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return loss
# def dev_mrr(model,dev_dataloader):


def dev_loss(model,dev_dataloader):
    model.eval()
    loss_tensor = torch.tensor([], device=DEVICE)
    with torch.no_grad():
        for batch_idx, source in enumerate(tqdm(dev_dataloader), start=1):
            # 维度转换 [batch, 3, seq_len] -> [batch * 3, sql_len]
            real_batch_num = source.get('input_ids').shape[0]
            input_ids = source.get('input_ids').view(real_batch_num * 3, -1).to(DEVICE)
            attention_mask = source.get('attention_mask').view(real_batch_num * 3, -1).to(DEVICE)
            token_type_ids = source.get('token_type_ids').view(real_batch_num * 3, -1).to(DEVICE)
            # 训练
            out = model(input_ids, attention_mask, token_type_ids)
            eval_loss=torch.unsqueeze(simcse_sup_loss(out),-1)
            loss_tensor=torch.cat([loss_tensor,eval_loss])
    return torch.mean(loss_tensor,dim=0,keepdim=False).cpu().numpy()


def eval(model, dataloader) -> float:
    """模型评估函数 
    批量预测, 计算cos_sim, 转成numpy数组拼接起来, 一次性求spearman相关度
    """
    model.eval()
    sim_tensor = torch.tensor([], device=DEVICE)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source['input_ids'].squeeze(1).to(DEVICE)
            source_attention_mask = source['attention_mask'].squeeze(1).to(DEVICE)
            source_token_type_ids = source['token_type_ids'].squeeze(1).to(DEVICE)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target['input_ids'].squeeze(1).to(DEVICE)
            target_attention_mask = target['attention_mask'].squeeze(1).to(DEVICE)
            target_token_type_ids = target['token_type_ids'].squeeze(1).to(DEVICE)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))  
    # corrcoef       
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation
        

def train(model, train_dl, dev_dl, optimizer) -> None:
    """模型训练函数 
    """
    model.train()
    global best
    early_stop_batch = 0
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 3, seq_len] -> [batch * 3, sql_len]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 3, -1).to(DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num * 3, -1).to(DEVICE)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 3, -1).to(DEVICE)
        # 训练
        out = model(input_ids, attention_mask, token_type_ids)
        loss = simcse_sup_loss(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    dev_losses = dev_loss(model, dev_dl)
    model.train()
    if best > dev_losses:
        best = dev_losses
        torch.save(model.state_dict(), SAVE_PATH)
        logger.info(f"best dev loss: {best:.5f} in batch: {batch_idx}, save model")
def infernece_embedding(model:SimcseModel,test_data_path:str,batch_size:int,dest_embedding_path:str,start_index:int):
        '''
        获得test_data_path中每句话的embedding
        并插入faiss 索引
        并写入dest_index_path
        '''
        # np.set_printoptions(precision=8)
        test_data=load_data('sup_embedding',test_data_path)
        test_data_loader=DataLoader(TestDataset(test_data),batch_size=batch_size,shuffle=False,drop_last=False)
        model.eval()
        item_index=start_index
        with codecs.open(dest_embedding_path,'w') as f:
            with torch.no_grad():
                for source in test_data_loader:
                    source_input_ids = source.get('input_ids').squeeze(1).to(DEVICE)
                    source_attention_mask = source.get('attention_mask').squeeze(1).to(DEVICE)
                    source_token_type_ids = source.get('token_type_ids').squeeze(1).to(DEVICE)
                    source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
                    noraml_embedding = torch.nn.functional.normalize(source_pred, p=2, dim=1, eps=1e-12).cpu().numpy()
                    embedding_shape=noraml_embedding.shape
                    for i in range(embedding_shape[0]):
                        item_index+=1
                        per_embedding = list(noraml_embedding[i])
                        str_embedding=','.join([("%.8f" % i) for i in per_embedding])
                        f.write(str(item_index)+'\t'+str_embedding+'\n')

                        

if __name__ == '__main__':

    logger.info(f'device: {DEVICE}, pooling: {POOLING}, model path: {model_path}')
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # load data
    data_base_path=config['data']['data_base_path']
    result_dir=os.path.join(data_base_path,config['data']['result'])
    corpus_result_path=os.path.join(result_dir,'doc_embedding')
    query_result_path=os.path.join(result_dir,'query_embedding')
    final4train_data_dir=config['data']['final4train']
    source_data_dir=config['data']['source_data_dir']
    train_data_path=os.path.join(data_base_path,final4train_data_dir+'train.txt')
    dev_data_path=os.path.join(data_base_path,final4train_data_dir+'dev.txt')
    test_data_path=os.path.join(data_base_path,source_data_dir+'corpus.tsv')
    test_query_path=os.path.join(data_base_path,source_data_dir+'dev.query.txt')
    
    train_data = load_data('sup_embedding', train_data_path) 
    dev_data = load_data('sup_embedding', dev_data_path)
    test_data=load_data('sup_embedding', test_data_path)
    test_query_data=load_data('sup_embedding',test_query_path)

    train_dataloader = DataLoader(TrainDataset(train_data), batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    dev_dataloader = DataLoader(TrainDataset(dev_data),batch_size=BATCH_SIZE,shuffle=False,drop_last=False)
    test_dataloader = DataLoader(TrainDataset(test_data),batch_size=BATCH_SIZE,shuffle=False,drop_last=False)
    test_query_dataloader = DataLoader(TrainDataset(test_query_data),batch_size=BATCH_SIZE,shuffle=False,drop_last=False)
    # load model    
    assert POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
    model = SimcseModel(pretrained_model=model_path, pooling=POOLING)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # train
    model.load_state_dict(torch.load(unsup_model_path))
    best = float('inf')
    for epoch in range(EPOCHS):
        logger.info(f'epoch: {epoch}')
        train(model, train_dataloader, dev_dataloader, optimizer)
    logger.info(f'train is finished, best model is saved at {SAVE_PATH}')

    #infernece
    model.load_state_dict(torch.load(SAVE_PATH))
    #获得corpus的embedding
    infernece_embedding(model=model,test_data_path=test_data_path,batch_size=INFERENCE_BATCH_SIZE,dest_embedding_path=corpus_result_path,start_index=0)
    #获得query的embedding
    infernece_embedding(model=model,test_data_path=test_query_path,batch_size=INFERENCE_BATCH_SIZE,dest_embedding_path=query_result_path,start_index=200000)