import random
import time
from typing import Dict, List
import codecs
import jsonlines
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
import yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer
import faiss
import os
logger.add('./log.txt')
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
f=codecs.open('/home/chezhonghao/projects/competition/ranking/competition_search/config.yml','r')
config=yaml.load(f)
# 基本参数
EPOCHS = 10
BATCH_SIZE = 90
LR = 1e-5
DROPOUT = 0.3
MAXLEN = int(config['model']['model_config']['max_position_size'])
POOLING = 'first-last-avg'   # choose in ['cls', 'pooler', 'first-last-avg', 'last-avg']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# 预训练模型目录
BERT = 'pretrained_model/bert_pytorch'
BERT_WWM_EXT = 'pretrained_model/bert_wwm_ext_pytorch'
ROBERTA = 'pretrained_model/roberta_wwm_ext_pytorch'
model_path = BERT_WWM_EXT 

# 微调后参数存放位置
model_base_path=config['model']['model_path']['model_saved_base_path']
unsup_only_document_model_dir=os.path.join(model_base_path,config['model']['model_path']['unsup_corpus_model_dir'])
unsup_only_document_model_path=os.path.join(unsup_only_document_model_dir,'model.pt')
SAVE_PATH = unsup_only_document_model_path

# 数据目录
SNIL_TRAIN = '/home/chezhonghao/projects/competition/ranking/data/SNLI/train.txt'
STS_TRAIN = './datasets/STS-B/cnsd-sts-train.txt'
STS_DEV = './datasets/STS-B/cnsd-sts-dev.txt'
STS_TEST = './datasets/STS-B/cnsd-sts-test.txt'

def load_data(name:str,path:str)->List:
    '''
    根据任务的不同加载数据集
    '''
    def unsup_simcse_documents(path):
        '''
        '''
        res=[]
        with codecs.open(path,'r') as f:
            line=f.readline()
            while line:
                ids,doc=line.replace('\n','').split('\t')
                res.append(doc)
                line=f.readline()
            return res
    def get_hard_negative(path):
        res=[]
        with codecs.open(path,'r') as f:
            line=f.readline()
            while line:
                ids,doc=line.replace('\n','').split('\t')
                res.append([ids,doc])
                line=f.readline()
            return res
    assert name in ["unsup_only_documents"] 
    if name == "unsup_only_documents":
        return    unsup_simcse_documents(path)
    if name == "get_hard_negative":
        return get_hard_negative(path)



# def load_data(name: str, path: str) -> List:
#     """根据名字加载不同的数据集"""
#     def load_snli_data(path):
#         with jsonlines.open(path, 'r') as f:
#             return [line.get('origin') for line in f]

#     def load_lqcmc_data(path):
#         with open(path, 'r', encoding='utf8') as f:
#             return [line.strip().split('\t')[0] for line in f]    

#     def load_sts_data(path):
#         with open(path, 'r', encoding='utf8') as f:            
#             return [(line.split("||")[1], line.split("||")[2], line.split("||")[3]) for line in f]
        
#     assert name in ["snli", "lqcmc", "sts"]
#     if name == 'snli':
#         return load_snli_data(path)
#     return load_lqcmc_data(path) if name == 'lqcmc' else load_sts_data(path) 


class TrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法"""
    def __init__(self, data: List):
        self.data = data
      
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        # 添加自身两次, 经过bert编码之后, 互为正样本
        return tokenizer([text, text], max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index: int):
        return self.text_2_id(self.data[index])
    

class TestDataset(Dataset):
    """测试数据集, 重写__getitem__和__len__方法"""
    def __init__(self, data: List):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        return tokenizer(text, max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index: int):
        da = self.data[index]        
        return self.text_2_id(da)


class SimcseModel(nn.Module):
    """Simcse无监督模型定义"""
    def __init__(self, pretrained_model, pooling):
        super(SimcseModel, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model)       
        config.attention_probs_dropout_prob = DROPOUT   # 修改config的dropout系数
        config.hidden_dropout_prob = DROPOUT           
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling
        
    def forward(self, input_ids, attention_mask, token_type_ids):

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
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)     # [batch, 768]
    
    
def simcse_unsup_loss(y_pred: 'tensor') -> 'tensor':
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]
    
    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=DEVICE)
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return loss
def dev(model, dataloader) -> float:
    """模型评估函数 
    批量预测, batch结果拼接, 一次性求spearman相关度
    """
    model.eval()
    eval_loss_mean=0.0
    loss_tensor = torch.tensor([], device=DEVICE)
    label_array = np.array([])
    with torch.no_grad():
        for source in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            real_batch_num = source.get('input_ids').shape[0]
            input_ids = source.get('input_ids').view(real_batch_num * 2, -1).to(DEVICE)
            attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1).to(DEVICE)
            token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1).to(DEVICE)
            source_pred = model(input_ids, attention_mask, token_type_ids)
            eval_loss=torch.unsqueeze(simcse_unsup_loss(source_pred),-1)
            loss_tensor=torch.cat([loss_tensor,eval_loss])
    return torch.mean(loss_tensor,dim=0,keepdim=False).cpu().numpy()

def eval(model, dataloader) -> float:
    """模型评估函数 
    批量预测, batch结果拼接, 一次性求spearman相关度
    """
    model.eval()
    sim_tensor = torch.tensor([], device=DEVICE)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(DEVICE)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(DEVICE)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(DEVICE)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(DEVICE)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(DEVICE)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(DEVICE)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)            
            label_array = np.append(label_array, np.array(label))
    # corrcoef 
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation
class InferenceItools():
    '''
    提供找到hard negative example的工具
    '''
    def __init__(self,model:SimcseModel,dim:int,gpu_id=1,embedding_index_path=''):
        '''
        model:unsupervised with only documents simcse
        dim:embedding的维度
        gpu_id:你希望由第几块卡做检索
        '''
        self.model=model
        if embedding_index_path=='':
            self.build_faiss_index(dim,gpu_id)
        else:
            self.embedding_index=index = faiss.read_index(embedding_index_path)
    def build_faiss_index(self,dim=768,gpu_id=1):
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = gpu_id
        self.embedding_index =faiss.GpuIndexFlatIP(res,dim,flat_config)
    
    def add_embedding_data(self,test_data_path:str,batch_size:int,dest_index_path:str):
        '''
        获得test_data_path中每句话的embedding
        并插入faiss 索引
        并写入dest_index_path
        '''
        test_data=load_data('unsup_only_documents',test_data_path)
        test_data_loader=DataLoader(TestDataset(test_data),batch_size=batch_size,shuffle=False,drop_last=False)
        self.model.eval()
        with torch.no_grad():
            for source in test_data_loader:
                source_input_ids = source.get('input_ids').squeeze(1).to(DEVICE)
                source_attention_mask = source.get('attention_mask').squeeze(1).to(DEVICE)
                source_token_type_ids = source.get('token_type_ids').squeeze(1).to(DEVICE)
                source_pred = self.model(source_input_ids, source_attention_mask, source_token_type_ids)
                noraml_embedding = torch.nn.functional.normalize(source_pred, p=2, dim=1, eps=1e-12).cpu().numpy()
                self.embedding_index.add(noraml_embedding)
        m=faiss.index_gpu_to_cpu(self.embedding_index)
        faiss.write_index(m, dest_index_path)
        logger.info("vec numbers: {}".format(self.embedding_index.ntotal))
    def get_negative(self,example_id:int,similarity_k:List[int],similarity:List[float],true_k:int,batch_id:int,threshold=0.45):
        '''
        获得negative的策略
        example_id :你需要找到id为example_id的doc的最相似的k个doc,编号从1开始
        similarity_k :最近的k个doc的编号，编号从1开始
        similarity：k个doc的相似度
        true_k:从最想似的k个中选出true_k个
        '''
        similar_ids=[]
        similar_cos=[]
        res=[]
        cos_res=[]
        for i in range(len(similarity_k)):
            if similarity[i] <= threshold:
                if similarity_k[i]!=example_id:
                    res.append(similarity_k[i])
                    cos_res.append(similarity[i])
        sample_index=random.sample(range(0,len(res)),true_k)
        for i in sample_index:
            similar_ids.append(res[i])
            similar_cos.append(cos_res[i])
        return similar_ids,similar_cos
    def get_similarity(self,test_data_path:str,k:int,dest_path:str,true_k=3):
        '''
        true_k:从最想似的k个中选出true_k个
        找到test_data_path中每个example最similarity的k个example 并返回 cosin和id
        并写入文件dest_path
        '''
        test_data=load_data('unsup_only_documents',test_data_path)
        test_data_loader=DataLoader(TestDataset(test_data),batch_size=60,shuffle=False,drop_last=False)
        self.model.eval()
        batch_id = 0
        with torch.no_grad():
            with codecs.open(dest_path,'w') as f:
                with codecs.open(test_data_path,'r') as t_f:
                    for source in test_data_loader:
                        batch_id+=1
                        source_input_ids = source.get('input_ids').squeeze(1).to(DEVICE)
                        source_attention_mask = source.get('attention_mask').squeeze(1).to(DEVICE)
                        source_token_type_ids = source.get('token_type_ids').squeeze(1).to(DEVICE)
                        source_pred = self.model(source_input_ids, source_attention_mask, source_token_type_ids)
                        noraml_embedding = torch.nn.functional.normalize(source_pred, p=2, dim=1, eps=1e-12).cpu().numpy()
                        cos,ids=self.embedding_index.search(noraml_embedding,k)
                        cos_shape=cos.shape#[batch,k]
                        #把test_data_path文件读cos_shape[0]次
                        source_batch_id_list=[]
                        for i in range(cos_shape[0]):
                            line=t_f.readline().replace('\n','')
                            per_ids,doc=line.split('\t')
                            source_batch_id_list.append(int(per_ids))
                        for i in range(cos_shape[0]):
                            #注意这个id是从1开始编号的，而faiss中的id是从0开始编号的,所以faiss中的id需要加1
                            per_source_id=source_batch_id_list[i]
                            similarity_ids=list(np.add(ids[i],1))
                            similarity=list(cos[i])
                            sample_ids,cos_sample=self.get_negative(per_source_id,similarity_ids,similarity,true_k,batch_id)
                            for j in range(len(sample_ids)):
                                f.write(str(per_source_id)+'\t'+str(sample_ids[j])+'\t'+str(cos_sample[j])+'\n')
                     
def train(model, train_dl, dev_dl, optimizer) -> None:
    """模型训练函数"""
    model.train()
    global best
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 2, seq_len] -> [batch * 2, sql_len]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 2, -1).to(DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1).to(DEVICE)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1).to(DEVICE)
        out = model(input_ids, attention_mask, token_type_ids)        
        loss = simcse_unsup_loss(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logger.info(f'loss: {loss.item():.4f}')
    dev_loss = dev(model, dev_dl)
    model.train()
    if best > dev_loss:
        best = dev_loss
        torch.save(model.state_dict(), SAVE_PATH)
        logger.info(f"higher eval_loss: {dev_loss:.4f} save model")
       
            
if __name__ == '__main__':
    data_base_path=config['data']['data_base_path']
    dest_unsup_train_data_dir=config['data']['unsup_corpus_dir']
    roberta_path=config['model']['model_path']['hugging_face_roberta_path']
    unsup_dev_data_path=os.path.join(data_base_path,dest_unsup_train_data_dir+'dev.txt')
    unsup_train_data_path=os.path.join(data_base_path,dest_unsup_train_data_dir+'train.txt')
    logger.info(f'device: {DEVICE}, pooling: {POOLING}, model path: {roberta_path}')
    tokenizer = BertTokenizer.from_pretrained(roberta_path)
    # load data
    train_data_unsup = load_data('unsup_only_documents', unsup_train_data_path)
    dev_data = load_data('unsup_only_documents', unsup_dev_data_path)
    train_dataloader = DataLoader(TrainDataset(train_data_unsup), batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    dev_dataloader = DataLoader(TrainDataset(dev_data), batch_size=BATCH_SIZE,shuffle=False,drop_last=True)
    # test_dataloader = DataLoader(TestDataset(test_data), batch_size=BATCH_SIZE)
    # load model
    assert POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
    model = SimcseModel(pretrained_model=roberta_path, pooling=POOLING).to(DEVICE)  
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # train
    # best = float('inf')
    # for epoch in range(EPOCHS):
    #     logger.info(f'epoch: {epoch}')
    #     train(model, train_dataloader, dev_dataloader, optimizer)
    # logger.info(f'train is finished, best model is saved at {SAVE_PATH}')
    
    
    # inference
    format_path=config['data']['format']
    filter_corpus_path=os.path.join(data_base_path,format_path+'filter_corpus_with_qrel.txt')
    dest_index_path='/home/chezhonghao/projects/competition/ranking/data/index/corpus.index'
    model.load_state_dict(torch.load(SAVE_PATH))
    it=InferenceItools(model=model,dim=768,gpu_id=1,embedding_index_path=dest_index_path)
    #获得source corpus.tsv的数据地址
    source_data_dir=config['data']['source_data_dir']
    corpus_data_path=os.path.join(data_base_path,source_data_dir+'corpus.tsv')
    # it.add_embedding_data(test_data_path=corpus_data_path,batch_size=200,dest_index_path=dest_index_path)
    hard_negative_example_path='/home/chezhonghao/projects/competition/ranking/data/hard_nagative_example/hard_nagative.txt'
    format_path=config['data']['format']
    filter_corpus_path=os.path.join(data_base_path,format_path+'filter_corpus_with_qrel.txt')
    it.get_similarity(test_data_path=filter_corpus_path,k=10000,dest_path=hard_negative_example_path,true_k=1)

    