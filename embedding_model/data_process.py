
"""snli数据预处理"""
import time
from typing import List,Dict,Union
import jsonlines
import codecs
import random
import yaml
from tqdm import tqdm
import os
import numpy as np
import math
class DataItools():
    @staticmethod
    def read_documents(data_path:str,percent=98,shuffle=True)->List[List[str]]:
        '''
        读取corpus模型
        return :[['1','盛弹盖文艺保温杯学生男女情侣车载时尚英文锁扣不锈钢真空水杯'],...] ,
        '''
        res=[]
        len_array=[]
        with codecs.open(data_path,'r') as f:
            line = f.readline()
            while line:
                array=line.replace('\n','').split('\t')
                ids=array[0]
                doc=' '.join(array[1:])
                res.append([ids,doc])
                len_array.append(len(doc))
                line = f.readline()
        array=np.array(len_array)
        if shuffle:
            random.shuffle(res)
        return res,math.floor(np.percentile(array,percent))+2
    # @staticmethod
    # def get_topk_similiarity():
    @staticmethod
    def judge_exists(path_list:List[str]):
        '''
        判断一个文件或者目录是否存在
        '''
        for i in range(len(path_list)):
            if not os.path.exists(path_list[i]):
                os.makedirs(path_list[i])

    @staticmethod
    def split_dataset(data:Union[List[List[str]],Dict[int,List[int]]],mod:str,ratio:float,dest_data_dir:str,train_file_name='train.txt',dev_file_name='dev.txt'):
        '''
        将data切分成 train_data ,dev_data
        data:存放数据
        train_data_path： split 出来的train data 要写入的文件地址
        dev_data_path : 
        mod  :无所谓了
        ratio: dev数据的比例是多少
        '''
        def split_doc(dev_index,train_data_path,dev_data_path):
            with codecs.open(train_data_path,'w') as t_f:
                with codecs.open(dev_data_path,'w') as d_f:
                    for i in range(len(data)):
                        write_res=''
                        for j in range(len(data[i])):
                            if j==len(data[i])-1:
                                temp=str(data[i][j])+'\n'
                            else:
                                temp=str(data[i][j])+'\t'
                            write_res=write_res+temp
                        if i in dev_index:
                            d_f.write(write_res)
                        else:
                            t_f.write(write_res)
        
        def split_qrels(dev_index,data_keys):

            dev_dict=dict()
            train_dict=dict()
            for i in range(len(data_keys)):
                if i in dev_index:
                    dev_dict[data_keys[i]]=data[data_keys[i]]
                else:
                    train_dict[data_keys[i]]=data[data_keys[i]]
            return train_dict,dev_dict
        if mod == 'doc':
            data_len=len(data)
            dev_index = set(random.sample(range(0,data_len),int(ratio*data_len)))
            DataItools.judge_exists([dest_data_dir])
            train_data_path=os.path.join(dest_data_dir,train_file_name)
            dev_data_path=os.path.join(dest_data_dir,dev_file_name)
            split_doc(dev_index,train_data_path,dev_data_path)
        if mod == 'qrels':
            data_keys=list(data.keys())
            data_len=len(data_keys)
            dev_index = set(random.sample(range(0,data_len),int(ratio*data_len)))
            return split_qrels(dev_index,data_keys)
    @staticmethod
    def write_file(data_path:str,data:List[List[Union[int,str,float]]]):
        with codecs.open(data_path,'w') as d_f:
            for i in range(len(data)):
                write_res=''
                for j in range(len(data[i])):
                    if j==len(data[i])-1:
                        temp=str(data[i][j])+'\n'
                    else:
                        temp=str(data[i][j])+'\t'
                    write_res=write_res+temp
                d_f.write(write_res)
    @staticmethod
    def get_train_doc(qrels_path:str,corpus_path:str,dest_path:str):
        '''
        qrels:qrels.train.tsv
        为了只用doc训练unsup的simcse 来寻找hard negative ,所以需要根据qrels.train.tsv来过滤corpus.tsv
        然后根据过滤的corpus.tsv来推理 找最hard negative
        '''
        corpus_list=[]
        with codecs.open(dest_path,'w') as d_f:
            with codecs.open(qrels_path,'r') as r_f:
                with codecs.open(corpus_path,'r') as c_f:
                    c_line=c_f.readline()
                    while c_line:
                        ids,doc=c_line.replace('\n','').split('\t')
                        corpus_list.append(doc)
                        c_line=c_f.readline()
                    r_line=r_f.readline()
                    while r_line:
                        q_id,d_id=r_line.replace('\n','').split('\t')
                        doc=corpus_list[int(d_id)-1]
                        d_f.write(d_id+'\t'+doc+'\n')
                        r_line=r_f.readline()
    @staticmethod
    def read_refs(refs_path:str)->Dict[int,List[int]]:
        res=dict()
        with codecs.open(refs_path,'r') as f:
            line = f.readline()
            while line:
                array=line.replace('\n','').split('\t')
                source_id,target_id=array[0],array[1]
                if int(source_id) not in res.keys():
                    res[int(source_id)]=[int(target_id)]
                else:
                    res[int(source_id)].append(int(target_id))
                line = f.readline()
        return res
    @staticmethod
    def filter_with_qrels(q2positive,positive2negative,query_list,doc_list):
        res=[]
        for q_id,positive_id_list in q2positive.items():
            for i in range(len(positive_id_list)):
                for j in range(len(positive2negative[positive_id_list[i]])):
                    q=query_list[q_id-1]
                    positive=doc_list[positive_id_list[i]-1]
                    per_negative=doc_list[positive2negative[positive_id_list[i]][j]-1]
                    res.append([q[1],positive[1],per_negative[1]])
        return res

    @staticmethod
    def get_train_data(qrels_path:str,query_path:str,corpus_path:str,hard_negative_rel_path:str,dest_data_dir:str,train_data_name='train.txt',dev_data_name='dev.txt'):
        '''
        生成 [query,positive_example,negative_example]
        dest_data_dir :存放 train.txt和dev.txt的地址
        '''
        res=[]
        #读query_path
        query_list,_=DataItools.read_documents(query_path,shuffle=False)
        #读corpus_path
        doc_list,_=DataItools.read_documents(corpus_path,shuffle=False)
        #读hard_negative_rel_path
        positive2negative=DataItools.read_refs(hard_negative_rel_path)
        #读qrels_path
        q2positive=DataItools.read_refs(qrels_path)
        #切分qrels
        train_qrels_list,dev_qrels_list = DataItools.split_dataset(data=q2positive,mod='qrels',ratio=0.1,dest_data_dir='')
        
        sup_train_data_list=DataItools.filter_with_qrels(
                                q2positive=train_qrels_list,
                                positive2negative=positive2negative,
                                query_list=query_list,
                                doc_list=doc_list)
        sup_dev_data_list=DataItools.filter_with_qrels(
                                q2positive=dev_qrels_list,
                                positive2negative=positive2negative,
                                query_list=query_list,
                                doc_list=doc_list)
        random.shuffle(sup_train_data_list)
        train_data_path=os.path.join(dest_data_dir,train_data_name)
        dev_data_path=os.path.join(dest_data_dir,dev_data_name)
        DataItools.write_file(data_path=train_data_path,data=sup_train_data_list)
        DataItools.write_file(data_path=dev_data_path,data=sup_dev_data_list)


        # DataItools.split_dataset(data=res,mod='doc',ratio=0.1,dest_data_dir=dest_data_dir)

            
#   class DataProcess():
            
        

    



def timer(func):
    """ time-consuming decorator 
    """
    def wrapper(*args, **kwargs):
        ts = time.time()
        res = func(*args, **kwargs)
        te = time.time()
        print(f"function: `{func.__name__}` running time: {te - ts:.4f} secs")
        return res
    return wrapper


@timer
def snli_preprocess(src_path: str, dst_path:str) -> None:
    """处理原始的中文snli数据
    Args:
        src_path (str): 原始文件地址
        dst_path (str): 输出文件地址
    """
    # 组织数据
    all_data = {}
    with jsonlines.open(src_path, 'r') as reader:
        for line in tqdm(reader):
            sent1 = line.get('sentence1')
            sent2 = line.get('sentence2')
            label = line.get('gold_label')
            if not sent1:
                continue
            if sent1 not in all_data:
                all_data[sent1] = {}
            if label == 'entailment':
                all_data[sent1]['entailment'] = sent2                
            elif label == 'contradiction':
                all_data[sent1]['contradiction'] = sent2  
    # 筛选
    out_data = [
            {'origin': k, 'entailment': v.get('entailment'), 'contradiction': v.get('contradiction')} 
            for k, v in all_data.items() if v.get('entailment') and v.get('contradiction')
        ]
    # 写文件
    with jsonlines.open(dst_path, 'w') as writer:
        writer.write_all(out_data)
        
            
            
if __name__ == '__main__':
    f=codecs.open('/home/chezhonghao/projects/competition/ranking/competition_search/config.yml','r')
    config=yaml.load(f)
    data_base_path=config['data']['data_base_path']
    source_data_dir=config['data']['source_data_dir']
    final4train_dir=config['data']['final4train']
    corpus_path = os.path.join(data_base_path,source_data_dir+'corpus.tsv')
    dest_unsup_train_data_dir=config['data']['unsup_corpus_dir']
    dest_unsup_simcse_corpus_base_data_path=os.path.join(data_base_path,dest_unsup_train_data_dir)
    di=DataItools()
    # corpus_list,length=di.read_documents(data_path=corpus_path,percent=98)
    # di.split_dataset(data=corpus_list,mod='doc',ratio=0.1,dest_data_dir=dest_unsup_simcse_corpus_base_data_path)
    

    qrels_path=os.path.join(data_base_path,source_data_dir+'qrels.train.tsv')
    query_path=os.path.join(data_base_path,source_data_dir+'train.query.txt')

    format_path=config['data']['format']
    final4train_dirs=os.path.join(data_base_path,final4train_dir)
    hard_negative_example_path=os.path.join(data_base_path,'hard_nagative_example/hard_nagative.txt')
    filter_corpus_path=os.path.join(data_base_path,format_path+'filter_corpus_with_qrel.txt')
    # di.get_train_doc(qrels_path=qrels_path,corpus_path=corpus_path,dest_path=filter_corpus_path)
    di.get_train_data(
        qrels_path=qrels_path,query_path=query_path,
        corpus_path=corpus_path,
        hard_negative_rel_path=hard_negative_example_path,
        dest_data_dir=final4train_dirs
        )
    # print(config)
    
    # dev_src, dev_dst = '/home/chezhonghao/projects/competition/ranking/data/SNLI/cnsd_snli_v1.0.dev.jsonl', '/home/chezhonghao/projects/competition/ranking/data/SNLI/dev.txt'
    # test_src, test_dst = '/home/chezhonghao/projects/competition/ranking/data/SNLI/cnsd_snli_v1.0.test.jsonl', '/home/chezhonghao/projects/competition/ranking/data/SNLI/test.txt'
    # train_src, train_dst = '/home/chezhonghao/projects/competition/ranking/data/SNLI/cnsd_snli_v1.0.train.jsonl', '/home/chezhonghao/projects/competition/ranking/data/SNLI/train.txt'
    
    # snli_preprocess(train_src, train_dst)
    # snli_preprocess(test_src, test_dst)
    # snli_preprocess(dev_src, dev_dst)
