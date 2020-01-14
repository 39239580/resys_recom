# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:42:13 2019

@author: Administrator
"""

# 先处理数据，将数据格式转成代用的
import pandas as pd
import os
import numpy as np
import re
import pickle as pk
import gc

pd.set_option('display.max_columns', None)
data_path ="../original_data"
pkl_data_path ="../pkl_data"
gender_embed={"unknown":0,
              "male":1,
              "female":2}

visitfre_embed={"new":0,
                "daily":1,
                "weekly":2,
                "monthly":3,
                "unknown":4}
# 对嵌入 进行重新编码
def parse_str(d):
    return np.array(list(map(float,d.split()))) # 转成数组的形式 
    # 做的操作为 将"1 2 3"  转成列表的结果是  ['1','2','3']

# 解析数字
def parse_num(d):
    return "".join(re.findall(r"\d+",d))

def parse_gender(d):
#    print(d)
    return gender_embed[d]
    
def parse_visit_fre(d):
    return visitfre_embed[d]
    
def parse_care_topic(d):  # 对关心话题进行解析
    if d=="-1":
        return[0]
    else:
        num=list(map(lambda x: int(x), re.findall(r"\d+",d)))
        return num

def parse_interest_topic(d): # 对感兴趣话题进行解析
    if d=="-1":
        return {}
    else:
        return dict([int(z.split(":")[0][1:]),float(z.split(":")[1])] for z in d.split(","))

def reduce_mem_usage(df):
    """
    修改数据类型减少内存的占用
    """
    start_mem = df.memory_usage().sum()/1024**2
    print("Memory usage of dataframe is {:.2f}MB".format(start_mem))
    for col in df.columns: #遍历所有的列
        col_type =df[col].dtype
        if col_type!=object:
            c_min =df[col].min()
            c_max =df[col].max()
            if str(col_type)[:3] =="int":
                if c_min >np.iinfo(np.int8).min and c_max< np.iinfo(np.int8).max:  # 根据数据范围，转成响应的数据范围
                    df[col] =df[col].astype(np.int8)
                elif c_min >np.iinfo(np.int16).min and c_max<np.iinfo(np.int16).max:
                    df[col] =df[col].astype(np.int16)
                elif c_min >np.iinfo(np.int32).min and c_max<np.iinfo(np.int32).max:
                    df[col] =df[col].astype(np.int32)
                elif c_min >np.iinfo(np.int64).min  and c_max <np.iinfo(np.int64).max:
                    df[col] =df[col].astype(np.int64)
    end_mem =df.memory_usage().sum()/1024**2
    print("Memory usage after optimization is:{:.2f}MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100*(start_mem -end_mem)/start_mem))
    return df

def block_preprocess(file_name,params,chunk_size=1000000): # 块处理  默认处理100W条
    """
    分块读入数据模块，
    """
    print("re")
    reader=pd.read_csv(os.path.join(data_path,file_name),names =params,header =0,
                   iterator=True,encoding="utf-8",sep ="\t")  #文本文件读取对象

    print("nwefuw")
    loop =True
    cnt =0
    while loop:
        try:
            chunk =reader.get_chunk(chunk_size)
            print(chunk.head(2))
            newdf=preprocess_answaer_mudle(chunk,params)
            print(newdf.head(2))
            with open(os.path.join(pkl_data_path,file_name.split(".")[0]+"_"+str(cnt)+".pkl"),"wb") as file:
                pk.dump(newdf,file)
            cnt+=1
            del chunk,newdf 
        except StopIteration:
            loop = False
            print("Iteration is stopped!")
            del cnt
            gc.collect()
            
#    return newdf
def preprocess_answaer_mudle(df,params):
    """
    预处理回答数据模块
    """
    for i in range(3):
        df[params[i]] =df[params[i]].apply(parse_num)
    df["day"]=df["AnswerTime"].apply(lambda x:int(x.split('-')[0][1:])).astype(np.int16)
    df["hour"]=df["AnswerTime"].apply(lambda x:int(x.split('-')[1][1:])).astype(np.int8)
    for i in range(4,6):
        df[params[i]] =df[params[i]].apply(lambda x: parse_care_topic(x)) 
    df.drop(["AnswerTime"],axis=1,inplace=True)  # c删除6列用不到的数据
    df =reduce_mem_usage(df)
    return df

def read_txt(file_name,params,data_type):  # 
    """
    读取原始的文件,并进行简单的处理
    """
    try:
        if data_type in ["vectors","question","answer","member","invite"]:
            if data_type =="vectors":
                df=pd.read_csv(os.path.join(data_path,file_name),names=params,sep="\t")  # 第一列与后面的列中间为跳格键隔开
                
                print(df.head(2))
                df[params[0]]=df[params[0]].apply(parse_num)  # 去掉字母 剩下数字
                df[params[1]] =df[params[1]].apply(parse_str)  # 将数字变成一个列表
                save_pkl(df,file_name.split(".")[0]+".pkl",data_type)
            elif data_type =="member":
                df=pd.read_csv(os.path.join(data_path,file_name),names =params,sep="\t")
                print(df.head(2))
                df["uid"]=df["uid"].apply(parse_num)  # 去掉字母 剩下数字
                df["Gender"]=df["Gender"].apply(lambda x: parse_gender(x))  # 性别编码
                df["VisitFre"]=df["VisitFre"].apply(lambda x:parse_visit_fre(x)) # 参观频率编码
                df["CareTopicSerise"]=df["CareTopicSerise"].apply(lambda x: parse_care_topic(x)) 
                df["InterestTopicSerise"]=df["InterestTopicSerise"].apply(parse_interest_topic)
                df.drop(["CreateKeySerise","CreatNumLevel","CreateHotLevel",
                         "RegisterType","RegisterPlat"],axis=1,inplace=True)  # c删除6列用不到的数据
                print(df.head(2))
                df =reduce_mem_usage(df)
                save_pkl(df,file_name.split(".")[0]+".pkl",data_type)
            elif data_type =="invite":  # 邀请文件
                df =pd.read_csv(os.path.join(data_path,file_name),names =params,sep="\t") #
                print(df.head(2))
                df["qid"]=df["qid"].apply(parse_num)
                df["uid"]=df["uid"].apply(parse_num)
                df["day"]=df["CreateTime"].apply(lambda x:int(x.split('-')[0][1:])).astype(np.int16)
                df["hour"]=df["CreateTime"].apply(lambda x:int(x.split('-')[1][1:])).astype(np.int8)
                df.drop(["CreateTime"],axis=1,inplace=True)  # c删除6列用不到的数据
                df =reduce_mem_usage(df)
                print(df.head(2))
                save_pkl(df,file_name.split(".")[0]+".pkl",data_type)
            elif data_type =="question":
                df=pd.read_csv(os.path.join(data_path,file_name),names =params,sep ="\t")
                print(df.head(2))
                df["qid"] =df["qid"].apply(parse_num)
                df["day"]=df["CreateTime"].apply(lambda x:int(x.split('-')[0][1:])).astype(np.int16)
                df["hour"]=df["CreateTime"].apply(lambda x:int(x.split('-')[1][1:])).astype(np.int8)
                df["qTitleSingleSerise"] =df["qTitleSingleSerise"].apply(lambda x: parse_care_topic(x)) 
                df["qTitleWordSerise"] =df["qTitleWordSerise"].apply(lambda x: parse_care_topic(x)) 
                df["qDiscribSingleSerise"] =df["qDiscribSingleSerise"].apply(lambda x: parse_care_topic(x)) 
                df["qDiscribWordSerise"] =df["qDiscribWordSerise"].apply(lambda x: parse_care_topic(x)) 
                df["topicId"] =df["topicId"].apply(lambda x: parse_care_topic(x)) 
                df =reduce_mem_usage(df)
                df.drop(["CreateTime"],axis=1,inplace=True)  # c删除6列用不到的数据
                print(df.head(2))
                save_pkl(df,file_name.split(".")[0]+".pkl",data_type)
            elif data_type =="answer":  # 回答模块由于数据量庞大，导致一次读取数据可能会引爆内存条
                print("ok")
                block_preprocess(file_name,params) #默认100万行数据
#                print(df.head(2))
            del df
            gc.collect()
    except Exception as e:
        print(e)
        gc.collect()
#    print(df.head(2))
    
def save_pkl(df,filename,data_type):  # 保存文件
    if data_type =="vectors": # 向量类
        with open(os.path.join(pkl_data_path,filename),"wb") as file:
            pk.dump(df,file)
    elif data_type =="member":
        with open(os.path.join(pkl_data_path,filename),"wb") as file:
            pk.dump(df,file)
    elif data_type =="invite":
        with open(os.path.join(pkl_data_path,filename),"wb") as file:
            pk.dump(df,file)
    elif data_type =="question":
        with open(os.path.join(pkl_data_path,filename),"wb") as file:
            pk.dump(df,file)
    elif data_type =="answer":
        with open(os.path.join(pkl_data_path,filename),"wb") as file:
            pk.dump(df,file)

params={"topic":["topicId","topicEmbed"],
        "word":["wordId","wordEmbed"],
        "single_word":["singleId","singleEmbed"],
        "member":["uid","Gender","CreateKeySerise","CreatNumLevel","CreateHotLevel",
                  "RegisterType","RegisterPlat","VisitFre","BinFeat_A","BinFeat_B",
                  "BinFeat_C","BinFeat_D","BinFeat_E","ClassFeat_A","ClassFeat_B",
                  "ClassFeat_C","ClassFeat_D","ClassFeat_E","SaltValue","CareTopicSerise",
                  "InterestTopicSerise"],
        "invite_val":["qid","uid","CreateTime"],
        "invite_train":["qid","uid","CreateTime","ALabel"],
        "answer":["ID","qid","AuthorId","AnswerTime","ContextSingleSerise","ContextWordSerise","IsFine",
                  "IsRecommed","IsRecive","IsImage","IsVideo","AnswerNum","RecThumb-up","RecThumb-cancel",
                  "RecCommentNum","Recollect","ReThanks","ReComplain","ReNoUse","ReAgainst"],
        "question":["qid","CreateTime","qTitleSingleSerise","qTitleWordSerise",
                    "qDiscribSingleSerise","qDiscribWordSerise","topicId"]
        }

def sample_process():
    if not os.path.exists(pkl_data_path):
        print("create dir:%s"%pkl_data_path)
        os.mkdir(pkl_data_path)
    read_txt("topic_vectors_64d.txt",params["topic"],"vectors")

    read_txt("word_vectors_64d.txt",params["word"],"vectors")
    read_txt("single_word_vectors_64d.txt",params["single_word"],"vectors")
    read_txt("member_info_0926.txt",params["member"],"member")
    read_txt("invite_info_0926.txt",params["invite_train"],"invite")
    read_txt("invite_info_evaluate_1_0926.txt",params["invite_val"],"invite")

    read_txt("question_info_0926.txt",params["question"],"question")
    read_txt("answer_info_0926.txt",params["answer"],"answer")


if __name__=="__main__":
    sample_process()
    