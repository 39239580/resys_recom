# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:31:38 2019

@author: Administrator
"""
#import numpy as np
#import pandas as pd
#import gc
#import time
import os
import pickle as pk
from collections import Counter
import pandas as pd
import gc
pd.set_option('display.max_columns', None)
pkl_path ="../pkl_data"
save_pak_path ="../second_pkl_data"

def read_precoss_pkl(filename):   # 加载数据
    with open(os.path.join(pkl_path,filename),"rb") as file:
        df=pk.load(file)
    return df

"""
总的数据结构
1.topic     两列  topicId:int    topicEmbed:list
2.single    两列  singleId:int   singleEmbeded:list
3.word      两列  wordId: int   wordEmbeded:list
4.member    16列  uid:int  Gender:3c  VisitFre:5c  BinFeat_A:2c  BinFeat_B:2c  BinFeat_C:2c
                  BinFeat_D:2c BinFeat_E:2c ClassFeat_A:2c ClassFeat_B:2c ClassFeat_C:2c 
                  ClassFeat_D:2c ClassFeat_E:2  SaltValue:int   CareTopicSerise:list
                  InterestTopicSerise:dict
5.question  8列  qid:int, qTitleSingleSerise: list, qTitleWordSerise:list,
                 qDiscribSingleSerise:list, qDiscribWordSerise:list, topicId:list
                 day:int   hour:int
6.invite_train    5列  qid:int  uid:int   hour:int  day:int  Alabel:2c  2c代表2分类 
7.invite_test     4列  qid:int  uid:int   hour:int  day:int 
8.answer               ID:int   qid:int   AuthorId:int   day:int  hour:int   ContextSingleSerise:list
                       ContextWordSerise:list  IsFine:2c  IsRecommed:2  IsRecive:2c IsImage:2c  IsVideo:2c 
                       AnswerNum:int   RecThumb-up:int  RecThumb-cancel:int
                       RecCommentNum:int   Recollect:int   ReThanks:int  ReComplain:int 
                       ReNoUse:int   ReAgainst:int
将上述数据进行拆分，1-3不动,6与7 不动
将4 中的列表改成统计量的形式,即将用户关心的话题，转成字典的形式  保存。
将5  list也改成字典的形式。
将8  list也改成字典的形式。
将上述的文件进行组装，整形数据进行一起保存， 将list 转成字典一起保存。将文件进行拆分
"""

def pk2dict(params):
    df=read_precoss_pkl(params["filename"])  # 读取数据
#    print(type(df))
    print(df.head(2))  # 显示前两行
    filename=params["filename"].split(".")[0]
    for col in params["serise_list"]:
        df[col]=df[col].apply(list2dict)
    print(df.head(2))
    if params["data_type"] =="member": # 将dict 数据与int 数据进行分开保存
        split2n(df,filename,params["col_name"],["uid"])
    elif params["data_type"] =="answer": # 将dict 数据与int 数据进行分开保存
        split2n(df,filename,params["col_name"],["ID","qid","AuthorId"])
    elif params["data_type"] =="question": # 将dict 数据与int 数据进行分开保存
        split2n(df,filename,params["col_name"],["qid"])
    gc.collect()
        
#    with open(os.path.join(save_pak_path,filename),"wb") as file:
#        pk.dump(df,file)


def split2n(df,filename,col_name,add_col_name):  # 将数据进行切割， list类型的数据与 数值型的数据进行分开保存
    df1=df[col_name+add_col_name]
    all_columName=df.columns.values.tolist()
    df2_columName=list(set(all_columName)-set(col_name))
    df2=df[df2_columName]
    del df
    print('///////////////////////////////////')
    print(df1.head(2))
    print(df2.head(2))
    with open(os.path.join(save_pak_path,filename+"_value.pkl"),"wb") as file:
        pk.dump(df1,file)
    with open(os.path.join(save_pak_path,filename+"_dict.pkl"),"wb") as file:
        pk.dump(df2,file)
    print("After preservation")
    del df1,df2
    


def list2dict(df):# 统计用户关心 的主题
    dict0=Counter(df)
    return {k:v/float(len(df)) for k,v in dict0.items()}


all_dic=[{"data_type":"member",
        "filename":"member_info_0926.pkl",
        "serise_list":["CareTopicSerise"],  # 被处理的列
        "col_name":["CareTopicSerise","InterestTopicSerise"]
        },
        {"data_type":"question",
        "filename":"question_info_0926.pkl",
        "serise_list":["qTitleSingleSerise", "qTitleWordSerise",
                       "qDiscribSingleSerise", "qDiscribWordSerise", "topicId"],  # 被处理的列
        "col_name":["qTitleSingleSerise","qTitleWordSerise",
             "qDiscribSingleSerise", "qDiscribWordSerise", "topicId"]
                },
        {"data_type":"answer",
        "filename":"question_info_0926.pkl",
        "serise_list":["ContextSingleSerise","ContextWordSerise"],  # 被处理的列
        "col_name":["ContextSingleSerise","ContextWordSerise"]
                },
        ]



def split_log():
    if not os.path.exists(save_pak_path):
        print("create dir:%s"%save_pak_path)
        os.mkdir(save_pak_path) 
    for new_dict in all_dic:
        if new_dict["data_type"] =="answer":
            for i in range(5):
                new_dict["filename"]="answer_info_0926"+"_"+str(i)+".pkl"
                pk2dict(new_dict)
        else:    
            pk2dict(new_dict)
        
def check_df(filename):
    with open(os.path.join(save_pak_path,filename+"_value.pkl"),"rb") as file:
        print(pk.load(file).head())



if __name__=="__main__":
    split_log()
#    check_df()
