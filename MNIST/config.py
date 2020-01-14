# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 12:02:01 2019

@author: Administrator
"""


# file
train_file = 'data/dataset1.csv'
valid_file = 'data/dataset2.csv'
test_file = 'data/dataset3.csv'

train_save_file = 'data/dataset1.txt'
valid_save_file = 'data/dataset2.txt'
test_save_file = 'data/dataset3.txt'

label_name = 'label'  # 标签

# features
numeric_features = ['all_launch_count', 'last_launch', 'all_video_count', 'last_video', 'all_video_day',
                    'all_action_count', 'last_action',
                    'all_action_day', 'register_day']  # 数值特征9个
single_features = ['register_type', 'device_type']   # 单值特征2个
multi_features = []

num_embedding = True
single_feature_frequency = 10   
multi_feature_frequency = 0

# model, 默认使用DFM模型操作

FM_layer = True
DNN_layer = True
CIN_layer = False

use_numerical_embedding = False


embedding_size = 16

dnn_net_size = [128,64,32]   # DNN 三层
cross_layer_size = [10,10,10]  #cin 部分也是3层
cross_direct = False  # 不直接输出
cross_output_size = 1 # 交叉输出尺寸1

# train
batch_size = 4096
epochs = 4000   # 4000千轮，将整个数据都训练一遍
learning_rate = 0.01