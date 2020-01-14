from XDeepFM_DeepFM_test.dataprd import Parse
from code_tools.FM_util.XdeepFM_batch_next import ffmasvm2deepfm_v1
import pandas as pd
from XDeepFM_DeepFM_test.config import embedding_size

def feat_process():
    # 特征转换并保存
    # pa = Parse()
    with open("data_conf.txt", "r") as f:
        line = f.readline()
        line = line.split("\t")
        total_emb_len, single_size, numerical_size, multi_size = int(line[0]), int(line[1]), int(line[2]),int(line(3))

    field_size = single_size + numerical_size + multi_size  # field个数
    embedding_length = field_size * embedding_size

    return field_size ,embedding_length


    # xi, xv, label = ffmasvm2deepfm_v1(filepath="./data/dataset1.txt", feat_len=feat_index_len)




field_sie, embedding_length = feat_process()  # 获取的 字段个数，





