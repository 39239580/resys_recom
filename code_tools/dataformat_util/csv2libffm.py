# import numpy as np
import pandas as pd
import xlearn as xl
# import pickle as pl

"""
在线将csv格式文件数据转成libffm的格式
"""
"""
csv 文件转换libffm
csv 文件格式为
   y value_1 value_2 .. value_n

   0      0.1     0.2     0.2   ...
   1      0.2     0.3     0.1   ...
   0      0.1     0.2     0.4   ...
"""


def read_csv(filepath):
    data = pd.read_csv(filepath, header=None,  sep="\t")
    return data


def split_x_y(df):
    x = df[df.columns[1:]]   # 除了第一列其余的值
    y = df[0]   # 第一列的值
    return x, y


def change2liffm(df_data, usefield=False, field_map=None):  # 转成 Dmatrix
    """
    :param df_data:   df 数据
    :param usefield:   #  是否使用
    :param field_map:   Dataframe/series, array or list   推荐使用list
    :return:
    """
    x_, y_ = split_x_y(df_data)
    if usefield:
        xdm_data = xl.DMatrix(x_, y_, field_map)
    else:
        xdm_data = xl.DMatrix(x_, y_)
    return xdm_data


def get_libffm(filepath, usefield=False, field_map=None):
    print("ok")
    return change2liffm(read_csv(filepath), usefield, field_map)


if __name__ == "__main__":
    filepath = "F:/kanshancup/def/FMdata/data/house_price/house_price_train.txt"
    df = get_libffm(filepath)   # 数据格式为DMatrix
    print(df)
