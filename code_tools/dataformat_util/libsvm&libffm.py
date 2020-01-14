import os
import gc
import time
import numpy as np
# from scipy import sparse
import pandas as pd
from functools import partial
from multiprocessing import Pool   # 进程模块
"""
此模块，主要运用单线程或多线程，将非libsvm，libffm 文件等文本文件，转成libsvm或libffm格式文件
，再将转格式后的文件进行离线保存。便于离线计算。
"""
# ---------------------------------------------------------------------
# 单线程
# ---------------------------------------------------------------------


class Converter(object):
    def __init__(self, x, y, file_path, fields=None, threshold=1e-6):
        """
        Write data to libsvm or libffm
        >>  详情参考https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/datasets/svmlight_format.py
        :param X:  array-like,  Feature matrix in numpy or sparse format
        :param y:  array-like,  Label in numpy or sparse format
        :param file_path:  file loacation for writing data to
        :param fields:    An array specifying fields in each columns of X. It should have same length
           as the number of columns in X. When set to None, convert data to libsvm format else
           libffm format.
        :param threshold:   x =x if x>threshold else 0
        :return:
        """
        self.X = x
        self.y = y
        self.file_path = file_path
        self.fields = fields
        self.threshold = threshold

        self.dtype_kind = x.dtype.kind  # 整形  数据类型
        self.is_ffm_format = True if fields is not None else False
        self.X_is_sp = int(hasattr(x, "tocsr"))
        self.y_is_sp = int(hasattr(y, "tocsr"))

        assert len(X.shape) == 2  # 假设等于2

    def label_format(self, value):
        """
        :param value:  label_value
        :return:
        """
        # return "%d" % value if self.dtype_kind == "i" else ("%.6g" % value)
        if self.dtype_kind == "i":
            return "%d" % value
        else:
            return ("%.6g" % value)

    def fm_format(self, feature_id, value):
        """   libsvm  数据格式
        :param feature_id:  feature_index
        :param value: feature_value
        :return:
        """
        # return "%d:%d" % (feature_id, value) if self.dtype_kind == "i" else ("%d:%.6g" %(feature_id))
        if self.dtype_kind == "i":
            return "%d:%d" % (feature_id, value)
        else:
            return ("%d:%.6g" % (feature_id, value))

    def ffm_format(self, field_id, feature_id, value):
        """   libffm 数据格式
        :param field_id:   特征域id()某个字段可以分属多个类别，(对于多分类变量，包含多个列索引同属一个field) [0,0,0,1,2]
        :param feature_id:    特征id(列索引)
        :param value:     特征值
        :return:
        """
        return "%d:%s" % (field_id, self.fm_format(feature_id, value))

    def process_row(self, row_idx):
        """
        :param row_idx: 根据索引行进行转换，行号
        :return:
        """
        if self.X_is_sp:
            span = slice(self.X.indptr[row_idx], self.X.indptr[row_idx + 1])
            x_indices = self.X.indices[span]  # x的索引
            if self.is_ffm_format:
                row = zip(self.fields[x_indices], x_indices, self.X.data[span])
            else:
                row = zip(x_indices, self.X.data[span])
        else:
            nz = self.X[row_idx] != 0
            if self.is_ffm_format:
                row = zip(self.fields[nz], np.where(nz)[0], self.X[row_idx, nz])
            else:
                row = zip(np.where(nz)[0], self.X[row_idx,nz])

        if self.is_ffm_format:
            s = " ".join(self.ffm_format(f, j, x) for f, j, x in row)
        else:
            s = " ".join(self.fm_format(j, x) for j, x in row)

        if self.y_is_sp:
            label_str = self.label_format(self.y.data[row_idx])
        else:
            label_str = self.label_format(self.y[row_idx])

        return "%s %s" % (label_str, s)

    def convert(self):
        with open(self.file_path, "w") as f:
            start = time.time()
            for row_idx in range(self.X.shape[0]):  # 遍历数据每行
                f.write(f"{self.process_row(row_idx)}\n")  # 逐行写入
            print(f"All Finished . cost:{time.time()- start}s")

# ------------------------------------------------------------------------------
#  多线程
# ------------------------------------------------------------------------------


def _label_format(value, dtype_kind):
    if dtype_kind == 'i':
        return "%d" % value
    else:
        return ("%.6g" % value)


def _fm_format(feature_id, value, dtype_kind):
    if dtype_kind == 'i':
        return "%d:%d" % (feature_id, value)
    else:
        return ("%d:%.6g" % (feature_id, value))


def _ffm_format(field_id, feature_id, value, dtype_kind):
    return "%d:%s" % (field_id, _fm_format(feature_id, value, dtype_kind))


def _process_row(x, y, fields, row_idx, x_is_sp, y_is_sp, is_ffm_format, dtype_kind):
    """
    根据行索引转换每一行
    :param row_idx: 行号
    :return:
    """
    if x_is_sp:
        span = slice(x.indptr[row_idx], x.indptr[row_idx + 1])
        x_indices = x.indices[span]
        if is_ffm_format:
            row = zip(fields[x_indices], x_indices, x.data[span])
        else:
            row = zip(x_indices, x.data[span])
    else:
        nz = x[row_idx] != 0
        # print(nz)
        if is_ffm_format:
            row = zip(fields[nz], np.where(nz)[0], x[row_idx, nz])
        else:
            row = zip(np.where(nz)[0], x[row_idx, nz])

    if is_ffm_format:
        s = " ".join(_ffm_format(f, j, x, dtype_kind) for f, j, x in row)
    else:
        s = " ".join(_fm_format(j, x, dtype_kind) for j, x in row)

    if y_is_sp:
        labels_str = _label_format(y.data[row_idx], dtype_kind)
    else:
        labels_str = _label_format(y[row_idx], dtype_kind)

    return "%s %s\n" % (labels_str, s)


def _process_chunk(x_is_sp, y_is_sp, is_ffm_format, dtype_kind, chunk):
    lines = []
    if is_ffm_format:
        x, y, fields = chunk[0], chunk[1], chunk[2]
        print(f'Process-{os.getpid()}, chunk_size={x.shape[0]}')
        for row_idx in range(x.shape[0]):
            lines.append(_process_row(x, y, fields, row_idx, x_is_sp, y_is_sp, is_ffm_format, dtype_kind))
    else:
        x, y = chunk[0], chunk[1]
        print(f'Process-{os.getpid()}, chunk_size={x.shape[0]}')
        for row_idx in range(x.shape[0]):
            lines.append(_process_row(x, y, None, row_idx, x_is_sp, y_is_sp, is_ffm_format, dtype_kind))
    return ''.join(lines)


def convert_by_parallel(x, y, file_path, fields=None, n_jobs=8, chunk_size=10):
    """
    多进程版本，转换为 libsvm、libffm 数据格式
    :param x:           仅支持 scipy.csr_matrix 或 np.array
    :param y:           标签集 np.array
    :param file_path:   保存路径
    :param fields:      特征域 np.array
    :param n_jobs:      进程数
    :param chunk_size:  分块大小
    :return:
    """
    # 根据索引分块
    chunks = []
    for i in range(0, x.shape[0], chunk_size):
        indices = np.arange(i, i+chunk_size if i + chunk_size < x.shape[0] else x.shape[0])
        # print(indices)
        if fields is not None:
            chunks.append((x[indices], y[indices], fields))
        else:
            chunks.append((x[indices], y[indices]))

    is_ffm_format = True if fields is not None else False
    dtype_kind = x.dtype.kind  # i:整型
    x_is_sp = int(hasattr(x, "tocsr"))
    y_is_sp = int(hasattr(y, "tocsr"))

    del x, y
    gc.collect()

    with open(file_path, "w") as f_handle, Pool(processes=n_jobs) as pool:
        start = time.time()
        # 多进程、保证顺序、chunksize(每个进程分配的元素个数)
        for res in pool.imap(func=partial(_process_chunk, x_is_sp, y_is_sp, is_ffm_format, dtype_kind),
                             iterable=chunks, chunksize=10):
            f_handle.write(res)
            f_handle.flush()
        print(f'All Finished. cost: {time.time()-start}s')


def df2txt(params):
    """
    :param params:  文件转换格式的参数
    :return:
    """
    df = pd.read_csv(params["filepath"], header=None, sep="\t")  # 读取数据， 转成 df 格式数据
    # 手动对数据进行拆分 header 不需要 表头数据
    y = df[0]   # label 数据
    x = df[df.columns[1:]]   # x 数据
    ifmultip = params.get("multip", True)
    if ifmultip:  # 开启多进程
        convert_by_parallel(x.values, y.values, params["outputpath"], params.get("fields", None),
                            params.get("n_jobs", 8), params.get("chunk_size", 100))
    else:
        Converter(x.values, y.values, params["outputpath"], params.get("fields", None)).convert()
    # x.values   类型为np.array


if __name__ == '__main__':
    # X = np.random.randint(0, 11, (1000, 10)) * 1e-5  # 0  到10 范围内的1000*10数组
    # X = sparse.csr_matrix(np.random.randint(0, 5, (100, 5)))
    # y = np.random.randint(0, 2, 1000)  # 0  到10 范围内的1000*10数组
    # print(X)
    # print(y)
    # fields = np.random.randint(0, 11, 10)
    # convert_by_parallel(X, y, 'tmp01.txt', fields=fields, n_jobs=8, chunk_size=100)   # 多线程
    # ctr = Converter(X, y, file_path="tmp02.txt", fields=fields)   # 单线程处理
    # ctr.convert()

    params = {"filepath": "F:/kanshancup/def/FMdata/data/house_price/house_price_test.txt",
              "outputpath": "F:/kanshancup/def/FMdata/data/house_price/libffm.txt"}

    df2txt(params)  # 数据转换
