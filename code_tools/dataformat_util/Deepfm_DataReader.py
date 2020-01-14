import pandas as pd
import time
# from threading import Thread   #  线程模块
# from multiprocessing  import Process,Pool   # 进程模块
# from multiprocessing import Value, Array  #  内存共享模块

pd.set_option('display.max_columns', None)

"""
 数据格式
 libsvm format:

   y index_1:value_1 index_2:value_2 ... index_n:value_n

   0   0:0.1   1:0.5   3:0.2   ...
   0   0:0.2   2:0.3   5:0.1   ...
   1   0:0.2   2:0.3   5:0.1   ...

CSV format:

   y value_1 value_2 .. value_n

   0      0.1     0.2     0.2   ...
   1      0.2     0.3     0.1   ...
   0      0.1     0.2     0.4   ...

libffm format:

   y field_1:index_1:value_1 field_2:index_2:value_2   ...

   0   0:0:0.1   1:1:0.5   2:3:0.2   ...
   0   0:0:0.2   1:2:0.3   2:5:0.1   ...
   1   0:0:0.2   1:2:0.3   2:5:0.1   ...

deepfm format
 xi:[[ind1_1, ind1_2, ...],...,[indi_1, ..., indi_j, ...]]  indi_j   第i个样本 第j个字段的索引值
 xv:[[val1_1, val1_2, ...],...,[vali_1, ..., vali_j, ...]]  vali_j   第i个样本 第j个字段的值
 y: [label1,...,labeli,....]   第i个样本的标签，1/0分类 ，或直接使用量化数字进行 回归处理

 对于上述样本

 xi :[[0,1,3],[0,2,5],[0,2,5]]
 xv :[[0.1,0.5,0.2],[0.2,0.3,0.1],[0.2,0.3,0.1]]
 y =[0,0,1]
"""


def ffmasvm2deepfm_v1(filepath, feat_len):  # 简单版本
    #  直接使用csv 进行读取， 进行数据补全
    #  libsvm  与 libffm进行数据格式转换
    """
    :param filepath:  libffm 文件路径
    :param feat_len:
    :return:
    """
    label = []
    xi = []
    xv = []
    norm_feat = [i for i in range(feat_len)]
    with open(filepath, "r") as f:
        f = f.readlines()
        for line in f:  # 每行数据
            # print(line)
            new_context = line.split(" ")
            # print(new_context[0])
            label.append(int(new_context[0]))
            # print(label)
            xii = []
            xvv = []
            new_context.pop(0)
            for feat_context in new_context:
                new_feat = feat_context.split(":")
                # print(new_feat)
                xii.append(int(new_feat[-2]))
                xvv.append(float(new_feat[-1]))

            diff = list(set(norm_feat) ^ set(xii))
            xii = xii+diff
            xvv = xvv + [0.0]*len(diff)
            xi.append(xii)
            xv.append(xvv)

    return xi, xv, label


class DeefmReade(object):
    def __init__(self,trainfile=None, testfile=None,
                 dftrain=None, dftest=None,
                 method="apply",
                 feat_len=None,
                 padding=False,
                 ismulprocess=True
                 ):
        """
        :param trainfile:   # 训练集文件
        :param testfile:   # 测试集文件
        :param dftrain:
        :param dftest:
        :param method:  # 使用哪种操作方式对pd进行处理
        :param feat_len:  # 进行最大特征feature_index 的长度
        :param padding:   # 是否进行补零
        :param ismulprocess: # 是否使用多线程
        训练集与测试集可以只有一个
        """
        assert not((trainfile is None) and (dftrain is None))  # 均不能为空
        assert not((trainfile is not None) and (dftrain is not None))  # 均不能同时出现
        assert ((padding is True) and (feat_len is not None))  # 补零操作，且必须指定最大feat_index长度
        self.trainfile = trainfile
        self.testfile = testfile
        self.dftrain = dftrain
        self.dftest = dftest
        self.method = method
        self.feat_len = feat_len
        self.padding = padding
        self.ismulprocess = ismulprocess
        if not self.feat_len:
            row_num, col_num = pd.read_csv(filepathx, header=None, sep=" ").shape
            self.feat_len = col_num
        #        print(self.feat_len)
        self.norm_feat = [i for i in range(self.feat_len)]

    def ffm2deepfm(self):  #libsvm 或者libffm  数据转deepfm 格式操作
        # start=time.time()
        if self.dftrain:
            dftrain = self.dftrain
        else:
            dftrain = pd.read_csv(self.trainfile, header=None)

        if self.dftest:
            dftest = self.dftest
        elif self.testfile:
            dftest = pd.read_csv(self.testfile, header=None)
        else:
            dftest = None

        if dftest:
            new_df = pd.concat([dftrain, dftest], axis=1)  # 按列进行拼接
        else:
            new_df = dftrain
#        print(new_df.head(2))

        # way1 apply 
        if self.method == "apply":
            new_df = new_df.apply(lambda x: self.row_process(x), axis=1)
        # way2 applymap
        else:
            new_df = new_df.applymap(lambda x: self.row_process(x))

        # print(new_df)
        # t1 = time.time()
        # print("耗时1：%.8f s"%(t1-start))
        deepfm_out = {"label": [],
                      "feat_index": [],
                      "feat_value": []}

        for rows in new_df.values:
            if self.method == "applymap":
                rows = rows[0]
            deepfm_out["label"].append(int(rows["label"]))
            deepfm_out["feat_index"].append(rows["feat_index"])
            deepfm_out["feat_value"].append(rows["feat_value"])
        # print("耗时2：%.8f s" % (time.time() - t1))
        return deepfm_out

    def row_process(self, rows):
        if self.method == "apply":
            rows = rows.values.tolist()[0].split(" ")
        else:  # applymap
            rows = rows.split(" ")
        label = rows.pop(0)
        feat_dict = {"label": label,
                     "feat_index": [],
                     "feat_value": []}
        # t0 = time.time()
        for per_row in rows:  # 改进
            feat_dict = self.feat_process(per_row, feat_dict)
        # print("耗时wei：%.8f s" % (time.time() - t0))
        if self.padding:
            diff = list(set(self.norm_feat) ^ set(feat_dict["feat_index"]))
#            print(diff)
            feat_dict["feat_index"] += diff
            feat_dict["feat_value"] += len(diff)*[0.0]        
        return feat_dict

    def feat_process(self, feat, feat_dict):   
        feat_dict["feat_index"].append(int(feat.split(":")[-2]))
        feat_dict["feat_value"].append(float(feat.split(":")[-1]))
        return feat_dict

    # def multp_process(self, n_job = 8 ):  # 默认使用8进程  操作  后期进行改进
    #     pool = Pool(Process=n_job)
    #     pool.join()


if __name__ == "__main__":
    # filepath = "./tmp02.txt"
    t1 = time.time()
    filepathx = "F:/kanshancup/def/FMdata/data/house_price/libffm.txt"
    #
    xi, xv, label = ffmasvm2deepfm_v1(filepath=filepathx, feat_len=444)
    t2 = time.time()
    print("花费时间：%.6f s" % (t2-t1))
    dfR = DeefmReade(trainfile=filepathx, method="apply", padding=True, feat_len=444)
    ffm2deepfm = dfR.ffm2deepfm()
    # print(ffm2deepfm)
    xii, xvv, labell = ffm2deepfm["feat_index"], ffm2deepfm["feat_value"], ffm2deepfm["label"]
    t3 = time.time()
    print("花费时间：%.6f s" % (t3-t2))
    # 测试读取数据正常
    #   读取数据正常


#    print(ffm2deepfm["label"])
#    print(ffm2deepfm["feat_index"])
#    print(len(ffm2deepfm["feat_value"][0]))
#    q=pd.read_csv(filepathx,header=None,sep=" ")
#    print(q.shape)

    # print(xv)
    print(type(xv))
    for c1 in xv:
        print(c1)
    time.sleep(2)
    print(type(xvv))
    for c2 in xvv:
        print(c2)
