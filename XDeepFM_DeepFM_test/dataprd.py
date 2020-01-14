import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from XDeepFM_DeepFM_test import config


class Parse(object):
    def __init__(self):
        self.global_emb_idx = 0
        self.label_num = 0         # 标签数量
        self.single_num = 0        # 单值型
        self.multi_num = 0         # 多值型
        self.train = pd.read_csv(config.train_file, index_col=0)  # 获取数据
        self.test = pd.read_csv(config.test_file, index_col=0)
        self.valid = pd.read_csv(config.valid_file, index_col=0)

        scalar = MinMaxScaler()  # 实例化对象
        all_data = pd.concat([self.train, self.valid, self.test])
        # print(all_data)
        print("transform data ...")
        for s in config.numeric_features:
            scalar.fit(all_data[s].values.reshape(-1, 1))  # 获取每一列的值
            self.train[s] = scalar.transform(self.train[s].values.reshape(-1, 1))  # 按列进行了归一化操作
            self.valid[s] = scalar.transform(self.valid[s].values.reshape(-1, 1))
            self.test[s] = scalar.transform(self.test[s].values.reshape(-1, 1))
            # print(scalar.transform(self.train[s].values.reshape(-1,1)))
            # raise

        self.check()  # 检查尺寸是否一致
        self.num_features = config.numeric_features  # 列表
        self.single_features = config.single_features
        self.multi_features = config.multi_features
        self.backup_dict = {}

        self.num_dict = {}
        self.single_dict = {}
        self.multi_dict = {}
        self.get_dict()
        raise
        # print(self.train)
        self.trans_data(self.train, config.train_save_file)
        self.trans_data(self.valid, config.valid_save_file)
        self.trans_data(self.test, config.test_save_file)

        self.save_conf()

    def get_dict(self):
        print("pepare dict...")
        self.global_emb_idx = 0  # 初始化为 0
        if self.num_features and config.num_embedding:  # 特征名称存在
            for s in self.num_features:  # 遍历每个数值化特征名称，从0 开始编号
                self.num_dict[s] = self.global_emb_idx
                self.global_emb_idx += 1
                # for Nan  对于空值
                self.backup_dict[s] = self.global_emb_idx
                self.global_emb_idx += 1
                # print(s)
                # print(self.global_emb_idx)
            # print(self.global_emb_idx)  # 18   2*9=18
            # raise

        if self.single_features:  # 对于单值特征
            for s in self.single_features:  # 每个field
                # print(s)
                # 每个field
                frequency_dict = {}
                current_dict = {}
                values = pd.concat([self.train, self.valid, self.test])[s]   # 获取对应每个单值列
                # print(values)

                for v in values:  # 遍历每行数据,  统计频次 每个值出现的次数，存入字典  #
                    # print(v)
                    if v in frequency_dict:  #
                        frequency_dict[v] += 1
                    else:
                        frequency_dict[v] = 1



                for k, v in frequency_dict.items():  # 对于频次 大于10 的。存入 single_dict 字典，
                    # 频次小于10 的存入backup 字典
                    # print(k,v)
                    if v > config.single_feature_frequency:  # 频次大于10次的,
                        current_dict[k] = self.global_emb_idx   # 从19开始编号
                        self.global_emb_idx += 1
                    # print(self.global_emb_idx)
                if s =="device_type":
                    print(len(current_dict))
                    raise


                self.single_dict[s] = current_dict
                self.backup_dict[s] = self.global_emb_idx
                # for Nan and low frequency word
                # 为每个filed 留出2个emb的位置来处理不在词典中的值和缺失值
                self.global_emb_idx += 1  # 继续编号下去   # 用于预留缺失值和不在字典中的值对应的feat_index 的
            # print(self.single_dict)
            """
            {'register_type': {0: 18, 1: 19, 2: 20, 7: 21}, 'device_type': {34: 30, 43: 31, 13: 32}
            """
            # print(self.backup_dict)
            """
            {'all_launch_count': 1, 'last_launch': 3, 'all_video_count': 5, 'last_video': 7, 
            'all_video_day': 9, 'all_action_count': 11, 'last_action': 13, 'all_action_day': 15, 
            'register_day': 17, 'register_type': 29, 'device_type': 729}
            """
            # print(self.global_emb_idx)   # 730

        if self.multi_features:  # 多值特征  与上述进行同样的操作
            for s in self.multi_features:
                # 每个field
                frequency_dict = {}
                current_dict = {}
                values = pd.concat([self.train, self.valid, self.test])[s]
                for vs in values:
                    for v in vs.split("|"):
                        v = int(v)
                        if v in frequency_dict:
                            frequency_dict[v] += 1
                        else:
                            frequency_dict[v] = 1

                for k, v in frequency_dict.items():
                    if v > config.multi_feature_frequency:
                        current_dict[k] = self.global_emb_idx
                        self.global_emb_idx += 1
                self.multi_dict[s] = current_dict
                self.backup_dict[s] = self.global_emb_idx
                # for Nan and low frequency word
                # 为每个field留出两个emb的位置来处理不在字典中的缺失值
        print(self.num_dict)
        print(self.single_dict)
        print(self.multi_dict)
        print(self.backup_dict)

    def trans_data(self, data, save_file):   # 数据组装操作
        print("trans data..."+save_file)
        # label index1:value1  index2:value2

        with open(save_file, "w") as f:
            # label, index : value

            def write_to_file(line):  # 每行的操作
                label = line[config.label_name]  # label  值
                f.write(str(label) + ",")   # 转成字符串操作 label,
                self.label_num += 1  # 标签数量
                for s in self.single_features:  # 单值特征 ，遍历每一个特征值
                    now_v = line[s]
                    if now_v in self.single_dict[s]:  # 查询字典
                        now_idx = self.single_dict[s][now_v]  # 获取  获取feat_index值
                    else:
                        now_idx = self.backup_dict[s]  # 直接使用备用字典进行填充
                    f.write(str(now_idx)+":" + str(1)+",")  # feat_index: value , 单行写入
                    self.single_num += 1    # 单值数量+1

                for s in self.num_features:  # 遍历数值型特征
                    now_v = line[s]
                    f.write(str(self.num_dict[s])+":"+str(now_v) + ",")
                    self.single_num += 1
                for s in self.multi_features:  # 遍历多值型特征
                    now_v = line[s]
                    if "|" not in now_v:
                        idxs = [now_v]
                    else:
                        idxs = now_v.split("|")
                    idxs = [x for x in idxs if int(x) in self.multi_dict[s]]
                    print(idxs)
                    if idxs:
                        f.write(str("|".join(idxs))+":" + str(1)+",")
                    else:
                        f.write(str(self.backup_dict[s])+":" + str(1)+",")
                    self.multi_num += 1
                f.write("\n")

            data.apply(lambda x: write_to_file(x), axis=1)  # 对每一列进行操作  data 为一个df 格式数据

    def check(self):  # 检查   列数是不是一致
        if self.train.shape[1] == self.test.shape[1] == self.test.shape[1]:
            return True
        else:
            print("error, all dataset must have same shape")

    # 保存数据处理的信息 总的embeding大小，单值离散型特征数量，数值型特征数量，多值离散型特征数量
    def save_conf(self):  # 进行保存
        with open('data_conf.txt', 'w') as f:
            f.write(str(self.global_emb_idx)+"\t")   # 总共的 index 编号数量
            f.write(str(len(self.single_features))+"\t")   # 单值特征数量  field 个数
            f.write(str(len(self.num_features))+"\t")   # 连续性特征数量  field 个数
            f.write(str(len(self.multi_features)))   # 多值特征数量  field 个数


if __name__ == "__main__":
    pa = Parse()
