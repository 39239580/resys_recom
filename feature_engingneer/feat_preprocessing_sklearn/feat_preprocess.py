from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MultiLabelBinarizer


class DataPrProcess(object):
    def __init__(self, process_type):  # 特征处理
        if process_type == "Binary":  # 二值化处理
            self.processmodule = Binarizer(copy=True, threshold=0.0)
            # 大于 threshold 的映射为1， 小于 threshold 的映射为0

        elif process_type == "MinMax":  # 归一化处理
            self.processmodule = MinMaxScaler(feature_range=(0, 1), copy=True)

        elif process_type == "Stand":  # 标准化处理
            self.processmodule = StandardScaler(copy=True, with_mean=True, with_std=True)

        elif process_type == "Normal":
            self.processmodule = Normalizer(copy=True, norm="l2")

        elif process_type == "MultiLabelBinar":   # 多标签2值话处理
            self.processmodule = MultiLabelBinarizer(sparse_output=True)

        else:
            raise ValueError("please select a correct process_type")

    def _fit(self, data):   # 无输出
        self.processmodule.fit(data)

    def _transform(self, data):  # 有输出
        return self.processmodule.transform(data)

    def _fit_transform(self, data):
        return self.processmodule.fit_transform(data)


if __name__ == "__main__":
    data1 = [[1., -1.,  2.], [2.,  0.,  0.], [0.,  1., -1.]]  # 二值化
    print(data1)
    print(DataPrProcess("Binary")._fit_transform(data1))

    data2 = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]  # 极大极小值归一化
    print(data2)
    print(DataPrProcess("MinMax")._fit_transform(data2))

    data3 = [[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]]  # 正则化
    print(data3)
    print(DataPrProcess("Normal")._fit_transform(data3))

    data4 = [[0, 0], [0, 0], [1, 1], [1, 1]]
    print(data4)
    print(DataPrProcess("Stand")._fit_transform(data4))   # 标准化

    data5 = [(1, 2), (3, 4), (5,)]
    print(data5)
    print(DataPrProcess("MultiLabelBinar")._fit_transform(data5))
