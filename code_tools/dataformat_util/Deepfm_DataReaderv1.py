from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MultiLabelBinarizer


def data_process(data, process_type):  # 特征处理
    if process_type == "Binary":  # 二值化处理
        processmodule = Binarizer(copy=True, threshold=0.0)
        # 大于 threshold 的映射为1， 小于 threshold 的映射为0

    elif process_type == "MinMax":  # 归一化处理
        processmodule = MinMaxScaler(feature_range=(0, 1), copy=True)

    elif process_type == "Stand":  # 标准化处理
        processmodule = StandardScaler(copy=True, with_mean=True, with_std=True)

    elif process_type == "Normal":
        processmodule = Normalizer(copy=True, norm="l2")

    elif process_type == "MultiLabelBinar":   # 多标签2值话处理
        processmodule = MultiLabelBinarizer(sparse_output=True)

    else:
        raise ValueError("please select a correct process_type")

    result = processmodule.fit_transform(data)
    return result


if __name__ == "__main__":
    # data1 = [[1., -1.,  2.], [2.,  0.,  0.], [0.,  1., -1.]]  # 二值化
    # print(data1)
    # print(data_process(data1, "Binary"))

    # data2 = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]  # 极大极小值归一化
    # print(data2)
    # print(data_process(data2, "MinMax"))

    # data3 = [[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]]  # 正则化
    # print(data3)
    # print(data_process(data3, "Normal"))

    # data4 = [[0, 0], [0, 0], [1, 1], [1, 1]]
    # print(data4)
    # print(data_process(data4, "Stand"))   # 标准化

    data5 = [(1, 2), (3, 4), (5,)]
    print(data5)
    print(data_process(data5, "MultiLabelBinar"))
