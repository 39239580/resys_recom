import numpy as np
from code_tools.dataformat_util.Deepfm_DataReader import ffmasvm2deepfm_v1, DeefmReade


class BacthDataset(object):  # 制作自己的数据集
    def __init__(self, xi, xv, labels):

        rng_state = np.random.get_state()  # 获取随机排列三个列表
        np.random.shuffle(xi)
        np.random.set_state(rng_state)
        np.random.shuffle(xv)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        if isinstance(labels, list):
            xi = np.array(xi)
            xv = np.array(xv)
            labels = np.array(labels)

        self._num_examples = int(len(labels))  # 数据的长度
        self._xi = xi
        self._xv = xv
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._train_lens = 0
        self._test_lens = 0

    @property
    def xi(self):
        return self._xi

    @property
    def xv(self):
        return self._xv

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def test_lens(self):
        return self._test_lens

    @property
    def train_lens(self):
        return self._train_lens

    def next_batch(self, batch_size, shuffle=True):

        start = self._index_in_epoch  # 批量的索引
        # 第一轮开始混合
        if self._epochs_completed == 0 and start == 0 and shuffle:  # 随机打乱
            perm0 = np.arange(self._num_examples)

            # print(perm0)
            np.random.shuffle(perm0)  # 首先进行打乱
            # print(perm0)
            self._xi = self.xi[perm0]
            self._xv = self.xv[perm0]
            self._labels = self.labels[perm0]
        # 开始下一轮
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start  # 剩下的样本数
            xi_rest_part = self._xi[start:self._num_examples]
            xv_rest_part = self._xv[start:self._num_examples]
            label_rest_part = self._labels[start: self._num_examples]

            # 打乱数据
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._xi = self.xi[perm]
                self._xv = self.xv[perm]
                self._labels = self.labels[perm]
            # Start next epoch 开始下一轮
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            xi_new_part = self._xi[start:end]
            xv_new_part = self._xv[start:end]
            label_new_part = self._labels[start:end]

            return np.concatenate((xi_rest_part, xi_new_part), axis=0).tolist(), np.concatenate(
                (xv_rest_part, xv_new_part), axis=0).tolist(), np.concatenate(
                (label_rest_part, label_new_part), axis=0).tolist()

        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._xi[start:end].tolist(), self._xv[start:end].tolist(), self._labels[start:end].tolist()

    def train_test_split(self, test_size=0.3):  # 分割数据集   默认 7:3 的比例进行分割数据集
        batch_size = int(self.num_examples * test_size)  # 测试样本数
        xi_test, xv_test, y_test = self.next_batch(batch_size)
        xi_train, xv_train, y_train = self.next_batch(self.num_examples - batch_size)
        self._test_lens = int(len(y_test))
        self._train_lens = int(len(y_train))
        return xi_train, xi_test, xv_train, xv_test, y_test, y_test


def get_deepfm(filepath, read_way="normal"):
    if read_way == "normal":
        xi, xv, label = ffmasvm2deepfm_v1(filepath=filepath, feat_len=444)
    else:
        deefmreade_out = DeefmReade(trainfile=filepath, method="apply", padding=True, feat_len=444).ffm2deepfm()
        xi, xv, label = deefmreade_out["feat_index"], deefmreade_out["feat_value"], deefmreade_out["label"]
    dataset = BacthDataset(xi, xv, label)
    return dataset


if __name__ == "__main__":
    filepath = "F:/kanshancup/def/FMdata/data/house_price/libffm.txt"
    dataset = get_deepfm(filepath)

    print(dataset.next_batch(20))
    # print(df.head())
