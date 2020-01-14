from sklearn.cluster import KMeans
import numpy as np
"""
api接口地址 http://lijiancheng0614.github.io/scikit-learn/modules/generated/sklearn.cluster.KMeans.html#
sklearn.cluster.KMeans
"""


class KMEANS(object):
    def __init__(self, n_clusters=8,  init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs=None, algorithm='auto'):
        """
        :param n_clusters:  簇数，int 可选项
        :param init:  "k-means++", "randoms"  或一个ndarray ,默认kmeans++
        :param n_init: int 默认10
        :param max_iter:  最大迭代数 int  默认300
        :param tol:关于惯性宣布收敛的相对容忍度 1e-4,
        :param precompute_distances:  "auto" True, False,预计算距离，快速但耗内存。
        自动时，如果n_samples * n_clusters > 1200万，则不计算距离. True,计算， False 不计算
        :param verbose: 默认为0， int， 冗余模型
        :param random_state:  随机种子
        :param copy_x:
        :param n_jobs:  # 进程数  int
        :param algorithm: 默认为auto, {"auto", "full", "elkan"},
        """
        self.model = KMeans(n_clusters=n_clusters, init=init, n_init=n_init,
                            max_iter=max_iter, tol=tol, precompute_distances=precompute_distances,
                            verbose=verbose, random_state=random_state, copy_x=copy_x,
                            n_jobs=n_jobs, algorithm=algorithm)

    def fit(self, x, y=None, sample_weight=None):  # 计算k-means 聚类
        self.model.fit(X=x, y=y, sample_weight=sample_weight)

    def transform(self, x):  # 将x 转换到簇距离空间中
        return self.model.transform(X=x)

    def fit_transform(self, x, y=None, sample_weight=None):
        return self.model.fit_transform(X=x, y=y, sample_weight=sample_weight)

    def fit_predict(self, x, y=None, sample_weight=None):  # == fit+predict 计算簇中心预测每个样本归属簇
        return self.model.fit_predict(X=x, y=y, sample_weight=sample_weight)

    def get_params(self, deep):
        return self.model.get_params(deep=deep)

    def predict(self, x):  # U预测样本属于最近的哪个簇
        return self.model.predict(X=x)

    def score(self, x, y=None, sample_weight=None):  # 与k-均值目标上的x值相反
        return self.model.score(X=x, y=y, sample_weight=sample_weight)

    def set_params(self, **params):
        self.model.set_params(**params)

    def get_attributes(self):   # 生成模型之后，才能获取
        cluster_centers = self.model.cluster_centers_  # 簇中心坐标 array, [n_clusters, n_features]
        labels = self.model.labels_  # 每个点的标签
        inertia = self.model.inertia_  # 样本到距离他们最近的簇中心的距离
        n_iter = self.model.n_iter_   # 运行的迭代数
        return cluster_centers, labels, inertia, n_iter


if __name__ == "__main__":
    x = np.array([[1, 2], [1, 4], [1, 0],
                  [10, 2], [10, 4], [10, 0]])
    kmeans = KMEANS(n_clusters=2, random_state=0)

    kmeans.fit(x)

    cluster_center, label, inertias, n_iters = kmeans.get_attributes()
    print(cluster_center)
    print(label)
    print(inertias)
    print(n_iters)
