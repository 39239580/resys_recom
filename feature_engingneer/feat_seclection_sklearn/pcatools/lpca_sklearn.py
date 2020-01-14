from sklearn.decomposition import IncrementalPCA
from sklearn.datasets import load_digits
from scipy import sparse
"""
API接口地址 https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html#
sklearn.decomposition.IncrementalPCA
"""


class IPCA(object):
    def __init__(self, n_components=None, whiten=False, copy=True, batch_size=None):
        """
        :param n_components:   default为None ，int 或None， 想要保留的分量数，None 时，
        min(n_samples, n_features)
        :param whiten:   bool型，可选项， 默认为False, 当true（默认情况下为false）时，components_ 向量除以
        n_samples*components_以确保具有单位组件级方差的不相关输出。
        :param copy: 默认为True,  False时，x 将被覆盖，将节约能存，但存在不安全
        :param batch_size: default None， 批量样本数，   只在fit 中使用，设为None,系统自动设成5*n_features,
        以保持经度与内存开销的平衡
        """
        self.model = IncrementalPCA(n_components=n_components,
                                    whiten=whiten,
                                    copy=copy,
                                    batch_size=batch_size)

    def fit(self, x, y=None):
        self.model.fit(X=x, y=y)

    def transform(self, x):
        return self.model.transform(X=x)

    def fit_transform(self, x, y=None):
        return self.model.fit_transform(X=x, y=y)

    def get_params(self, deep=True):   # 获取评估器的参数
        return self.model.get_params(deep=deep)

    def set_params(self, **params):   # 设置评估器的参数
        self.model.set_params(**params)

    def inverse_transform(self, x):   # 与 fit_tansform 刚好相反的两个操作
        return self.model.inverse_transform(X=x)

    def get_precision(self):    # 根据生成模型计算精度矩阵
        return self.model.get_precision()

    def get_covariance(self):    # 根据生成模型获取协方差
        return self.model.get_covariance()

    def partial_fit(self, x, y=None, check_input=True):  # 增量训练
        self.model.partial_fit(X=x, y=y, check_input=check_input)

    def get_attributes(self):
        component = self.model.components_
        explained_variance = self.model.explained_variance_
        explained_variance_ratio = self.model.explained_variance_ratio_
        singular_values = self.model.singular_values_
        means = self.model.mean_     # 每个特征的均值
        var = self.model.var_   # 每个特征的方差
        noise_variance = self.model.noise_variance_  # 评估的噪声协方差
        n_component = self.model.n_components_
        n_samples_seen = self.model.n_samples_seen_
        return component, explained_variance, explained_variance_ratio, singular_values, means, var, noise_variance, \
               n_component, n_samples_seen


if __name__ == "__main__":
    x, _ = load_digits(return_X_y=True)
    transformer = IPCA(n_components=7, batch_size=200)
    transformer.partial_fit(x[:100, :])
    x_sparse = sparse.csr_matrix(x)
    print(type(x_sparse))
    print(x_sparse)
    x_transformed = transformer.fit_transform(x_sparse)
    print(x_transformed.shape)
