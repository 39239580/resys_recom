from sklearn.decomposition import FactorAnalysis
from sklearn.datasets import load_digits
"""
API接口地址https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html#
sklearn.decomposition.FactorAnalysis
"""


class FA(object):   # 因子分解
    def __init__(self, n_components=None, tol=1e-2, copy=True, max_iter=1000,
                 noise_variance_init=None, svd_method='randomized',
                 iterated_power=3, random_state=0):
        """
        :param n_components:   int 想要的分量个数， 默认为None，表示全部需要
        :param tol: float  停止对数斯然估计的增加的容忍度
        :param copy:  bool   False时，在fit阶段，数据会被覆盖
        :param max_iter: # 最大迭代次数
        :param noise_variance_init:   # None | array, shape=(n_features,)  每个特征的噪声方差的初始化猜测，
        如果为None，  默认为np.ones(n_features)
        :param svd_method:  {"lapack","randomized"}, "lapack", 使用标注的svd, "randomized"  使用快速的随机svd
        :param iterated_power: int, 可选项。 默认为3， 幂方法的迭代次数
        :param random_state: 随机种子
        """
        self.model = FactorAnalysis(n_components=n_components,
                                    tol=tol,
                                    copy=copy,
                                    max_iter=max_iter,
                                    noise_variance_init=noise_variance_init,
                                    svd_method=svd_method,
                                    iterated_power=iterated_power,
                                    random_state=random_state)

    def fit(self, x, y=None):
        return self.model.fit(X=x, y=y)

    def transform(self, x):
        self.model.transform(X=x)

    def fit_transform(self, x, y=None):
        return self.model.fit_transform(X=x, y=y)

    def get_covariance(self):
        return self.model.get_covariance()

    def get_params(self, deep):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        self.model.set_params(**params)

    def get_precision(self):   # 用因子分解模型生成 精度矩阵
        return self.model.get_precision()

    def score(self, x, y=None):
        return self.model.score(X=x, y=y)

    def score_sample(self, x):
        return self.model.score_samples(X=x)

    def get_attributes(self):
        component = self.model.components_   # 分量值
        loglike = self.model.loglike_  # 对数似然
        noise_var = self.model.noise_variance_   # 每个特征的评估噪声方差 arry  shape(n_features,)
        n_iter = self.model.n_iter_  # int, 运行的迭代次数
        mean = self.model.mean_  # array,shape(n_features,) 训练集中评估的特征均值

        return component, loglike, noise_var, n_iter, mean


if __name__ == "__main__":
    x, _ = load_digits(return_X_y=True)
    transformer = FA(n_components=7, random_state=0)
    x_transformer = transformer.fit_transform(x)
    print(x_transformer.shape)
