from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random
"""
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD
"""


class TruncSvd(object):  # 截断 svd
    def __init__(self, n_components=2, algorithm="randomized", n_iter=5,
                 random_state=None, tol=0.):
        """
        :param n_components:  默认为2，int 输出数据的期望维数。必须严格少于特征的数量。
        默认值对于可视化是有用的。对于LSA，建议值为100。
        :param algorithm: string, 默认为 randomized. randomized  arpack,
        :param n_iter: 默认为5，整形，可选项 随机SVD的迭代次数，
        :param random_state: 随机种子，默认为None
        :param tol: float ,可选项 arpack的容忍度。 0 意味着机器经度，  使用随机算法时，不起作用
        """
        self.model = TruncatedSVD(n_components=n_components,
                                  algorithm=algorithm,
                                  n_iter=n_iter,
                                  random_state=random_state,
                                  tol=tol)

    def fit(self, x, y=None):
        self.model.fit(X=x, y=y)

    def transform(self, x):
        return self.model.transform(X=x)

    def fit_transform(self, x, y=None):
        return self.model.fit_transform(X=x, y=y)

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        self.model.set_params(**params)

    def inverse_transform(self, x):
        self.model.inverse_transform(X=x)

    def get_attributes(self):
        components = self.model.components_   # 分量 shape（n_components，n_feature）
        explained_variance = self.model.explained_variance_  # 解释方差 通过投影转换到每个分量的训练样本的方差。
        explained_variance_ratio = self.model.explained_variance_ratio_   # 解释方差率shape(n_components,),
        # 每个选定分量部分解释的差异百分比。
        singular_value = self.model.singular_values_   # 奇异值  shape(n_components,)
        return components, explained_variance, explained_variance_ratio, singular_value


if __name__ == "__main__":
    x = sparse_random(100, 100, density=0.1,
                      format="csr", random_state=42)
    svd = TruncSvd(n_components=5, n_iter=7, random_state=42)
    svd.fit(x)
    component, explained_variances, explained_variance_ratios, singular_values = svd.get_attributes()
    print(explained_variance_ratios)
    print(explained_variance_ratios.sum())
    print(singular_values)
