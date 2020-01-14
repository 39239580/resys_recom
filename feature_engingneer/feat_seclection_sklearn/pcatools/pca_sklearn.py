from sklearn.decomposition import PCA
import numpy as np
"""
代码地址  https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
"""


class PcaMethod(object):
    def __init__(self, n_components, copy=True, whiten=False, svd_solver="auto",
                 tol=0.0, iterated_power="auto", random_state=None):
        """
        :param n_components:  #保持的分量数， 若不设，则保存所有的分量   n_components == min(n_samples, n_features)
        :param copy:  默认为True, 如果使用False, 运行fit.transform数据不会得到想要的值
        :param whiten: 默认为False,分量向量乘以n个样本的平方根，然后除以奇异值，以确保具有单位分量方差的不相关输出。
        :param svd_solver:  {"auto", "full", "arpack", "randomized"} 中一种，
        "auto"： 求解器由基于x.shape和n_Components的默认策略选择：如果输入数据大于500x500，
        并且要提取的分量数低于数据最小维数的80%，则启用更有效的“随机”方法。否则，将计算出精确的完整SvD，然后可选地截断。
        "full":通过scipy.linalg.svd运行调用标准lapack求解器的完全svd，并通过后处理选择组件
        "arpack": 运行svd截断为n_Components，通过scipy.sparse.linalg.svds调用arpack求解器。
        它严格要求0<；n_components<；min（x.shape）
        "randomized": 运行随机的SVD
        :param tol:  float, 默认为0.0  通过svd_solver=="arpack"  计算奇异值公差
        :param iterated_power: 默认为自动，int,>0 计算幂方法的迭代次数， 通过svd_solver=="randomized"
        :param random_state: 默认为None， int,
       上述的  使用方法
       1.n_components 不设，  n_components= n_components == min(n_samples, n_features)
       2.If 0 < n_components < 1 and svd_solver == 'full',
       选择方差的数量，以便需要解释的方差量大于n_Components指定的百分比。
       3.如果svd_solver=‘arpack’，则分量的数量必须严格小于n_features和n_sample的最小值。
       4.设为None，n_components == min(n_samples, n_features) - 1

        输入X 为数组 array shape =[n_samples, n_features]

        """
        self.model = PCA(n_components=n_components, copy=copy, whiten=whiten, svd_solver=svd_solver,
                         tol=tol, iterated_power=iterated_power, random_state=random_state)

    def fit(self, x, y=None):
        self.model.fit(X=x, y=y)

    def transform(self, x):
        return self.model.transform(X=x)

    def fit_transform(self, x, y=None):
        return self.model.fit_transform(X=x, y=y)

    def get_covariance(self):  # 用生成模型计算数据协方差。
        return self.model.get_covariance()

    def get_params(self, deep=True):  # 获取此估计器的参数。
        return self.model.get_params(deep=deep)

    def get_precision(self):   # 用生成模型计算数据精确度矩阵。
        return self.model.get_precision()

    def inverse_transform(self, x):  # 将数据转换回原来的空间。
        return self.model.inverse_transform(X=x)

    def score(self, x, y):  # 返回所有样本的平均对数似然函数。
        return self.model.score(X=x, y=y)   # 当前模型下，样本的平均似然相似度， float

    def score_samples(self, x):   # 返回每个样本的对数似然函数
        return self.model.score_samples(X=x)   # 当前模型下，每个样本的似然相似度，数组

    def set_params(self, **params):   # 设置评估器的参数
        self.model.set_params(**params)

    def get_attributes(self):
        explained_variance_ratio_ = self.model.explained_variance_ratio_   # 每个选定组成分量表达的方差百分比。 没设定为1.0
        components_ = self.model.components_   # shape(n_components_,n_features_),  array
        # 特征空间中的主轴，表示数据中最大方差的方向。
        explained_variance_ = self.model.explained_variance_    # array, shape (n_components,)由每个选定分量表达的方差量。
        # 等于x的协方差矩阵的n_分量最大特征值
        singular_values_ = self.model.singular_values_  # shape (n_components,)
        # 对应于每个选定分量的奇异值。奇异值等于低维空间中n_分量变量的2-范数。
        mean_ = self.model.mean_   # 每个特征的经验平均值，从训练集估计。 与 X.mean(axis=0).相等
        n_components_ = self.model.n_components_
        n_features_ = self.model.n_features_  # 训练集中的特征数量
        n_samples_ = self.model.n_samples_  # 训练集中的样本量
        noise_variance_ = self.model.noise_variance_   # 噪音方差 等于X协方差矩阵的平均
        # （min（n_features，n_samples）-n_分量）最小特征值。
        return explained_variance_ratio_, components_, explained_variance_, singular_values_,\
               mean_, n_components_, n_features_, n_samples_, noise_variance_


if __name__ == "__main__":
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PcaMethod(n_components=2)
    pca.fit(X)
    explained_variance_ratio_, components_, explained_variance_, singular_values_, \
    mean_, n_components_, n_features_, n_samples_, noise_variance_ = pca.get_attributes()
    print(explained_variance_ratio_)
    print(singular_values_)


