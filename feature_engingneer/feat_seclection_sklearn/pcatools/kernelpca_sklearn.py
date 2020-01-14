from sklearn.decomposition import KernelPCA
from sklearn.datasets import load_digits
"""
API 接口地址 https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#
sklearn.decomposition.KernelPCA
"""


class KPCA(object):  # 核pca
    def __init__(self, n_components=None,  kernel="linear", gamma=None, degree=3, coef0=1,
                 kernel_params=None, alpha=1.0, fit_inverse_transform=False, eigen_solver='auto',
                 tol=0, max_iter=None, remove_zero_eig=False, random_state=None, copy_x=True, n_jobs=None):
        """
        :param n_components:  # 想要降维的维度数
        :param kernel: "linear/poly/rbf/sigmoid/cosine/preconputed"  默认为linear
        :param gamma:  float 默认为1/n_features  rbf, poly, sigmoid核系数，其他核忽略掉
        :param degree: int 默认为3  poly核的度， 其他核全部忽略
        :param coef0: float 默认为1 poly与sigmoid核的独立项目，其他核忽略掉
        :param kernel_params: 默认为None
        :param alpha: int 默认为1.0  学习方向变换的脊回归超参数（fit_inverse_transform=True）
        :param fit_inverse_transform: bool 默认为False， 反向计算
        :param eigen_solver: [‘auto’|’dense’|’arpack’] 默认为自动，选择要使用的特征转换器。如果n_分量远小
        于训练样本的数量，arpack可能比密集的本征溶胶更有效。
        :param tol: float default =0  对Arpack的收敛公差。如果为0，最优值将由arpack选择。
        :param max_iter:int default =None  arpack的最大迭代次数。如果没有，最优值将由arpack选择。
        :param remove_zero_eig: 布尔型， 默认为False  ,True时，去掉0特征。 输出的分量维度小于 n_components，
        设为None时，不管n_components设为多少，0特征都会被删除
        :param random_state: 随机种子， 可选项 默认为None
        :param copy_x: bool  default=True   如果为真，则输入x由模型复制并存储在x_fit_属性中
        :param n_jobs: int or None  default =None   并行个数
        """
        self.model = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0,
                               kernel_params=kernel_params, alpha=alpha, fit_inverse_transform=fit_inverse_transform,
                               eigen_solver=eigen_solver, tol=tol, max_iter=max_iter,
                               remove_zero_eig=remove_zero_eig, random_state=random_state,
                               copy_X=copy_x, n_jobs=n_jobs)

    def fit(self, x, y):
        self.model.fit(X=x, y=y)

    def transform(self, x):
        return self.model.transform(X=x)

    def fit_transform(self, x, y):
        return self.model.fit_transform(X=x, y=y)

    def inverse_transform(self, x):
        return self.model.inverse_transform(X=x)

    def get_params(self):
        return self.model.set_params()

    def set_params(self, **params):
        self.model.set_params(**params)

    def get_attributes(self):
        lambdas = self.model.lambdas_
        alphas = self.model.alphas_
        dual_coef = self.model.dual_coef_
        x_transformed_fit = self.model.X_transformed_fit_
        x_fit = self.model.X_fit_
        return lambdas, alphas, dual_coef, x_transformed_fit, x_fit


if __name__ == "__main__":
    x, _ = load_digits(return_X_y=True)
    kpca = KernelPCA(n_components=7, kernel="linear")
    x_transformed = kpca.fit_transform(X=x)
    print(x_transformed.shape)
