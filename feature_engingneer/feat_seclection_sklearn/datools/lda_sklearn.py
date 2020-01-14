from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
"""
api接口 https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.
LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis
"""


class LDA(object):
    def __init__(self,solver="svd", shrinkage=None, priors=None,  n_components=None,
                 store_covariance=False, tol=1e-4):
        """
        :param solver: string， 可选项，"svd","lsqr", "eigen"。 默认使用svd, 不计算协方差矩阵，适用于大量特征
        的数据， 最小二乘 lsqr, 结合shrinkage 使用。 eigen 特征值分解， 集合shrinkage  使用
        :param shrinkage: str/float 可选项，概率值，默认为None, "auto", 自动收缩， 0到1内的float, 固定的收缩参数
        :param priors: array, optional, shape (n_classes,) 分类优先
        :param n_components:  # 分量数， 默认None， int， 可选项
        :param store_covariance:  bool, 可选项， 只用于”svd“ 额外计算分类协方差矩阵
        :param tol: 浮点型，默认1e-4, 在svd 中，用于排序评估的阈值
        """
        self.model = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, priors=priors,
                                                n_components=n_components, store_covariance=store_covariance, tol=tol)

    def fit(self, x, y):
        self.model.fit(X=x, y=y)

    def transform(self, x):
        return self.model.transform(X=x)

    def fit_transform(self, x, y):
        return self.model.fit_transform(X=x, y=y)

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        self.model.set_params(**params)

    def decision_function(self,x):
        self.model.decision_function(X=x)

    def predict(self, x):
        return self.model.predict(X=x)

    def predict_log_proba(self,x):
        return self.model.predict_log_proba(X=x)

    def predict_proba(self,x):
        return self.model.predict_proba(X=x)

    def score(self, x, y, sample_weight):
        return self.model.score(X=x, y=y, sample_weight=sample_weight)

    def get_attributes(self):  # 生成模型之后才能获取相关属性值
        coef = self.model.coef_        # 权重向量，
        intercept = self.model.intercept_   # 截距项
        covariance = self.model.covariance_  # 协方差矩阵
        explained_variance_ratio = self.model.explained_variance_ratio_
        means = self.model.means_
        priors = self.model.priors_   # 分类等级， 求和为1 shape (n_classes)
        scalings = self.model.scalings_   # shape(rank,n_classes-1). 缩放
        xbar = self.model.xbar_   # 所有的均值
        classes = self.model.classes_  # 分类标签

        return coef, intercept, covariance, explained_variance_ratio, means, priors, scalings, xbar, classes


if __name__ == "__main__":
    x = np.array([[-1, -1], [-2, 1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y = np.array([1, 1, 1, 2, 2, 2])
    clf = LDA()
    clf.fit(x, y)
    print(clf.predict([[-0.8, -1]]))
    print(clf.predict_proba([[-0.7, 0.2]]))  # 输出对应的概率值
