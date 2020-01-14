from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
"""
api接口地址 https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.
QuadraticDiscriminantAnalysis.html#sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
"""



class QDA(object):
    def __init__(self, priors=None, reg_param=0., store_covariance=False, tol=1.0e-4):
        """
        :param priors:  分来优先级， array, 可选项， shape=[n_classes]
        :param reg_param:  float, 可选项，将协方差估计正规化
        :param store_covariance: boolean 如果为真，则计算并存储协方差矩阵到self.covariance_中
        :param tol:  使用排序评估的阈值
        """
        self.model = QuadraticDiscriminantAnalysis(priors=priors,
                                                   reg_param=reg_param,
                                                   store_covariance=store_covariance,
                                                   tol=tol)

    def fit(self,x, y):
        self.model.fit(X=x, y=y)

    def get_params(self,deep=True):
        return self.model.get_params(deep=deep)

    def predict(self,x):
        return self.model.predict(X=x)

    def predict_log_dict(self,x):
        return self.model.predict_log_proba(X=x)

    def predict_proba(self,x):
        return self.model.predict_proba(X=x)

    def score(self):
        return self.model.score(X=x, y=y, sample_weight=sample_weight)

    def set_params(self, **params):
        self.model.set_params(**params)

    def decision_function(self, x): # 将决策函数应用于样本数组。
        return self.model.decision_function(X=x)

    def get_attribute(self):
        covariance = self.model.covariance_  # 每个种类的协方差矩阵， list of array-like of shape (n_features, n_features)
        means = self.model.means   # 种类均值， array-like of shape (n_classes, n_features)
        priors = self.model.priors_  # 种类占比， 求和为1， array-like of shape (n_classes)
        rotations = self.model.rotations_  # n_k = min(n_features, number of elements in class k) list_array,
        # 高斯分布的旋转
        scalings = self.model.scalings_  # list_array, 每个种类k，shape[n_k]的数组，包含高斯分布的缩放，
        # 如，旋转坐标系中的方差
        classes = self.model.classes_  # array-like, shape(n_classes,), 不同种类标签

        return covariance, means, priors, rotations, scalings, classes


if __name__ == "__main__":
    x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y = np.array([1, 1, 1, 2, 2, 2])
    clf = QDA()
    clf.fit(x, y)
    print(clf.predict([[-0.8, -1]]))
