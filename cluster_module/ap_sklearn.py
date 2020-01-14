from sklearn.cluster import AffinityPropagation
import numpy as np
"""
api 接口地址 https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#
sklearn.cluster.AffinityPropagation
"""


class AP(object):
    def __init__(self, damping=.5, max_iter=200, convergence_iter=15,
                 copy=True, preference=None, affinity='euclidean',
                 verbose=False):
        """
        :param damping:
        :param max_iter:
        :param convergence_iter:
        :param copy:
        :param preference:
        :param affinity:
        :param verbose:
        """
        self.model = AffinityPropagation(damping=damping,
                                         max_iter=max_iter,
                                         convergence_iter=convergence_iter,
                                         copy=copy,
                                         preference=preference,
                                         affinity=affinity,
                                         verbose=verbose)

    def fit(self, x, y=None):
        self.model.fit(X=x, y=y)

    def fit_predict(self, x, y=None):
        return self.model.fit_predict(X=x, y=y)

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def predict(self, x):
        return self.model.predict(X=x)

    def set_params(self, **params):
        self.model.set_params(**params)

    def get_attributes(self):
        cluster_centers = self.model.cluster_centers_
        cluster_centers_indices = self.model.cluster_centers_indices_
        labels = self.model.labels_
        affinity_matrix = self.model.affinity_matrix_
        n_iter = self.model.n_iter_

        return cluster_centers, cluster_centers_indices, labels, affinity_matrix, n_iter


if __name__ == "__main__":
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 4], [4, 0]])
    cluster = AP()
    cluster.fit(X)
    cluster_center, cluster_centers_indice, label, affinity_matrixs, n_iters = cluster.get_attributes()

    print(label)
    print(cluster.predict([[0, 0], [4, 4]]))
    print(cluster_center)
