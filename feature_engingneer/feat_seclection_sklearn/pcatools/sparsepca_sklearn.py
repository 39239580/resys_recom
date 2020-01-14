from sklearn.decomposition import SparsePCA
from sklearn.datasets import make_friedman1
import numpy as np
"""
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html#
sklearn.decomposition.SparsePCA
"""


class SPCA(object):
    def __init__(self, n_components=None, alpha=1, ridge_alpha=0.01,
                 max_iter=1000, tol=1e-8, method='lars', n_jobs=None,
                 U_init=None, V_init=None, verbose=False, random_state=None,
                 normalize_components='deprecated'):
        """
        :param n_components:
        :param alpha:
        :param ridge_alpha:
        :param max_iter:
        :param tol:
        :param method:
        :param n_jobs:
        :param U_init:
        :param V_init:
        :param verbose:
        :param random_state:
        :param normalize_components:
        """
        self.model = SparsePCA(n_components=n_components,
                               alpha=alpha,
                               ridge_alpha=ridge_alpha,
                               max_iter=max_iter,
                               tol=tol,
                               method=method,
                               n_jobs=n_jobs,
                               U_init=U_init,
                               V_init=V_init,
                               verbose=verbose,
                               random_state=random_state,
                               normalize_components=normalize_components)

    def fit(self, x, y):
        self.model.fit(X=x, y=y)

    def transform(self,x):
        self.model.transform(X=x)

    def fit_transform(self, x, y=None):
        return self.model.fit_transform(X=x, y=y)

    def get_params(self):
        return self.model.get_params(deep=True)

    def set_params(self, **params):
        return self.model.set_params(**params)

    def get_attributes(self):
        components = self.model.components_
        error = self.model.error_
        n_iter = self.model.n_iter_
        mean = self.model.mean_
        return components, error, n_iter, mean


if __name__ == "__main__":
    x, _ = make_friedman1(n_samples=200, n_features=30, random_state=0)
    transformer = SPCA(n_components=5, random_state=0)
    x_transformed = transformer.fit_transform(x=x)
    print(x_transformed.shape)
    components, error, n_iter,mean =transformer.get_attributes()
    print(np.mean(components == 0))
