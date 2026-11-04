"""All concrete tools inherit this node contract, directly or through a category base."""
from abc import ABC
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from .data import Spectra, matrix


class Node(BaseEstimator, ABC):
    input_kind = "matrix"
    output_kind = "matrix"
    requires_fit = False

    def fit(self, X, y=None, **metadata):
        self._values(X)
        self.is_fitted_ = True
        return self

    def transform(self, X, **metadata):
        raise NotImplementedError(f"{type(self).__name__} does not provide transform().")

    def predict(self, X, **metadata):
        raise NotImplementedError(f"{type(self).__name__} does not provide predict().")

    def fit_transform(self, X, y=None, **metadata):
        return self.fit(X, y, **metadata).transform(X, **metadata)

    def _check_fitted(self):
        if self.requires_fit and not getattr(self, "is_fitted_", False):
            raise NotFittedError(f"Call {type(self).__name__}.fit() first.")

    @staticmethod
    def _values(X):
        return X.intensity if isinstance(X, Spectra) else matrix(X)

    @staticmethod
    def _wrap(X, values):
        return X.with_values(values) if isinstance(X, Spectra) else values
