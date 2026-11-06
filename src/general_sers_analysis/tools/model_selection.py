"""Small classical nodes used by the workflow search layer."""
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from ..core import Node
from .estimators import RegressionNode


class RidgeRegressor(RegressionNode):
    estimator_type = Ridge


class KernelRidgeRegressor(RegressionNode):
    estimator_type = KernelRidge


class TargetScaledRegressor(Node):
    """Fit a regression Node in training-only standardized target coordinates."""
    requires_fit = True
    output_kind = 'target_values'

    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2 or not np.isfinite(y).all():
            raise ValueError('Finite 2D targets required.')
        self.mean_ = y.mean(axis=0)
        std = y.std(axis=0)
        self.scale_ = np.where(std > 0, std, 1.)
        self.regressor_ = clone(self.regressor).fit(X, (y - self.mean_) / self.scale_)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        self._check_fitted()
        return self.regressor_.predict(X) * self.scale_ + self.mean_


class ConstantPredictor(Node):
    """Training mean/majority baseline with the same fit/predict interface."""
    requires_fit = True
    def __init__(self, classification=False):
        self.classification = classification

    def fit(self, X, y):
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[:, None]
        self.value_ = ((y.mean(axis=0) >= .5).astype(int) if self.classification else y.mean(axis=0))
        self.is_fitted_ = True
        return self

    def predict(self, X):
        self._check_fitted()
        return np.tile(self.value_, (len(self._values(X)), 1))


class IndependentPresence(Node):
    """One classifier per molecule, with a constant for unobserved fold classes."""
    requires_fit = True
    output_kind = 'class_labels'

    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, X, y):
        y = np.asarray(y)
        if y.ndim != 2 or not np.isin(y, [0, 1]).all():
            raise ValueError('Expected binary component-presence matrix.')
        self.models_ = []
        for col in y.T:
            model = (ConstantPredictor(True) if len(np.unique(col)) == 1 else clone(self.classifier))
            self.models_.append(model.fit(X, col))
        self.is_fitted_ = True
        return self

    def predict(self, X):
        self._check_fitted()
        return np.column_stack([m.predict(X).reshape(-1) for m in self.models_])
