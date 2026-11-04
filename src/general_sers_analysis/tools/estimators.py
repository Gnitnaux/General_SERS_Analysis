"""Shared estimator adapters; no workflow-level search or data splitting."""
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor
from ..core import Node, Spectra


class EstimatorNode(Node):
    requires_fit = True

    def __init__(self, parameters=None):
        self.parameters = parameters

    def _remember(self, X):
        a = self._values(X)
        self.n_features_in_ = a.shape[1]
        self.axis_ = X.axis.copy() if isinstance(X, Spectra) else None
        return a

    def _input(self, X):
        self._check_fitted(); a = self._values(X)
        if a.shape[1] != self.n_features_in_: raise ValueError("Feature count differs from fit.")
        if self.axis_ is not None and (not isinstance(X, Spectra) or not np.array_equal(X.axis, self.axis_)):
            raise ValueError("Spectral axis differs from fit; align it explicitly.")
        return a


class ClassificationNode(EstimatorNode):
    output_kind = "class_labels"

    def __init__(self, parameters=None, multilabel=False):
        self.parameters = parameters; self.multilabel = multilabel

    def fit(self, X, y):
        a = self._remember(X); labels = np.asarray(y)
        if any(v is None or isinstance(v, (float, np.floating)) and not np.isfinite(v) for v in labels.flat):
            raise ValueError("Classification labels cannot be missing or nonfinite.")
        params = dict(self.parameters or {}); params.setdefault("random_state", 0)
        if self.estimator_type.__name__ == "SVC": params.setdefault("probability", True)
        model = self.estimator_type(**params)
        if self.multilabel:
            if labels.ndim != 2 or not np.isin(labels, [0, 1]).all():
                raise ValueError("Multilabel targets must be a binary samples x components matrix.")
            if any(len(np.unique(col)) != 2 for col in labels.T):
                raise ValueError("Each component needs both positive and negative training examples.")
            self.classes_ = np.arange(labels.shape[1]); model = OneVsRestClassifier(model)
            target = labels
        else:
            if labels.ndim != 1: raise ValueError("Use multilabel=True for component-presence matrices.")
            self.label_encoder_ = LabelEncoder().fit(labels)
            self.classes_ = self.label_encoder_.classes_
            if len(self.classes_) < 2: raise ValueError("Classification needs at least two classes.")
            target = self.label_encoder_.transform(labels)
        self.model_ = model.fit(a, target); self.is_fitted_ = True
        return self

    def predict(self, X):
        a = self._input(X); result = self.model_.predict(a)
        return result if self.multilabel else self.label_encoder_.inverse_transform(np.asarray(result, dtype=int))

    def predict_proba(self, X):
        a = self._input(X)
        if not hasattr(self.model_, "predict_proba"): raise NotImplementedError("This estimator has no probability output.")
        return np.asarray(self.model_.predict_proba(a))


class RegressionNode(EstimatorNode):
    output_kind = "target_values"

    def fit(self, X, y):
        a = self._remember(X); target = np.asarray(y, dtype=float)
        if target.ndim == 1: target = target[:, None]
        if target.ndim != 2 or not np.isfinite(target).all(): raise ValueError("Targets must be finite samples x targets.")
        params = dict(self.parameters or {})
        if self.estimator_type.__name__ in {"RandomForestRegressor", "XGBRegressor"}: params.setdefault("random_state", 0)
        estimator = self.estimator_type(**params)
        if self.estimator_type.__name__ in {"SVR", "XGBRegressor"}:
            estimator = MultiOutputRegressor(estimator)
        self.model_ = estimator.fit(a, target); self.is_fitted_ = True
        return self

    def predict(self, X):
        a = self._input(X); result = np.asarray(self.model_.predict(a))
        return result[:, None] if result.ndim == 1 else result
