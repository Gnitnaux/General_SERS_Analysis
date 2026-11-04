import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from .estimators import ClassificationNode, EstimatorNode


class RFClassifier(ClassificationNode):
    estimator_type = RandomForestClassifier


class SVMClassifier(ClassificationNode):
    estimator_type = SVC


class XGBoostClassifier(ClassificationNode):
    estimator_type = XGBClassifier


class LogisticClassifier(ClassificationNode):
    estimator_type = LogisticRegression


class PLSDAClassifier(EstimatorNode):
    """PLS regression onto one-hot class targets; argmax decision, no fake probabilities."""
    output_kind = "class_labels"

    def fit(self, X, y):
        a = self._remember(X); y = np.asarray(y)
        if y.ndim != 1: raise ValueError("Initial PLS-DA supports single-label classification only.")
        if any(v is None or isinstance(v, (float, np.floating)) and not np.isfinite(v) for v in y):
            raise ValueError("Classification labels cannot be missing or nonfinite.")
        self.label_encoder_ = LabelEncoder().fit(y); self.classes_ = self.label_encoder_.classes_
        if len(self.classes_) < 2: raise ValueError("PLS-DA needs at least two classes.")
        encoded = np.eye(len(self.classes_))[self.label_encoder_.transform(y)]
        self.model_ = PLSRegression(**dict(self.parameters or {})).fit(a, encoded)
        self.is_fitted_ = True
        return self

    def predict_scores(self, X):
        a = self._input(X); return self.model_.predict(a)

    def predict(self, X):
        scores = self.predict_scores(X)
        return self.label_encoder_.inverse_transform(scores.argmax(axis=1))

    def predict_proba(self, X):
        raise NotImplementedError("PLS-DA scores are not calibrated probabilities; use predict_scores().")
