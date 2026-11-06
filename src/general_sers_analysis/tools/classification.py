import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from .estimators import ClassificationNode, EstimatorNode


class RFClassifier(ClassificationNode):
    """Random forest classification. Configuration, fit/predict and learned state follow the inherited estimator adapter.

    Attributes
    ----------
    estimator_type: Underlying estimator class instantiated during fit.

    Inherited interface
    -------------------
    See ClassificationNode for methods and attributes not overridden here.
    """
    estimator_type = RandomForestClassifier


class SVMClassifier(ClassificationNode):
    """Support-vector classification. Configuration, fit/predict and learned state follow the inherited estimator adapter.

    Attributes
    ----------
    estimator_type: Underlying estimator class instantiated during fit.

    Inherited interface
    -------------------
    See ClassificationNode for methods and attributes not overridden here.
    """
    estimator_type = SVC


class XGBoostClassifier(ClassificationNode):
    """XGBoost classification. Configuration, fit/predict and learned state follow the inherited estimator adapter.

    Attributes
    ----------
    estimator_type: Underlying estimator class instantiated during fit.

    Inherited interface
    -------------------
    See ClassificationNode for methods and attributes not overridden here.
    """
    estimator_type = XGBClassifier


class LogisticClassifier(ClassificationNode):
    """Logistic classification. Configuration, fit/predict and learned state follow the inherited estimator adapter.

    Attributes
    ----------
    estimator_type: Underlying estimator class instantiated during fit.

    Inherited interface
    -------------------
    See ClassificationNode for methods and attributes not overridden here.
    """
    estimator_type = LogisticRegression


class PLSDAClassifier(EstimatorNode):
    """PLS regression on one-hot class targets; raw scores are not calibrated probabilities.

    Attributes
    ----------
    classes_: Output class order; multilabel adapters use target-column indices.
    is_fitted_: Whether fitting completed successfully; typically created or set by fit.
    label_encoder_: Training mapping between original single-label classes and integer codes.
    model_: Fitted third-party estimator, projection or scaler; usually absent before fit.
    output_kind: Declared output category.

    Inherited interface
    -------------------
    See EstimatorNode for methods and attributes not overridden here.
    """
    output_kind = "class_labels"

    def fit(self, X, y):
        """Fit class encoding and PLS regression on one-hot training class targets.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.
        y: Ground-truth targets aligned with rows, usually (n_rows, n_targets); never predictions.

        Returns and state
        -----------------
        self. Learned attributes are stored on the instance; see the class attribute descriptions.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
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
        """Return PLS-DA raw class scores in classes_ order, not calibrated probabilities.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Return PLS-DA raw class scores in classes_ order, not calibrated probabilities.
        """
        a = self._input(X); return self.model_.predict(a)

    def predict(self, X):
        """Take the maximum raw class score and restore the original label encoding.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Take the maximum raw class score and restore the original label encoding.
        """
        scores = self.predict_scores(X)
        return self.label_encoder_.inverse_transform(scores.argmax(axis=1))

    def predict_proba(self, X):
        """Always raise NotImplementedError: PLS scores are not calibrated probabilities.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Always raise NotImplementedError: PLS scores are not calibrated probabilities.

        Raises
        ------
        NotImplementedError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        raise NotImplementedError("PLS-DA scores are not calibrated probabilities; use predict_scores().")
