"""Shared estimator adapters; no workflow-level search or data splitting."""
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor
from ..core import Node, Spectra


class EstimatorNode(Node):
    """Shared estimator adapter recording training width and optional spectral coordinates.

    Attributes
    ----------
    axis_: Training coordinate copy, or None for ordinary matrix inputs.
    n_features_in_: Training feature count; checked against inference inputs.
    parameters: Keyword arguments forwarded to the underlying estimator; None uses its defaults.
    requires_fit: Whether the executor must fit this tool before using its output method.

    Inherited interface
    -------------------
    See Node for methods and attributes not overridden here.
    """
    requires_fit = True

    def __init__(self, parameters=None):
        """Store configuration without fitting training data. See the class attributes for constructor parameter meanings.

        Parameters
        ----------
        parameters: Keyword arguments forwarded to the underlying estimator; None uses its defaults.

        Returns and state
        -----------------
        None. Stores configuration only.
        """
        self.parameters = parameters

    def _remember(self, X):
        """Extract the numeric matrix and record training width and optional coordinates.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Input matrix; also records feature count and optional axis.
        """
        a = self._values(X)
        self.n_features_in_ = a.shape[1]
        self.axis_ = X.axis.copy() if isinstance(X, Spectra) else None
        return a

    def _input(self, X):
        """Check fitted state, feature count and optional coordinate equality before inference.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Validated input matrix.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        self._check_fitted(); a = self._values(X)
        if a.shape[1] != self.n_features_in_: raise ValueError("Feature count differs from fit.")
        if self.axis_ is not None and (not isinstance(X, Spectra) or not np.array_equal(X.axis, self.axis_)):
            raise ValueError("Spectral axis differs from fit; align it explicitly.")
        return a


class ClassificationNode(EstimatorNode):
    """Single-label encoded or multilabel component-wise classifier adapter.

    Attributes
    ----------
    classes_: Output class order; multilabel adapters use target-column indices.
    is_fitted_: Whether fitting completed successfully; typically created or set by fit.
    label_encoder_: Training mapping between original single-label classes and integer codes.
    model_: Fitted third-party estimator, projection or scaler; usually absent before fit.
    multilabel: Use a binary component-presence matrix instead of a single-label class vector.
    output_kind: Declared output category.
    parameters: Keyword arguments forwarded to the underlying estimator; None uses its defaults.

    Inherited interface
    -------------------
    See EstimatorNode for methods and attributes not overridden here.
    """
    output_kind = "class_labels"

    def __init__(self, parameters=None, multilabel=False):
        """Store configuration without fitting training data. See the class attributes for constructor parameter meanings.

        Parameters
        ----------
        parameters: Keyword arguments forwarded to the underlying estimator; None uses its defaults.
        multilabel: Use a binary component-presence matrix instead of a single-label class vector.

        Returns and state
        -----------------
        None. Stores configuration only.
        """
        self.parameters = parameters; self.multilabel = multilabel

    def fit(self, X, y):
        """Validate labels, fit class encoding or multilabel adaptation, and train the underlying classifier.

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
        """Return original single-label classes or a multilabel binary target matrix.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Return original single-label classes or a multilabel binary target matrix.
        """
        a = self._input(X); result = self.model_.predict(a)
        return result if self.multilabel else self.label_encoder_.inverse_transform(np.asarray(result, dtype=int))

    def predict_proba(self, X):
        """Return class probabilities in classes_ order, if the underlying model supports them.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Return class probabilities in classes_ order, if the underlying model supports them.

        Raises
        ------
        NotImplementedError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        a = self._input(X)
        if not hasattr(self.model_, "predict_proba"): raise NotImplementedError("This estimator has no probability output.")
        return np.asarray(self.model_.predict_proba(a))


class RegressionNode(EstimatorNode):
    """Regression adapter returning two-dimensional sample-by-target predictions.

    Attributes
    ----------
    is_fitted_: Whether fitting completed successfully; typically created or set by fit.
    model_: Fitted third-party estimator, projection or scaler; usually absent before fit.
    output_kind: Declared output category.

    Inherited interface
    -------------------
    See EstimatorNode for methods and attributes not overridden here.
    """
    output_kind = "target_values"

    def fit(self, X, y):
        """Normalize targets to two dimensions and fit the underlying regression adapter.

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
        """Return (n_samples, n_targets) predictions in training target order and units.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Return (n_samples, n_targets) predictions in training target order and units.
        """
        a = self._input(X); result = np.asarray(self.model_.predict(a))
        return result[:, None] if result.ndim == 1 else result
