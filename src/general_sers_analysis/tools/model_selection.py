"""Small classical nodes used by the workflow search layer."""
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from ..core import Node
from .estimators import RegressionNode


class RidgeRegressor(RegressionNode):
    """Ridge regression. Configuration, fit/predict and learned state follow the inherited estimator adapter.

    Attributes
    ----------
    estimator_type: Underlying estimator class instantiated during fit.

    Inherited interface
    -------------------
    See RegressionNode for methods and attributes not overridden here.
    """
    estimator_type = Ridge


class KernelRidgeRegressor(RegressionNode):
    """Kernel ridge regression. Configuration, fit/predict and learned state follow the inherited estimator adapter.

    Attributes
    ----------
    estimator_type: Underlying estimator class instantiated during fit.

    Inherited interface
    -------------------
    See RegressionNode for methods and attributes not overridden here.
    """
    estimator_type = KernelRidge


class TargetScaledRegressor(Node):
    """Fit a regression Node in training-standardized target coordinates and restore source units at inference.

    Attributes
    ----------
    is_fitted_: Whether fitting completed successfully; typically created or set by fit.
    mean_: Per-target training mean used to restore source units.
    output_kind: Declared output category.
    regressor: Unfitted regression Node prototype.
    regressor_: Cloned regression Node fitted to standardized training targets.
    requires_fit: Whether the executor must fit this tool before using its output method.
    scale_: Per-target training standard deviation; constant columns use one.

    Inherited interface
    -------------------
    See Node for methods and attributes not overridden here.
    """
    requires_fit = True
    output_kind = 'target_values'

    def __init__(self, regressor):
        """Store configuration without fitting training data. See the class attributes for constructor parameter meanings.

        Parameters
        ----------
        regressor: Unfitted regression Node prototype.

        Returns and state
        -----------------
        None. Stores configuration only.
        """
        self.regressor = regressor

    def fit(self, X, y):
        """Learn target means/scales, clone the regressor, and fit standardized training targets.

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
        """Predict standardized targets and restore source units as prediction * scale_ + mean_.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Predict standardized targets and restore source units as prediction * scale_ + mean_.
        """
        self._check_fitted()
        return self.regressor_.predict(X) * self.scale_ + self.mean_


class ConstantPredictor(Node):
    """Training-mean regression or binary-majority classification baseline; ties predict one.

    Attributes
    ----------
    classification: True learns binary majority labels; False learns training means.
    is_fitted_: Whether fitting completed successfully; typically created or set by fit.
    requires_fit: Whether the executor must fit this tool before using its output method.
    value_: Fixed output vector learned from training target columns.

    Inherited interface
    -------------------
    See Node for methods and attributes not overridden here.
    """
    requires_fit = True
    def __init__(self, classification=False):
        """Store configuration without fitting training data. See the class attributes for constructor parameter meanings.

        Parameters
        ----------
        classification: True learns binary majority labels; False learns training means.

        Returns and state
        -----------------
        None. Stores configuration only.
        """
        self.classification = classification

    def fit(self, X, y):
        """Learn per-target training means or binary-majority labels into value_.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.
        y: Ground-truth targets aligned with rows, usually (n_rows, n_targets); never predictions.

        Returns and state
        -----------------
        self. Learned attributes are stored on the instance; see the class attribute descriptions.
        """
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[:, None]
        self.value_ = ((y.mean(axis=0) >= .5).astype(int) if self.classification else y.mean(axis=0))
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Repeat value_ for each input row, returning a two-dimensional target matrix.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Repeat value_ for each input row, returning a two-dimensional target matrix.
        """
        self._check_fitted()
        return np.tile(self.value_, (len(self._values(X)), 1))


class IndependentPresence(Node):
    """One classifier per molecular target; constant-label columns use constants.

    Attributes
    ----------
    classifier: Unfitted classifier prototype cloned separately for each target.
    is_fitted_: Whether fitting completed successfully; typically created or set by fit.
    models_: Per-target fitted classifiers or constants, ordered as the columns of y.
    output_kind: Declared output category.
    requires_fit: Whether the executor must fit this tool before using its output method.

    Inherited interface
    -------------------
    See Node for methods and attributes not overridden here.
    """
    requires_fit = True
    output_kind = 'class_labels'

    def __init__(self, classifier):
        """Store configuration without fitting training data. See the class attributes for constructor parameter meanings.

        Parameters
        ----------
        classifier: Unfitted classifier prototype cloned separately for each target.

        Returns and state
        -----------------
        None. Stores configuration only.
        """
        self.classifier = classifier

    def fit(self, X, y):
        """Clone and fit one classifier per target, substituting constants for single-label columns.

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
        """Return a binary sample-by-component matrix in training target order.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Return a binary sample-by-component matrix in training target order.
        """
        self._check_fitted()
        return np.column_stack([m.predict(X).reshape(-1) for m in self.models_])
