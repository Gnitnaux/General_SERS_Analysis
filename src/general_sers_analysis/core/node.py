"""All concrete tools inherit this node contract, directly or through a category base."""
from abc import ABC
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from .data import Spectra, matrix


class Node(BaseEstimator, ABC):
    """Common tool interface and numeric validation helpers.

    Attributes
    ----------
    input_kind: Declared input category; a hint rather than a complete data schema.
    is_fitted_: Whether fitting completed successfully; typically created or set by fit.
    output_kind: Declared output category.
    requires_fit: Whether the executor must fit this tool before using its output method.

    Inherited interface
    -------------------
    See BaseEstimator, ABC for methods and attributes not overridden here.
    """
    input_kind = "matrix"
    output_kind = "matrix"
    requires_fit = False

    def fit(self, X, y=None, **metadata):
        """Validate input, set is_fitted_, and return self; the base implementation learns no statistical parameters.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.
        y: Ground-truth targets aligned with rows, usually (n_rows, n_targets); never predictions.
        metadata: Additional named metadata forwarded according to the tool interface.

        Returns and state
        -----------------
        self. Learned attributes are stored on the instance; see the class attribute descriptions.
        """
        self._values(X)
        self.is_fitted_ = True
        return self

    def transform(self, X, **metadata):
        """Always raise NotImplementedError; concrete tools implement meaningful transformations.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.
        metadata: Additional named metadata forwarded according to the tool interface.

        Returns and state
        -----------------
        Always raise NotImplementedError; concrete tools implement meaningful transformations.

        Raises
        ------
        NotImplementedError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        raise NotImplementedError(f"{type(self).__name__} does not provide transform().")

    def predict(self, X, **metadata):
        """Always raise NotImplementedError; concrete predictors provide their own output methods.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.
        metadata: Additional named metadata forwarded according to the tool interface.

        Returns and state
        -----------------
        Always raise NotImplementedError; concrete predictors provide their own output methods.

        Raises
        ------
        NotImplementedError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        raise NotImplementedError(f"{type(self).__name__} does not provide predict().")

    def fit_transform(self, X, y=None, **metadata):
        """Fit on X and return its training-set transformation; do not use this to refit validation data.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.
        y: Ground-truth targets aligned with rows, usually (n_rows, n_targets); never predictions.
        metadata: Additional named metadata forwarded according to the tool interface.

        Returns and state
        -----------------
        Fit on X and return its training-set transformation; do not use this to refit validation data.
        """
        return self.fit(X, y, **metadata).transform(X, **metadata)

    def _check_fitted(self):
        """Raise NotFittedError if this learned tool has not been fitted.

        Returns and state
        -----------------
        None; raises when unfitted.

        Raises
        ------
        NotFittedError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        if self.requires_fit and not getattr(self, "is_fitted_", False):
            raise NotFittedError(f"Call {type(self).__name__}.fit() first.")

    @staticmethod
    def _values(X):
        """Extract Spectra.intensity or copy and validate a finite two-dimensional matrix.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Two-dimensional numeric matrix.
        """
        return X.intensity if isinstance(X, Spectra) else matrix(X)

    @staticmethod
    def _wrap(X, values):
        """Preserve spectral coordinates/identities for Spectra inputs; otherwise return the numeric values.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.
        values: Aggregated matrix with shape (n_groups, n_features).

        Returns and state
        -----------------
        Spectra or numeric values according to the original input type.
        """
        return X.with_values(values) if isinstance(X, Spectra) else values
