import numpy as np
from scipy.optimize import nnls
from ..core import Node, Spectra, UnmixingResult, matrix


class CLSUnmixer(Node):
    """Least-squares decomposition against known independent reference spectra, not blind component discovery.

    Attributes
    ----------
    axis_: Training coordinate copy, or None for ordinary matrix inputs.
    component_names: Optional caller-provided reference-component names; None generates names.
    is_fitted_: Whether fitting completed successfully; typically created or set by fit.
    names_: Validated unique reference-component names.
    output_kind: Declared output category.
    references_: Reference matrix copy with shape (n_components, n_points).
    requires_fit: Whether the executor must fit this tool before using its output method.

    Inherited interface
    -------------------
    See Node for methods and attributes not overridden here.
    """
    requires_fit = True
    output_kind = "unmixing_result"

    def __init__(self, component_names=None):
        """Store configuration without fitting training data. See the class attributes for constructor parameter meanings.

        Parameters
        ----------
        component_names: Optional caller-provided reference-component names; None generates names.

        Returns and state
        -----------------
        None. Stores configuration only.
        """
        self.component_names = component_names

    def fit(self, reference_spectra, y=None):
        """Store reference spectra/coordinates/names and require independent reference columns; y is unused.

        Parameters
        ----------
        reference_spectra: Known component spectra, with components in rows and coordinates in columns.
        y: Ground-truth targets aligned with rows, usually (n_rows, n_targets); never predictions.

        Returns and state
        -----------------
        self. Learned attributes are stored on the instance; see the class attribute descriptions.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        self.references_ = self._values(reference_spectra).copy()
        if np.linalg.matrix_rank(self.references_.T) < len(self.references_):
            raise ValueError("Reference components must be linearly independent.")
        self.axis_ = reference_spectra.axis.copy() if isinstance(reference_spectra, Spectra) else None
        self.names_ = tuple(self.component_names) if self.component_names is not None else tuple(f"component_{i}" for i in range(len(self.references_)))
        if len(self.names_) != len(self.references_) or len(set(self.names_)) != len(self.names_):
            raise ValueError("Provide one unique name per reference component.")
        self.is_fitted_ = True
        return self

    def _solve(self, a):
        """Solve sample-by-component coefficients against the stored reference spectra.

        Parameters
        ----------
        a: Validated two-dimensional numeric input matrix.

        Returns and state
        -----------------
        Sample-by-component coefficient matrix.
        """
        return np.linalg.lstsq(self.references_.T, a.T, rcond=None)[0].T

    def transform(self, X):
        """Return UnmixingResult with coefficients, reconstruction and residual norms.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Return UnmixingResult with coefficients, reconstruction and residual norms.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        self._check_fitted(); a = self._values(X)
        if a.shape[1] != self.references_.shape[1]: raise ValueError("Mixture/reference feature counts differ.")
        if self.axis_ is not None and (not isinstance(X, Spectra) or not np.array_equal(X.axis, self.axis_)):
            raise ValueError("Mixture/reference axes must match exactly.")
        coefficients = self._solve(a); reconstruction = coefficients @ self.references_
        return UnmixingResult(coefficients, self.references_.copy(), reconstruction,
                             np.linalg.norm(a-reconstruction, axis=1), self.names_,
                             None if self.axis_ is None else self.axis_.copy(), X.sample_ids if isinstance(X, Spectra) else None)

    def predict(self, X):
        """Return only the sample-by-reference coefficient matrix, not an assumed concentration.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Return only the sample-by-reference coefficient matrix, not an assumed concentration.
        """
        return self.transform(X).coefficients


class NNLSUnmixer(CLSUnmixer):
    """Nonnegative known-reference decomposition, inheriting the CLS interface and state.

    Inherited interface
    -------------------
    See CLSUnmixer for methods and attributes not overridden here.
    """
    def _solve(self, a):
        """Solve sample-by-component coefficients against the stored reference spectra.

        Parameters
        ----------
        a: Validated two-dimensional numeric input matrix.

        Returns and state
        -----------------
        Sample-by-component coefficient matrix.
        """
        return np.vstack([nnls(self.references_.T, row)[0] for row in a])
