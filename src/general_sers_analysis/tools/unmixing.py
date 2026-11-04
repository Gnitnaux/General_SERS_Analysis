import numpy as np
from scipy.optimize import nnls
from ..core import Node, Spectra, UnmixingResult, matrix


class CLSUnmixer(Node):
    """Known reference spectra -> least-squares coefficients; no blind component discovery."""
    requires_fit = True
    output_kind = "unmixing_result"

    def __init__(self, component_names=None):
        self.component_names = component_names

    def fit(self, reference_spectra, y=None):
        self.references_ = self._values(reference_spectra).copy()
        if np.linalg.matrix_rank(self.references_.T) < len(self.references_):
            raise ValueError("Reference components must be linearly independent.")
        self.axis_ = reference_spectra.axis.copy() if isinstance(reference_spectra, Spectra) else None
        self.names_ = tuple(self.component_names) if self.component_names is not None else tuple(f"component_{i}" for i in range(len(self.references_)))
        if len(self.names_) != len(self.references_) or len(set(self.names_)) != len(self.names_):
            raise ValueError("Provide one unique name per reference component.")
        self.is_fitted_ = True
        return self

    def _solve(self, a): return np.linalg.lstsq(self.references_.T, a.T, rcond=None)[0].T

    def transform(self, X):
        self._check_fitted(); a = self._values(X)
        if a.shape[1] != self.references_.shape[1]: raise ValueError("Mixture/reference feature counts differ.")
        if self.axis_ is not None and (not isinstance(X, Spectra) or not np.array_equal(X.axis, self.axis_)):
            raise ValueError("Mixture/reference axes must match exactly.")
        coefficients = self._solve(a); reconstruction = coefficients @ self.references_
        return UnmixingResult(coefficients, self.references_.copy(), reconstruction,
                             np.linalg.norm(a-reconstruction, axis=1), self.names_,
                             None if self.axis_ is None else self.axis_.copy(), X.sample_ids if isinstance(X, Spectra) else None)

    def predict(self, X): return self.transform(X).coefficients


class NNLSUnmixer(CLSUnmixer):
    def _solve(self, a): return np.vstack([nnls(self.references_.T, row)[0] for row in a])
