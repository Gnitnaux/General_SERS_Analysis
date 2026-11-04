"""Feature extraction and aggregation with explicit sample/group semantics."""
from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import PCA, FactorAnalysis
from ..core import Node, Spectra
from .estimators import EstimatorNode


class _FeatureModel(EstimatorNode):
    output_kind = "features"

    def fit(self, X, y=None):
        a = self._remember(X); p = dict(self.parameters or {}); p.setdefault("random_state", 0)
        self.model_ = self.estimator_type(**p).fit(a); self.is_fitted_ = True
        return self

    def transform(self, X):
        a = self._input(X); return self.model_.transform(a)


class PCAFeatures(_FeatureModel):
    estimator_type = PCA


class FactorAnalysisFeatures(_FeatureModel):
    estimator_type = FactorAnalysis


class PeakExtractor(Node):
    """Fixed windows give stable feature columns; not fitted peak-shape deconvolution."""
    output_kind = "features"

    def __init__(self, bands, statistic="height"):
        self.bands = bands; self.statistic = statistic

    def transform(self, X):
        if not isinstance(X, Spectra): raise TypeError("Peak extraction requires a spectral axis.")
        if self.statistic not in {"height", "position", "area"}: raise ValueError("Use height, position or area.")
        result = []
        for lo, hi in self.bands:
            mask = (X.axis >= lo) & (X.axis <= hi)
            if lo > hi or not mask.any(): raise ValueError("A peak window contains no measured points.")
            y = X.intensity[:, mask]; x = X.axis[mask]
            if self.statistic == "height": values = y.max(axis=1)
            elif self.statistic == "position": values = x[y.argmax(axis=1)]
            else:
                if len(x) < 2: raise ValueError("Peak area needs at least two coordinates.")
                values = np.sum((y[:, 1:] + y[:, :-1]) * np.diff(x) / 2, axis=1)
            result.append(values)
        if not result: raise ValueError("Supply at least one peak window.")
        return np.column_stack(result)


class PeakDetector(Node):
    """Variable-length local-peak lists for inspection, not directly a fixed-width ML matrix."""
    output_kind = "peak_lists"

    def __init__(self, prominence=None, distance=None):
        self.prominence = prominence; self.distance = distance

    def transform(self, X):
        if not isinstance(X, Spectra): raise TypeError("Peak detection requires a spectral axis.")
        result = []
        for row in X.intensity:
            indices, _ = find_peaks(row, prominence=self.prominence, distance=self.distance)
            result.append({"positions": X.axis[indices].copy(), "intensities": row[indices].copy()})
        return result


@dataclass(frozen=True)
class AggregationResult:
    values: np.ndarray
    group_ids: tuple


class _Reducer(Node):
    """Without groups: reduce features per row. With groups: pool rows within sample."""
    output_kind = "aggregated_values"

    def transform(self, X, *, groups=None):
        a = self._values(X)
        if groups is None:
            return self.function(a, axis=1, keepdims=True)
        groups = list(groups)
        if len(groups) != len(a): raise ValueError("groups must match input rows.")
        ids = tuple(dict.fromkeys(groups))
        values = np.vstack([self.function(a[[g == key for g in groups]], axis=0) for key in ids])
        if isinstance(X, Spectra): return Spectra(values, X.axis, ids)
        return AggregationResult(values, ids)


class MeanAggregator(_Reducer):
    function = staticmethod(np.mean)


class MedianAggregator(_Reducer):
    function = staticmethod(np.median)


class MaxAggregator(_Reducer):
    function = staticmethod(np.max)


class MinAggregator(_Reducer):
    function = staticmethod(np.min)


class SpectralDifference(Node):
    """Subtract aligned rows or one explicitly broadcast background spectrum."""
    output_kind = "spectra"

    def transform(self, X, *, reference):
        if not isinstance(X, Spectra) or not isinstance(reference, Spectra):
            raise TypeError("Both inputs must be Spectra with explicit axes.")
        if not np.array_equal(X.axis, reference.axis): raise ValueError("Axes must match exactly.")
        if len(reference.intensity) not in {1, len(X.intensity)}: raise ValueError("Reference must have one or matching rows.")
        if len(reference.intensity) > 1:
            if X.sample_ids is None or reference.sample_ids != X.sample_ids:
                raise ValueError("Multiple reference rows require matching sample_ids in matching order.")
        return X.with_values(X.intensity-reference.intensity)
