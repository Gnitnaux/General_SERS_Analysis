"""Feature extraction and aggregation with explicit sample/group semantics."""
from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import PCA, FactorAnalysis
from ..core import Node, Spectra
from .estimators import EstimatorNode


class _FeatureModel(EstimatorNode):
    """Shared training-only PCA/factor-analysis projection adapter.

    Attributes
    ----------
    is_fitted_: Whether fitting completed successfully; typically created or set by fit.
    model_: Fitted third-party estimator, projection or scaler; usually absent before fit.
    output_kind: Declared output category.

    Inherited interface
    -------------------
    See EstimatorNode for methods and attributes not overridden here.
    """
    output_kind = "features"

    def fit(self, X, y=None):
        """Fit the training-only PCA or factor-analysis projection and store model_.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.
        y: Ground-truth targets aligned with rows, usually (n_rows, n_targets); never predictions.

        Returns and state
        -----------------
        self. Learned attributes are stored on the instance; see the class attribute descriptions.
        """
        a = self._remember(X); p = dict(self.parameters or {}); p.setdefault("random_state", 0)
        self.model_ = self.estimator_type(**p).fit(a); self.is_fitted_ = True
        return self

    def transform(self, X):
        """Apply the learned projection and return a two-dimensional feature matrix.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Apply the learned projection and return a two-dimensional feature matrix.
        """
        a = self._input(X); return self.model_.transform(a)


class PCAFeatures(_FeatureModel):
    """Principal-component feature projection using the inherited estimator interface.

    Attributes
    ----------
    estimator_type: Underlying estimator class instantiated during fit.

    Inherited interface
    -------------------
    See _FeatureModel for methods and attributes not overridden here.
    """
    estimator_type = PCA


class FactorAnalysisFeatures(_FeatureModel):
    """Factor-analysis feature projection using the inherited estimator interface.

    Attributes
    ----------
    estimator_type: Underlying estimator class instantiated during fit.

    Inherited interface
    -------------------
    See _FeatureModel for methods and attributes not overridden here.
    """
    estimator_type = FactorAnalysis


class PeakExtractor(Node):
    """Fixed-width height, position or area features from declared spectral bands.

    Attributes
    ----------
    bands: Fixed list of (lower, upper) coordinate intervals.
    output_kind: Declared output category.
    statistic: Peak feature: height, position or trapezoidal area.

    Inherited interface
    -------------------
    See Node for methods and attributes not overridden here.
    """
    output_kind = "features"

    def __init__(self, bands, statistic="height"):
        """Store configuration without fitting training data. See the class attributes for constructor parameter meanings.

        Parameters
        ----------
        bands: Fixed list of (lower, upper) coordinate intervals.
        statistic: Peak feature: height, position or trapezoidal area.

        Returns and state
        -----------------
        None. Stores configuration only.
        """
        self.bands = bands; self.statistic = statistic

    def transform(self, X):
        """Return one declared height, position or trapezoidal-area feature per band.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Return one declared height, position or trapezoidal-area feature per band.

        Raises
        ------
        TypeError, ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
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
    """Variable-length per-spectrum peak lists for inspection, not a fixed-width model matrix.

    Attributes
    ----------
    distance: Optional minimum peak spacing in points, forwarded to scipy.find_peaks.
    output_kind: Declared output category.
    prominence: Optional minimum prominence forwarded to scipy.find_peaks.

    Inherited interface
    -------------------
    See Node for methods and attributes not overridden here.
    """
    output_kind = "peak_lists"

    def __init__(self, prominence=None, distance=None):
        """Store configuration without fitting training data. See the class attributes for constructor parameter meanings.

        Parameters
        ----------
        prominence: Optional minimum prominence forwarded to scipy.find_peaks.
        distance: Optional minimum peak spacing in points, forwarded to scipy.find_peaks.

        Returns and state
        -----------------
        None. Stores configuration only.
        """
        self.prominence = prominence; self.distance = distance

    def transform(self, X):
        """Return variable-length per-row peak position/intensity dictionaries.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Return variable-length per-row peak position/intensity dictionaries.

        Raises
        ------
        TypeError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        if not isinstance(X, Spectra): raise TypeError("Peak detection requires a spectral axis.")
        result = []
        for row in X.intensity:
            indices, _ = find_peaks(row, prominence=self.prominence, distance=self.distance)
            result.append({"positions": X.axis[indices].copy(), "intensities": row[indices].copy()})
        return result


@dataclass(frozen=True)
class AggregationResult:
    """Aggregated numeric matrix with explicit output-group ordering.

    Attributes
    ----------
    group_ids: Original source-group identifiers aligned with output rows; not proof of preparation independence.
    values: Aggregated matrix with shape (n_groups, n_features).
    """
    values: np.ndarray
    group_ids: tuple


class _Reducer(Node):
    """Reduce rows within groups, or features within each row when groups are omitted.

    Attributes
    ----------
    output_kind: Declared output category.

    Inherited interface
    -------------------
    See Node for methods and attributes not overridden here.
    """
    output_kind = "aggregated_values"

    def transform(self, X, *, groups=None):
        """Without groups, reduce each row to one column; with groups, pool rows into Spectra or AggregationResult.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.
        groups: Source-group identifiers aligned with current rows; determine pooling membership.

        Returns and state
        -----------------
        Without groups, reduce each row to one column; with groups, pool rows into Spectra or AggregationResult.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
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
    """Arithmetic-mean reduction using the inherited grouping contract.

    Attributes
    ----------
    function: NumPy reduction function applied along the explicitly selected axis.

    Inherited interface
    -------------------
    See _Reducer for methods and attributes not overridden here.
    """
    function = staticmethod(np.mean)


class FeatureConcat(Node):
    """Concatenate feature columns after the batch executor aligns row identities.

    Inherited interface
    -------------------
    See Node for methods and attributes not overridden here.
    """
    def transform(self, X, *, other):
        """Concatenate X and other along columns; row identity alignment is the executor responsibility.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.
        other: Feature branch already aligned to X by row identifiers.

        Returns and state
        -----------------
        Concatenate X and other along columns; row identity alignment is the executor responsibility.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        a, b = self._values(X), self._values(other)
        if len(a) != len(b):
            raise ValueError('Feature branches must have matching row counts.')
        return np.column_stack((a, b))


class MedianAggregator(_Reducer):
    """Median reduction using the inherited grouping contract.

    Attributes
    ----------
    function: NumPy reduction function applied along the explicitly selected axis.

    Inherited interface
    -------------------
    See _Reducer for methods and attributes not overridden here.
    """
    function = staticmethod(np.median)


class MaxAggregator(_Reducer):
    """Maximum reduction using the inherited grouping contract.

    Attributes
    ----------
    function: NumPy reduction function applied along the explicitly selected axis.

    Inherited interface
    -------------------
    See _Reducer for methods and attributes not overridden here.
    """
    function = staticmethod(np.max)


class MinAggregator(_Reducer):
    """Minimum reduction using the inherited grouping contract.

    Attributes
    ----------
    function: NumPy reduction function applied along the explicitly selected axis.

    Inherited interface
    -------------------
    See _Reducer for methods and attributes not overridden here.
    """
    function = staticmethod(np.min)


class SpectralDifference(Node):
    """Subtract aligned spectra or one explicitly broadcast reference spectrum.

    Attributes
    ----------
    output_kind: Declared output category.

    Inherited interface
    -------------------
    See Node for methods and attributes not overridden here.
    """
    output_kind = "spectra"

    def transform(self, X, *, reference):
        """Subtract a matching reference batch or one broadcast background spectrum; check axes and row identities.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.
        reference: One broadcast background spectrum or a row/axis-matched reference Spectra batch.

        Returns and state
        -----------------
        Subtract a matching reference batch or one broadcast background spectrum; check axes and row identities.

        Raises
        ------
        TypeError, ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        if not isinstance(X, Spectra) or not isinstance(reference, Spectra):
            raise TypeError("Both inputs must be Spectra with explicit axes.")
        if not np.array_equal(X.axis, reference.axis): raise ValueError("Axes must match exactly.")
        if len(reference.intensity) not in {1, len(X.intensity)}: raise ValueError("Reference must have one or matching rows.")
        if len(reference.intensity) > 1:
            if X.sample_ids is None or reference.sample_ids != X.sample_ids:
                raise ValueError("Multiple reference rows require matching sample_ids in matching order.")
        return X.with_values(X.intensity-reference.intensity)
