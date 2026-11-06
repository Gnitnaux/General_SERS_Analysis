"""Thresholded independent-event occupancy calibration, NOT a universal digital-SERS assay."""
import numpy as np
from ..core import Node, Spectra


class DigitalSERSRegressor(Node):
    """Thresholded independent-event occupancy calibration; rejects all-positive saturated groups.

    Attributes
    ----------
    axis_: Training coordinate copy, or None for ordinary matrix inputs.
    band: Optional (lower, upper) coordinate interval; None uses all columns.
    is_fitted_: Whether fitting completed successfully; typically created or set by fit.
    last_event_counts_: Per-group total-event counts from the most recent prediction.
    last_group_ids_: Group order produced by the most recent digital-SERS prediction.
    last_positive_counts_: Per-group positive-event counts from the most recent prediction.
    n_features_in_: Training feature count; checked against inference inputs.
    output_kind: Declared output category.
    requires_fit: Whether the executor must fit this tool before using its output method.
    sensitivity_: Positive zero-intercept concentration-to-occupancy calibration factor.
    threshold: Independently chosen finite event threshold; positivity uses strict greater-than.

    Inherited interface
    -------------------
    See Node for methods and attributes not overridden here.
    """
    requires_fit = True
    output_kind = "group_target_values"

    def __init__(self, threshold, band=None):
        """Store configuration without fitting training data. See the class attributes for constructor parameter meanings.

        Parameters
        ----------
        threshold: Independently chosen finite event threshold; positivity uses strict greater-than.
        band: Optional (lower, upper) coordinate interval; None uses all columns.

        Returns and state
        -----------------
        None. Stores configuration only.
        """
        self.threshold = threshold; self.band = band

    def _occupancy(self, X, groups):
        """Count threshold-positive events and compute -log(1-p) per group; reject saturation.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.
        groups: Source-group identifiers aligned with current rows; determine pooling membership.

        Returns and state
        -----------------
        Tuple (group IDs, occupancy, positive counts, total counts).

        Raises
        ------
        TypeError, ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        a = self._values(X)
        if not np.isfinite(self.threshold): raise ValueError("Supply a finite, independently chosen detection threshold.")
        if self.band is not None:
            if not isinstance(X, Spectra): raise TypeError("A detection band requires Spectra.")
            lo, hi = self.band; mask = (X.axis >= lo) & (X.axis <= hi)
            if lo > hi or not mask.any(): raise ValueError("Detection band contains no measured points.")
            a = a[:, mask]
        groups = list(groups)
        if len(groups) != len(a): raise ValueError("groups must identify every event row.")
        ids = list(dict.fromkeys(groups)); positive = a.max(axis=1) > self.threshold
        counts = np.array([sum(positive[i] for i,g in enumerate(groups) if g == key) for key in ids])
        totals = np.array([groups.count(key) for key in ids]); p = counts / totals
        if np.any(p == 1): raise ValueError("All-positive groups are saturated; finite occupancy concentration cannot be estimated.")
        return ids, -np.log1p(-p), counts, totals

    def fit(self, X, y, *, groups):
        """Validate group concentrations and learn a positive zero-intercept concentration/occupancy factor.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.
        y: Ground-truth targets aligned with rows, usually (n_rows, n_targets); never predictions.
        groups: Source-group identifiers aligned with current rows; determine pooling membership.

        Returns and state
        -----------------
        self. Learned attributes are stored on the instance; see the class attribute descriptions.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        self.n_features_in_ = self._values(X).shape[1]
        self.axis_ = X.axis.copy() if isinstance(X, Spectra) else None
        ids, occupancy, _, _ = self._occupancy(X, groups)
        target = np.asarray(y, dtype=float).reshape(-1); group_values = list(groups)
        if len(target) != len(group_values) or not np.isfinite(target).all() or (target < 0).any():
            raise ValueError("Provide a finite nonnegative concentration for every event row.")
        c = []
        for key in ids:
            values = target[[g == key for g in group_values]]
            if not np.all(values == values[0]): raise ValueError("One group must have one concentration label.")
            c.append(values[0])
        if len(np.unique(c)) < 2 or occupancy @ occupancy == 0:
            raise ValueError("Calibration needs multiple concentration levels and positive events.")
        self.sensitivity_ = float(occupancy @ np.asarray(c) / (occupancy @ occupancy))
        if self.sensitivity_ <= 0: raise ValueError("Calibration slope must be positive.")
        self.is_fitted_ = True
        return self

    def predict(self, X, *, groups):
        """Return one concentration per group and record group order and event counts.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.
        groups: Source-group identifiers aligned with current rows; determine pooling membership.

        Returns and state
        -----------------
        Return one concentration per group and record group order and event counts.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        self._check_fitted()
        if self._values(X).shape[1] != self.n_features_in_: raise ValueError("Feature count differs from calibration.")
        if self.axis_ is not None and (not isinstance(X, Spectra) or not np.array_equal(X.axis, self.axis_)):
            raise ValueError("Spectral axis differs from calibration.")
        ids, occupancy, counts, totals = self._occupancy(X, groups)
        self.last_group_ids_ = tuple(ids); self.last_positive_counts_ = counts; self.last_event_counts_ = totals
        return (self.sensitivity_ * occupancy)[:, None]
