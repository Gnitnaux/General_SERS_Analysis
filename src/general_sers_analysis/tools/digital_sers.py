"""Thresholded independent-event occupancy calibration, NOT a universal digital-SERS assay."""
import numpy as np
from ..core import Node, Spectra


class DigitalSERSRegressor(Node):
    requires_fit = True
    output_kind = "group_target_values"

    def __init__(self, threshold, band=None):
        self.threshold = threshold; self.band = band

    def _occupancy(self, X, groups):
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
        self._check_fitted()
        if self._values(X).shape[1] != self.n_features_in_: raise ValueError("Feature count differs from calibration.")
        if self.axis_ is not None and (not isinstance(X, Spectra) or not np.array_equal(X.axis, self.axis_)):
            raise ValueError("Spectral axis differs from calibration.")
        ids, occupancy, counts, totals = self._occupancy(X, groups)
        self.last_group_ids_ = tuple(ids); self.last_positive_counts_ = counts; self.last_event_counts_ = totals
        return (self.sensitivity_ * occupancy)[:, None]
