"""Spectral transformations; learned scalers fit on training data only."""
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pybaselines import Baseline
from ..core import Node, Spectra


def _spectra(X):
    if not isinstance(X, Spectra):
        raise TypeError("This tool requires Spectra with an explicit RamanShift axis.")
    return X


class SGFilter(Node):
    def __init__(self, window_length=11, polyorder=3, deriv=0):
        self.window_length = window_length; self.polyorder = polyorder; self.deriv = deriv

    def transform(self, X):
        a = self._values(X)
        if self.window_length % 2 != 1 or not 0 <= self.polyorder < self.window_length <= a.shape[1]:
            raise ValueError("SG requires an odd window <= points and 0 <= polyorder < window.")
        if not 0 <= self.deriv <= self.polyorder:
            raise ValueError("deriv must lie between 0 and polyorder.")
        delta = 1.0
        if isinstance(X, Spectra):
            steps = np.diff(X.axis)
            if len(steps) and not np.allclose(steps, steps[0]):
                raise ValueError("SG requires an equally spaced spectral grid; interpolate explicitly.")
            delta = float(steps[0]) if len(steps) else 1.0
        return self._wrap(X, savgol_filter(a, self.window_length, self.polyorder, deriv=self.deriv, delta=delta, axis=1))


class AirPLS(Node):
    def __init__(self, lam=1e5, diff_order=2, max_iter=50, tol=1e-3):
        self.lam = lam; self.diff_order = diff_order; self.max_iter = max_iter; self.tol = tol

    def transform(self, X):
        a = self._values(X)
        if self.lam <= 0 or self.diff_order < 1 or self.max_iter < 1 or self.tol <= 0:
            raise ValueError("Invalid airPLS parameters.")
        fitter = Baseline(x_data=X.axis if isinstance(X, Spectra) else None)
        corrected = [row - fitter.airpls(row, lam=self.lam, diff_order=self.diff_order,
                     max_iter=self.max_iter, tol=self.tol)[0] for row in a]
        return self._wrap(X, np.asarray(corrected))


class Interpolate(Node):
    def __init__(self, target_axis, kind="linear"):
        self.target_axis = target_axis; self.kind = kind

    def transform(self, X):
        X = _spectra(X); target = np.asarray(self.target_axis, dtype=float)
        if target.ndim != 1 or len(target) < 2 or not np.isfinite(target).all() or not np.all(np.diff(target) > 0):
            raise ValueError("target_axis must be finite and strictly increasing.")
        if target[0] < X.axis[0] or target[-1] > X.axis[-1]:
            raise ValueError("Extrapolation outside the measured range is not permitted.")
        y = interp1d(X.axis, X.intensity, axis=1, kind=self.kind, bounds_error=True)(target)
        return X.with_values(y, axis=target)


class SpectralCut(Node):
    def __init__(self, lower, upper):
        self.lower = lower; self.upper = upper

    def transform(self, X):
        X = _spectra(X)
        if self.lower >= self.upper:
            raise ValueError("lower must be less than upper.")
        mask = (X.axis >= self.lower) & (X.axis <= self.upper)
        if mask.sum() < 2:
            raise ValueError("Cut contains fewer than two measured coordinates.")
        return X.with_values(X.intensity[:, mask], axis=X.axis[mask])


class PeakNormalize(Node):
    """Divide each spectrum by its maximum absolute intensity (or a given band)."""
    def __init__(self, band=None):
        self.band = band

    def transform(self, X):
        a = self._values(X); region = a
        if self.band is not None:
            X = _spectra(X); lo, hi = self.band
            if lo > hi: raise ValueError("Invalid normalization band.")
            mask = (X.axis >= lo) & (X.axis <= hi)
            if not mask.any(): raise ValueError("Normalization band has no measured points.")
            region = a[:, mask]
        scale = np.max(np.abs(region), axis=1, keepdims=True)
        return self._wrap(X, a / np.where(scale == 0, 1.0, scale))


class _Scale(Node):
    requires_fit = True

    def _check_axis(self, X):
        if self.axis_ is not None and (not isinstance(X, Spectra) or not np.array_equal(X.axis, self.axis_)):
            raise ValueError("Scaler requires the same spectral axis used at fit.")

    def fit(self, X, y=None):
        if self.scope not in {"feature", "spectrum"}: raise ValueError("scope is feature or spectrum.")
        self.axis_ = X.axis.copy() if isinstance(X, Spectra) else None
        self.n_features_in_ = self._values(X).shape[1]
        if self.scope == "feature": self.model_ = self._make_scaler().fit(self._values(X))
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._check_fitted(); self._check_axis(X); a = self._values(X)
        if a.shape[1] != self.n_features_in_: raise ValueError("Feature count differs from fit.")
        if self.scope == "feature": out = self.model_.transform(a)
        else: out = self._make_scaler().fit_transform(a.T).T
        return self._wrap(X, out)


class MinMaxNormalize(_Scale):
    def __init__(self, scope="spectrum", feature_range=(0, 1)):
        self.scope = scope; self.feature_range = feature_range

    def _make_scaler(self): return MinMaxScaler(feature_range=self.feature_range)


class Standardize(_Scale):
    def __init__(self, scope="feature", with_mean=True, with_std=True):
        self.scope = scope; self.with_mean = with_mean; self.with_std = with_std

    def _make_scaler(self): return StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
