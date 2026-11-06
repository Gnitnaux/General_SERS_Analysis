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
    """Row-wise Savitzky-Golay smoothing or derivatives; explicit spectral grids must be equally spaced.

    Attributes
    ----------
    deriv: SG derivative order; zero smooths. Explicit coordinates determine derivative spacing.
    polyorder: SG local polynomial order, smaller than window_length.
    window_length: Odd SG window length in points; must not exceed the input width.

    Inherited interface
    -------------------
    See Node for methods and attributes not overridden here.
    """
    def __init__(self, window_length=11, polyorder=3, deriv=0):
        """Store configuration without fitting training data. See the class attributes for constructor parameter meanings.

        Parameters
        ----------
        window_length: Odd SG window length in points; must not exceed the input width.
        polyorder: SG local polynomial order, smaller than window_length.
        deriv: SG derivative order; zero smooths. Explicit coordinates determine derivative spacing.

        Returns and state
        -----------------
        None. Stores configuration only.
        """
        self.window_length = window_length; self.polyorder = polyorder; self.deriv = deriv

    def transform(self, X):
        """Smooth or differentiate each spectrum; use actual spacing when Spectra coordinates are provided.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Smooth or differentiate each spectrum; use actual spacing when Spectra coordinates are provided.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
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
    """Estimate and subtract an airPLS baseline independently for each input spectrum.

    Attributes
    ----------
    diff_order: Positive airPLS difference-penalty order.
    lam: Positive airPLS smoothing penalty.
    max_iter: Maximum number of airPLS iterations.
    tol: Positive airPLS convergence tolerance.

    Inherited interface
    -------------------
    See Node for methods and attributes not overridden here.
    """
    def __init__(self, lam=1e5, diff_order=2, max_iter=50, tol=1e-3):
        """Store configuration without fitting training data. See the class attributes for constructor parameter meanings.

        Parameters
        ----------
        lam: Positive airPLS smoothing penalty.
        diff_order: Positive airPLS difference-penalty order.
        max_iter: Maximum number of airPLS iterations.
        tol: Positive airPLS convergence tolerance.

        Returns and state
        -----------------
        None. Stores configuration only.
        """
        self.lam = lam; self.diff_order = diff_order; self.max_iter = max_iter; self.tol = tol

    def transform(self, X):
        """Estimate and subtract each row baseline independently, preserving the input wrapper.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Estimate and subtract each row baseline independently, preserving the input wrapper.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        a = self._values(X)
        if self.lam <= 0 or self.diff_order < 1 or self.max_iter < 1 or self.tol <= 0:
            raise ValueError("Invalid airPLS parameters.")
        fitter = Baseline(x_data=X.axis if isinstance(X, Spectra) else None)
        corrected = [row - fitter.airpls(row, lam=self.lam, diff_order=self.diff_order,
                     max_iter=self.max_iter, tol=self.tol)[0] for row in a]
        return self._wrap(X, np.asarray(corrected))


class Interpolate(Node):
    """Resample Spectra to explicit coordinates without extrapolation.

    Attributes
    ----------
    kind: Interpolation method forwarded to scipy.interpolate.interp1d.
    target_axis: Explicit increasing interpolation grid; extrapolation is forbidden.

    Inherited interface
    -------------------
    See Node for methods and attributes not overridden here.
    """
    def __init__(self, target_axis, kind="linear"):
        """Store configuration without fitting training data. See the class attributes for constructor parameter meanings.

        Parameters
        ----------
        target_axis: Explicit increasing interpolation grid; extrapolation is forbidden.
        kind: Expected coarse payload category, or None to skip this check.

        Returns and state
        -----------------
        None. Stores configuration only.
        """
        self.target_axis = target_axis; self.kind = kind

    def transform(self, X):
        """Interpolate to target_axis and reject extrapolation outside measured coordinates.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Interpolate to target_axis and reject extrapolation outside measured coordinates.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        X = _spectra(X); target = np.asarray(self.target_axis, dtype=float)
        if target.ndim != 1 or len(target) < 2 or not np.isfinite(target).all() or not np.all(np.diff(target) > 0):
            raise ValueError("target_axis must be finite and strictly increasing.")
        if target[0] < X.axis[0] or target[-1] > X.axis[-1]:
            raise ValueError("Extrapolation outside the measured range is not permitted.")
        y = interp1d(X.axis, X.intensity, axis=1, kind=self.kind, bounds_error=True)(target)
        return X.with_values(y, axis=target)


class SpectralCut(Node):
    """Select spectral columns inside an inclusive coordinate interval without changing sample rows.

    Attributes
    ----------
    lower: Inclusive lower spectral bound; must be smaller than upper.
    upper: Inclusive upper spectral bound.

    Inherited interface
    -------------------
    See Node for methods and attributes not overridden here.
    """
    def __init__(self, lower, upper):
        """Store configuration without fitting training data. See the class attributes for constructor parameter meanings.

        Parameters
        ----------
        lower: Inclusive lower spectral bound; must be smaller than upper.
        upper: Inclusive upper spectral bound.

        Returns and state
        -----------------
        None. Stores configuration only.
        """
        self.lower = lower; self.upper = upper

    def transform(self, X):
        """Keep columns in the inclusive interval and reject fewer than two retained points.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Keep columns in the inclusive interval and reject fewer than two retained points.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        X = _spectra(X)
        if self.lower >= self.upper:
            raise ValueError("lower must be less than upper.")
        mask = (X.axis >= self.lower) & (X.axis <= self.upper)
        if mask.sum() < 2:
            raise ValueError("Cut contains fewer than two measured coordinates.")
        return X.with_values(X.intensity[:, mask], axis=X.axis[mask])


class PeakNormalize(Node):
    """Divide each spectrum by its maximum absolute intensity, optionally in a band; zero spectra remain zero.

    Attributes
    ----------
    band: Optional (lower, upper) coordinate interval; None uses all columns.

    Inherited interface
    -------------------
    See Node for methods and attributes not overridden here.
    """
    def __init__(self, band=None):
        """Store configuration without fitting training data. See the class attributes for constructor parameter meanings.

        Parameters
        ----------
        band: Optional (lower, upper) coordinate interval; None uses all columns.

        Returns and state
        -----------------
        None. Stores configuration only.
        """
        self.band = band

    def transform(self, X):
        """Divide each row by its reference-band maximum absolute value; zero divisors use one.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Divide each row by its reference-band maximum absolute value; zero divisors use one.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
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
    """Shared feature-population or independent per-spectrum scaling implementation.

    Attributes
    ----------
    axis_: Training coordinate copy, or None for ordinary matrix inputs.
    is_fitted_: Whether fitting completed successfully; typically created or set by fit.
    model_: Fitted third-party estimator, projection or scaler; usually absent before fit.
    n_features_in_: Training feature count; checked against inference inputs.
    requires_fit: Whether the executor must fit this tool before using its output method.

    Inherited interface
    -------------------
    See Node for methods and attributes not overridden here.
    """
    requires_fit = True

    def _check_axis(self, X):
        """Require the same explicit spectral coordinates used during fit.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        None; raises on mismatched coordinates.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        if self.axis_ is not None and (not isinstance(X, Spectra) or not np.array_equal(X.axis, self.axis_)):
            raise ValueError("Scaler requires the same spectral axis used at fit.")

    def fit(self, X, y=None):
        """Record input width/axis. Feature mode learns training-population statistics; spectrum mode records structure only.

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
        if self.scope not in {"feature", "spectrum"}: raise ValueError("scope is feature or spectrum.")
        self.axis_ = X.axis.copy() if isinstance(X, Spectra) else None
        self.n_features_in_ = self._values(X).shape[1]
        if self.scope == "feature": self.model_ = self._make_scaler().fit(self._values(X))
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Use fitted feature statistics or independently scale each spectrum; preserve input wrapping.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Use fitted feature statistics or independently scale each spectrum; preserve input wrapping.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        self._check_fitted(); self._check_axis(X); a = self._values(X)
        if a.shape[1] != self.n_features_in_: raise ValueError("Feature count differs from fit.")
        if self.scope == "feature": out = self.model_.transform(a)
        else: out = self._make_scaler().fit_transform(a.T).T
        return self._wrap(X, out)


class MinMaxNormalize(_Scale):
    """Scale to a requested interval, independently per spectrum by default.

    Attributes
    ----------
    feature_range: Requested min-max output interval.
    scope: feature uses training-population statistics; spectrum scales each row independently.

    Inherited interface
    -------------------
    See _Scale for methods and attributes not overridden here.
    """
    def __init__(self, scope="spectrum", feature_range=(0, 1)):
        """Store configuration without fitting training data. See the class attributes for constructor parameter meanings.

        Parameters
        ----------
        scope: feature uses training-population statistics; spectrum scales each row independently.
        feature_range: Requested min-max output interval.

        Returns and state
        -----------------
        None. Stores configuration only.
        """
        self.scope = scope; self.feature_range = feature_range

    def _make_scaler(self):
        """Construct a new unfitted sklearn scaler from the current configuration.

        Returns and state
        -----------------
        New unfitted sklearn scaler.
        """
        return MinMaxScaler(feature_range=self.feature_range)


class Standardize(_Scale):
    """Mean/standard-deviation scaling; feature mode learns training statistics, spectrum mode implements row-wise SNV.

    Attributes
    ----------
    scope: feature uses training-population statistics; spectrum scales each row independently.
    with_mean: Whether to subtract the mean.
    with_std: Whether to divide by the standard deviation.

    Inherited interface
    -------------------
    See _Scale for methods and attributes not overridden here.
    """
    def __init__(self, scope="feature", with_mean=True, with_std=True):
        """Store configuration without fitting training data. See the class attributes for constructor parameter meanings.

        Parameters
        ----------
        scope: feature uses training-population statistics; spectrum scales each row independently.
        with_mean: Whether to subtract the mean.
        with_std: Whether to divide by the standard deviation.

        Returns and state
        -----------------
        None. Stores configuration only.
        """
        self.scope = scope; self.with_mean = with_mean; self.with_std = with_std

    def _make_scaler(self):
        """Construct a new unfitted sklearn scaler from the current configuration.

        Returns and state
        -----------------
        New unfitted sklearn scaler.
        """
        return StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
