"""Explicit axes and sample identity; no implicit alignment/interpolation."""
from dataclasses import dataclass
import numpy as np


def matrix(values):
    out = np.array(values, dtype=float, copy=True)
    if out.ndim != 2 or min(out.shape) == 0 or not np.isfinite(out).all():
        raise ValueError("Expected a finite, non-empty (samples, features) matrix.")
    return out


@dataclass(frozen=True)
class Spectra:
    intensity: np.ndarray
    axis: np.ndarray
    sample_ids: tuple | None = None

    def __post_init__(self):
        y = matrix(self.intensity)
        x = np.array(self.axis, dtype=float, copy=True)
        if x.ndim != 1 or len(x) != y.shape[1] or not np.isfinite(x).all():
            raise ValueError("axis must be finite and match the spectral columns.")
        if len(x) > 1 and not np.all(np.diff(x) > 0):
            raise ValueError("RamanShift must be strictly increasing; sort explicitly before use.")
        ids = None if self.sample_ids is None else tuple(self.sample_ids)
        if ids is not None and len(ids) != y.shape[0]:
            raise ValueError("sample_ids must match rows.")
        x.setflags(write=False); y.setflags(write=False)
        object.__setattr__(self, "axis", x)
        object.__setattr__(self, "intensity", y)
        object.__setattr__(self, "sample_ids", ids)

    def with_values(self, values, axis=None, sample_ids=None):
        return Spectra(values, self.axis if axis is None else axis,
                       self.sample_ids if sample_ids is None else sample_ids)


@dataclass(frozen=True)
class UnmixingResult:
    coefficients: np.ndarray       # samples x components; not automatically concentrations
    component_spectra: np.ndarray  # components x spectral points
    reconstruction: np.ndarray
    residual_norm: np.ndarray
    component_names: tuple[str, ...]
    axis: np.ndarray | None
    sample_ids: tuple | None
