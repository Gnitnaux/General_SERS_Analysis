"""Read standardized database CSVs and visualize raw within-group mean spectra."""
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np

from .core.data import load_dataset, read_spectrum as _read_spectrum
from .util import natural_key as _natural_key


@dataclass(frozen=True)
class GroupMean:
    """One source group's arithmetic mean for visualization, without implicit interpolation or baseline correction.

    Attributes
    ----------
    axis: Strictly increasing one-dimensional Raman-shift coordinates, normally in cm^-1.
    group_id: Original source-group folder identifier.
    intensity: One-dimensional mean intensity array, ordered by axis; not an immutable Spectra matrix.
    n_spectra: Number of spectra contributing to this group mean.
    """
    group_id: str
    n_spectra: int
    axis: np.ndarray
    intensity: np.ndarray


def load_group_means(database_dir, data_id, *, groups=None):
    """Strict raw-axis visualization using the shared reader; no implicit alignment."""
    dataset = load_dataset(database_dir, data_id, groups=groups, read_labels=False,
                           label_policy='include_flagged')
    means = []
    for group in sorted(set(dataset.group_ids), key=_natural_key):
        indices = np.flatnonzero(dataset.group_ids == group)
        spectra = [dataset.spectra[i] for i in indices]
        axis = spectra[0].axis
        if any(not np.array_equal(s.axis, axis) for s in spectra):
            raise ValueError(f'RamanShift mismatch within {group}; align spectra explicitly before averaging.')
        means.append(GroupMean(group, len(spectra), axis,
                               np.mean([s.intensity[0] for s in spectra], axis=0)))
    if groups is None:
        for key, observed in (('n_group', len(means)), ('n_spectra', len(dataset.spectra))):
            recorded = dataset.metadata.get(key, '').strip()
            if recorded and recorded.lower() != 'null' and int(recorded) != observed:
                warnings.warn(f'{data_id}: index {key}={recorded}, observed {observed}.', stacklevel=2)
    return dataset.metadata, means


def visualize_dataset(database_dir, data_id, output_dir, *, groups=None, show=False):
    """Save one overlay plot; each curve represents one group's repeat-spectrum mean."""
    metadata, means = load_group_means(database_dir, data_id, groups=groups)
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
        try:
            colors = plt.get_cmap("turbo")
            for index, group in enumerate(means):
                ax.plot(group.axis, group.intensity, linewidth=1,
                        color=colors(index / max(len(means) - 1, 1)),
                        label=f"{group.group_id} (n={group.n_spectra})")
            details = " | ".join(value for value in (metadata.get("task"), metadata.get("modality")) if value)
            ax.set_title(f"{data_id}: group mean spectra\n{details}")
            ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
            ax.set_ylabel("Intensity (source units)")
            ax.grid(alpha=0.18)
            if len(means) <= 20:
                ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
            else:
                ax.text(.01, .99, f"{len(means)} group means; use --groups for a labeled subset",
                        transform=ax.transAxes, va="top", fontsize=8)
            target = Path(output_dir).resolve() / f"{data_id}_group_means.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, dpi=180)
            if show:
                plt.show()
        finally:
            plt.close(fig)
    return target, means
