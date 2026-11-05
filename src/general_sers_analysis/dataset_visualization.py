"""Read standardized database CSVs and visualize raw within-group mean spectra."""
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import numpy as np

from .core import Spectra


@dataclass(frozen=True)
class GroupMean:
    group_id: str
    n_spectra: int
    axis: np.ndarray
    intensity: np.ndarray


def _natural_key(path):
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r"(\d+)", path.name))


def _read_spectrum(path):
    """Accept increasing or decreasing axes; reverse decreasing data without interpolation."""
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
            if [column.strip() for column in header] != ["RamanShift", "intensity"]:
                raise ValueError("Expected exactly two columns: RamanShift,intensity.")
            values = np.loadtxt(stream, delimiter=",", ndmin=2)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
            raise ValueError("A spectrum needs at least two rows and exactly two numeric columns.")
        if np.all(np.diff(values[:, 0]) < 0):
            values = values[::-1]
        spectrum = Spectra(values[:, 1][None, :], values[:, 0])
        return spectrum.axis, spectrum.intensity[0]
    except (ValueError, OSError) as error:
        raise ValueError(f"Invalid spectrum {path}: {error}") from error


def load_group_means(database_dir, data_id, *, groups=None):
    """Return index metadata and one raw arithmetic mean per group (no preprocessing)."""
    root = Path(database_dir).resolve()
    if not data_id or data_id in {".", ".."} or any(c in data_id for c in "/\\"):
        raise ValueError("data_id must be an exact dataset identifier, not a path.")
    with (root / "content.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "Data_id" not in reader.fieldnames:
            raise ValueError("content.csv must contain a Data_id column.")
        rows = [row for row in reader if row["Data_id"] == data_id]
    if len(rows) != 1:
        raise ValueError(f"Expected one index row for {data_id}; found {len(rows)}.")
    metadata = rows[0]
    dataset = root / "data" / data_id
    if not dataset.is_dir():
        raise ValueError(f"Dataset data directory does not exist: {dataset}")
    folders = sorted((path for path in dataset.iterdir() if path.is_dir() and not path.name.startswith(".")),
                     key=_natural_key)
    selected = set(groups) if groups is not None else None
    if selected is not None:
        missing = selected - {path.name for path in folders}
        if missing:
            raise ValueError(f"Unknown group folders: {', '.join(sorted(missing))}")
        folders = [path for path in folders if path.name in selected]
    if not folders:
        raise ValueError(f"No group directories found for {data_id}.")

    means = []
    for folder in folders:
        paths = sorted((path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".csv"),
                       key=_natural_key)
        if not paths:
            raise ValueError(f"Group contains no spectrum CSV files: {folder}")
        axis, average = _read_spectrum(paths[0])
        average = average.copy()
        for count, path in enumerate(paths[1:], start=2):
            next_axis, values = _read_spectrum(path)
            if not np.array_equal(next_axis, axis):
                raise ValueError(f"RamanShift mismatch within {folder.name}: {paths[0].name} vs {path.name}. "
                                 "Align spectra explicitly before averaging.")
            # Online arithmetic mean keeps memory independent of the number of repeats.
            average += (values - average) / count
        means.append(GroupMean(folder.name, len(paths), axis, average))

    if selected is None:
        for key, observed in (("n_group", len(means)), ("n_spectra", sum(g.n_spectra for g in means))):
            recorded = metadata.get(key, "").strip()
            if recorded and recorded.lower() != "null" and int(recorded) != observed:
                warnings.warn(f"{data_id}: index {key}={recorded}, observed {observed}.", stacklevel=2)
    return metadata, means


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
