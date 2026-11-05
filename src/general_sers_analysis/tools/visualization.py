"""Reusable visualization Nodes. transform saves a figure and returns its input unchanged."""
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, mean_squared_error, r2_score

from ..core import Node, Spectra
from .aggregation import MeanAggregator


class _PlotNode(Node):
    def _pyplot(self):
        import matplotlib
        if not self.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt

    def _save(self, fig, plt):
        target = Path(self.output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=180)
        self.last_output_path_ = str(target)
        if self.show:
            plt.show()


class GroupSpectrumPlot(_PlotNode):
    """One within-group arithmetic mean per curve; no baseline correction or normalization."""
    def __init__(self, output_path, show=False, title="Group mean spectra",
                 ylabel="Intensity (source units)"):
        self.output_path = output_path
        self.show = show
        self.title = title
        self.ylabel = ylabel

    def transform(self, X, *, groups=None):
        if not isinstance(X, Spectra):
            raise TypeError("GroupSpectrumPlot requires Spectra with an explicit axis.")
        ids = list(groups) if groups is not None else list(X.sample_ids or range(len(X.intensity)))
        means = MeanAggregator().transform(X, groups=ids)
        self.group_ids_ = means.sample_ids
        plt = self._pyplot()
        fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
        try:
            colors = plt.get_cmap("turbo")
            for index, (group, values) in enumerate(zip(means.sample_ids, means.intensity)):
                ax.plot(means.axis, values, lw=1, color=colors(index / max(len(means.intensity)-1, 1)),
                        label=f"{group} (n={ids.count(group)})")
            ax.set(xlabel=r"Raman shift (cm$^{-1}$)", ylabel=self.ylabel, title=self.title)
            if len(means.intensity) <= 20:
                ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
            self._save(fig, plt)
        finally:
            plt.close(fig)
        return X


class PredictionPlot(_PlotNode):
    """Predicted versus true numeric targets, one subplot per molecule.

    X is the prediction matrix; y_true has identical sample/target ordering.
    Optional groups averages true and predicted values before plotting.
    """
    def __init__(self, output_path, molecule_names=None, unit="", show=False,
                 title="Predicted vs true"):
        self.output_path = output_path
        self.molecule_names = molecule_names
        self.unit = unit
        self.show = show
        self.title = title

    def transform(self, X, *, y_true, groups=None):
        pred = np.asarray(X, dtype=float)
        truth = np.asarray(y_true, dtype=float)
        if pred.ndim == 1:
            pred = pred[:, None]
        if truth.ndim == 1:
            truth = truth[:, None]
        if pred.ndim != 2 or pred.shape != truth.shape or not pred.size:
            raise ValueError("Predictions and true values must have matching nonempty samples x targets shapes.")
        if not np.isfinite(pred).all() or not np.isfinite(truth).all():
            raise ValueError("Predictions and true values must be finite.")
        if groups is not None:
            ids = list(groups)
            pred = MeanAggregator().transform(pred, groups=ids).values
            truth = MeanAggregator().transform(truth, groups=ids).values
        count = pred.shape[1]
        names = list(self.molecule_names) if self.molecule_names is not None else [f"Target {i+1}" for i in range(count)]
        if len(names) != count:
            raise ValueError("molecule_names must match the target count.")
        plt = self._pyplot()
        cols = min(count, 3)
        fig, axes = plt.subplots(int(np.ceil(count / cols)), cols, squeeze=False,
                                 figsize=(4.3 * cols, 4 * int(np.ceil(count / cols))), layout="constrained")
        self.metrics_ = []
        try:
            for i, ax in enumerate(axes.flat):
                if i >= count:
                    ax.set_visible(False)
                    continue
                lo, hi = min(truth[:, i].min(), pred[:, i].min()), max(truth[:, i].max(), pred[:, i].max())
                padding = (hi - lo) * .08 if hi > lo else max(abs(lo) * .08, .1)
                limits = (lo-padding, hi+padding)
                ax.scatter(truth[:, i], pred[:, i], s=40, alpha=.8)
                ax.plot(limits, limits, "--", color="gray", lw=1, label="Ideal: predicted = true")
                rmse = float(np.sqrt(mean_squared_error(truth[:, i], pred[:, i])))
                r2 = float(r2_score(truth[:, i], pred[:, i])) if len(truth) > 1 and np.ptp(truth[:, i]) else None
                self.metrics_.append({"RMSE": rmse, "R2": r2})
                score = f"{r2:.3g}" if r2 is not None else "undefined"
                suffix = f" ({self.unit})" if self.unit else ""
                ax.set(xlim=limits, ylim=limits, xlabel="True" + suffix, ylabel="Predicted" + suffix,
                       title=f"{names[i]}\nRMSE={rmse:.3g}; R²={score}; n={len(truth)}")
                ax.set_aspect("equal", adjustable="box")
                ax.legend(fontsize=7)
            fig.suptitle(self.title)
            self._save(fig, plt)
        finally:
            plt.close(fig)
        return X


class ConfusionMatrixPlot(_PlotNode):
    """Single-label confusion matrix, or one binary matrix per multilabel component.

    X must contain predicted labels, not probabilities. Rows are true labels.
    labels explicitly controls single-label class order; omitted order uses observed labels.
    """
    def __init__(self, output_path, labels=None, class_names=None, normalize=None,
                 multilabel=False, show=False, title="Confusion matrix"):
        self.output_path = output_path
        self.labels = labels
        self.class_names = class_names
        self.normalize = normalize
        self.multilabel = multilabel
        self.show = show
        self.title = title

    def transform(self, X, *, y_true):
        pred, truth = np.asarray(X), np.asarray(y_true)
        if pred.shape != truth.shape or not pred.size:
            raise ValueError("Predicted and true labels must have matching nonempty shapes.")
        if self.normalize not in {None, "true", "pred", "all"}:
            raise ValueError("normalize must be None, true, pred or all.")
        if self.multilabel:
            if pred.ndim != 2 or not np.isin(pred, [0, 1]).all() or not np.isin(truth, [0, 1]).all():
                raise ValueError("Multilabel inputs must be binary samples x components matrices.")
            if self.labels is not None:
                raise ValueError("labels is for single-label class ordering only.")
            matrices = multilabel_confusion_matrix(truth, pred).astype(float)
            names = list(self.class_names) if self.class_names is not None else [f"Component {i+1}" for i in range(pred.shape[1])]
            if len(names) != len(matrices):
                raise ValueError("class_names must match the component count.")
            ticks = ["Absent", "Present"]
        else:
            if pred.ndim != 1:
                raise ValueError("Single-label inputs must be label vectors; probabilities are not accepted.")
            observed = np.unique(np.concatenate([truth, pred]))
            labels = list(self.labels) if self.labels is not None else list(observed)
            if len(set(labels)) != len(labels) or not set(observed).issubset(labels):
                raise ValueError("labels must uniquely include every observed class.")
            matrices = confusion_matrix(truth, pred, labels=labels)[None, :].astype(float)
            ticks = list(self.class_names) if self.class_names is not None else [str(label) for label in labels]
            if len(ticks) != len(labels):
                raise ValueError("class_names must match labels.")
            names = [self.title]
        if self.normalize is not None:
            axis = {"true": 2, "pred": 1, "all": (1, 2)}[self.normalize]
            denominator = matrices.sum(axis=axis, keepdims=True)
            matrices = np.divide(matrices, denominator, out=np.zeros_like(matrices), where=denominator != 0)
        self.matrices_ = matrices
        plt = self._pyplot()
        count, cols = len(matrices), min(len(matrices), 3)
        fig, axes = plt.subplots(int(np.ceil(count/cols)), cols, squeeze=False,
                                 figsize=(5*cols, 4.5*int(np.ceil(count/cols))), layout="constrained")
        try:
            for i, ax in enumerate(axes.flat):
                if i >= count:
                    ax.set_visible(False)
                    continue
                matrix = matrices[i]
                picture = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if self.normalize else None)
                fig.colorbar(picture, ax=ax, shrink=.8)
                for (row, col), value in np.ndenumerate(matrix):
                    ax.text(col, row, f"{value:.2f}" if self.normalize else f"{value:.0f}",
                            ha="center", va="center", color="white" if value > matrix.max()/2 else "black")
                ax.set(xticks=range(len(ticks)), yticks=range(len(ticks)), xticklabels=ticks,
                       yticklabels=ticks, xlabel="Predicted label", ylabel="True label", title=names[i])
                ax.tick_params(axis="x", labelrotation=30)
            if self.multilabel:
                fig.suptitle(self.title)
            self._save(fig, plt)
        finally:
            plt.close(fig)
        return X
