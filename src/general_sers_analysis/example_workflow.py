"""GSAD0004 example configured for the reusable Workflow executor."""
import csv
from dataclasses import asdict
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from .core import Edge, Spectra
from .workflow import Workflow, NodeSpec
from .core.data import load_dataset
from .tools.preprocessing import SpectralCut, Interpolate, AirPLS
from .tools.aggregation import MeanAggregator
from .tools.regression import PLSRRegressor
from .tools.visualization import GroupSpectrumPlot, PredictionPlot
from .workflow_graph import draw_workflow_graph
from .util import visualization_run_dir


def run_example(database_dir, output_dir, visualization_dir, *, lower=600, upper=1800,
                step=2, n_components=2, test_size=0.25, seed=0, lam=1e5, show=False):
    """Train multi-output PLSR on GSAD0004, holding out complete groups."""
    if not all(np.isfinite(v) for v in (lower, upper, step, lam)) or lower >= upper or step <= 0:
        raise ValueError("Supply a finite lower < upper and positive grid step.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must lie between 0 and 1.")
    dataset = load_dataset(database_dir, 'GSAD0004')
    metadata = dataset.metadata
    molecules = metadata['molecules'].split('_')
    if metadata['task'] != 'Multi-Component Quantification':
        raise ValueError('GSAD0004 must describe a multi-component task.')
    raw = list(dataset.spectra)
    y = dataset.y
    groups = dataset.group_ids.tolist()
    files = list(dataset.sample_ids)
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
                       .split(np.zeros(len(raw)), y, dataset.split_blocks))
    # Fixed, declared target grid; no test data or labels select preprocessing parameters.
    grid = np.arange(lower + step, upper, step)
    if len(grid) < 2:
        raise ValueError("The requested alignment grid has fewer than two points.")
    if not isinstance(n_components, int) or not 1 <= n_components <= min(len(train)-1, len(grid)):
        raise ValueError("n_components exceeds the training matrix limit.")
    cut = SpectralCut(lower, upper)
    output = Path(output_dir)
    gallery = visualization_run_dir(output, visualization_dir, Path(__file__).resolve().parents[2] / "outputs")
    figure = gallery / "GSAD0004_example_baseline_corrected.png"
    prediction_figure = gallery / "GSAD0004_example_pred_vs_true.png"
    nodes = {"cut": cut, "align": Interpolate(grid), "baseline": AirPLS(lam=lam),
             "visualize": GroupSpectrumPlot(figure, show=show,
                 title="GSAD0004: held-out group means after cut / alignment / airPLS",
                 ylabel="Baseline-corrected intensity (source units)"),
             "prediction_plot": PredictionPlot(prediction_figure, molecule_names=molecules,
                 unit=metadata["quant_unit"], show=show, title="GSAD0004: held-out group means"),
             "plsr": PLSRRegressor(parameters={"n_components": n_components, "scale": True})}
    specs = {key: NodeSpec(tool) for key, tool in nodes.items()}
    specs["cut"].map_samples = True
    specs["align"].map_samples = True
    specs["align"].stack_output = True
    specs["plsr"].method = "predict"
    specs["visualize"].stages = ("evaluate",)
    specs["prediction_plot"].stages = ("evaluate",)
    edges = [Edge("input", "cut", source_port="X", data_kind="spectra")]
    edges += [Edge(source, target, data_kind="spectra") for source,target in
              [("cut", "align"), ("align", "baseline"),
               ("baseline", "plsr"), ("baseline", "visualize")]]
    edges += [Edge("input", "plsr", source_port="y", target_port="fit:y", data_kind="target_values"),
              Edge("input", "visualize", source_port="groups", target_port="groups", data_kind="groups"),
              Edge("plsr", "prediction_plot", data_kind="target_values"),
              Edge("input", "prediction_plot", source_port="y", target_port="y_true", data_kind="target_values"),
              Edge("input", "prediction_plot", source_port="groups", target_port="groups", data_kind="groups")]
    workflow = Workflow(specs, edges, output_node="plsr")
    workflow.fit([raw[i] for i in train], y[train])
    test_groups = [groups[i] for i in test]
    prediction = workflow.predict([raw[i] for i in test])
    prediction_trace = list(workflow.last_trace_)
    workflow.evaluate([raw[i] for i in test], y[test], groups=test_groups)
    nodes = workflow.nodes_
    mean_prediction = MeanAggregator().transform(prediction, groups=test_groups)
    mean_target = MeanAggregator().transform(y[test], groups=test_groups)
    metrics = {}
    for column, molecule in enumerate(molecules):
        truth = mean_target.values[:, column]
        pred = mean_prediction.values[:, column]
        r2 = float(r2_score(truth, pred)) if len(truth) > 1 and np.ptp(truth) > 0 else None
        metrics[molecule] = {"RMSE": float(np.sqrt(mean_squared_error(truth, pred))),
                             "MAE": float(mean_absolute_error(truth, pred)), "R2": r2}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    graph_paths = draw_workflow_graph(nodes, edges, output,
        note="Workflow executor: fit training groups; predict without labels; evaluate plots held-out groups.\n"
             "fit:y edges supply labels only during training. Plot nodes run only during evaluation.")
    summary = {"data_id": "GSAD0004", "unit": metadata["quant_unit"], "molecules": molecules,
               "train_groups": sorted({groups[i] for i in train}),
               "test_groups": sorted(set(test_groups)), "train_spectra": len(train), "test_spectra": len(test),
               "metrics_on_group_means": metrics, "seed": seed, "test_size": test_size,
               "cut": [lower, upper], "grid": [float(grid[0]), float(grid[-1]), step],
               "executor": "Workflow", "spectral_plot_subset": "held-out groups only",
               "nodes": {key: {"tool": type(node).__name__, "parameters": node.get_params(deep=False)}
                         for key, node in nodes.items() if key not in {"align", "visualize"}},
               "edges": [asdict(edge) for edge in edges], "n_components": n_components,
               "figure": str(figure.resolve()),
               "workflow_graph": graph_paths,
               "visualization_dir": str(gallery),
               "figures": {"group_means": str(figure.resolve()),
                           "pred_vs_true": str(prediction_figure.resolve())},
               "limitations": "Fixed demonstration split; only 11 condition groups. Group holdout does not "
               "establish batch/matrix transfer or independent preparations. Targets and ppm follow source "
               "labels. Negative PLSR predictions are retained; no clipping or hyperparameter tuning."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False, default=str), encoding="utf-8")
    with (output / "test_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["spectrum", "group"] + [f"true_{name}" for name in molecules] +
                        [f"pred_{name}" for name in molecules])
        for index, pred in zip(test, prediction):
            writer.writerow([files[index], groups[index], *y[index], *pred])
    (output / "workflow_spec.json").write_text(json.dumps(workflow.to_dict(), indent=2), encoding="utf-8")
    (output / "execution_trace.json").write_text(json.dumps({"fit": workflow.fit_trace_, "predict": prediction_trace,
                 "evaluate": workflow.last_trace_}, indent=2), encoding="utf-8")
    joblib.dump({"workflow": workflow, "nodes": nodes, "edges": edges, "molecules": molecules,
                 "unit": metadata["quant_unit"]}, output / "workflow.joblib")
    return summary
