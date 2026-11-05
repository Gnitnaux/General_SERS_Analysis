# General SERS Analysis

Composable, non-deep-learning tools for Raman and SERS analysis.

Research context, agreed dataset conventions and development scope are recorded in [agent.md](agent.md) (Chinese).

This version implements **nodes, edge descriptions, individual tools, and a main.py entry point with dataset visualization and a fixed PLSR example**. It implements a reusable unconditional DAG executor but does not implement AutoML search, general model-training commands, data acquisition/import conversion, cross-validation management or data-volume planning. The workflow-design and data-planning directories are reserved for later review.

## Layout

```text
General_SERS_Analysis/
├── data/
│   ├── database/            # content.csv, data/<Data_id>/<group>/*.csv, molecules/
│   └── test_data/           # Empty placeholder for your small local test dataset
├── logs/                   # Training/run logs
├── outputs/                # Trained models, numerical results and records
├── visualizations/         # Plots and other generated visual results
├── src/general_sers_analysis/
│   ├── core/               # Node, Edge, Condition, Spectra, UnmixingResult
│   ├── tools/              # Individual preprocessing/ML/unmixing/aggregation tools
│   ├── workflow_design/    # Reserved; no implementation yet
│   └── data_planning/      # Reserved; no implementation yet
├── configs/                # Reserved experiment configurations
├── scripts/                # Reserved entry scripts
├── tests/                  # Synthetic, in-memory behavior tests
├── docs/tool_library.md    # Chinese tool/API guide
├── main.py                 # Shared entry point; currently visualization only
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

The `.gitkeep` placeholders keep empty directories in Git. Files under data/ are intended to be tracked in Git. Logs, generated outputs, plots and environments are ignored. No existing database has been copied or modified. After moving your database, its index should be `data/database/content.csv`, with `data/database/data/` and `data/database/molecules/` beside it, not nested beneath another database-name folder.

## Installation

Python 3.10 or newer; a virtual environment is recommended. From this repository on Windows:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
```

`requirements.txt` contains runtime dependencies; the `dev` extra adds pytest. No GPU/deep-learning packages are required. Version ranges describe compatibility expectations; they are not a reproducible lockfile for all platforms. Record exact installed versions with each future experiment.

## Visualize a dataset

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe main.py visualize --data-id GSAD0004
```

The database root defaults to `data/database/`, independently of the shell working directory. The command finds the exact Data_id in `content.csv`, reads the CSV files directly inside each group folder in `data/<Data_id>/`, and plots **one arithmetic mean spectrum per group**. Replicates are averaged within their existing group only; identical labels in different group folders are not merged. No smoothing, baseline correction, normalization or model fitting is performed. A group containing one spectrum is plotted as-is.

Each CSV must have the columns `RamanShift,intensity`, finite numeric values, and at least two distinct spectral points. Decreasing axes are reversed together with their intensities. Axes must otherwise match exactly within a group; mismatches fail with filenames rather than silently interpolating. Groups may have different axes because each mean curve is plotted on its own grid. For example, the current GSAD0005 groups mix disjoint spectral windows and are rejected by this strict averaging task; resolve the acquisition/grouping semantics before combining them. Index count discrepancies are reported as warnings; the plot uses the files actually present. Empty groups and malformed spectra produce errors.

The default output is `visualizations/<Data_id>_group_means.png`; running the same ID again replaces that figure. Intensities retain their source units. A legend lists each group and its repeat count for up to 20 groups. For larger datasets all group means are drawn without a large legend; select groups to inspect labeled curves:

```powershell
.\.venv\Scripts\python.exe main.py visualize --data-id GSAD0004 --groups Quan_0_0_0_000001 Quan_1_1_0_000005 --show
```

Use `--database-dir "PATH"` to select another database root, `--output-dir "PATH"` to change the figure destination, and `--show` to additionally open a plot window on a desktop with a working graphical backend. Without `--show` the command saves the PNG without requiring a display. `python main.py --help` and `python main.py visualize --help` list the available options. No other task commands are implemented yet.

## Data and API contract

- Numeric arrays use **rows = spectra/samples**, **columns = spectral points/features**.
- `Spectra(intensity, axis, sample_ids)` binds an intensity matrix to a finite, strictly increasing RamanShift axis in cm^-1. It copies and protects the input arrays against accidental mutation. One shared axis is required per batch; alignment of different grids is explicit.
- `fit` learns training-dependent state and returns the node. `transform` converts data. `predict` returns target predictions. Fixed-rule transformations can run without `fit`.
- Every concrete tool inherits `Node` (sometimes through a category base). `parameters={...}` configures sklearn/XGBoost-backed estimators; preprocessing tools expose named parameters.
- `Edge` describes source/target IDs, named ports, data kind and optional comparisons. It does not run a graph, check global cycles or validate the existence of node IDs. Conditions use structured comparisons, never `eval`.
- Feature transforms return a two-dimensional feature matrix. Predictors preserve row order. Future orchestration must carry sample identity/units/component order alongside those arrays.
- Classifier probability columns follow `classes_`. Multilabel presence matrices require `multilabel=True`; each target needs positive and negative training examples. PLS-DA currently supports single-label classes, and exposes uncalibrated `predict_scores`, not probabilities.
- Regressors return `(samples, targets)` even for one target. Outputs inherit the units/order of training labels; missing targets must be handled explicitly before fitting.

```python
import numpy as np
from general_sers_analysis import Spectra, Edge, Condition
from general_sers_analysis.tools.preprocessing import SGFilter, Standardize
from general_sers_analysis.tools.regression import PLSRRegressor

# Arrays below must come from YOUR explicitly separated training and test data.
# X_train, X_test: rows of spectra; axis: RamanShift; y_train: known targets.
# train = Spectra(X_train, axis)
# test = Spectra(X_test, axis)
# smoother = SGFilter(window_length=11, polyorder=3)
# scaler = Standardize(scope="feature")
# train_features = scaler.fit_transform(smoother.transform(train))
# test_features = scaler.transform(smoother.transform(test))
# model = PLSRRegressor(parameters={"n_components": 2}).fit(train_features, y_train)
# prediction = model.predict(test_features)

edge = Edge("presence_classifier", "quantifier", source_port="presence_probability",
            condition=Condition("presence_probability", "ge", 0.5))
```

The example is a manual call sequence, not an automatic workflow implementation. Learned preprocessing must be fitted inside each training fold. This version does not automatically implement out-of-fold predictions or protect against upstream-prediction leakage: future executors must implement grouped splitting and cross-fitting before training predictive cascades.

## Available tools

| Category/module | Concrete tools |
| --- | --- |
| `tools.preprocessing` | `SGFilter`, `AirPLS`, `Interpolate`, `SpectralCut`, `PeakNormalize`, `MinMaxNormalize`, `Standardize` |
| `tools.classification` | `RFClassifier`, `PLSDAClassifier`, `SVMClassifier`, `XGBoostClassifier`, `LogisticClassifier` |
| `tools.clustering` | `KMeansClusterer`, `TSNEEmbedding` |
| `tools.regression` | `PLSRRegressor`, `RFRegressor`, `SVRRegressor`, `XGBoostRegressor`, `DigitalSERSRegressor`, `LinearRegressor` |
| `tools.unmixing` | `CLSUnmixer`, `NNLSUnmixer` |
| `tools.aggregation` | `PCAFeatures`, `PeakExtractor`, `PeakDetector`, `FactorAnalysisFeatures`, `MeanAggregator`, `MedianAggregator`, `MaxAggregator`, `MinAggregator`, `SpectralDifference` |

## Important interpretations

- **SG** assumes equal spectral spacing. Interpolation refuses extrapolation. Spectral cut, peak-window operations and subtraction require explicit axes.
- **Peak normalization** divides by the maximum absolute intensity, optionally within a specified band; a zero spectrum stays zero. This definition is explicit because peak normalization conventions vary.
- **Min-max** defaults to independent per-spectrum scaling. **Standardization** defaults to per-feature training-set mean/standard deviation. Both also support the other scope explicitly. Per-spectrum standardization corresponds to row-wise centering/scaling, not training-population standardization.
- **t-SNE** is for training-set visualization, not class prediction or transformation of new samples. `fit_transform` returns the fitted embedding; `transform` and `predict` deliberately reject these unsupported operations.
- **CLS/NNLS** fit against known reference component spectra. They return coefficients, reference spectra, reconstruction and residual norms. They do not discover unknown pure spectra, and coefficients are not automatically molar concentrations. References must have matching axes and independent columns. NNLS constrains coefficients to be nonnegative; CLS does not.
- **Aggregation** without `groups` reduces features within each row to one value. With `groups`, it pools spectra within each sample and returns group identity. The two operations are deliberately distinct. Average probability/target vectors only when component order and physical meaning are compatible.
- **Digital SERS** here is an explicitly chosen occupancy-calibration variant: an event is positive if its maximum intensity (optionally in a band) exceeds a supplied threshold. Within each group, convert the positive fraction to Poisson occupancy by taking the negative natural logarithm of one minus that fraction. Fit a nonnegative, zero-intercept concentration scale from standards. This assumes independent comparable detection events and a stable assay/detection efficiency; ordinary correlated map pixels do not establish those assumptions. All-positive groups are saturated and rejected. Prediction returns one concentration per group; `last_group_ids_`, `last_positive_counts_` and `last_event_counts_` identify it. Threshold selection, detection efficiency validation, uncertainty estimation and digital-assay validation are not implemented. This is not claimed as a complete reproduction of a published dCERS protocol.

## Sources and dependencies

- airPLS is delegated to [pybaselines](https://pybaselines.readthedocs.io/en/latest/generated/api/pybaselines.Baseline.airpls.html), following Zhang, Chen and Liang, *Analyst* 135, 1138–1146 (2010), [doi:10.1039/B922045C](https://doi.org/10.1039/B922045C). The authors' LGPL Python implementation is not copied into this repository.
- Estimators, PCA, factor analysis and t-SNE use [scikit-learn](https://scikit-learn.org/stable/); boosted trees use [XGBoost](https://xgboost.readthedocs.io/).
- SG, interpolation, peak detection and NNLS use [SciPy](https://docs.scipy.org/doc/scipy/reference/). NNLS follows the documented constrained least-squares solver; linear unmixing is a known-reference linear-mixture assumption, not a general nonlinear SERS interaction model.
- The digital counting concept is motivated by Bi et al., *Nature* 628, 771–775 (2024), [doi:10.1038/s41586-024-07218-1](https://doi.org/10.1038/s41586-024-07218-1). The calibration convention above is this implementation's explicit choice.

All tests use generated numerical data and verify software behavior, not analytical validation on real catecholamine spectra. No clinical-performance or chemical-generalization claim is implied.

## Fixed GSAD0004 workflow example

```powershell
.\.venv\Scripts\python.exe main.py example
# Optional desktop display and explicit preprocessing/model settings:
.\.venv\Scripts\python.exe main.py example --lower 600 --upper 1800 --step 2 --lam 100000 --n-components 2 --show
```

The example now declares NodeSpec and Edge objects for the reusable Workflow executor. SpectralCut and Interpolate map over raw spectra; aligned spectra are stacked before AirPLS. The corrected spectra feed PLSR and a separate evaluation-only plot. PLSR feeds an evaluation-only PredictionPlot. There is no hand-written PLSR fit/predict logic.

Default alignment uses a fixed 602–1798 cm^-1 grid with a 2 cm^-1 step inside the cut interval. This is coordinate resampling, not peak-shift calibration. No extrapolation is permitted. Whole groups are held out (8 training, 3 test at defaults); output errors retain source ppm units. The default first two target R2 scores remain negative. Only 11 condition groups exist and this split does not establish preparation independence or matrix transfer.

Outputs include workflow_spec.json (declaration), execution_trace.json (fit/predict/evaluate operations), workflow.joblib (fitted executor plus tools), predictions/summary and graph files. Business plots remain in visualizations/GSAD0004_example/. The mean spectrum plot now uses held-out groups only. Only load trusted joblib files.

## Visualization tools

All three tools live in `general_sers_analysis.tools.visualization`, inherit Node, and implement `transform`. They save a PNG, optionally display it with `show=True`, then return the original input unchanged. They do not fit models. `last_output_path_` records the saved image path. Sample and target order must already match; plots do not align labels by identifier.

| Node | Input and options | Figure |
| --- | --- | --- |
| GroupSpectrumPlot | Spectra; optional `groups`; optional title/ylabel | Arithmetic mean per group on the common spectral axis. Without groups, uses sample_ids (or treats each row as a separate group). |
| PredictionPlot | Predicted values X; required `y_true`; optional groups, molecule_names, unit | One subplot per target, scatter of predicted vs true, y=x line, RMSE and R2; grouping averages truth and predictions first. Constant-target R2 is undefined. |
| ConfusionMatrixPlot | Predicted labels X; required `y_true`; optional labels, class_names, normalize | True labels on rows and predicted labels on columns. Counts by default; normalize=true/pred/all divides by row/column/total. Does not threshold probabilities. With multilabel=True, binary target matrices produce one 2x2 subplot per component. |

```python
from general_sers_analysis.tools.visualization import PredictionPlot, ConfusionMatrixPlot

# y_pred and y_true must have the same samples x targets shape and target order.
plot = PredictionPlot("outputs/run/figures/pred_vs_true.png",
                      molecule_names=["A", "B"], unit="ppm")
# plot.transform(y_pred, y_true=y_true)

confusion = ConfusionMatrixPlot("outputs/run/figures/confusion.png",
                                labels=["A", "B"], normalize="true")
# confusion.transform(predicted_labels, y_true=true_labels)
```

The example graph includes an Edge from PLSR to PredictionPlot (`data_kind="target_values"`). Its result plot uses the same held-out group means as the reported metrics. It saves average spectra and pred-vs-true subplots only in `visualizations/GSAD0004_example/`. The actual Node/Edge workflow diagram is saved under `outputs/GSAD0004_example/` as workflow_graph.png and workflow_graph.svg, with workflow_graph.json describing the same nodes, parameters and edges. `summary.json` records both paths in `figures`. GSAD0004 is a regression example, so no classification confusion matrix is fabricated for this run.


## Output directory convention

```text
outputs/
  GSAD0004_example/
    workflow_graph.png
    workflow_graph.svg
    workflow_graph.json
    summary.json
    test_predictions.csv
    workflow.joblib
visualizations/
  GSAD0004_example/
    GSAD0004_example_baseline_corrected.png
    GSAD0004_example_pred_vs_true.png
```

The graph is rendered from the actual Node/Edge objects, with tool names, selected parameters and edge data kinds; it is not a manually maintained diagram. The renderer rejects cycles and unknown endpoints. It documents the declared tool graph; group labels and held-out true targets are plotting side inputs noted in the diagram.

`--visualization-dir` specifies the visualization **root**. A run at `outputs/<relative run path>` writes its plots to `visualizations/<relative run path>` (including nested folders). If --output-dir is outside the project outputs root, its final directory name is used under the visualization root. Workflow graphs stay beside numerical/model outputs; scientific plots are not copied there. summary.json records both workflow_graph and figures. The standalone visualize command retains its explicitly selected output directory because it does not create a model-run outputs directory.


## Reusable workflow executor

Workflow and NodeSpec are exported from general_sers_analysis.workflow. The executor is independent of the example dataset, algorithm selection and dataset splitting. The caller is responsible for independent training/validation data.

```python
from general_sers_analysis import Workflow, NodeSpec, Edge
from general_sers_analysis.tools.preprocessing import Standardize
from general_sers_analysis.tools.regression import PLSRRegressor

workflow = Workflow(
    nodes={
        "scale": NodeSpec(Standardize()),
        "regression": NodeSpec(PLSRRegressor({"n_components": 2}), method="predict"),
    },
    edges=[
        Edge("input", "scale", source_port="X"),
        Edge("scale", "regression"),
        Edge("input", "regression", source_port="y", target_port="fit:y"),
    ],
    output_node="regression",
)
# workflow.fit(X_train, y_train)
# predictions = workflow.predict(X_test)  # no true labels required
```

1. **fit:** clone configured tools, fit requires_fit nodes using training inputs, then invoke their declared output methods. Stateless tools only transform. Refitting creates fresh tools.
2. **predict:** never fit; use saved state and return output_node's value.
3. **evaluate:** never fit; accept true targets and optional groups, run evaluation plots, and return a dictionary of node outputs.

NodeSpec.method explicitly selects transform/predict/predict_proba/predict_scores. No tool name is treated specially. Stages default to fit/predict/evaluate; plots use stages=("evaluate",). Learned nodes participate in all three stages.

The reserved external node input exposes X, y, groups and caller-supplied keyword ports. Tools currently expose one output port. target_port=input binds X; named ports bind method keyword arguments. fit:y supplies training labels only to fit; fit:X can override the fit input for reference-based tools. Metadata are not broadcast implicitly. map_samples=True maps a stateless tool over a list of Spectra; stack_output=True stacks identical-axis results and preserves sample IDs.

Execution order comes from edges, including branches and named-input merges. Validation rejects missing/duplicate input ports, unknown endpoints, cycles, incompatible method arguments and incompatible stage dependencies. data_kind checks coarse data shapes/types, not units, molecule order, sample identity or experimental independence. Labels and groups must be correctly aligned by the caller.

Current limits: conditional/sample-mask routing, loops, automatic multi-output-port extraction, automatic metadata reindexing after group aggregation, grouped cross-validation and out-of-fold training are not implemented. Learning downstream of prediction nodes is rejected: training-internal predictions must not be used to train a second learner without out-of-fold handling. The executor does not split data or search workflows.

workflow.to_dict() exports a JSON-compatible declaration. Workflow.from_dict(spec, registry) rebuilds it using a caller-supplied allowlist of tool names/classes, without importing arbitrary code. Reconstructed workflows must be fitted; JSON does not contain learned parameters. joblib preserves a trusted fitted executor. fit_trace_ and last_trace_ describe stages, fit calls, output methods and timing. Upper-level algorithms can generate declarations, validate/build them and evaluate them using a separately controlled validation scheme.
