# General SERS Analysis

Composable, non-deep-learning tools for Raman and SERS analysis.

Research context, agreed dataset conventions and development scope are recorded in [agent.md](agent.md) (Chinese).

This initial version implements **nodes, edge descriptions and individual tools**. It intentionally does not implement a graph executor, AutoML search, command-line task runner, database importer, cross-validation manager or data-volume planning algorithm. The workflow-design and data-planning directories are reserved for later review.

## Layout

```text
General_SERS_Analysis/
├── data/
│   ├── database/            # Empty placeholder: move your database contents here
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
