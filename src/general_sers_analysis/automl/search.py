"""Budgeted classical AutoML over real Node/Edge workflow declarations.

Search uses blocked training-only CV. A single untouched holdout is evaluated
after selection, along with baselines fixed before search. No test-score feedback.
"""
from dataclasses import dataclass, asdict
import csv
import importlib.metadata
import itertools
import json
from pathlib import Path
import time
import warnings
import joblib
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from threadpoolctl import threadpool_limits

from ..core import Spectra, Edge
from ..workflow import Workflow, NodeSpec
from ..core.data import load_dataset, read_dataset_index
from .representation import prepare_group_means
from ..tools.preprocessing import SGFilter, AirPLS, PeakNormalize, Standardize
from ..tools.aggregation import PCAFeatures
from ..tools.regression import PLSRRegressor, RFRegressor, SVRRegressor, XGBoostRegressor
from ..tools.classification import SVMClassifier, LogisticClassifier, RFClassifier
from ..tools.model_selection import (RidgeRegressor, KernelRidgeRegressor, TargetScaledRegressor,
                                    IndependentPresence, ConstantPredictor)
from ..tools.visualization import PredictionPlot, ConfusionMatrixPlot
from ..workflow_graph import draw_workflow_graph
from ..util import write_json as _json, subset_spectra as subset, visualization_run_dir


@dataclass(frozen=True)
class Candidate:
    preprocessing: str = 'raw'
    normalization: str = 'none'
    pca: int = 0
    model: str = 'pls'
    parameter: float = 2


def build_workflow(candidate, *, classification=False, seed=0, jobs=1, features_only=False):
    """Generate an executable, serializable DAG from a bounded search declaration."""
    c = candidate
    nodes, edges = {}, []
    previous = 'input'

    def append(key, tool, method='transform'):
        nonlocal previous
        nodes[key] = NodeSpec(tool, method=method)
        edges.append(Edge(previous, key, source_port='X' if previous == 'input' else 'output'))
        previous = key

    if c.preprocessing in ('sg', 'derivative', 'airpls_sg'):
        if c.preprocessing == 'airpls_sg':
            append('baseline', AirPLS(lam=1e5))
        append('sg', SGFilter(window_length=11, polyorder=2, deriv=int(c.preprocessing == 'derivative')))
    elif c.preprocessing == 'airpls':
        append('baseline', AirPLS(lam=1e5))
    elif c.preprocessing != 'raw':
        raise ValueError('Unknown preprocessing.')
    if c.normalization == 'snv':
        append('row_scale', Standardize(scope='spectrum'))
    elif c.normalization == 'peak':
        append('peak', PeakNormalize())
    elif c.normalization != 'none':
        raise ValueError('Unknown normalization.')
    # Training-fitted feature scaling is always inside each CV fold.
    append('feature_scale', Standardize())
    if c.pca:
        append('pca', PCAFeatures({'n_components': c.pca, 'random_state': seed, 'svd_solver': 'full'}))
    if not features_only:
        p = c.parameter
        if c.model == 'constant':
            model = ConstantPredictor(classification)
        elif classification:
            if c.model == 'svm':
                base = SVMClassifier({'C': p, 'gamma': 'scale', 'probability': False, 'random_state': seed})
            elif c.model == 'logistic':
                base = LogisticClassifier({'C': p, 'max_iter': 3000, 'class_weight': 'balanced', 'random_state': seed})
            elif c.model == 'rf':
                base = RFClassifier({'n_estimators': 160, 'min_samples_leaf': int(p), 'n_jobs': jobs,
                                     'class_weight': 'balanced', 'random_state': seed})
            else:
                raise ValueError('Unknown classification model.')
            model = IndependentPresence(base)
        else:
            if c.model == 'pls':
                base = PLSRRegressor({'n_components': int(p), 'scale': False, 'max_iter': 1000})
            elif c.model == 'ridge':
                base = RidgeRegressor({'alpha': p})
            elif c.model == 'kernel_ridge':
                base = KernelRidgeRegressor({'alpha': p, 'kernel': 'rbf'})
            elif c.model == 'svr':
                base = SVRRegressor({'C': p, 'epsilon': .03, 'gamma': 'scale'})
            elif c.model == 'rf':
                base = RFRegressor({'n_estimators': 160, 'min_samples_leaf': int(p), 'n_jobs': jobs, 'random_state': seed})
            elif c.model == 'xgb':
                base = XGBoostRegressor({'n_estimators': 160, 'max_depth': int(p), 'learning_rate': .04,
                                        'n_jobs': jobs, 'random_state': seed, 'verbosity': 0})
            else:
                raise ValueError('Unknown regression model.')
            model = TargetScaledRegressor(base)
        append('model', model, 'predict')
        edges.append(Edge('input', 'model', source_port='y', target_port='fit:y'))
    return Workflow(nodes, edges, previous)


def candidate_space(classification, n_trials, seed):
    """Seeded random search with explicit simple-model coverage before exploration."""
    if classification:
        models = [('svm', p) for p in (.1, 1, 10, 100)] + [('logistic', p) for p in (.01, .1, 1, 10)] + [('rf', p) for p in (1, 3)]
        initial = [Candidate(model=m, parameter=p) for m, p in models]
    else:
        models = ([('pls', p) for p in (1, 2, 3, 5, 8, 12)] + [('ridge', p) for p in (.01, 1, 100, 10000)] +
                  [('svr', p) for p in (.1, 1, 10, 100)] + [('kernel_ridge', p) for p in (.001, .1, 1)] +
                  [('rf', p) for p in (1, 3)] + [('xgb', p) for p in (2, 4)])
        initial = [Candidate(model=m, parameter=p) for m, p in models[:10]]
        initial += [Candidate(preprocessing='airpls', model='pls', parameter=p) for p in (1, 2, 3, 5)]
        initial += [Candidate(preprocessing='derivative', model='pls', parameter=p) for p in (2, 5)]
    space = [Candidate(pre, norm, pca, model, p) for pre, norm, pca, (model, p) in
             itertools.product(('raw', 'sg', 'derivative', 'airpls', 'airpls_sg'),
                               ('none', 'snv', 'peak'), (0, 3, 8), models)]
    rng = np.random.default_rng(seed)
    remaining = [space[i] for i in rng.permutation(len(space)) if space[i] not in initial]
    return (initial + remaining)[:n_trials]


def metrics(y, prediction, classification, molecules):
    if classification:
        return {'macro_f1': _macro_f1(y, prediction),
                'subset_accuracy': float(accuracy_score(y, prediction)),
                'per_target': {name: {'f1': float(f1_score(y[:, i], prediction[:, i], zero_division=0)),
                                      'accuracy': float(accuracy_score(y[:, i], prediction[:, i])),
                                      'positive_groups': int(y[:, i].sum()),
                                      'predicted_positive_groups': int(prediction[:, i].sum())}
                               for i, name in enumerate(molecules)}}
    return {'per_target': {name: {'RMSE': float(np.sqrt(mean_squared_error(y[:, i], prediction[:, i]))),
                                  'MAE': float(mean_absolute_error(y[:, i], prediction[:, i])),
                                  'R2': float(r2_score(y[:, i], prediction[:, i])) if len(y) > 1 and np.ptp(y[:, i]) > 0 else None}
                          for i, name in enumerate(molecules)}}


def cv_score(y, pred, train_y, classification):
    if classification:
        return _macro_f1(y, pred)
    scale = train_y.std(axis=0)
    scale = np.where(scale > 0, scale, 1.)
    return -float(np.mean(np.sqrt(np.mean(((y - pred) / scale) ** 2, axis=0))))


def _macro_f1(y, pred):
    # Explicit columns preserve the positive-presence convention for one target too.
    return float(np.mean([f1_score(a, b, zero_division=0) for a, b in zip(y.T, pred.T)]))


def split_holdout(groups, test_size, seed, source_types=None):
    """Preserve an explicit author test partition when supplied; otherwise block shuffle."""
    groups = np.asarray(groups)
    if not 0 < test_size < 1 or len(np.unique(groups)) < 4:
        raise ValueError('Need >=4 split blocks and 0 < test_size < 1.')
    if source_types is not None and 'test' in source_types:
        test_blocks = {g for g, t in zip(groups, source_types) if t == 'test'}
        test = np.flatnonzero(np.isin(groups, list(test_blocks)))
        train = np.flatnonzero(~np.isin(groups, list(test_blocks)))
        kind = 'author test markers, expanded to complete connected split blocks'
    else:
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(groups, groups=groups))
        kind = 'fixed blocked random holdout'
    if len(np.unique(groups[train])) < 3 or not len(test):
        raise ValueError('Need >=3 development blocks and a nonempty holdout.')
    return train, test, kind


def run_automl(database_dir, data_id, output_dir, visualization_dir, *, n_trials=64,
               cv=3, seed=0, test_size=.25, time_budget=1800., jobs=1,
               lower=600., upper=1380., step=2., max_spectra_per_group=0):
    """Search once, refit on development data, evaluate the untouched holdout once."""
    if n_trials < 1 or cv < 2 or jobs < 1 or not np.isfinite(time_budget) or time_budget <= 0:
        raise ValueError('Positive trials/jobs/budget and cv>=2 required.')
    output = Path(output_dir)
    # Avoid quietly overwriting a prior experiment or resuming with a new test score.
    if output.exists() and any(p.name != '.gitkeep' for p in output.iterdir()):
        raise ValueError(f'Output directory is not empty: {output}. Choose a new run directory.')
    output.mkdir(parents=True, exist_ok=True)
    raw_dataset = load_dataset(database_dir, data_id,
                               max_spectra_per_group=max_spectra_per_group, seed=seed)
    dataset = prepare_group_means(raw_dataset, lower=lower, upper=upper, step=step)
    X, y, groups = dataset.X, dataset.y, dataset.groups
    classification = dataset.metadata['task'] == 'Classification'
    molecules = dataset.metadata['molecules'].split('_')
    train, test, split_kind = split_holdout(groups, test_size, seed, dataset.audit['source_types'])
    folds = list(GroupKFold(n_splits=min(cv, len(np.unique(groups[train]))), shuffle=True,
                           random_state=seed).split(train, groups=groups[train]))
    _json(output / 'data_audit.json', dataset.audit)
    _json(output / 'representation_spec.json', dataset.representation_workflow.to_dict())
    split = {'kind': split_kind, 'seed': seed, 'train': train.tolist(), 'test': test.tolist(),
             'row_ids': list(X.sample_ids), 'split_blocks': groups.tolist(),
             'cv': [{'train': train[a].tolist(), 'validation': train[b].tolist()} for a, b in folds]}
    _json(output / 'splits.json', split)
    _json(output / 'config.json', {'data_id': data_id, 'n_trials': n_trials, 'cv': cv, 'seed': seed,
                                  'test_size': test_size, 'time_budget': time_budget, 'jobs': jobs,
                                  'lower': lower, 'upper': upper, 'step': step,
                                  'max_spectra_per_group': max_spectra_per_group})
    history, cache = [], {}
    started = time.perf_counter()
    best_score, best = -np.inf, None
    print(f'{data_id}: {len(train)} development / {len(test)} holdout folder means; {len(folds)} CV folds', flush=True)
    # Limit BLAS as well as model workers: small spectral matrices suffer from oversubscription.
    with threadpool_limits(limits=jobs):
        for i, candidate in enumerate(candidate_space(classification, n_trials, seed)):
            if i and time.perf_counter() - started >= time_budget:
                break
            tick = time.perf_counter()
            record = {'trial': i, 'candidate': asdict(candidate)}
            scores, messages = [], []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    for fold, (a, b) in enumerate(folds):
                        a, b = train[a], train[b]
                        if candidate.model == 'pls' and candidate.parameter > min(len(a)-1, X.intensity.shape[1]):
                            raise ValueError('PLS components exceed fold training rank limit.')
                        if candidate.pca and candidate.pca > min(len(a), X.intensity.shape[1]):
                            raise ValueError('PCA components exceed fold matrix limit.')
                        key = (candidate.preprocessing, candidate.normalization, candidate.pca, fold)
                        if key not in cache:
                            feature_workflow = build_workflow(candidate, features_only=True, seed=seed, jobs=jobs)
                            feature_workflow.fit(subset(X, a))
                            cache[key] = (feature_workflow.predict(subset(X, a)), feature_workflow.predict(subset(X, b)))
                        xa, xb = cache[key]
                        # The same model Node that appears in the final DAG is fitted on cached fold-local features.
                        model = build_workflow(candidate, classification=classification, seed=seed, jobs=jobs).specs['model'].tool
                        model.fit(xa, y[a])
                        pred = np.asarray(model.predict(xb))
                        if not np.isfinite(pred).all():
                            raise ValueError('Nonfinite predictions.')
                        scores.append(cv_score(y[b], pred, y[a], classification))
                    messages = sorted(set(str(w.message) for w in caught))
                score = float(np.mean(scores))
                if not np.isfinite(score):
                    raise ValueError('Nonfinite CV score.')
                record.update(status='ok', score=score, fold_scores=scores, score_std=float(np.std(scores)), warnings=messages)
                if score > best_score:
                    best_score, best = score, candidate
            except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError) as error:
                record.update(status='failed', error=str(error), fold_scores=scores)
            record['seconds'] = time.perf_counter() - tick
            history.append(record)
            # A durable record is written after every candidate, including failures.
            _json(output / 'trials.json', history)
            print(f"  trial {i+1}/{n_trials}: {candidate.model}/{candidate.preprocessing}/{candidate.normalization} "
                  f"{record.get('score', record.get('error'))}", flush=True)
        if best is None:
            raise ValueError('All candidates failed. See trials.json.')
        workflow = build_workflow(best, classification=classification, seed=seed, jobs=jobs)
        workflow.fit(subset(X, train), y[train])
        prediction = workflow.predict(subset(X, test))
        baseline_candidates = {'constant': Candidate(model='constant')}
        if not classification:
            baseline_candidates['fixed_pls2'] = Candidate(preprocessing='airpls', model='pls', parameter=2)
        baseline_results = {}
        for name, c in baseline_candidates.items():
            baseline = build_workflow(c, classification=classification, seed=seed, jobs=jobs).fit(subset(X, train), y[train])
            baseline_results[name] = metrics(y[test], baseline.predict(subset(X, test)), classification, molecules)
            joblib.dump({'workflow': baseline, 'axis': X.axis, 'molecules': molecules,
                         'unit': dataset.metadata['quant_unit'], 'candidate': asdict(c),
                         'input_representation': dataset.audit['representation']}, output / f'baseline_{name}.joblib')
    gallery = visualization_run_dir(output, visualization_dir, Path(__file__).resolve().parents[3] / 'outputs')
    gallery.mkdir(parents=True, exist_ok=True)
    plot_path = gallery / ('confusion_matrix.png' if classification else 'pred_vs_true.png')
    if classification:
        ConfusionMatrixPlot(plot_path, multilabel=True, class_names=molecules,
                            title=f'{data_id}: untouched holdout groups').transform(prediction, y_true=y[test])
    else:
        PredictionPlot(plot_path, molecule_names=molecules, unit=dataset.metadata['quant_unit'],
                        title=f'{data_id}: untouched holdout groups').transform(prediction, y_true=y[test])
    graph = draw_workflow_graph(workflow.nodes_, workflow.edges, output,
                               note='Selected by development-only blocked CV. Inputs: fixed-grid source-folder means.\n'
                                    'Training target standardization is contained in model. Test groups never select candidates.')
    summary = {'data_id': data_id, 'task': dataset.metadata['task'], 'modality': dataset.metadata.get('modality'),
               'unit': dataset.metadata['quant_unit'], 'molecules': molecules, 'best_candidate': asdict(best),
               'selection_metric': 'macro F1 (higher is better)' if classification else 'negative mean RMSE / fold-training target SD (higher is better)',
               'best_cv_score': best_score, 'trials_completed': len(history),
               'trials_succeeded': sum(r['status'] == 'ok' for r in history),
               'holdout_metrics': metrics(y[test], prediction, classification, molecules),
               'baseline_holdout_metrics': baseline_results, 'development_groups': len(train), 'holdout_groups': len(test),
               'development_split_blocks': len(np.unique(groups[train])), 'holdout_split_blocks': len(np.unique(groups[test])),
               'split_kind': split_kind, 'seconds': time.perf_counter()-started, 'workflow_graph': graph,
               'figures': {'holdout': str(plot_path.resolve())}, 'input_sha256': dataset.audit['input_sha256'],
               'versions': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'xgboost', 'joblib', 'pybaselines')},
               'limitations': ['This is bounded classical pipeline search, not a learned general-purpose graph designer.',
                              'One fixed holdout; small panels have large uncertainty. No holdout-guided tuning or seed selection.',
                              'Folder means are evaluated; pixel-level detection and independent-preparation transfer are not established.',
                              'Source-label conflicts and spectra outside the declared window are excluded; see data_audit.json.'] + dataset.audit['source_limitations']}
    _json(output / 'summary.json', summary)
    _json(output / 'workflow_spec.json', workflow.to_dict())
    _json(output / 'execution_trace.json', {'fit': workflow.fit_trace_, 'predict': workflow.last_trace_})
    with (output / 'test_predictions.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['group', 'split_block'] + ['true_'+m for m in molecules] + ['pred_'+m for m in molecules])
        for row, pred in zip(test, prediction):
            writer.writerow([X.sample_ids[row], groups[row], *y[row], *pred])
    joblib.dump({'workflow': workflow, 'axis': X.axis, 'molecules': molecules, 'unit': dataset.metadata['quant_unit'],
                 'input_representation': dataset.audit['representation'], 'best_candidate': asdict(best)}, output / 'workflow.joblib')
    return summary


def run_benchmark(database_dir, output_dir, visualization_dir, **options):
    """Run each indexed task independently, preserving completed runs and failures."""
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new empty benchmark directory.')
    output.mkdir(parents=True, exist_ok=True)
    ids = [r['Data_id'] for r in read_dataset_index(database_dir)]
    records = []
    for data_id in ids:
        try:
            result = run_automl(database_dir, data_id, output / data_id, visualization_dir, **options)
            records.append({'data_id': data_id, 'status': 'ok', 'best_cv_score': result['best_cv_score'],
                            'holdout_metrics': result['holdout_metrics'], 'baseline_holdout_metrics': result['baseline_holdout_metrics'],
                            'summary': str((output / data_id / 'summary.json').resolve())})
        except (ValueError, OSError) as error:
            records.append({'data_id': data_id, 'status': 'failed', 'error': str(error)})
            print(f'{data_id}: failed: {error}', flush=True)
        _json(output / 'benchmark.json', records)
    return records
