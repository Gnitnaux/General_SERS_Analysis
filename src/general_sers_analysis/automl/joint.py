"""Grouped, row-aware structure/parameter search on original CSV spectra."""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import csv
import itertools
import time
import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from ..core.data import load_dataset
from ..core.batch import DataBatch
from ..core import Edge
from ..workflow import Workflow, NodeSpec
from ..tools.preprocessing import Interpolate
from ..tools.aggregation import MeanAggregator, MedianAggregator, PCAFeatures, FeatureConcat
from ..util import write_json
from ..workflow_graph import draw_workflow_graph, visualization_run_dir
from ..tools.visualization import PredictionPlot, ConfusionMatrixPlot
from .search import Candidate, build_workflow, candidate_space, split_holdout, cv_score, metrics


@dataclass(frozen=True)
class JointCandidate:
    """Immutable structure/parameter configuration for bounded joint search.

    Attributes
    ----------
    aggregation: Pooling position: none, before_preprocessing, before_model or after_model.
    base: Candidate containing preprocessing, feature and predictor settings.
    feature_branch: Add a parallel two-component PCA branch before feature concatenation.
    lam: Positive airPLS smoothing penalty.
    lower: Inclusive lower spectral bound; must be smaller than upper.
    reducer: Pooling statistic: mean or median.
    reverse: Swap baseline and SG order when both nodes are present.
    step: Positive interpolation-grid spacing.
    upper: Inclusive upper spectral bound.
    window: SG window length in points for a joint candidate.
    """
    base: Candidate
    aggregation: str = 'before_model'
    reducer: str = 'mean'
    lower: float = 600
    upper: float = 1380
    step: float = 2
    lam: float = 1e5
    window: int = 11
    reverse: bool = False
    feature_branch: bool = False


def build_joint(c, classification=False, seed=0, jobs=1):
    if c.aggregation not in ('none', 'before_preprocessing', 'before_model', 'after_model'):
        raise ValueError('Invalid aggregation position.')
    if c.reducer not in ('mean', 'median') or c.step <= 0 or c.lower >= c.upper:
        raise ValueError('Invalid aggregation/grid parameters.')
    base = build_workflow(c.base, classification=classification, seed=seed, jobs=jobs)
    chain = [(k, v) for k, v in base.specs.items()]
    for key, spec in chain:
        if key == 'baseline': spec.tool.lam = c.lam
        if key == 'sg': spec.tool.window_length = c.window
    if c.reverse:
        # Change baseline/smoothing order, preserving normalization -> feature stages.
        positions = [i for i, (k, _) in enumerate(chain) if k in ('baseline', 'sg')]
        if len(positions) == 2:
            a, b = positions
            chain[a], chain[b] = chain[b], chain[a]
    if c.aggregation != 'none':
        reducer = MeanAggregator() if c.reducer == 'mean' else MedianAggregator()
        aggregate = ('aggregate', NodeSpec(reducer, row_action='aggregate'))
        position = {'before_preprocessing': 0, 'before_model': len(chain)-1, 'after_model': len(chain)}[c.aggregation]
        chain.insert(position, aggregate)
    nodes = {'align': NodeSpec(Interpolate(np.arange(c.lower, c.upper+c.step*1e-6, c.step)),
                               map_samples=True, stack_output=True)}
    edges = [Edge('input', 'align', source_port='X')]
    previous = 'align'
    for key, spec in chain:
        nodes[key] = spec
        edges.append(Edge(previous, key))
        if key == 'aggregate':
            edges.append(Edge('input', key, source_port='groups', target_port='groups'))
        if key == 'model':
            edges.append(Edge('input', key, source_port='y', target_port='fit:y'))
        previous = key
    if c.feature_branch:
        # Parallel PCA of standardized spectra, joined by exact row IDs before pooling/modeling.
        source = 'pca' if c.base.pca else 'feature_scale'
        consumer = next(e.target for e in edges if e.source == source and e.target_port == 'input')
        nodes['branch_pca'] = NodeSpec(PCAFeatures({'n_components': 2, 'svd_solver': 'full'}))
        nodes['feature_union'] = NodeSpec(FeatureConcat())
        edges = [e for e in edges if not (e.source == source and e.target == consumer and e.target_port == 'input')]
        edges += [Edge('feature_scale', 'branch_pca'), Edge(source,'feature_union'),
                  Edge('branch_pca','feature_union',target_port='other'),Edge('feature_union',consumer)]
    return Workflow(nodes, edges, previous)


def group_result(batch, classification=False):
    """One fixed group-level scoring unit for every candidate structure."""
    result = batch.aggregate(MeanAggregator().transform(batch.X, groups=batch.groups))
    values = result.X.intensity if hasattr(result.X, 'intensity') else result.X
    if classification:
        values = (values >= .5).astype(int)
    return result, np.asarray(values)


def proposals(n, lower, upper, step, classification, seed):
    bases = candidate_space(classification, 450 if classification else 945, seed)
    # Deterministic initial coverage makes all aggregation positions observable.
    first = [JointCandidate(bases[0], a, lower=lower, upper=upper, step=step)
             for a in ('none', 'before_preprocessing', 'before_model', 'after_model')]
    rng = np.random.default_rng(seed)
    windows = [(lower, upper), (lower+(upper-lower)*.15, upper), (lower, upper-(upper-lower)*.15)]
    seen = set(first)
    yield from first[:n]
    for _ in range(max(0, n-len(first))):
        for attempt in range(10000):
            lo, hi = windows[int(rng.integers(len(windows)))]
            c = JointCandidate(bases[int(rng.integers(len(bases)))],
                               str(rng.choice(['none','before_preprocessing','before_model','after_model'])),
                               str(rng.choice(['mean','median'])), lo, hi, float(rng.choice([step, step*2])),
                               float(rng.choice([1e4,1e5,1e6])), int(rng.choice([5,11,21])), bool(rng.integers(2)), bool(rng.integers(2)))
            if c not in seen:
                seen.add(c); yield c; break


def run_joint(database_dir, data_id, output_dir, visualization_dir, *, n_trials=64, cv=3,
              seed=0, test_size=.25, time_budget=1800, jobs=1, lower=600, upper=1380,
              step=2, max_spectra_per_group=0):
    if n_trials < 1 or cv < 2 or jobs < 1 or time_budget <= 0 or not np.isfinite(time_budget):
        raise ValueError('Invalid search budget.')
    if not all(np.isfinite(v) for v in (lower, upper, step)) or lower >= upper or step <= 0:
        raise ValueError('Invalid grid.')
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new empty run directory.')
    output.mkdir(parents=True, exist_ok=True)
    data = load_dataset(database_dir, data_id, max_spectra_per_group=max_spectra_per_group, seed=seed)
    # Declare the eligible cohort once, before splitting; no candidate can drop hard test rows.
    eligible = [i for i, s in enumerate(data.spectra) if s.axis[0] <= lower and s.axis[-1] >= upper]
    if not eligible:
        raise ValueError('No spectra cover the declared common window.')
    indices = np.asarray(eligible)
    train, test, kind = split_holdout(data.split_blocks[indices], test_size, seed,
                                     [data.audit['source_types'][i] for i in indices])
    train, test = indices[train], indices[test]
    classification = data.metadata['task'] == 'Classification'
    names = data.metadata['molecules'].split('_')
    folds = [(train[a], train[b]) for a,b in GroupKFold(n_splits=min(cv,len(set(data.split_blocks[train]))),
              shuffle=True,random_state=seed).split(train,groups=data.split_blocks[train])]
    split = {'kind':kind,'train':train.tolist(),'test':test.tolist(),'row_ids':list(data.sample_ids),
             'split_blocks':data.split_blocks.tolist(), 'excluded_window_rows':sorted(set(range(len(data.spectra)))-set(eligible)),
             'cv':[{'train':a.tolist(),'validation':b.tolist()} for a,b in folds]}
    write_json(output/'splits.json',split)
    write_json(output/'data_audit.json',data.audit)
    write_json(output/'config.json',dict(n_trials=n_trials,cv=cv,seed=seed,test_size=test_size,
                                        lower=lower,upper=upper,step=step,time_budget=time_budget,jobs=jobs,
                                        max_spectra_per_group=max_spectra_per_group,search_mode='joint'))
    best, best_score, history = None, -np.inf, []
    start = time.perf_counter()
    with threadpool_limits(limits=jobs):
        for c in proposals(n_trials,lower,upper,step,classification,seed):
            if history and time.perf_counter()-start >= time_budget: break
            tick=time.perf_counter(); record={'candidate':asdict(c)}
            try:
                scores=[]
                for a,b in folds:
                    rows = len(set(data.group_ids[a])) if c.aggregation == 'before_preprocessing' else len(a)
                    points = len(np.arange(c.lower,c.upper+c.step*1e-6,c.step))
                    if points < c.window and c.base.preprocessing in ('sg','derivative','airpls_sg'):
                        raise ValueError('SG window exceeds candidate spectral grid.')
                    if c.base.pca > min(rows,points) or (c.feature_branch and min(rows,points)<2):
                        raise ValueError('Feature rank exceeds training-fold matrix limit.')
                    model_rows = len(set(data.group_ids[a])) if c.aggregation in ('before_preprocessing','before_model') else len(a)
                    features = (c.base.pca or points)+(2 if c.feature_branch else 0)
                    if c.base.model == 'pls' and c.base.parameter > min(model_rows-1,features):
                        raise ValueError('PLS rank exceeds training-fold matrix limit.')
                    model=build_joint(c,classification,seed,jobs)
                    training_batch = DataBatch.from_dataset(data, a)
                    validation_batch = DataBatch.from_dataset(data, b)
                    model.fit(training_batch)
                    raw_validation_result = model.predict(validation_batch)
                    validation_result, validation_predictions = group_result(
                        raw_validation_result, classification)
                    truth=DataBatch.from_dataset(data,a)
                    # Fold-training group labels define target scale independently of repeat count.
                    truth=truth.aggregate(MeanAggregator().transform(np.zeros((len(a),1)),groups=truth.groups))
                    scores.append(cv_score(validation_result.y, validation_predictions,
                                           truth.y, classification))
                score=float(np.mean(scores))
                if not np.isfinite(score): raise ValueError('Nonfinite validation score.')
                record.update(status='ok',score=score,fold_scores=scores)
                if score > best_score: best,best_score=c,score
            except (ValueError,TypeError,RuntimeError,np.linalg.LinAlgError) as error:
                record.update(status='failed',error=str(error))
            record['seconds']=time.perf_counter()-tick; history.append(record)
            write_json(output/'trials.json',history)
            print(f"{data_id} joint trial {len(history)}: {record.get('score',record.get('error'))}",flush=True)
        if best is None: raise ValueError('All joint candidates failed; see trials.json.')
        model=build_joint(best,classification,seed,jobs).fit(DataBatch.from_dataset(data,train))
        result,values=group_result(model.predict(DataBatch.from_dataset(data,test)),classification)
        baselines={}
        for name,c in [('constant',JointCandidate(Candidate(model='constant'),lower=lower,upper=upper,step=step)),
                       ('fixed',JointCandidate(Candidate(model='logistic' if classification else 'pls',parameter=1 if classification else 2),lower=lower,upper=upper,step=step))]:
            baseline=build_joint(c,classification,seed,jobs).fit(DataBatch.from_dataset(data,train))
            batch,p=group_result(baseline.predict(DataBatch.from_dataset(data,test)),classification)
            baselines[name]=metrics(batch.y,p,classification,names)
        graph=draw_workflow_graph(model.nodes_,model.edges,output,note='Raw spectra; row-aware aggregation; grouped development-only CV.')
        gallery=visualization_run_dir(output,visualization_dir,Path(__file__).resolve().parents[3]/'outputs')
        if classification: ConfusionMatrixPlot(gallery/'confusion_matrix.png',multilabel=True,class_names=names).transform(values,y_true=result.y)
        else: PredictionPlot(gallery/'pred_vs_true.png',molecule_names=names,unit=data.metadata['quant_unit']).transform(values,y_true=result.y)
        summary={'data_id':data_id,'best_candidate':asdict(best),'best_cv_score':best_score,
                 'holdout_metrics':metrics(result.y,values,classification,names), 'baseline_holdout_metrics':baselines,
                 'trials_completed':len(history),'trials_succeeded':sum(r['status']=='ok' for r in history),
                 'workflow_graph':graph,'evaluation_unit':'original source group; mean vote for classification',
                 'limitations':['Bounded random structural search; no cross-task learning or prediction cascades.',
                                'One exploratory blocked holdout; unknown preparation independence remains unknown.']}
        write_json(output/'summary.json',summary); write_json(output/'workflow_spec.json',model.to_dict())
        write_json(output/'execution_trace.json',{'fit':model.fit_trace_,'predict':model.last_trace_})
        with (output/'test_predictions.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f); writer.writerow(['group']+['true_'+n for n in names]+['pred_'+n for n in names])
            for g,y,p in zip(result.row_ids,result.y,values): writer.writerow([g,*y,*p])
        joblib.dump({'workflow':model,'molecules':names,'unit':data.metadata['quant_unit'],
                     'input_representation':'raw_batch','classification':classification},output/'workflow.joblib')
    return summary
