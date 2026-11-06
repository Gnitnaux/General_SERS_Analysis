"""Explicit representation for the historical group-mean AutoML baseline.

This is a fixed preparation workflow, not a searched aggregation strategy.
Raw acquisition/labels/dependencies are loaded only by core.data.load_dataset.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
from ..core import Edge
from ..tools.preprocessing import Interpolate
from ..tools.aggregation import MeanAggregator
from ..workflow import Workflow, NodeSpec


@dataclass
class PreparedDataset:
    """Historical group-mean preparation result; each row is one source-folder mean.

    Attributes
    ----------
    X: Shared-grid Spectra with one arithmetic-mean row per original source folder.
    audit: Input paths, counts, exclusions, fingerprints and the effective loading/preparation policy.
    groups: Dependency-block IDs aligned with the prepared source-folder means.
    metadata: Dataset-level content.csv fields, including task, molecule order and units.
    representation_workflow: Actual interpolation/pooling preparation workflow, exportable as a declaration.
    y: Ground-truth targets aligned with rows, usually (n_rows, n_targets); never predictions.
    """
    X: object
    y: np.ndarray
    groups: np.ndarray
    metadata: dict
    audit: dict
    representation_workflow: Workflow


def prepare_group_means(dataset, *, lower=600., upper=1380., step=2.):
    if not all(np.isfinite(v) for v in (lower, upper, step)) or lower >= upper or step <= 0:
        raise ValueError('Invalid alignment grid.')
    grid = np.arange(lower, upper + step * 1e-6, step)
    if not 3 <= len(grid) <= 20000:
        raise ValueError('Grid must contain 3..20000 points.')
    selected, exclusions = [], list(dataset.audit['exclusions'])
    sample_ids = dataset.sample_ids
    for i, spectrum in enumerate(dataset.spectra):
        if spectrum.axis[0] > grid[0] or spectrum.axis[-1] < grid[-1]:
            exclusions.append({'file': sample_ids[i], 'reason': 'does not cover declared window',
                               'range': [float(spectrum.axis[0]), float(spectrum.axis[-1])]})
        else:
            selected.append(i)
    ids = tuple(dict.fromkeys(dataset.group_ids[selected]))
    for group in dict.fromkeys(dataset.group_ids):
        if group not in ids:
            exclusions.append({'group': str(group), 'reason': 'no spectrum covers declared window'})
    if len(ids) < 4:
        raise ValueError('fewer than four eligible groups in the declared alignment window.')
    workflow = Workflow({'align': NodeSpec(Interpolate(grid), map_samples=True, stack_output=True),
                         'group_mean': NodeSpec(MeanAggregator())},
                        [Edge('input', 'align', source_port='X', data_kind='spectra'),
                         Edge('align', 'group_mean', data_kind='spectra'),
                         Edge('input', 'group_mean', source_port='groups', target_port='groups', data_kind='groups')],
                        output_node='group_mean')
    raw = [dataset.spectra[i] for i in selected]
    source_groups = dataset.group_ids[selected].tolist()
    X = workflow.fit(raw, groups=source_groups).predict(raw, groups=source_groups)
    first = {group: source_groups.index(group) for group in ids}
    indices = [selected[first[g]] for g in ids]
    fingerprint = hashlib.sha256()
    files = [sample_ids[i] for i in selected]
    for file in files:
        fingerprint.update(file.encode('utf-8'))
        fingerprint.update((Path(dataset.audit['database_dir']) / file).read_bytes())
    original_counts = {r['group']: r for r in dataset.audit['counts']}
    audit = {**dataset.audit, 'representation': 'one arithmetic mean per original folder on a fixed declared grid',
             'grid': [float(grid[0]), float(grid[-1]), step], 'exclusions': exclusions,
             'files': files, 'input_sha256': fingerprint.hexdigest(),
             'source_types': [dataset.audit['source_types'][i] for i in indices],
             'group_ids': list(ids), 'split_blocks': dataset.split_blocks[indices].tolist(),
             'counts': [{'group': g, 'available': original_counts[g]['available'],
                         'used': source_groups.count(g)} for g in ids]}
    return PreparedDataset(X, dataset.y[indices], dataset.split_blocks[indices], dataset.metadata, audit, workflow)
