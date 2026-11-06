"""Explicit axes and sample identity; no implicit alignment/interpolation."""
from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def matrix(values):
    out = np.array(values, dtype=float, copy=True)
    if out.ndim != 2 or min(out.shape) == 0 or not np.isfinite(out).all():
        raise ValueError("Expected a finite, non-empty (samples, features) matrix.")
    return out


@dataclass(frozen=True)
class Spectra:
    """Immutable shared-grid spectral matrix with optional row identifiers.

    Attributes
    ----------
    axis: Strictly increasing one-dimensional Raman-shift coordinates, normally in cm^-1.
    intensity: Read-only float matrix with shape (n_samples, n_points).
    sample_ids: Optional row identifiers in the same order as the intensity matrix.
    """
    intensity: np.ndarray
    axis: np.ndarray
    sample_ids: tuple | None = None

    def __post_init__(self):
        """Validate aligned members and normalize their internal representation.

        Returns and state
        -----------------
        None. Normalizes members or rejects invalid data.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
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
        """Create a new immutable Spectra, optionally replacing coordinates or row IDs.

        Parameters
        ----------
        values: Aggregated matrix with shape (n_groups, n_features).
        axis: Strictly increasing one-dimensional Raman-shift coordinates, normally in cm^-1.
        sample_ids: Optional row identifiers in the same order as the intensity matrix.

        Returns and state
        -----------------
        New immutable Spectra.
        """
        return Spectra(values, self.axis if axis is None else axis,
                       self.sample_ids if sample_ids is None else sample_ids)


@dataclass(frozen=True)
class UnmixingResult:
    """Known-reference unmixing outputs, reconstruction and residuals.

    Attributes
    ----------
    axis: Shared spectral coordinates, or None when references were supplied as plain matrices.
    coefficients: Sample-by-component unmixing coefficients; not automatically concentrations.
    component_names: Validated component-name tuple in coefficient-column order.
    component_spectra: Reference component spectra with shape (n_components, n_points).
    reconstruction: Sample spectra reconstructed from the fitted coefficients and references.
    residual_norm: Per-sample L2 norm of the reconstruction residual.
    sample_ids: Optional row identifiers in the same order as the intensity matrix.
    """
    coefficients: np.ndarray       # samples x components; not automatically concentrations
    component_spectra: np.ndarray  # components x spectral points
    reconstruction: np.ndarray
    residual_norm: np.ndarray
    component_names: tuple[str, ...]
    axis: np.ndarray | None
    sample_ids: tuple | None


@dataclass
class Dataset:
    """Raw individual spectra with row-aligned labels, groups and split dependencies. Loading does not pool or align spectra.

    Attributes
    ----------
    audit: Input paths, counts, exclusions, fingerprints and the effective loading/preparation policy.
    group_ids: Original source-group identifiers aligned with output rows; not proof of preparation independence.
    group_metadata: Optional source information keyed by source-group folder name; empty when unavailable.
    metadata: Dataset-level content.csv fields, including task, molecule order and units.
    spectra: Tuple of individual Spectra objects; original coordinate grids may differ.
    split_blocks: Dependency-block IDs aligned with spectra. Pass these to grouped splitters; do not use them as pooling IDs.
    y: Ground-truth targets aligned with rows, usually (n_rows, n_targets); never predictions.

    Row alignment
    -------------
    spectra[i], y[i], group_ids[i] and split_blocks[i] describe the same original spectrum. X aliases spectra; sample_ids is a computed file-ID tuple.
    """
    spectra: tuple
    y: np.ndarray
    group_ids: np.ndarray
    split_blocks: np.ndarray
    metadata: dict
    audit: dict
    group_metadata: dict

    @property
    def X(self):
        """Return spectra as an alias without copying; accessed as dataset.X.

        Returns and state
        -----------------
        The original spectra tuple without copying.
        """
        return self.spectra

    @property
    def sample_ids(self):
        """Return original spectrum file identifiers in dataset order.

        Returns and state
        -----------------
        Tuple of original spectrum file identifiers.
        """
        return tuple(s.sample_ids[0] for s in self.spectra)


def read_spectrum(path):
    """Read the common two-column CSV format and return axis, intensity."""
    path = Path(path)
    try:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            header = next(csv.reader(stream), [])
            if [column.strip() for column in header] != ['RamanShift', 'intensity']:
                raise ValueError('Expected exactly two columns: RamanShift,intensity.')
            values = np.loadtxt(stream, delimiter=',', ndmin=2)
        if values.shape[1] != 2 or len(values) < 2:
            raise ValueError('A spectrum needs at least two rows and exactly two numeric columns.')
        if np.all(np.diff(values[:, 0]) < 0):
            values = values[::-1]
        spectrum = Spectra(values[:, 1][None, :], values[:, 0])
        return spectrum.axis, spectrum.intensity[0]
    except (ValueError, OSError) as error:
        raise ValueError(f'Invalid spectrum {path}: {error}') from error


def read_dataset_index(database_dir):
    """Return index records; shared by consumers that enumerate available tasks."""
    with (Path(database_dir) / 'content.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or 'Data_id' not in reader.fieldnames:
            raise ValueError('content.csv must contain a Data_id column.')
        return list(reader)


def load_dataset(database_dir, data_id, *, groups=None, read_labels=True,
                 label_policy="include_flagged", max_spectra_per_group=0, seed=0,
                 use_metadata=True):
    """Read raw per-spectrum records without interpolation, averaging or splitting.

    group_ids identify original folders; split_blocks encode known dependencies.
    Metadata label flags are conservative source-review flags, not corrected labels.
    read_labels=False supports inspection of legacy indexes without target fields.
    use_metadata=False ignores optional metadata.json and spectrum_manifest.csv;
    labels then come exclusively from the index/folders and duplicates from files.
    """
    if max_spectra_per_group < 0 or label_policy not in {'exclude_flagged', 'include_flagged', 'error'}:
        raise ValueError('Invalid spectrum limit or label policy.')
    root = Path(database_dir)
    rows = [r for r in read_dataset_index(root) if r['Data_id'] == data_id]
    if len(rows) != 1:
        raise ValueError(f"Expected one index row for {data_id}.")
    info = rows[0]
    # Use the index identifier only after exact lookup and enforce a single path component.
    if Path(data_id).name != data_id or '/' in data_id or '\\' in data_id:
        raise ValueError("Invalid dataset identifier.")
    directory = root / 'data' / data_id
    meta_path = directory / 'metadata.json'
    provenance = json.loads(meta_path.read_text(encoding='utf-8')) if use_metadata and meta_path.exists() else {}
    group_meta = {g['folder']: g for g in provenance.get('groups', [])}
    folders = sorted(p for p in directory.iterdir() if p.is_dir() and not p.name.startswith('.'))
    parent = {p.name: p.name for p in folders}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    identities = {}
    for folder in folders:
        gm = group_meta.get(folder.name, {})
        for key in ('shared_sample_id', 'physical_sample_id', 'physical_preparation_id',
                    'independent_preparation_id', 'duplicate_connected_split_block'):
            value = gm.get(key)
            if value is not None:
                token = (key, str(value))
                if token in identities:
                    union(folder.name, identities[token])
                identities[token] = folder.name
    # Manifest hashes cover even spectra omitted by the optional compute limit.
    manifest = directory / 'spectrum_manifest.csv'
    seen_hash, manifest_digest = {}, hashlib.sha256()
    if use_metadata and manifest.exists():
        manifest_digest.update(manifest.read_bytes())
        with manifest.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                g, digest = row.get('group_id'), row.get('sha256')
                if g in parent and digest:
                    if digest in seen_hash:
                        union(g, seen_hash[digest])
                    seen_hash[digest] = g
    selected = set(groups) if groups is not None else None
    if selected is not None:
        missing = selected - set(parent)
        if missing:
            raise ValueError(f"Unknown group folders: {', '.join(sorted(missing))}")
        folders = [p for p in folders if p.name in selected]
    if not folders:
        raise ValueError(f'No group directories found for {data_id}.')
    n_targets = int(info['n_classes']) if read_labels else 0
    if read_labels and len(info['molecules'].split('_')) != n_targets:
        raise ValueError("Molecule count does not match metadata.")
    classification = info.get('task') == 'Classification'
    if read_labels and info['task'] not in ('Classification', 'Single-Component Quantification',
                             'Multi-Component Quantification'):
        raise ValueError("Unsupported task.")
    X, y, ids, used_files, exclusions, counts, types = [], [], [], [], [], [], []
    quality_flags = {}
    observed_hashes = {}
    rng = np.random.default_rng(seed)
    for folder in folders:
        gm = group_meta.get(folder.name, {})
        flagged = (gm.get('label_consistent_with_checked_SI_table') is False or
                   'conflict' in gm.get('label_review_status', ''))
        quality_flags[folder.name] = gm.get('label_review_status') or ('flagged' if flagged else 'not_reviewed')
        if flagged:
            if label_policy == 'error':
                raise ValueError(f'{folder.name}: unresolved source-label review flag.')
            if label_policy == 'exclude_flagged':
                exclusions.append({'group': folder.name, 'reason': 'unresolved source-label review flag'})
                continue
        if read_labels:
            tokens = folder.name.split('_')
            if len(tokens) != n_targets + 2 or tokens[0] != ('Class' if classification else 'Quan'):
                raise ValueError(f'Invalid group label: {folder.name}')
            target = np.asarray(tokens[1:-1], dtype=float)
            if not np.isfinite(target).all() or (target < 0).any() or (classification and not np.isin(target, [0, 1]).all()):
                raise ValueError(f'Invalid targets: {folder.name}')
            if 'labels' in gm and not np.array_equal(target, np.asarray(gm['labels'], dtype=float)):
                raise ValueError(f'Folder and metadata labels disagree: {folder.name}')
        else:
            target = np.empty(0)
        paths = sorted(folder.glob('*.csv'))
        if not paths:
            raise ValueError(f"Empty group: {folder.name}")
        total = len(paths)
        if max_spectra_per_group and total > max_spectra_per_group:
            paths = [paths[i] for i in sorted(rng.choice(total, max_spectra_per_group, replace=False))]
        values, selected = [], []
        for p in paths:
            axis, intensity = read_spectrum(p)
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest in observed_hashes:
                union(folder.name, observed_hashes[digest])
            observed_hashes[digest] = folder.name
            values.append(Spectra(intensity[None, :], axis, (str(p.relative_to(root)),)))
            selected.append(str(p.relative_to(root)))
        X.extend(values); y.extend([target] * len(values)); ids.extend([folder.name] * len(values))
        used_files.extend(selected); counts.append({'group': folder.name, 'available': total, 'used': len(values)})
        types.extend([gm.get('type')] * len(values))
    fingerprint = hashlib.sha256()
    for p in used_files:
        fingerprint.update(p.encode('utf-8')); fingerprint.update((root / p).read_bytes())
    if not X:
        raise ValueError('No eligible spectra remain under the selected label policy.')
    audit = {'representation': 'raw individual spectra; original axes retained', 'database_dir': str(root.resolve()),
             'max_spectra_per_group': max_spectra_per_group, 'label_policy': label_policy,
             'seed': seed, 'counts': counts, 'files': used_files, 'exclusions': exclusions,
             'source_types': types, 'input_sha256': fingerprint.hexdigest(),
             'manifest_sha256': manifest_digest.hexdigest() if use_metadata and manifest.exists() else None,
             'use_metadata': use_metadata,
             'source_limitations': provenance.get('limitations', []),
             'split_rule': 'union of source folders, known shared sample/preparation IDs and duplicate hashes',
             'group_ids': ids, 'split_blocks': [find(g) for g in ids], 'quality_flags': quality_flags}
    return Dataset(tuple(X), np.asarray(y, dtype=int if classification and read_labels else float).reshape(len(X), n_targets),
                   np.asarray(ids), np.asarray([find(g) for g in ids]), info, audit, group_meta)
