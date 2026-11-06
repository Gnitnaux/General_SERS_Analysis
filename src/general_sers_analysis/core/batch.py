"""Minimal row-aligned execution data for transforms, selections and aggregation."""
from dataclasses import dataclass
import numpy as np
from .data import Spectra


def row_count(X):
    return len(X.intensity) if isinstance(X, Spectra) else len(X)


@dataclass
class DataBatch:
    """Execution payload carrying data, ground truth, row identities, groups and lineage. X may become predictions; y remains ground truth.

    Attributes
    ----------
    X: Current payload: individual spectra, shared-grid Spectra, features or predictions.
    groups: Source-group identifiers aligned with current rows; determine pooling membership.
    lineage: For each output row, the tuple of original row IDs contributing to it.
    row_ids: Unique identifiers aligned with the current rows of X and y.
    y: Ground-truth targets aligned with rows, usually (n_rows, n_targets); never predictions.
    """
    X: object
    row_ids: tuple
    groups: tuple
    y: object = None
    lineage: tuple = ()

    def __post_init__(self):
        """Validate aligned members and normalize their internal representation.

        Returns and state
        -----------------
        None. Normalizes members or rejects invalid data.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        self.row_ids = tuple(self.row_ids)
        self.groups = tuple(self.groups)
        n = row_count(self.X)
        if not n or len(self.row_ids) != n or len(set(self.row_ids)) != n or len(self.groups) != n:
            raise ValueError('Batch data, unique row IDs and groups must match.')
        if self.y is not None:
            self.y = np.array(self.y, copy=True)
            if len(self.y) != n:
                raise ValueError('Batch labels must match rows.')
        self.lineage = tuple(self.lineage) or tuple((r,) for r in self.row_ids)
        if len(self.lineage) != n:
            raise ValueError('Lineage must match rows.')

    def replace(self, X):
        """Create a batch with new values and the same row count, retaining labels and identities.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        New DataBatch with unchanged identities.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        if row_count(X) != len(self.row_ids):
            raise ValueError('Row-changing node requires an explicit row mapping.')
        return DataBatch(X, self.row_ids, self.groups, self.y, self.lineage)

    def take(self, indices):
        """Select/reorder payload, targets, groups and lineage using unique valid integer positions.

        Parameters
        ----------
        indices: Unique integer source-row positions; their order determines output order.

        Returns and state
        -----------------
        New DataBatch in selected row order.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.dtype.kind not in 'iu' or len(set(indices.tolist())) != len(indices):
            raise ValueError('Selection needs unique integer indices.')
        if (indices < 0).any() or (indices >= len(self.row_ids)).any():
            raise ValueError('Selection index out of bounds.')
        X = self.X
        if isinstance(X, Spectra):
            X = Spectra(X.intensity[indices], X.axis, tuple(self.row_ids[i] for i in indices))
        elif isinstance(X, (tuple, list)):
            X = [X[i] for i in indices]
        else:
            X = np.asarray(X)[indices]
        return DataBatch(X, tuple(self.row_ids[i] for i in indices), tuple(self.groups[i] for i in indices),
                         None if self.y is None else self.y[indices], tuple(self.lineage[i] for i in indices))

    def aggregate(self, result):
        """Validate output-group ordering and equal member labels, then merge identities and lineage.

        Parameters
        ----------
        result: Aggregator output with explicit ordered group IDs.

        Returns and state
        -----------------
        New group-level DataBatch with validated labels and merged lineage.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        ids = tuple(dict.fromkeys(self.groups))
        output_ids = result.sample_ids if isinstance(result, Spectra) else result.group_ids
        if tuple(output_ids) != ids:
            raise ValueError('Aggregator output IDs do not match input group order.')
        labels, lineage = [], []
        for group in ids:
            indices = [i for i, g in enumerate(self.groups) if g == group]
            if self.y is not None:
                if not all(np.array_equal(self.y[i], self.y[indices[0]]) for i in indices):
                    raise ValueError('Cannot aggregate records with inconsistent target labels.')
                labels.append(self.y[indices[0]])
            lineage.append(tuple(r for i in indices for r in self.lineage[i]))
        X = result if isinstance(result, Spectra) else result.values
        return DataBatch(X, ids, ids, None if self.y is None else np.asarray(labels), tuple(lineage))

    @classmethod
    def from_dataset(cls, dataset, indices):
        """Construct a raw batch from selected dataset rows without interpolation or pooling.

        Parameters
        ----------
        dataset: Raw Dataset returned by the shared loader.
        indices: Unique integer source-row positions; their order determines output order.

        Returns and state
        -----------------
        New raw-spectrum DataBatch.
        """
        return cls([dataset.spectra[i] for i in indices], tuple(dataset.sample_ids[i] for i in indices),
                   tuple(dataset.group_ids[i] for i in indices), dataset.y[indices])


@dataclass
class RowSelection:
    """Explicit source-row positions returned by a filtering Node.

    Attributes
    ----------
    indices: Unique integer source-row positions; their order determines output order.
    """
    indices: object
