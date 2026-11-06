"""Small shared helpers; dataset semantics and algorithms belong in their own modules."""
import json
from pathlib import Path
import re


def natural_key(path):
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r'(\d+)', Path(path).name))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False, default=str), encoding='utf-8')


def subset_spectra(X, indices):
    from .core.data import Spectra
    return Spectra(X.intensity[indices], X.axis,
                   tuple(X.sample_ids[i] for i in indices) if X.sample_ids else None)


def visualization_run_dir(output_dir, visualization_root, output_root):
    output = Path(output_dir).resolve()
    try:
        relative = output.relative_to(Path(output_root).resolve())
    except ValueError:
        relative = Path(output.name)
    return Path(visualization_root).resolve() / relative
