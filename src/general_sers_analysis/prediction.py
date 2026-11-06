"""Predict new source-folder means using a trusted saved AutoML workflow."""
import csv
from pathlib import Path
import joblib
import numpy as np
from .core import Spectra
from .core.data import read_spectrum


def predict_directory(model_path, input_dir, output_file):
    """One folder of CSVs or immediate group subfolders; no target labels required.

    joblib is a trusted-local-artifact format: loading it can execute Python.
    Every supplied spectrum must cover the saved coordinate grid. This function
    mirrors training's interpolation-then-mean representation and rejects omissions.
    """
    output = Path(output_file)
    if output.exists():
        raise ValueError(f'Prediction output already exists: {output}')
    saved = joblib.load(model_path)
    axis = np.asarray(saved['axis'])
    root = Path(input_dir)
    direct = sorted(root.glob('*.csv'))
    folders = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    if direct and folders:
        raise ValueError('Input must contain spectra directly or group folders, not both.')
    folders = [root] if direct else folders
    values, ids, counts = [], [], []
    for folder in folders:
        files = sorted(folder.glob('*.csv'))
        if not files:
            raise ValueError(f'Empty input group: {folder}')
        spectra = []
        for f in files:
            x, y = read_spectrum(f)
            if x[0] > axis[0] or x[-1] < axis[-1]:
                raise ValueError(f'{f}: does not cover saved model axis; extrapolation is forbidden.')
            spectra.append(np.interp(axis, x, y))
        values.append(np.mean(spectra, axis=0)); ids.append(folder.name); counts.append(len(spectra))
    if not values:
        raise ValueError('No input spectra found.')
    prediction = np.asarray(saved['workflow'].predict(Spectra(np.asarray(values), axis, tuple(ids))))
    if prediction.shape != (len(ids), len(saved['molecules'])) or not np.isfinite(prediction).all():
        raise ValueError('Invalid prediction shape or values.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['group', 'n_spectra'] + ['pred_' + m for m in saved['molecules']])
        for group, count, pred in zip(ids, counts, prediction):
            writer.writerow([group, count, *pred])
    return output
