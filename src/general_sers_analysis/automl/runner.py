"""One experiment entry point for a single dataset or the database index."""
from pathlib import Path
from ..core.data import read_dataset_index
from ..util import write_json
from .joint import run_joint


def run_experiment(database_dir, data_id, output_dir, visualization_dir, **options):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Choose a new empty run directory: {output}')
    all_datasets = data_id == 'all'
    ids = [row['Data_id'] for row in read_dataset_index(database_dir)] if all_datasets else [data_id]
    if not ids:
        raise ValueError('No datasets in the database index.')
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for identifier in ids:
        destination = output / identifier if all_datasets else output
        try:
            summary = run_joint(database_dir, identifier, destination, visualization_dir, **options)
            records.append({'data_id': identifier, 'status': 'ok',
                            'summary': str((destination / 'summary.json').resolve())})
        except (OSError, ValueError) as error:
            records.append({'data_id': identifier, 'status': 'failed', 'error': str(error)})
        write_json(output / 'runs.json', records)
    return records
