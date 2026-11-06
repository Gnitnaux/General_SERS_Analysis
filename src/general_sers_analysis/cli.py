"""Command-line definitions, grouped by command with shared argument helpers."""
import argparse
from pathlib import Path


def _add_database_argument(parser, root):
    parser.add_argument("--database-dir", type=Path, default=root / "data" / "database",
                        help="Database root containing content.csv and data/.")


def _add_visualization_argument(parser, root):
    parser.add_argument("--visualization-dir", type=Path, default=root / "visualizations")


def _add_grid_arguments(parser, *, upper):
    parser.add_argument("--lower", type=float, default=600)
    parser.add_argument("--upper", type=float, default=upper)
    parser.add_argument("--step", type=float, default=2)


def _add_split_arguments(parser):
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=.25)


def _configure_visualize(parser, root):
    parser.add_argument("--data-id", required=True, help="Exact Data_id from content.csv.")
    _add_database_argument(parser, root)
    parser.add_argument("--output-dir", type=Path, default=root / "visualizations",
                           help="Directory for the generated PNG.")
    parser.add_argument("--groups", nargs="+", help="Optional exact group folder names to plot.")
    parser.add_argument("--show", action="store_true", help="Also open an interactive plot window.")


def _configure_example(parser, root):
    _add_database_argument(parser, root)
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "GSAD0004_example")
    _add_visualization_argument(parser, root)
    _add_grid_arguments(parser, upper=1800)
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--lam", type=float, default=1e5)
    parser.add_argument("--show", action="store_true")
    _add_split_arguments(parser)


def _configure_automl(parser, root):
    parser.add_argument("--data-id", required=True, help="Exact Data_id, or 'all' to benchmark the database.")
    _add_database_argument(parser, root)
    parser.add_argument("--output-dir", type=Path, required=True, help="New empty run directory (benchmark root for 'all').")
    _add_visualization_argument(parser, root)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--time-budget", type=float, default=1800, help="Search seconds per dataset; checked between trials.")
    parser.add_argument("--jobs", type=int, default=1, help="CPU/BLAS worker limit.")
    _add_grid_arguments(parser, upper=1380)
    parser.add_argument("--max-spectra-per-group", type=int, default=0, help="0=all; positive=seeded compute-limited subset.")
    _add_split_arguments(parser)


def _configure_predict(parser, root):
    parser.add_argument('--model', type=Path, required=True, help='Trusted local workflow.joblib.')
    parser.add_argument('--input-dir', type=Path, required=True, help='A folder of spectra or group subfolders.')
    parser.add_argument('--output-file', type=Path, required=True, help='New predictions CSV file.')


COMMANDS = (
    ('visualize', 'Plot one mean spectrum per dataset group.', _configure_visualize),
    ('example', 'Run GSAD0004 cut/align/airPLS/plot/multi-output PLSR.', _configure_example),
    ('automl', 'Search classical workflows with blocked CV and an untouched holdout.', _configure_automl),
    ('predict', 'Predict group means from a trusted saved AutoML model.', _configure_predict),
)


def build_parser(project_root):
    """Assemble commands; individual configurators own their argument definitions."""
    root = Path(project_root)
    parser = argparse.ArgumentParser(description="General Raman/SERS analysis tasks.")
    tasks = parser.add_subparsers(dest="command", required=True)
    for name, help_text, configure in COMMANDS:
        configure(tasks.add_parser(name, help=help_text), root)
    return parser


def run_automl_command(args):
    """Translate CLI options once; experiment orchestration belongs to AutoML."""
    from .automl.runner import run_experiment
    option_names = ('cv', 'seed', 'test_size', 'time_budget', 'jobs', 'lower',
                    'upper', 'step', 'max_spectra_per_group')
    options = {name: getattr(args, name) for name in option_names}
    records = run_experiment(args.database_dir, args.data_id, args.output_dir,
                             args.visualization_dir, n_trials=args.trials, **options)
    failures = [record for record in records if record['status'] == 'failed']
    for record in failures:
        print(f"{record['data_id']}: {record['error']}")
    print(f"Completed datasets: {len(records)-len(failures)}/{len(records)}")
    print(f"Results: {args.output_dir}")
    return int(bool(failures))
