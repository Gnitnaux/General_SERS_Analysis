"""Main entry point for spectral visualization, workflows and AutoML."""
import argparse
from pathlib import Path

from general_sers_analysis.dataset_visualization import visualize_dataset


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description="General Raman/SERS analysis tasks.")
    tasks = parser.add_subparsers(dest="command", required=True)
    visualize = tasks.add_parser("visualize", help="Plot one mean spectrum per dataset group.")
    visualize.add_argument("--data-id", required=True, help="Exact Data_id from content.csv.")
    visualize.add_argument("--database-dir", type=Path, default=PROJECT_ROOT / "data" / "database",
                           help="Database root containing content.csv and data/.")
    visualize.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "visualizations",
                           help="Directory for the generated PNG.")
    visualize.add_argument("--groups", nargs="+", help="Optional exact group folder names to plot.")
    visualize.add_argument("--show", action="store_true", help="Also open an interactive plot window.")
    example = tasks.add_parser("example", help="Run GSAD0004 cut/align/airPLS/plot/multi-output PLSR.")
    example.add_argument("--database-dir", type=Path, default=PROJECT_ROOT / "data" / "database")
    example.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "GSAD0004_example")
    example.add_argument("--visualization-dir", type=Path, default=PROJECT_ROOT / "visualizations")
    example.add_argument("--lower", type=float, default=600)
    example.add_argument("--upper", type=float, default=1800)
    example.add_argument("--step", type=float, default=2)
    example.add_argument("--n-components", type=int, default=2)
    example.add_argument("--test-size", type=float, default=0.25)
    example.add_argument("--seed", type=int, default=0)
    example.add_argument("--lam", type=float, default=1e5)
    example.add_argument("--show", action="store_true")
    automl = tasks.add_parser("automl", help="Search classical workflows with blocked CV and an untouched holdout.")
    automl.add_argument("--data-id", required=True, help="Exact Data_id, or 'all' to benchmark the database.")
    automl.add_argument("--database-dir", type=Path, default=PROJECT_ROOT / "data" / "database")
    automl.add_argument("--output-dir", type=Path, required=True, help="New empty run directory (benchmark root for 'all').")
    automl.add_argument("--visualization-dir", type=Path, default=PROJECT_ROOT / "visualizations")
    automl.add_argument("--trials", type=int, default=64)
    automl.add_argument("--cv", type=int, default=3)
    automl.add_argument("--seed", type=int, default=0)
    automl.add_argument("--test-size", type=float, default=.25)
    automl.add_argument("--time-budget", type=float, default=1800, help="Search seconds per dataset; checked between trials.")
    automl.add_argument("--jobs", type=int, default=1, help="CPU/BLAS worker limit.")
    automl.add_argument("--lower", type=float, default=600)
    automl.add_argument("--upper", type=float, default=1380)
    automl.add_argument("--step", type=float, default=2)
    automl.add_argument("--max-spectra-per-group", type=int, default=0, help="0=all; positive=seeded compute-limited subset.")
    predict = tasks.add_parser('predict', help='Predict group means from a trusted saved AutoML model.')
    predict.add_argument('--model', type=Path, required=True, help='Trusted local workflow.joblib.')
    predict.add_argument('--input-dir', type=Path, required=True, help='A folder of spectra or group subfolders.')
    predict.add_argument('--output-file', type=Path, required=True, help='New predictions CSV file.')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "visualize":
            output, means = visualize_dataset(args.database_dir, args.data_id,
                                              args.output_dir, groups=args.groups, show=args.show)
            print(f"Dataset: {args.data_id}")
            print(f"Groups: {len(means)}; spectra: {sum(group.n_spectra for group in means)}")
            print(f"Saved: {output}")
            if len(means) > 20:
                print("Legend omitted for more than 20 groups; use --groups to inspect a subset.")
        elif args.command == "example":
            from general_sers_analysis.example_workflow import run_example
            summary = run_example(args.database_dir, args.output_dir, args.visualization_dir,
                                  lower=args.lower, upper=args.upper, step=args.step,
                                  n_components=args.n_components, test_size=args.test_size,
                                  seed=args.seed, lam=args.lam, show=args.show)
            print(f"Train groups: {len(summary['train_groups'])}; test groups: {len(summary['test_groups'])}")
            print(f"Workflow graph: {summary['workflow_graph']['png']}")
            print(f"Plots: {summary['visualization_dir']}")
            print(f"Results: {args.output_dir}")
            print("Metrics on held-out group means:", summary["metrics_on_group_means"])
        elif args.command == "automl":
            from general_sers_analysis.automl import run_automl, run_benchmark
            options = dict(n_trials=args.trials, cv=args.cv, seed=args.seed, test_size=args.test_size,
                           time_budget=args.time_budget, jobs=args.jobs, lower=args.lower,
                           upper=args.upper, step=args.step, max_spectra_per_group=args.max_spectra_per_group)
            if args.data_id == 'all':
                result = run_benchmark(args.database_dir, args.output_dir, args.visualization_dir, **options)
                print(f"Completed datasets: {sum(r['status'] == 'ok' for r in result)}/{len(result)}")
                print(f"Benchmark: {args.output_dir / 'benchmark.json'}")
                if any(r['status'] != 'ok' for r in result):
                    return 1
            else:
                result = run_automl(args.database_dir, args.data_id, args.output_dir, args.visualization_dir, **options)
                print("Selected:", result['best_candidate'])
                print("Holdout:", result['holdout_metrics'])
            print(f"Results: {args.output_dir}")
        elif args.command == 'predict':
            from general_sers_analysis.prediction import predict_directory
            print('Saved:', predict_directory(args.model, args.input_dir, args.output_file))
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
