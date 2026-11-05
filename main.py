"""Main entry point; visualization and a fixed PLSR example are implemented."""
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
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
