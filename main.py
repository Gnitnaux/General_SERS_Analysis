"""Main entry point for spectral visualization, workflows and AutoML."""
from pathlib import Path

from general_sers_analysis.cli import build_parser as create_parser
from general_sers_analysis.dataset_visualization import visualize_dataset


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser():
    return create_parser(PROJECT_ROOT)


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
            from general_sers_analysis.cli import run_automl_command
            return run_automl_command(args)

        elif args.command == 'predict':
            from general_sers_analysis.prediction import predict_directory
            print('Saved:', predict_directory(args.model, args.input_dir, args.output_file))

    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
