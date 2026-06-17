from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
QUICK_MAX_ROWS = 5000
QUICK_EPOCHS = 3


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_step(command: list[str], title: str) -> None:
    print(f"\n=== {title} ===", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def add_if_value(command: list[str], option: str, value: int | float | str | None) -> None:
    if value is not None:
        command.extend([option, str(value)])


def build_prepare_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.data.prepare_from_metadata",
        "--metadata",
        str(args.metadata),
        "--html-root",
        str(args.html_root),
        "--out",
        str(args.data),
        "--min-html-length",
        str(args.min_html_length),
    ]


def build_baseline_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.training.train_baselines",
        "--data",
        str(args.data),
        "--out-dir",
        str(args.out_dir),
        "--results-dir",
        str(args.results_dir),
        "--figures-dir",
        str(args.figures_dir),
        "--html-max-chars",
        str(args.html_max_chars),
    ]
    add_if_value(command, "--max-rows", args.max_rows if args.max_rows > 0 else None)
    return command


def build_deep_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.training.train_deep_models",
        "--data",
        str(args.data),
        "--model",
        args.deep_model,
        "--out-dir",
        str(args.out_dir),
        "--results-dir",
        str(args.results_dir),
        "--figures-dir",
        str(args.figures_dir),
        "--html-max-chars",
        str(args.html_max_chars),
        "--max-html-len",
        str(args.max_html_len),
    ]
    add_if_value(command, "--max-rows", args.max_rows if args.max_rows > 0 else None)
    add_if_value(command, "--epochs", args.epochs)
    add_if_value(command, "--batch-size", args.batch_size)
    add_if_value(command, "--embedding-dim", args.embedding_dim)
    if args.cpu:
        command.append("--cpu")
    return command


def build_compare_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.evaluation.compare_models",
        "--results-dir",
        str(args.results_dir),
        "--figures-dir",
        str(args.figures_dir),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command training pipeline for URL + HTML phishing detection."
    )
    parser.add_argument("--metadata", default="output/mendeley_metadata.csv")
    parser.add_argument("--html-root", default="dataset")
    parser.add_argument("--data", default="data/processed/mendeley_url_html_label.csv.gz")
    parser.add_argument("--out-dir", default="models/saved")
    parser.add_argument("--results-dir", default="reports/results/mendeley")
    parser.add_argument("--figures-dir", default="reports/figures/mendeley")
    parser.add_argument("--min-html-length", type=int, default=100)
    parser.add_argument("--html-max-chars", type=int, default=2000)
    parser.add_argument("--max-html-len", type=int, default=2000)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means use the full dataset.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Shortcut for a fast smoke run: --max-rows {QUICK_MAX_ROWS} --epochs {QUICK_EPOCHS} --cpu.",
    )
    parser.add_argument(
        "--deep-model",
        default="all",
        choices=["url_cnn", "url_lstm", "html_cnn", "dual_branch_cnn", "all"],
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--cpu", action="store_true", help="Force PyTorch training on CPU.")
    parser.add_argument("--force-prepare", action="store_true", help="Rebuild the processed gzip dataset.")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-deep", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.quick:
        if args.max_rows == 0:
            args.max_rows = QUICK_MAX_ROWS
        if args.epochs is None:
            args.epochs = QUICK_EPOCHS
        args.cpu = True
    return args


def validate_inputs(args: argparse.Namespace) -> None:
    metadata_path = project_path(args.metadata)
    html_root = project_path(args.html_root)
    data_path = project_path(args.data)

    if args.force_prepare or not data_path.exists():
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        if not html_root.exists():
            raise FileNotFoundError(f"HTML dataset folder not found: {html_root}")


def main() -> None:
    args = normalize_args(parse_args())
    validate_inputs(args)
    data_path = project_path(args.data)

    if args.force_prepare or not data_path.exists():
        run_step(build_prepare_command(args), "Prepare dataset")
    else:
        print(f"Using existing processed dataset: {data_path}", flush=True)

    if not args.skip_baselines:
        run_step(build_baseline_command(args), "Train baseline models")

    if not args.skip_deep:
        run_step(build_deep_command(args), "Train deep models")

    if not args.skip_evaluation:
        run_step(build_compare_command(args), "Export comparison table and charts")
        print("\nDone. Results: reports/results/mendeley/model_comparison_sorted.csv", flush=True)
        print("Chart: reports/figures/mendeley/model_comparison_metrics.png", flush=True)
    else:
        print("\nDone. Evaluation export skipped.", flush=True)


if __name__ == "__main__":
    main()
