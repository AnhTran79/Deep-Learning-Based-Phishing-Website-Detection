from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
QUICK_MAX_ROWS = 1200
QUICK_EPOCHS = 2


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
    command = [
        sys.executable,
        "-m",
        "src.data.prepare_phish360",
        "--root",
        str(args.root),
        "--out",
        str(args.data),
    ]
    if args.allow_missing_html:
        command.append("--allow-missing-html")
    return command


def build_train_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.training.train_phish360_multimodal",
        "--data",
        str(args.data),
        "--model",
        args.model,
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
        "--image-size",
        str(args.image_size),
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
        "--comparison-file",
        "phish360_model_comparison.csv",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Separate Phish360 multi-modal training pipeline for URL + HTML + screenshot."
    )
    parser.add_argument("--root", default="data/external/Phish360")
    parser.add_argument("--data", default="data/processed/phish360_url_html_screenshot.csv")
    parser.add_argument("--out-dir", default="models/saved/phish360")
    parser.add_argument("--results-dir", default="reports/results/phish360")
    parser.add_argument("--figures-dir", default="reports/figures/phish360")
    parser.add_argument(
        "--model",
        default="all",
        choices=[
            "url_cnn",
            "url_lstm",
            "html_cnn",
            "screenshot_cnn",
            "dual_branch_cnn",
            "tri_branch_cnn",
            "all",
        ],
    )
    parser.add_argument("--html-max-chars", type=int, default=2000)
    parser.add_argument("--max-html-len", type=int, default=2000)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means use the full Phish360 dataset.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Shortcut for smoke runs: --max-rows {QUICK_MAX_ROWS} --epochs {QUICK_EPOCHS} --cpu.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument(
        "--allow-missing-html",
        action="store_true",
        help="Index samples without HTML. Leave off for URL + HTML + screenshot training.",
    )
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.quick:
        if args.max_rows == 0:
            args.max_rows = QUICK_MAX_ROWS
        if args.epochs is None:
            args.epochs = QUICK_EPOCHS
        if args.batch_size is None:
            args.batch_size = 16
        args.cpu = True
    return args


def validate_inputs(args: argparse.Namespace) -> None:
    root = project_path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Phish360 dataset folder not found: {root}")


def main() -> None:
    args = normalize_args(parse_args())
    validate_inputs(args)
    data_path = project_path(args.data)

    if args.force_prepare or not data_path.exists():
        run_step(build_prepare_command(args), "Prepare Phish360 URL + HTML + screenshot index")
    else:
        print(f"Using existing Phish360 index: {data_path}", flush=True)

    if args.skip_training:
        print("\nDone. Phish360 training skipped.", flush=True)
        return

    run_step(build_train_command(args), "Train Phish360 multi-modal models")
    if not args.skip_evaluation:
        run_step(build_compare_command(args), "Export Phish360 comparison table and charts")
    print("\nDone. Results: reports/results/phish360/phish360_model_comparison.csv", flush=True)
    print("Figures: reports/figures/phish360", flush=True)


if __name__ == "__main__":
    main()
