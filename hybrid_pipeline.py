from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_INDEX = Path("data/processed/phish360_url_html_screenshot.csv")
AUDITED_INDEX = Path("data/processed/phish360_plus_audited.csv")


def batch_name(value: str | None) -> str:
    value = value or date.today().isoformat().replace("-", "_")
    if re.fullmatch(r"\d{4}_\d{2}_\d{2}", value):
        value = f"week_{value}"
    if not re.fullmatch(r"week_\d{4}_\d{2}_\d{2}", value):
        raise ValueError("Batch must be YYYY_MM_DD or week_YYYY_MM_DD.")
    return value


def paths(batch: str) -> dict[str, Path]:
    return {
        "candidates": Path("data/candidates") / f"{batch}.csv",
        "candidate_summary": Path("data/candidates") / f"{batch}_summary.json",
        "dataset": Path("data/external_test") / batch,
        "results": Path("reports/results") / f"external_test_{batch}",
        "figures": Path("reports/figures") / f"external_test_{batch}",
        "audit": Path("data/audit") / f"{batch}_audit_queue.csv",
    }


def run(script: str, *args: object) -> None:
    command = [sys.executable, script, *(str(arg) for arg in args)]
    print(f"\n> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def collect(args: argparse.Namespace) -> None:
    batch = batch_name(args.batch)
    batch_paths = paths(batch)
    candidate_count = args.count * 3
    run(
        "fetch_weekly_candidates.py",
        "--tranco-file",
        "data/raw/top-1m.csv",
        "--out",
        batch_paths["candidates"],
        "--summary",
        batch_paths["candidate_summary"],
        "--phishing-count",
        candidate_count,
        "--legitimate-count",
        candidate_count,
        "--tranco-max-rows",
        max(50000, candidate_count * 1000),
        "--update-seen",
    )
    run(
        "collect_external_dataset.py",
        "--input",
        batch_paths["candidates"],
        "--output",
        batch_paths["dataset"],
        "--legitimate-count",
        args.count,
        "--phishing-count",
        args.count,
        "--source",
        batch,
        "--headless",
        "--resume",
    )


def evaluate(args: argparse.Namespace) -> None:
    batch = batch_name(args.batch)
    batch_paths = paths(batch)
    run(
        "evaluate_external_dataset.py",
        "--dataset",
        batch_paths["dataset"],
        "--results-dir",
        batch_paths["results"],
        "--figures-dir",
        batch_paths["figures"],
        "--models",
        "phish360_url_cnn",
        "phish360_url_lstm",
        "phish360_html_cnn",
        "phish360_screenshot_cnn",
        "phish360_dual_branch_cnn",
        "phish360_tri_branch_cnn",
    )
    run(
        "create_audit_queue.py",
        "--dataset",
        batch_paths["dataset"],
        "--predictions",
        batch_paths["results"] / "predictions.csv",
        "--out",
        batch_paths["audit"],
    )
    print(f"\nAudit file: {batch_paths['audit']}", flush=True)


def retrain(args: argparse.Namespace) -> None:
    batch = batch_name(args.batch)
    batch_paths = paths(batch)
    with batch_paths["audit"].open("r", encoding="utf-8-sig", newline="") as handle:
        approved = sum(
            (row.get("audit_status") or "").strip().lower() in {"approved", "corrected"}
            for row in csv.DictReader(handle)
        )
    if not approved:
        raise ValueError(f"No approved audit rows in {batch_paths['audit']}.")

    base_index = AUDITED_INDEX if AUDITED_INDEX.exists() else BASE_INDEX
    run(
        "build_audited_training_index.py",
        "--base-index",
        base_index,
        "--dataset",
        batch_paths["dataset"],
        "--audit-csv",
        batch_paths["audit"],
        "--out",
        AUDITED_INDEX,
    )
    run(
        "train_phish360.py",
        "--data",
        AUDITED_INDEX,
        "--model",
        "all",
        "--html-max-chars",
        args.html_chars,
        "--max-html-len",
        args.html_chars,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short commands for the hybrid update pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)

    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("batch", nargs="?")
    collect_parser.add_argument("--count", type=int, default=100)
    collect_parser.set_defaults(handler=collect)

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("batch")
    evaluate_parser.set_defaults(handler=evaluate)

    retrain_parser = commands.add_parser("retrain")
    retrain_parser.add_argument("batch")
    retrain_parser.add_argument("--html-chars", type=int, default=5000)
    retrain_parser.set_defaults(handler=retrain)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
