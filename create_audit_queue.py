from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_predictions(path: Path, focus_model: str) -> dict[str, list[dict]]:
    by_sample: dict[str, list[dict]] = defaultdict(list)
    if not path.is_file():
        return by_sample
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if focus_model and row.get("model") != focus_model:
                continue
            row["correct"] = int(row["correct"])
            row["true_label"] = int(row["true_label"])
            row["predicted_label"] = int(row["predicted_label"])
            row["phishing_probability"] = float(row["phishing_probability"])
            by_sample[row["sample_id"]].append(row)
    return by_sample


def risk_flags(row: dict, predictions: list[dict]) -> list[str]:
    flags: list[str] = []
    label = int(row["label"])
    html_length = int(row.get("html_length") or 0)
    if row["original_url"] != row["final_url"]:
        flags.append("redirect")
    if html_length < 1000:
        flags.append("short_html")
    if label == 0 and any(not pred["correct"] for pred in predictions):
        flags.append("false_positive")
    if label == 1 and any(not pred["correct"] for pred in predictions):
        flags.append("false_negative")
    if any(abs(pred["phishing_probability"] - float(pred.get("threshold", 0.5))) <= 0.1 for pred in predictions):
        flags.append("near_threshold")
    return flags


def priority(flags: list[str]) -> int:
    if "false_negative" in flags:
        return 1
    if "false_positive" in flags:
        return 2
    if "near_threshold" in flags:
        return 3
    if flags:
        return 4
    return 5


def screenshot_path(dataset: Path, sample_id: str) -> str:
    base_id = sample_id.split("_", 1)[0]
    return str(dataset / sample_id / "SCREEN-SHOT" / f"{base_id}.png")


def create_queue(args: argparse.Namespace) -> list[dict]:
    dataset = Path(args.dataset)
    predictions = read_predictions(Path(args.predictions), args.focus_model)
    rows: list[dict] = []
    with (dataset / "collection_results.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_predictions = predictions.get(row["sample_id"], [])
            flags = risk_flags(row, sample_predictions)
            probability_text = "; ".join(
                f"{pred['model']}={pred['phishing_probability']:.4f}" for pred in sample_predictions
            )
            rows.append(
                {
                    "priority": priority(flags),
                    "sample_id": row["sample_id"],
                    "current_label": row["label"],
                    "suggested_action": "review" if flags else "spot_check",
                    "flags": ";".join(flags),
                    "domain": row["domain"],
                    "final_url": row["final_url"],
                    "html_length": row.get("html_length", ""),
                    "screenshot_path": screenshot_path(dataset, row["sample_id"]),
                    "model_probabilities": probability_text,
                    "audit_status": "",
                    "audited_label": "",
                    "notes": "",
                }
            )
    rows.sort(key=lambda item: (item["priority"], item["sample_id"]))
    if args.limit > 0:
        rows = rows[: args.limit]
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "priority",
        "sample_id",
        "current_label",
        "suggested_action",
        "flags",
        "domain",
        "final_url",
        "html_length",
        "screenshot_path",
        "model_probabilities",
        "audit_status",
        "audited_label",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a semi-automatic audit queue for collected external samples.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", default="reports/results/external_test/predictions.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--focus-model", default="phish360_tri_branch_cnn")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = create_queue(args)
    write_csv(Path(args.out), rows)
    print(f"wrote {args.out} rows={len(rows)}")


if __name__ == "__main__":
    main()
