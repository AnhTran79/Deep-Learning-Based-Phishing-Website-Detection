from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_predictions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row["true_label"] = int(row["true_label"])
            row["predicted_label"] = int(row["predicted_label"])
            row["correct"] = int(row["correct"])
            row["phishing_probability"] = float(row["phishing_probability"])
            row["threshold"] = float(row["threshold"])
            rows.append(row)
    if not rows:
        raise ValueError(f"No predictions found: {path}")
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def model_error_summary(predictions: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for model in sorted({row["model"] for row in predictions}):
        model_rows = [row for row in predictions if row["model"] == model]
        wrong = [row for row in model_rows if not row["correct"]]
        false_positives = [row for row in wrong if row["true_label"] == 0]
        false_negatives = [row for row in wrong if row["true_label"] == 1]
        near_threshold = [
            row
            for row in wrong
            if abs(row["phishing_probability"] - row["threshold"]) <= 0.1
        ]
        rows.append(
            {
                "model": model,
                "input": model_rows[0].get("input", ""),
                "samples": len(model_rows),
                "wrong": len(wrong),
                "false_positives_legit_as_phish": len(false_positives),
                "false_negatives_phish_as_legit": len(false_negatives),
                "near_threshold_wrong_pm_0_1": len(near_threshold),
                "false_positive_domains": "; ".join(
                    domain for domain, _ in Counter(row["domain"] for row in false_positives).most_common(8)
                ),
                "false_negative_domains": "; ".join(
                    domain for domain, _ in Counter(row["domain"] for row in false_negatives).most_common(8)
                ),
            }
        )
    return rows


def hard_sample_summary(predictions: list[dict]) -> list[dict]:
    wrong_by_sample: dict[str, list[dict]] = defaultdict(list)
    all_by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        all_by_sample[row["sample_id"]].append(row)
        if not row["correct"]:
            wrong_by_sample[row["sample_id"]].append(row)

    rows: list[dict] = []
    for sample_id, wrong_rows in sorted(
        wrong_by_sample.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        first = wrong_rows[0]
        all_rows = all_by_sample[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "true_label": first["true_label"],
                "wrong_models": len(wrong_rows),
                "total_models": len(all_rows),
                "domain": first.get("domain", ""),
                "source": first.get("source", ""),
                "wrong_model_names": "; ".join(row["model"] for row in wrong_rows),
                "wrong_probabilities": "; ".join(
                    f"{row['model']}={row['phishing_probability']:.4f}" for row in wrong_rows
                ),
                "url": first.get("url", ""),
            }
        )
    return rows


def source_domain_summary(predictions: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in predictions:
        grouped[(row["model"], row.get("source", ""), row["true_label"])].append(row)

    rows: list[dict] = []
    for (model, source, true_label), items in sorted(grouped.items()):
        wrong = [row for row in items if not row["correct"]]
        rows.append(
            {
                "model": model,
                "source": source,
                "true_label": true_label,
                "samples": len(items),
                "wrong": len(wrong),
                "error_rate": len(wrong) / len(items) if items else 0.0,
                "top_wrong_domains": "; ".join(
                    domain for domain, _ in Counter(row["domain"] for row in wrong).most_common(8)
                ),
            }
        )
    return rows


def analyze(args: argparse.Namespace) -> dict:
    results_dir = Path(args.results_dir)
    predictions = read_predictions(results_dir / "predictions.csv")
    output_dir = Path(args.out_dir) if args.out_dir else results_dir / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_rows = model_error_summary(predictions)
    hard_rows = hard_sample_summary(predictions)
    source_rows = source_domain_summary(predictions)

    write_csv(
        output_dir / "model_error_summary.csv",
        model_rows,
        [
            "model",
            "input",
            "samples",
            "wrong",
            "false_positives_legit_as_phish",
            "false_negatives_phish_as_legit",
            "near_threshold_wrong_pm_0_1",
            "false_positive_domains",
            "false_negative_domains",
        ],
    )
    write_csv(
        output_dir / "hard_samples.csv",
        hard_rows,
        [
            "sample_id",
            "true_label",
            "wrong_models",
            "total_models",
            "domain",
            "source",
            "wrong_model_names",
            "wrong_probabilities",
            "url",
        ],
    )
    write_csv(
        output_dir / "source_domain_error_summary.csv",
        source_rows,
        ["model", "source", "true_label", "samples", "wrong", "error_rate", "top_wrong_domains"],
    )

    summary = {
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "models": len({row["model"] for row in predictions}),
        "samples": len({row["sample_id"] for row in predictions}),
        "hardest_samples": hard_rows[:10],
        "model_error_summary": model_rows,
    }
    (output_dir / "error_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze external evaluation errors by model, sample, source, and domain.")
    parser.add_argument("--results-dir", default="reports/results/external_test")
    parser.add_argument("--out-dir", default="")
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(f"wrote {summary['output_dir']}")
    for row in summary["model_error_summary"]:
        print(
            f"{row['model']}: wrong={row['wrong']} "
            f"fp={row['false_positives_legit_as_phish']} "
            f"fn={row['false_negatives_phish_as_legit']}"
        )


if __name__ == "__main__":
    main()
