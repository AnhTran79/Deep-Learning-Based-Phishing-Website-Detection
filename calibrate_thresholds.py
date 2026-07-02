from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_predictions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row["true_label"] = int(row["true_label"])
            row["phishing_probability"] = float(row["phishing_probability"])
            rows.append(row)
    if not rows:
        raise ValueError(f"No predictions found: {path}")
    return rows


def binary_metrics(labels: list[int], probabilities: list[float], threshold: float) -> dict[str, float | list[list[int]]]:
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    tn = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 0)
    fp = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 1)
    fn = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 0)
    tp = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def threshold_grid(step: float) -> list[float]:
    if step <= 0 or step > 0.5:
        raise ValueError("--step must be > 0 and <= 0.5")
    values = []
    threshold = 0.01
    while threshold < 1.0:
        values.append(round(threshold, 6))
        threshold += step
    if values[-1] != 0.99:
        values.append(0.99)
    return values


def score_row(row: dict, objective: str) -> float:
    if objective == "f1":
        return float(row["f1"])
    if objective == "accuracy":
        return float(row["accuracy"])
    if objective == "balanced_accuracy":
        return (float(row["recall"]) + float(row["specificity"])) / 2
    raise ValueError(f"Unsupported objective: {objective}")


def choose_threshold(
    labels: list[int],
    probabilities: list[float],
    objective: str,
    step: float,
    min_recall: float,
    max_false_positive_rate: float | None,
) -> dict:
    rows = [binary_metrics(labels, probabilities, threshold) for threshold in threshold_grid(step)]
    feasible = [
        row
        for row in rows
        if float(row["recall"]) >= min_recall
        and (max_false_positive_rate is None or float(row["false_positive_rate"]) <= max_false_positive_rate)
    ]
    constraints_satisfied = bool(feasible)
    if not feasible:
        feasible = rows
    chosen = max(
        feasible,
        key=lambda row: (
            score_row(row, objective),
            float(row["recall"]),
            -float(row["false_positive_rate"]),
            float(row["threshold"]),
        ),
    )
    chosen["constraints_satisfied"] = constraints_satisfied
    return chosen


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def calibrate(args: argparse.Namespace) -> dict:
    predictions_path = Path(args.predictions)
    predictions = read_predictions(predictions_path)
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        by_model[row["model"]].append(row)

    output: dict = {
        "threshold_source": str(predictions_path),
        "intended_use": "validation-calibrated threshold overrides; do not calibrate on final external test",
        "objective": args.objective,
        "step": args.step,
        "min_recall": args.min_recall,
        "max_false_positive_rate": args.max_false_positive_rate,
        "output": str(Path(args.out)),
        "models": {},
    }
    csv_rows: list[dict] = []
    for model, rows in sorted(by_model.items()):
        labels = [row["true_label"] for row in rows]
        probabilities = [row["phishing_probability"] for row in rows]
        chosen = choose_threshold(
            labels,
            probabilities,
            args.objective,
            args.step,
            args.min_recall,
            args.max_false_positive_rate,
        )
        output["models"][model] = chosen
        csv_rows.append({"model": model, **chosen})

    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(
        out_json.with_suffix(".csv"),
        csv_rows,
        [
            "model",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "false_positive_rate",
            "false_negative_rate",
            "confusion_matrix",
            "constraints_satisfied",
        ],
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate model thresholds from validation predictions.")
    parser.add_argument("--predictions", default="reports/results/external_validation/predictions.csv")
    parser.add_argument("--out", default="reports/results/external_validation/calibrated_thresholds.json")
    parser.add_argument("--objective", choices=["f1", "accuracy", "balanced_accuracy"], default="f1")
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--max-false-positive-rate", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    result = calibrate(parse_args())
    print(f"wrote {result['output']} -> {len(result['models'])} thresholds")
    for model, row in result["models"].items():
        constraint_text = "" if row["constraints_satisfied"] else " constraints_not_satisfied"
        print(
            f"{model}: threshold={row['threshold']:.2f} "
            f"f1={row['f1']:.4f} precision={row['precision']:.4f} recall={row['recall']:.4f} "
            f"fpr={row['false_positive_rate']:.4f}{constraint_text}"
        )


if __name__ == "__main__":
    main()
