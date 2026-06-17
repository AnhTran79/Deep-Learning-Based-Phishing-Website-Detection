from __future__ import annotations

import argparse
import ast
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def compare(results_dir: Path, comparison_file: str = "model_comparison.csv") -> list[dict]:
    path = results_dir / comparison_file
    if not path.exists():
        raise FileNotFoundError(f"No model comparison file found: {path}")
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    rows.sort(key=lambda row: parse_float(row.get("f1", "nan")), reverse=True)
    return rows


def write_sorted_results(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export_metric_chart(rows: list[dict], output_path: Path, title: str = "Model Comparison") -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_names = [row["model"] for row in rows]
    x = range(len(model_names))
    width = 0.15

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(10, len(model_names) * 1.6), 6))
    for index, metric in enumerate(metrics):
        offsets = [value + (index - 2) * width for value in x]
        values = []
        for row in rows:
            value = parse_float(row.get(metric, "nan"))
            values.append(0.0 if math.isnan(value) else value)
        plt.bar(offsets, values, width=width, label=metric)

    plt.xticks(list(x), model_names, rotation=30, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _labels_from_confusion_matrix(value: str) -> tuple[list[int], list[int]]:
    matrix = ast.literal_eval(value)
    if (
        not isinstance(matrix, list)
        or len(matrix) != 2
        or any(not isinstance(row, list) or len(row) != 2 for row in matrix)
    ):
        raise ValueError(f"Invalid confusion matrix format: {value}")
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    labels = [0] * (tn + fp) + [1] * (fn + tp)
    predictions = [0] * tn + [1] * fp + [0] * fn + [1] * tp
    return labels, predictions


def export_classification_reports(
    rows: list[dict],
    results_dir: Path,
    summary_name: str = "classification_reports.csv",
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / summary_name
    fieldnames = ["model", "label", "precision", "recall", "f1-score", "support"]
    summary_rows: list[dict[str, str | float | int]] = []

    for row in rows:
        model_name = row["model"]
        if not row.get("confusion_matrix"):
            continue
        labels, predictions = _labels_from_confusion_matrix(row["confusion_matrix"])
        report_text = classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=["legitimate", "phishing"],
            digits=4,
            zero_division=0,
        )
        (results_dir / f"classification_report_{model_name}.txt").write_text(report_text, encoding="utf-8")

        report = classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=["legitimate", "phishing"],
            output_dict=True,
            zero_division=0,
        )
        for label in ["legitimate", "phishing", "macro avg", "weighted avg"]:
            metrics = report[label]
            summary_rows.append(
                {
                    "model": model_name,
                    "label": label,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1-score": metrics["f1-score"],
                    "support": int(metrics["support"]),
                }
            )
        summary_rows.append(
            {
                "model": model_name,
                "label": "accuracy",
                "precision": "",
                "recall": "",
                "f1-score": report["accuracy"],
                "support": len(labels),
            }
        )

    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print model comparison sorted by F1-score.")
    parser.add_argument("--results-dir", default="reports/results/mendeley")
    parser.add_argument("--figures-dir", default="reports/figures/mendeley")
    parser.add_argument("--comparison-file", default="model_comparison.csv")
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison_file = Path(args.comparison_file).name
    comparison_stem = Path(comparison_file).stem
    rows = compare(Path(args.results_dir), comparison_file=comparison_file)
    if not args.no_export:
        results_dir = Path(args.results_dir)
        if comparison_file == "model_comparison.csv":
            sorted_name = "model_comparison_sorted.csv"
            summary_name = "classification_reports.csv"
            chart_name = "model_comparison_metrics.png"
            chart_title = "Model Comparison"
        elif comparison_file == "phish360_model_comparison.csv":
            sorted_name = "phish360_model_comparison_sorted.csv"
            summary_name = "classification_reports.csv"
            chart_name = "phish360_model_comparison_metrics.png"
            chart_title = "Phish360 Model Comparison"
        else:
            sorted_name = f"{comparison_stem}_sorted.csv"
            summary_name = f"{comparison_stem}_classification_reports.csv"
            chart_name = f"{comparison_stem}_metrics.png"
            chart_title = comparison_stem.replace("_", " ").title()
        write_sorted_results(rows, results_dir / sorted_name)
        export_classification_reports(rows, results_dir, summary_name=summary_name)
        export_metric_chart(rows, Path(args.figures_dir) / chart_name, title=chart_title)
    for row in rows:
        print(
            f"{row['model']}: input={row['input']} "
            f"f1={row['f1']} recall={row['recall']} roc_auc={row['roc_auc']}"
        )


if __name__ == "__main__":
    main()
