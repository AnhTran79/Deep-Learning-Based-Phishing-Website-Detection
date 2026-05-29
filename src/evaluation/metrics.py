from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classification(
    labels: list[int],
    probabilities: list[float],
    threshold: float = 0.5,
) -> dict[str, float | list[list[int]]]:
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    roc_auc = math.nan if len(set(labels)) < 2 else roc_auc_score(labels, probabilities)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _serialize_value(value: Any) -> Any:
    if isinstance(value, list):
        return str(value)
    return value


def write_result_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "input",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "false_positive_rate",
        "false_negative_rate",
        "confusion_matrix",
        "model_file",
    ]
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        rows = [existing for existing in rows if existing.get("model") != row.get("model")]

    rows.append({name: _serialize_value(row.get(name, "")) for name in fieldnames})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_metric_figures(
    model_name: str,
    labels: list[int],
    probabilities: list[float],
    figures_dir: Path,
    threshold: float = 0.5,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]

    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["legitimate", "phishing"]).plot(values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"confusion_matrix_{model_name}.png")
    plt.close()

    if len(set(labels)) > 1:
        RocCurveDisplay.from_predictions(labels, probabilities)
        plt.title(f"ROC Curve - {model_name}")
        plt.tight_layout()
        plt.savefig(figures_dir / f"roc_curve_{model_name}.png")
        plt.close()
