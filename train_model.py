from __future__ import annotations

"""Train deep-learning phishing models from a raw URL dataset.

The expected dataset format is:

url,label
https://example.com,0
https://bad.example/login,1

Label convention:
- 0 = legitimate
- 1 = phishing

It trains:
- a character-level CNN directly on URL text;
- an MLPClassifier over URL lexical features plus optional live/cached HTML
  features.
"""

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import joblib
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.features import COMBINED_FEATURE_NAMES, combined_feature_dict, fetch_html, normalize_url
from app.model import CharCnnUrlClassifier, encode_url


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT.parent / "balanced_dataset.csv"

ARTIFACT_DIR = "artifacts"
DATASET_DIR = "dataset"
CHART_DIR = "chart"

URL_CNN_FILENAME = "url_cnn.pt"
TABULAR_MODEL_FILENAME = "deep_learning_model.joblib"
CLASSIFICATION_REPORT_FILENAME = "classification_report.txt"
DEEP_METRICS_FILENAME = "deep_learning_metrics.json"
MODEL_RESULTS_FILENAME = "model_results.csv"

UrlRow = tuple[str, int]
Metrics = dict[str, float]


class UrlDataset(Dataset):
    def __init__(self, rows: list[UrlRow]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        url, label = self.rows[index]
        token_ids = torch.tensor(encode_url(url), dtype=torch.long)
        target = torch.tensor(label, dtype=torch.float32)
        return token_ids, target


def load_url_rows(data_path: Path, max_rows: int = 0, random_state: int = 42) -> list[UrlRow]:
    with data_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"No header found in dataset: {data_path}")
        missing = {"url", "label"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

        rows: list[UrlRow] = []
        for raw in reader:
            url = (raw.get("url") or "").strip()
            label_text = (raw.get("label") or "").strip()
            if not url or label_text == "":
                continue
            label = int(label_text)
            if label not in {0, 1}:
                raise ValueError(f"Labels must be 0/1. Invalid label: {label}")
            rows.append((url, label))

    if max_rows and len(rows) > max_rows:
        rng = random.Random(random_state)
        rows = rng.sample(rows, max_rows)

    counts = Counter(label for _, label in rows)
    if len(counts) != 2:
        raise ValueError("Training data must contain both label 0 and label 1.")
    if min(counts.values()) < 2:
        raise ValueError("Each label needs at least two rows for stratified splitting.")
    return rows


def split_rows(
    rows: list[UrlRow],
    test_size: float,
    random_state: int,
) -> tuple[list[UrlRow], list[UrlRow]]:
    if test_size <= 0 or test_size >= 1:
        raise ValueError("--test-size must be positive and less than 1.")

    labels = [label for _, label in rows]
    train_rows, test_rows = train_test_split(
        rows,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return train_rows, test_rows


def write_url_rows(path: Path, rows: list[UrlRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["url", "label", "label_name"])
        writer.writeheader()
        for url, label in rows:
            writer.writerow(
                {
                    "url": url,
                    "label": label,
                    "label_name": "phishing" if label == 1 else "legitimate",
                }
            )


def write_dataset_files(
    dataset_dir: Path,
    rows: list[UrlRow],
    train_rows: list[UrlRow],
    test_rows: list[UrlRow],
    config: dict,
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_url_rows(dataset_dir / "train.csv", train_rows)
    write_url_rows(dataset_dir / "test.csv", test_rows)
    summary = {
        "total_rows": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "class_counts": {
            "all": _label_counts(rows),
            "train": _label_counts(train_rows),
            "test": _label_counts(test_rows),
        },
        "source_format": "url,label",
        "label_mapping": {"0": "legitimate", "1": "phishing"},
    }
    (dataset_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (dataset_dir / "configs.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def _label_counts(rows: list[UrlRow]) -> dict[str, int]:
    counts = Counter(label for _, label in rows)
    return {"legitimate": counts[0], "phishing": counts[1]}


def evaluate_probabilities(labels: list[int], probabilities: list[float], threshold: float = 0.5) -> Metrics:
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    roc_auc = math.nan
    if len(set(labels)) > 1:
        roc_auc = roc_auc_score(labels, probabilities)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc,
    }


def evaluate_url_cnn(model: nn.Module, rows: list[UrlRow], batch_size: int) -> tuple[Metrics, list[float], list[int]]:
    loader = DataLoader(UrlDataset(rows), batch_size=batch_size, shuffle=False)
    model.eval()
    probabilities: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for token_ids, targets in loader:
            logits = model(token_ids)
            probs = torch.sigmoid(logits).cpu().tolist()
            probabilities.extend(float(prob) for prob in probs)
            labels.extend(int(label) for label in targets.cpu().tolist())
    return evaluate_probabilities(labels, probabilities), probabilities, labels


def train_url_cnn_model(
    train_rows: list[UrlRow],
    test_rows: list[UrlRow],
    output_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    threshold: float,
) -> dict:
    model = CharCnnUrlClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(UrlDataset(train_rows), batch_size=batch_size, shuffle=True)

    history = []
    best_state = None
    best_epoch = 0
    best_test_metrics: Metrics | None = None
    best_test_probabilities: list[float] = []
    best_test_labels: list[int] = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for token_ids, labels in loader:
            optimizer.zero_grad()
            logits = model(token_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        test_metrics, test_probabilities, test_labels = evaluate_url_cnn(model, test_rows, batch_size=batch_size)
        test_metrics = evaluate_probabilities(test_labels, test_probabilities, threshold=threshold)
        row = {"epoch": epoch + 1, "loss": total_loss, **test_metrics}
        history.append(row)
        print(f"url_cnn epoch={epoch + 1} loss={total_loss:.4f} test_f1={test_metrics['f1']:.4f}")
        if best_test_metrics is None or test_metrics["f1"] > best_test_metrics["f1"]:
            best_epoch = epoch + 1
            best_test_metrics = test_metrics
            best_test_probabilities = test_probabilities
            best_test_labels = test_labels
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = best_test_metrics or {}
    test_probabilities = best_test_probabilities
    test_labels = best_test_labels
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "char_cnn_url",
            "max_len": 200,
            "vocab_size": 128,
            "label_mapping": {"0": "legitimate", "1": "phishing"},
            "threshold": threshold,
            "best_epoch": best_epoch,
            "best_test_metrics": test_metrics,
        },
        output_path,
    )
    return {
        "model_type": "char_cnn_url",
        "model_file": str(output_path),
        "threshold": threshold,
        "best_epoch": best_epoch,
        "history": history,
        "test_metrics": test_metrics,
        "test_probabilities": test_probabilities,
        "test_labels": test_labels,
    }


def _cache_path(cache_dir: Path, url: str) -> Path:
    digest = hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.html"


def read_or_fetch_html(url: str, cache_dir: Path | None, fetch_enabled: bool, timeout: float) -> str | None:
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = _cache_path(cache_dir, url)
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    if not fetch_enabled:
        return None
    html = fetch_html(url, timeout=timeout)
    if html and cache_dir is not None:
        _cache_path(cache_dir, url).write_text(html, encoding="utf-8", errors="replace")
    return html


def build_feature_matrix(
    rows: list[UrlRow],
    cache_dir: Path | None,
    fetch_enabled: bool,
    fetch_timeout: float,
    html_max_rows: int,
) -> tuple[list[list[float]], list[int], int]:
    x: list[list[float]] = []
    y: list[int] = []
    html_available = 0
    for index, (url, label) in enumerate(rows, start=1):
        can_fetch = fetch_enabled and (html_max_rows <= 0 or index <= html_max_rows)
        html = read_or_fetch_html(url, cache_dir=cache_dir, fetch_enabled=can_fetch, timeout=fetch_timeout)
        values = combined_feature_dict(url, html)
        html_available += int(values["html_available"])
        x.append([float(values[name]) for name in COMBINED_FEATURE_NAMES])
        y.append(label)
        if index % 1000 == 0:
            print(f"features built rows={index} html_available={html_available}")
    return x, y, html_available


def _scaled_pipeline(classifier) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


def train_tabular_models(
    train_rows: list[UrlRow],
    test_rows: list[UrlRow],
    output_path: Path,
    cache_dir: Path | None,
    fetch_enabled: bool,
    fetch_timeout: float,
    html_max_rows: int,
    max_iter: int,
    random_state: int,
    threshold: float,
) -> dict:
    x_train, y_train, train_html = build_feature_matrix(
        train_rows,
        cache_dir=cache_dir,
        fetch_enabled=fetch_enabled,
        fetch_timeout=fetch_timeout,
        html_max_rows=html_max_rows,
    )
    x_test, y_test, test_html = build_feature_matrix(
        test_rows,
        cache_dir=cache_dir,
        fetch_enabled=fetch_enabled,
        fetch_timeout=fetch_timeout,
        html_max_rows=html_max_rows,
    )

    candidates = [
        (
            "baseline_logistic_regression",
            "baseline",
            _scaled_pipeline(LogisticRegression(max_iter=max_iter, random_state=random_state)),
        ),
        (
            "baseline_gaussian_naive_bayes",
            "baseline",
            Pipeline([("classifier", GaussianNB())]),
        ),
        (
            "deep_learning_url_html_mlp",
            "deep_learning",
            _scaled_pipeline(
                MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.0005,
                    learning_rate_init=0.001,
                    early_stopping=True,
                    max_iter=max_iter,
                    random_state=random_state,
                )
            ),
        ),
    ]

    results = []
    best_model: Pipeline | None = None
    best_name = ""
    for name, family, model in candidates:
        model.fit(x_train, y_train)
        test_probabilities = model.predict_proba(x_test)[:, 1].tolist()
        test_metrics = evaluate_probabilities(y_test, test_probabilities, threshold=threshold)
        row = {
            "name": name,
            "family": family,
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_roc_auc": test_metrics["roc_auc"],
            "selected_for_runtime": False,
        }
        results.append(row)
        if family == "deep_learning":
            best_model = model
            best_name = name

    if best_model is None:
        raise RuntimeError("No tabular deep-learning model was trained.")

    for row in results:
        row["selected_for_runtime"] = row["name"] == best_name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_type": "sklearn_mlp_url_html_features",
            "model_name": best_name,
            "pipeline": best_model,
            "feature_names": COMBINED_FEATURE_NAMES,
            "class_names": {0: "legitimate", 1: "phishing"},
            "html_fetch_enabled_during_training": fetch_enabled,
            "html_available_rows": {
                "train": train_html,
                "test": test_html,
            },
        },
        output_path,
    )

    test_probabilities = best_model.predict_proba(x_test)[:, 1].tolist()
    test_predictions = [1 if probability >= threshold else 0 for probability in test_probabilities]
    return {
        "model_type": "sklearn_mlp_url_html_features",
        "best_model": best_name,
        "model_file": str(output_path),
        "threshold": threshold,
        "candidate_results": results,
        "test_probabilities": test_probabilities,
        "test_predictions": test_predictions,
        "test_labels": y_test,
        "confusion_matrix": confusion_matrix(y_test, test_predictions, labels=[0, 1]).tolist(),
        "html_available_rows": {
            "train": train_html,
            "test": test_html,
        },
    }


def write_model_results(path: Path, cnn_metrics: dict, tabular_metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    test_metrics = cnn_metrics["test_metrics"]
    rows = [
        {
            "name": "url_cnn_deep_learning",
            "family": "deep_learning",
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_roc_auc": test_metrics["roc_auc"],
            "selected_for_runtime": True,
        },
        *tabular_metrics["candidate_results"],
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "name",
                "family",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_roc_auc",
                "selected_for_runtime",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_charts(chart_dir: Path, model_results_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    rows = list(csv.DictReader(model_results_path.open("r", encoding="utf-8", newline="")))
    names = [row["name"] for row in rows]
    metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    chart_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5.5))
    width = 0.15
    x_positions = list(range(len(names)))
    for metric_index, metric in enumerate(metrics):
        offsets = [x + (metric_index - 2) * width for x in x_positions]
        values = [float(row[metric]) for row in rows]
        plt.bar(offsets, values, width=width, label=labels[metric_index])
    plt.ylim(0, 1)
    plt.title("Raw URL model comparison")
    plt.ylabel("Test score")
    plt.xticks(x_positions, names, rotation=20, ha="right")
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(chart_dir / "model_comparison.png")
    plt.close()

    plt.figure(figsize=(8, 4.5))
    f1_scores = [float(row["test_f1"]) for row in rows]
    plt.bar(names, f1_scores)
    plt.ylim(0, 1)
    plt.title("Raw URL model comparison - Test F1")
    plt.ylabel("Test F1")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(chart_dir / "model_comparison_f1.png")
    plt.close()


def write_artifacts(
    output_dir: Path,
    data_path: Path,
    rows: list[UrlRow],
    train_rows: list[UrlRow],
    test_rows: list[UrlRow],
    cnn_metrics: dict,
    tabular_metrics: dict,
    args: argparse.Namespace,
) -> None:
    report = classification_report(
        tabular_metrics["test_labels"],
        tabular_metrics["test_predictions"],
        labels=[0, 1],
        target_names=["legitimate", "phishing"],
        digits=4,
        zero_division=0,
    )
    report_text = (
        "TRAINED MODEL RESULTS\n"
        "Dataset: raw url,label dataset\n"
        "Runtime models: char CNN URL model + URL/HTML feature MLP\n\n"
        f"Decision threshold: {args.threshold}\n"
        f"URL CNN best epoch: {cnn_metrics.get('best_epoch')}\n\n"
        "URL CNN test metrics:\n"
        f"{json.dumps(cnn_metrics['test_metrics'], indent=2)}\n\n"
        "URL/HTML MLP classification report:\n"
        f"{report}\n"
    )
    (output_dir / CLASSIFICATION_REPORT_FILENAME).write_text(report_text, encoding="utf-8")

    model_results_path = output_dir / MODEL_RESULTS_FILENAME
    write_model_results(model_results_path, cnn_metrics, tabular_metrics)
    write_charts(Path(args.chart_dir), model_results_path)

    metrics = {
        "dataset": {
            "path": str(data_path),
            "total_rows": len(rows),
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "class_counts": _label_counts(rows),
        },
        "url_cnn": {
            key: value
            for key, value in cnn_metrics.items()
            if key not in {"test_probabilities", "test_labels"}
        },
        "url_html_mlp": {
            key: value
            for key, value in tabular_metrics.items()
            if key not in {"test_probabilities", "test_predictions", "test_labels"}
        },
        "feature_names": COMBINED_FEATURE_NAMES,
        "html_fetch": {
            "enabled": args.fetch_html,
            "cache_dir": args.html_cache_dir,
            "timeout": args.fetch_timeout,
            "max_rows_per_split": args.html_max_rows,
        },
        "threshold": args.threshold,
    }
    (output_dir / DEEP_METRICS_FILENAME).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    artifact = {
        "artifact_version": "2.0",
        "model_type": "raw_url_deep_learning",
        "generated_by": "train_model.py",
        "source_data": str(data_path),
        "label_mapping": {"0": "legitimate", "1": "phishing"},
        "runtime_models": {
            "url_cnn": str(output_dir / URL_CNN_FILENAME),
            "url_html_mlp": str(output_dir / TABULAR_MODEL_FILENAME),
        },
        "training_config": {
            "epochs": args.epochs,
            "threshold": args.threshold,
            "url_cnn_best_epoch": cnn_metrics.get("best_epoch"),
        },
        "training_summary": metrics["dataset"],
        "note": (
            "This artifact is trained from raw URL strings. HTML features are learned "
            "when --fetch-html is enabled or when cached HTML files are present."
        ),
    }
    (output_dir / "best_model.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    metadata = {
        "project": "Phishing Website Detection Demo",
        "generated_by": "train_model.py",
        "artifact_version": "2.0",
        "training_file": str(data_path),
        "model_type": artifact["model_type"],
        "url_cnn_model_file": str(output_dir / URL_CNN_FILENAME),
        "url_html_model_file": str(output_dir / TABULAR_MODEL_FILENAME),
        "classification_report_file": str(output_dir / CLASSIFICATION_REPORT_FILENAME),
        "model_results_file": str(output_dir / MODEL_RESULTS_FILENAME),
        "label_mapping": artifact["label_mapping"],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to raw url,label CSV.")
    parser.add_argument("--out-dir", default=ARTIFACT_DIR)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--chart-dir", default=CHART_DIR)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional sampled row limit for quick experiments.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=150, help="Max iterations for sklearn MLP.")
    parser.add_argument("--threshold", type=float, default=0.4, help="Probability threshold for phishing metrics.")
    parser.add_argument("--fetch-html", action="store_true", help="Fetch live HTML while building URL/HTML features.")
    parser.add_argument("--html-cache-dir", default="dataset/html_cache")
    parser.add_argument("--fetch-timeout", type=float, default=4.0)
    parser.add_argument(
        "--html-max-rows",
        type=int,
        default=0,
        help="Maximum rows per split to fetch HTML for. 0 means no limit when --fetch-html is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    output_dir = Path(args.out_dir)
    dataset_dir = Path(args.dataset_dir)
    cache_dir = Path(args.html_cache_dir) if args.html_cache_dir else None

    rows = load_url_rows(data_path, max_rows=args.max_rows, random_state=args.random_state)
    train_rows, test_rows = split_rows(
        rows,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    split_config = {
        "source_data": str(data_path),
        "max_rows": args.max_rows,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "html_fetch_enabled": args.fetch_html,
        "html_cache_dir": args.html_cache_dir,
    }
    write_dataset_files(dataset_dir, rows, train_rows, test_rows, split_config)

    output_dir.mkdir(parents=True, exist_ok=True)
    cnn_metrics = train_url_cnn_model(
        train_rows=train_rows,
        test_rows=test_rows,
        output_path=output_dir / URL_CNN_FILENAME,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        threshold=args.threshold,
    )
    tabular_metrics = train_tabular_models(
        train_rows=train_rows,
        test_rows=test_rows,
        output_path=output_dir / TABULAR_MODEL_FILENAME,
        cache_dir=cache_dir,
        fetch_enabled=args.fetch_html,
        fetch_timeout=args.fetch_timeout,
        html_max_rows=args.html_max_rows,
        max_iter=args.max_iter,
        random_state=args.random_state,
        threshold=args.threshold,
    )
    write_artifacts(
        output_dir=output_dir,
        data_path=data_path,
        rows=rows,
        train_rows=train_rows,
        test_rows=test_rows,
        cnn_metrics=cnn_metrics,
        tabular_metrics=tabular_metrics,
        args=args,
    )

    print(f"wrote {output_dir / URL_CNN_FILENAME}")
    print(f"wrote {output_dir / TABULAR_MODEL_FILENAME}")
    print(f"wrote {output_dir / 'best_model.json'}")
    print(f"url_cnn_test_f1={cnn_metrics['test_metrics']['f1']:.4f}")


if __name__ == "__main__":
    main()
