from __future__ import annotations

"""Train phishing-detection models and write evaluation artifacts.

Default mode trains the Kaggle feature-based models used by the web demo:
- 2 baseline models for comparison
- 2 MLPClassifier neural networks for deep-learning comparison
- the best deep-learning model saved as deep_learning_model.joblib
- a Naive Bayes fallback saved inside best_model.json
- classification_report.txt for presentation/reporting

The optional `--task url-cnn` mode trains a character-level CNN from a CSV with
columns `url,label`.
"""

import argparse
import csv
import json
import math
import zipfile
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

from app.features import HTML_FEATURE_NAMES, KAGGLE_FEATURE_NAMES, extract_html_features
from app.model import CharCnnUrlClassifier


FeatureRow = tuple[str, dict[str, int]]
Metrics = dict[str, float]
ModelCandidate = dict[str, str | Pipeline]

PHISHING_LABEL = "-1"
LEGITIMATE_LABEL = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "url_dataset.csv"

DEEP_MODEL_FILENAME = "deep_learning_model.joblib"
CLASSIFICATION_REPORT_FILENAME = "classification_report.txt"
DEEP_METRICS_FILENAME = "deep_learning_metrics.json"
MODEL_RESULTS_FILENAME = "model_results.csv"

URL_CNN_FILENAME = "url_cnn.pt"
URL_CNN_MAX_LEN = 200
URL_CNN_VOCAB_SIZE = 128


# ---------------------------------------------------------------------------
# Character-level CNN training for raw URL datasets.
# ---------------------------------------------------------------------------


def encode_url(url: str) -> list[int]:
    ids = [min(ord(char), URL_CNN_VOCAB_SIZE - 1) for char in url[:URL_CNN_MAX_LEN]]
    return ids + [0] * (URL_CNN_MAX_LEN - len(ids))


class UrlDataset(Dataset):
    def __init__(self, urls: list[str], labels: list[int]) -> None:
        self.urls = urls
        self.labels = labels

    def __len__(self) -> int:
        return len(self.urls)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        url_ids = torch.tensor(encode_url(self.urls[index]), dtype=torch.long)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return url_ids, label


def load_url_label_rows(data_path: Path) -> list[tuple[str, int]]:
    rows = list(csv.DictReader(data_path.open("r", encoding="utf-8-sig", newline="")))
    if not rows:
        raise ValueError(f"No rows found in dataset: {data_path}")

    fieldnames = set(rows[0].keys())
    missing_columns = {"url", "label"} - fieldnames
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing_columns))}")

    parsed_rows: list[tuple[str, int]] = []
    for row in rows:
        url = (row.get("url") or "").strip()
        label_text = (row.get("label") or "").strip()
        if not url or not label_text:
            continue
        label = int(label_text)
        if label not in {0, 1}:
            raise ValueError(f"Labels must be binary 0/1. Invalid label: {label}")
        parsed_rows.append((url, label))

    if not parsed_rows:
        raise ValueError("CSV has no usable rows after dropping empty url/label values.")

    label_counts = Counter(label for _, label in parsed_rows)
    if len(label_counts) < 2:
        raise ValueError("Training data must contain both 0 and 1 labels.")
    if min(label_counts.values()) < 2:
        raise ValueError("Each label needs at least two rows for a stratified train/test split.")
    return parsed_rows


def evaluate_url_cnn(model: nn.Module, loader: DataLoader) -> Metrics:
    model.eval()
    probabilities: list[float] = []
    labels: list[int] = []

    with torch.no_grad():
        for token_ids, batch_labels in loader:
            logits = model(token_ids)
            probs = torch.sigmoid(logits).cpu().tolist()
            probabilities.extend(probs)
            labels.extend(batch_labels.int().cpu().tolist())

    predictions = [1 if prob >= 0.5 else 0 for prob in probabilities]
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


def train_url_cnn_model(
    data_path: Path,
    output_path: Path,
    epochs: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Metrics:
    rows = load_url_label_rows(data_path)
    train_rows, test_rows = train_test_split(
        rows,
        test_size=test_size,
        random_state=random_state,
        stratify=[label for _, label in rows],
    )

    train_loader = DataLoader(
        UrlDataset([url for url, _ in train_rows], [label for _, label in train_rows]),
        batch_size=64,
        shuffle=True,
    )
    test_loader = DataLoader(
        UrlDataset([url for url, _ in test_rows], [label for _, label in test_rows]),
        batch_size=64,
    )

    model = CharCnnUrlClassifier(vocab_size=URL_CNN_VOCAB_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    metrics: dict[str, float] = {}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for token_ids, labels in train_loader:
            optimizer.zero_grad()
            logits = model(token_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        metrics = evaluate_url_cnn(model, test_loader)
        print(f"epoch={epoch + 1} loss={total_loss:.4f} metrics={metrics}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    return metrics


# ---------------------------------------------------------------------------
# Categorical Naive Bayes fallback for pre-extracted feature datasets.
# ---------------------------------------------------------------------------


def _train_categorical_model(rows: list[FeatureRow], feature_names: list[str]) -> dict:
    label_counts = Counter(label for label, _ in rows)
    labels = (PHISHING_LABEL, LEGITIMATE_LABEL)
    missing_labels = [label for label in labels if label_counts[label] == 0]
    if missing_labels:
        readable = ", ".join(missing_labels)
        raise ValueError(f"Training data must include at least one row for each label. Missing: {readable}")

    observed_values = {
        name: sorted({str(features[name]) for _, features in rows if name in features})
        for name in feature_names
    }
    value_counts = {
        label: {name: Counter() for name in feature_names}
        for label in labels
    }

    for label, features in rows:
        if label not in value_counts:
            continue
        for name in feature_names:
            value_counts[label][name][str(features[name])] += 1

    total = sum(label_counts[label] for label in labels)
    priors = {
        label: math.log(label_counts[label] / total)
        for label in labels
    }
    likelihoods = {label: {} for label in labels}

    for label in labels:
        label_total = label_counts[label]
        for name in feature_names:
            likelihoods[label][name] = {}
            values = observed_values[name] or ["0"]
            denominator = label_total + len(values) + 1
            for value in values:
                count = value_counts[label][name][value]
                likelihoods[label][name][value] = math.log((count + 1) / denominator)
            likelihoods[label][name]["__UNK__"] = math.log(1 / denominator)

    return {
        "model_type": "categorical_naive_bayes",
        "feature_names": feature_names,
        "label_mapping": {"-1": "phishing", "1": "legitimate"},
        "class_counts": {
            "phishing": label_counts[PHISHING_LABEL],
            "legitimate": label_counts[LEGITIMATE_LABEL],
        },
        "priors": priors,
        "likelihoods": likelihoods,
    }


def train_url_model(data_path: Path) -> dict:
    model_rows = load_kaggle_feature_rows(data_path)
    artifact = _train_categorical_model(model_rows, KAGGLE_FEATURE_NAMES)
    artifact.update({
        "source": "Kaggle: akashkr/phishing-website-dataset",
    })
    return artifact


def load_kaggle_feature_rows(data_path: Path) -> list[FeatureRow]:
    rows = list(csv.DictReader(data_path.open("r", encoding="utf-8-sig", newline="")))
    if not rows:
        raise ValueError(f"No rows found in dataset: {data_path}")

    fieldnames = set(rows[0].keys())
    required_columns = set(KAGGLE_FEATURE_NAMES) | {"Result"}
    missing_columns = sorted(required_columns - fieldnames)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    model_rows = [
        (row["Result"], {name: int(row[name]) for name in KAGGLE_FEATURE_NAMES})
        for row in rows
        if row.get("Result") in (PHISHING_LABEL, LEGITIMATE_LABEL)
    ]
    return model_rows


def build_kaggle_feature_matrix(rows: list[FeatureRow]) -> tuple[list[list[int]], list[int]]:
    x = [[features[name] for name in KAGGLE_FEATURE_NAMES] for _, features in rows]
    y = [1 if label == PHISHING_LABEL else 0 for label, _ in rows]
    return x, y


def label_to_name(label: str | int) -> str:
    return "phishing" if label == PHISHING_LABEL or label == 1 else "legitimate"


def write_feature_rows(path: Path, rows: list[FeatureRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["label", "label_name", *KAGGLE_FEATURE_NAMES])
        writer.writeheader()
        for label, features in rows:
            writer.writerow({
                "label": 1 if label == PHISHING_LABEL else 0,
                "label_name": label_to_name(label),
                **{name: features[name] for name in KAGGLE_FEATURE_NAMES},
            })


def split_kaggle_rows(
    rows: list[FeatureRow],
    valid_size: float,
    test_size: float,
    random_state: int,
) -> tuple[list[FeatureRow], list[FeatureRow], list[FeatureRow]]:
    if valid_size <= 0 or test_size <= 0 or valid_size + test_size >= 1:
        raise ValueError("--valid-size and --test-size must be positive and sum to less than 1.")

    labels = [label for label, _ in rows]
    train_rows, temp_rows = train_test_split(
        rows,
        test_size=valid_size + test_size,
        random_state=random_state,
        stratify=labels,
    )
    temp_labels = [label for label, _ in temp_rows]
    relative_test_size = test_size / (valid_size + test_size)
    valid_rows, test_rows = train_test_split(
        temp_rows,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=temp_labels,
    )
    return train_rows, valid_rows, test_rows


def write_dataset_files(
    dataset_dir: Path,
    rows: list[FeatureRow],
    train_rows: list[FeatureRow],
    valid_rows: list[FeatureRow],
    test_rows: list[FeatureRow],
    config: dict,
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_feature_rows(dataset_dir / "features.csv", rows)
    write_feature_rows(dataset_dir / "train.csv", train_rows)
    write_feature_rows(dataset_dir / "valid.csv", valid_rows)
    write_feature_rows(dataset_dir / "test.csv", test_rows)

    summary = {
        "total_rows": len(rows),
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "test_rows": len(test_rows),
        "class_counts": {
            "all": Counter(label_to_name(label) for label, _ in rows),
            "train": Counter(label_to_name(label) for label, _ in train_rows),
            "valid": Counter(label_to_name(label) for label, _ in valid_rows),
            "test": Counter(label_to_name(label) for label, _ in test_rows),
        },
        "feature_count": len(KAGGLE_FEATURE_NAMES),
        "features": KAGGLE_FEATURE_NAMES,
    }
    (dataset_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (dataset_dir / "configs.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Model comparison for Kaggle feature datasets.
# ---------------------------------------------------------------------------


def _scaled_pipeline(classifier) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def build_model_candidates(max_iter: int, random_state: int) -> list[ModelCandidate]:
    return [
        {
            "name": "baseline_logistic_regression",
            "family": "baseline",
            "model": _scaled_pipeline(
                LogisticRegression(max_iter=max_iter, random_state=random_state)
            ),
        },
        {
            "name": "baseline_gaussian_naive_bayes",
            "family": "baseline",
            "model": Pipeline([("classifier", GaussianNB())]),
        },
        {
            "name": "deep_learning_mlp_small",
            "family": "deep_learning",
            "model": _scaled_pipeline(
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.0005,
                    learning_rate_init=0.001,
                    early_stopping=True,
                    max_iter=max_iter,
                    random_state=random_state,
                )
            ),
        },
        {
            "name": "deep_learning_mlp_deep",
            "family": "deep_learning",
            "model": _scaled_pipeline(
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
        },
    ]


def write_model_results(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "name",
                "family",
                "valid_accuracy",
                "valid_f1",
                "test_accuracy",
                "test_f1",
                "selected_for_runtime",
            ],
        )
        writer.writeheader()
        writer.writerows(results)


def write_evaluation_charts(
    chart_dir: Path,
    artifacts_dir: Path,
    model_results: list[dict],
    confusion_values: list[list[int]],
    test_labels: list[int],
    test_probabilities: list[float],
    best_model: Pipeline,
    x_test: list[list[int]],
) -> None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.inspection import permutation_importance
    except ModuleNotFoundError:
        return

    chart_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    names = [row["name"] for row in model_results]
    valid_scores = [float(row["valid_f1"]) for row in model_results]
    test_scores = [float(row["test_f1"]) for row in model_results]

    plt.figure(figsize=(8, 4.5))
    x_positions = range(len(names))
    plt.bar([x - 0.2 for x in x_positions], valid_scores, width=0.4, label="valid f1")
    plt.bar([x + 0.2 for x in x_positions], test_scores, width=0.4, label="test f1")
    plt.xticks(list(x_positions), names)
    plt.ylim(0, 1)
    plt.title("Model comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_dir / "model_comparison.png")
    plt.savefig(chart_dir / "model_comparison_fixed.png")
    plt.close()

    plt.figure(figsize=(5, 4.5))
    plt.imshow(confusion_values, cmap="Blues")
    plt.title("Confusion matrix")
    plt.xticks([0, 1], ["legitimate", "phishing"])
    plt.yticks([0, 1], ["legitimate", "phishing"])
    for row_index, row in enumerate(confusion_values):
        for col_index, value in enumerate(row):
            plt.text(col_index, row_index, str(value), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(chart_dir / "confusion_matrix.png")
    plt.close()

    metrics = ["accuracy", "f1"]
    best_row = next(row for row in model_results if row["selected_for_runtime"])
    metric_values = [float(best_row["test_accuracy"]), float(best_row["test_f1"])]
    plt.figure(figsize=(5, 4))
    plt.bar(metrics, metric_values, color=["#2f7666", "#24594f"])
    plt.ylim(0, 1)
    plt.title("Classification metrics")
    plt.tight_layout()
    plt.savefig(chart_dir / "cls_metrics.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(
        [score for score, label in zip(test_probabilities, test_labels) if label == 0],
        bins=20,
        alpha=0.65,
        label="legitimate",
    )
    plt.hist(
        [score for score, label in zip(test_probabilities, test_labels) if label == 1],
        bins=20,
        alpha=0.65,
        label="phishing",
    )
    plt.title("Prediction score distribution")
    plt.xlabel("Phishing probability")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts_dir / "score_distribution.png")
    plt.close()

    importance = permutation_importance(
        best_model,
        x_test,
        test_labels,
        n_repeats=5,
        random_state=42,
        scoring="f1",
    )
    top_indices = importance.importances_mean.argsort()[-12:]
    plt.figure(figsize=(8, 5))
    plt.barh(
        [KAGGLE_FEATURE_NAMES[index] for index in top_indices],
        [importance.importances_mean[index] for index in top_indices],
    )
    plt.title("Classification feature importance")
    plt.tight_layout()
    plt.savefig(chart_dir / "cls_feature_importance.png")
    plt.close()


def train_deep_learning_url_model(
    data_path: Path,
    output_dir: Path,
    dataset_dir: Path,
    chart_dir: Path,
    valid_size: float = 0.15,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 300,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_kaggle_feature_rows(data_path)
    labels = [label for label, _ in rows]
    if len(set(labels)) < 2:
        raise ValueError("Deep learning training requires both phishing and legitimate rows.")
    class_counts = Counter(labels)
    if min(class_counts.values()) < 2:
        raise ValueError("Each class needs at least two rows for a stratified train/test split.")

    train_rows, valid_rows, test_rows = split_kaggle_rows(
        rows,
        valid_size=valid_size,
        test_size=test_size,
        random_state=random_state,
    )
    split_config = {
        "source_data": str(data_path),
        "task": "phishing website classification",
        "train_ratio": round(len(train_rows) / len(rows), 4),
        "valid_ratio": round(len(valid_rows) / len(rows), 4),
        "test_ratio": round(len(test_rows) / len(rows), 4),
        "valid_size": valid_size,
        "test_size": test_size,
        "random_state": random_state,
    }
    write_dataset_files(dataset_dir, rows, train_rows, valid_rows, test_rows, split_config)

    x_train, y_train = build_kaggle_feature_matrix(train_rows)
    x_valid, y_valid = build_kaggle_feature_matrix(valid_rows)
    x_test, y_test = build_kaggle_feature_matrix(test_rows)

    results = []
    best_deep_learning_name = ""
    best_deep_learning_model: Pipeline | None = None
    best_deep_learning_score = -1.0

    for candidate in build_model_candidates(max_iter=max_iter, random_state=random_state):
        name = str(candidate["name"])
        family = str(candidate["family"])
        model = candidate["model"]
        if not isinstance(model, Pipeline):
            raise TypeError(f"Model candidate must be a sklearn Pipeline: {name}")

        model.fit(x_train, y_train)
        valid_predictions = model.predict(x_valid).tolist()
        test_predictions = model.predict(x_test).tolist()
        valid_f1 = f1_score(y_valid, valid_predictions, pos_label=1, zero_division=0)
        test_f1 = f1_score(y_test, test_predictions, pos_label=1, zero_division=0)
        row = {
            "name": name,
            "family": family,
            "valid_accuracy": accuracy_score(y_valid, valid_predictions),
            "valid_f1": valid_f1,
            "test_accuracy": accuracy_score(y_test, test_predictions),
            "test_f1": test_f1,
            "selected_for_runtime": False,
        }
        results.append(row)
        if family == "deep_learning" and valid_f1 > best_deep_learning_score:
            best_deep_learning_name = name
            best_deep_learning_model = model
            best_deep_learning_score = valid_f1

    if best_deep_learning_model is None:
        raise RuntimeError("No deep learning model was trained.")

    best_predictions = best_deep_learning_model.predict(x_test).tolist()
    best_probabilities = best_deep_learning_model.predict_proba(x_test)[:, 1].tolist()
    for row in results:
        row["selected_for_runtime"] = row["name"] == best_deep_learning_name

    report_body = classification_report(
        y_test,
        best_predictions,
        labels=[0, 1],
        target_names=["legitimate", "phishing"],
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, best_predictions, labels=[0, 1]).tolist()
    report_text = (
        "TRAINED MODEL RESULTS\n"
        f"Runtime deep learning model: {best_deep_learning_name}\n"
        "Task: phishing website detection\n\n"
        f"{report_body}\n"
    )

    model_path = output_dir / DEEP_MODEL_FILENAME
    report_path = output_dir / CLASSIFICATION_REPORT_FILENAME
    metrics_path = output_dir / DEEP_METRICS_FILENAME
    results_path = output_dir / MODEL_RESULTS_FILENAME
    joblib.dump(
        {
            "model_type": "sklearn_mlp_deep_learning",
            "model_name": best_deep_learning_name,
            "pipeline": best_deep_learning_model,
            "feature_names": KAGGLE_FEATURE_NAMES,
            "positive_class": 1,
            "class_names": {0: "legitimate", 1: "phishing"},
        },
        model_path,
    )
    report_path.write_text(report_text, encoding="utf-8")
    write_model_results(results_path, results)
    write_evaluation_charts(
        chart_dir=chart_dir,
        artifacts_dir=output_dir,
        model_results=results,
        confusion_values=matrix,
        test_labels=y_test,
        test_probabilities=best_probabilities,
        best_model=best_deep_learning_model,
        x_test=x_test,
    )

    metrics = {
        "model_type": "sklearn_mlp_deep_learning",
        "best_model": best_deep_learning_name,
        "model_file": str(model_path),
        "classification_report_file": str(report_path),
        "model_results_file": str(results_path),
        "dataset_dir": str(dataset_dir),
        "chart_dir": str(chart_dir),
        "test_size": test_size,
        "valid_size": valid_size,
        "random_state": random_state,
        "train_rows": len(y_train),
        "valid_rows": len(y_valid),
        "test_rows": len(y_test),
        "feature_names": KAGGLE_FEATURE_NAMES,
        "candidate_results": results,
        "confusion_matrix": {
            "labels": ["legitimate", "phishing"],
            "values": matrix,
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# Optional HTML feature model.
# ---------------------------------------------------------------------------


def _label_from_zip_path(path: str) -> str | None:
    parts = [part for part in path.replace("\\", "/").lower().split("/") if part]
    if "notphish" in parts:
        return LEGITIMATE_LABEL
    if "phish" in parts:
        return PHISHING_LABEL
    return None


def train_html_model(archive_path: Path) -> dict:
    rows: list[FeatureRow] = []
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            if entry.is_dir() or not entry.filename.lower().endswith((".html", ".htm")):
                continue
            label = _label_from_zip_path(entry.filename)
            if label is None:
                continue
            with archive.open(entry) as file:
                html = file.read().decode("utf-8", errors="ignore")
            features = extract_html_features(html, source_name=entry.filename).to_dict()
            rows.append((label, features))

    if not rows:
        raise ValueError(
            "No labelled HTML files found in archive. Expected files under Phish/ and NotPhish/ folders."
        )

    artifact = _train_categorical_model(rows, HTML_FEATURE_NAMES)
    artifact.update({
        "source": "Local HTML archive: phishing and non-phishing HTML snapshots",
    })
    return artifact


# ---------------------------------------------------------------------------
# Artifact writing and command-line entrypoint.
# ---------------------------------------------------------------------------


def train(data_path: Path, html_archive_path: Path | None = None, deep_learning_metrics: dict | None = None) -> dict:
    url_model = train_url_model(data_path)
    artifact = {
        "artifact_version": "1.2",
        "model_type": "url_html_ensemble",
        "generated_by": "train_model.py",
        "source": "Kaggle URL-feature dataset" + (" + local HTML archive" if html_archive_path else ""),
        "label_mapping": {"-1": "phishing", "1": "legitimate"},
        "url_model": url_model,
    }
    if html_archive_path is not None:
        artifact["html_model"] = train_html_model(html_archive_path)
    artifact["training_summary"] = {
        "url_dataset": {
            "path": str(data_path),
            "class_counts": artifact["url_model"]["class_counts"],
            "feature_count": len(artifact["url_model"]["feature_names"]),
        },
        "html_dataset": {
            "path": str(html_archive_path) if html_archive_path else None,
            "class_counts": artifact.get("html_model", {}).get("class_counts"),
            "feature_count": len(artifact.get("html_model", {}).get("feature_names", [])),
        },
        "estimator": "categorical Naive Bayes with Laplace smoothing",
        "note": "Numeric probabilities are generated from training data by this script, not manually entered.",
    }
    if deep_learning_metrics is not None:
        artifact["deep_learning_model"] = {
            "model_type": deep_learning_metrics["model_type"],
            "best_model": deep_learning_metrics["best_model"],
            "model_file": deep_learning_metrics["model_file"],
            "classification_report_file": deep_learning_metrics["classification_report_file"],
        }
        artifact["training_summary"]["deep_learning"] = {
            "best_model": deep_learning_metrics["best_model"],
            "train_rows": deep_learning_metrics["train_rows"],
            "valid_rows": deep_learning_metrics["valid_rows"],
            "test_rows": deep_learning_metrics["test_rows"],
            "classification_report_file": deep_learning_metrics["classification_report_file"],
            "model_results_file": deep_learning_metrics["model_results_file"],
            "dataset_dir": deep_learning_metrics["dataset_dir"],
            "chart_dir": deep_learning_metrics["chart_dir"],
            "estimator": "scikit-learn MLPClassifier neural network",
        }
    return artifact


def write_metadata(artifact: dict, output_dir: Path, data_path: Path, html_archive_path: Path | None) -> None:
    metadata = {
        "project": "Phishing Website Detection Demo",
        "generated_by": "train_model.py",
        "artifact_version": artifact.get("artifact_version"),
        "dataset_source": [
            "https://www.kaggle.com/datasets/akashkr/phishing-website-dataset",
        ],
        "training_file": str(data_path),
        "html_archive": str(html_archive_path) if html_archive_path else None,
        "model_file": str(output_dir / "best_model.json"),
        "deep_learning_model_file": str(output_dir / DEEP_MODEL_FILENAME)
        if artifact.get("deep_learning_model")
        else None,
        "classification_report_file": str(output_dir / CLASSIFICATION_REPORT_FILENAME)
        if artifact.get("deep_learning_model")
        else None,
        "model_results_file": str(output_dir / MODEL_RESULTS_FILENAME)
        if artifact.get("deep_learning_model")
        else None,
        "model_type": artifact["model_type"],
        "url_class_counts": artifact["url_model"]["class_counts"],
        "html_class_counts": artifact.get("html_model", {}).get("class_counts"),
        "note": (
            "The URL dataset contains pre-extracted website features. The optional HTML "
            "archive trains a second content-based model from local HTML snapshots. "
            "Runtime prediction fetches HTML from the submitted URL when possible. "
            "Runtime prediction prefers the trained MLP deep learning model when present, "
            "then falls back to the Naive Bayes URL/HTML ensemble and heuristic rules."
        ),
    }
    if html_archive_path:
        metadata["dataset_source"].append("Local archive.zip HTML snapshots")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["kaggle", "url-cnn"], default="kaggle")
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
        help=f"Path to dataset CSV. Default: {DEFAULT_DATA_PATH}",
    )
    parser.add_argument("--html-archive", help="Optional archive.zip with training/validation Phish/NotPhish HTML files")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--chart-dir", default="chart")
    parser.add_argument("--out", help="Output path for --task url-cnn")
    parser.add_argument("--epochs", type=int, default=5, help="Epoch count for --task url-cnn")
    parser.add_argument("--skip-deep-learning", action="store_true", help="Only train the legacy Naive Bayes artifact")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--valid-size", type=float, default=0.15)
    parser.add_argument("--max-iter", type=int, default=300, help="Max epochs/iterations for each MLPClassifier candidate")
    return parser.parse_args()


def run_url_cnn_task(args: argparse.Namespace, data_path: Path, output_dir: Path) -> None:
    output_path = Path(args.out) if args.out else output_dir / URL_CNN_FILENAME
    metrics = train_url_cnn_model(
        data_path=data_path,
        output_path=output_path,
        epochs=args.epochs,
        test_size=args.test_size,
    )
    print(f"wrote {output_path}")
    print(f"url_cnn_metrics={metrics}")


def run_kaggle_task(args: argparse.Namespace, data_path: Path, output_dir: Path) -> None:
    html_archive_path = Path(args.html_archive) if args.html_archive else None
    dataset_dir = Path(args.dataset_dir)
    chart_dir = Path(args.chart_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    deep_learning_metrics = None
    if not args.skip_deep_learning:
        deep_learning_metrics = train_deep_learning_url_model(
            data_path=data_path,
            output_dir=output_dir,
            dataset_dir=dataset_dir,
            chart_dir=chart_dir,
            valid_size=args.valid_size,
            test_size=args.test_size,
            max_iter=args.max_iter,
        )

    artifact = train(data_path, html_archive_path, deep_learning_metrics)
    (output_dir / "best_model.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    write_metadata(artifact, output_dir, data_path, html_archive_path)

    print(f"wrote {output_dir / 'best_model.json'}")
    print(f"wrote {output_dir / 'metadata.json'}")
    if deep_learning_metrics is not None:
        print(f"wrote {output_dir / DEEP_MODEL_FILENAME}")
        print(f"wrote {output_dir / CLASSIFICATION_REPORT_FILENAME}")
        print(f"wrote {output_dir / MODEL_RESULTS_FILENAME}")
        print(f"wrote {dataset_dir / 'train.csv'}")
        print(f"wrote {dataset_dir / 'valid.csv'}")
        print(f"wrote {dataset_dir / 'test.csv'}")
        print(f"best_deep_learning_model={deep_learning_metrics['best_model']}")
    print(f"url_class_counts={artifact['url_model']['class_counts']}")
    if "html_model" in artifact:
        print(f"html_class_counts={artifact['html_model']['class_counts']}")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "url-cnn":
        run_url_cnn_task(args, data_path, output_dir)
    else:
        run_kaggle_task(args, data_path, output_dir)


if __name__ == "__main__":
    main()
