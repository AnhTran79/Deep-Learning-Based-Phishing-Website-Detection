from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import torch
from sklearn.metrics import classification_report

from src.evaluation.compare_models import export_metric_chart
from src.evaluation.metrics import (
    evaluate_binary_classification,
    save_metric_figures,
)
from src.features.handcrafted_features import handcrafted_feature_vector
from src.models.dual_branch_cnn import DualBranchCnnClassifier
from src.models.html_cnn import HtmlCnnClassifier
from src.models.screenshot_cnn import ScreenshotCnnClassifier
from src.models.tri_branch_cnn import TriBranchCnnClassifier
from src.models.url_cnn import UrlCnnClassifier
from src.models.url_lstm import UrlLstmClassifier
from src.preprocessing.image_transforms import load_image_tensor
from src.preprocessing.text_cleaning import html_to_visible_text, normalize_url_text
from src.preprocessing.tokenizers import decode_config, encode_char_sequence


DEFAULT_DATASET = Path("data/external_test/dataset_100_clean")
DEFAULT_RESULTS = Path("reports/results/external_test")
DEFAULT_FIGURES = Path("reports/figures/external_test")

MODEL_SPECS = [
    ("mendeley", "baseline_tfidf_logreg", Path("models/saved/baseline_tfidf_logreg.joblib"), "URL + HTML"),
    ("mendeley", "baseline_random_forest", Path("models/saved/baseline_random_forest.joblib"), "URL + HTML"),
    ("mendeley", "url_cnn", Path("models/saved/url_cnn.pt"), "URL"),
    ("mendeley", "url_lstm", Path("models/saved/url_lstm.pt"), "URL"),
    ("mendeley", "html_cnn", Path("models/saved/html_cnn.pt"), "HTML"),
    ("mendeley", "dual_branch_cnn", Path("models/saved/dual_branch_cnn.pt"), "URL + HTML"),
    ("phish360", "phish360_url_cnn", Path("models/saved/phish360/phish360_url_cnn.pt"), "URL"),
    ("phish360", "phish360_url_lstm", Path("models/saved/phish360/phish360_url_lstm.pt"), "URL"),
    ("phish360", "phish360_html_cnn", Path("models/saved/phish360/phish360_html_cnn.pt"), "HTML"),
    (
        "phish360",
        "phish360_screenshot_cnn",
        Path("models/saved/phish360/phish360_screenshot_cnn.pt"),
        "Screenshot",
    ),
    (
        "phish360",
        "phish360_dual_branch_cnn",
        Path("models/saved/phish360/phish360_dual_branch_cnn.pt"),
        "URL + HTML",
    ),
    (
        "phish360",
        "phish360_tri_branch_cnn",
        Path("models/saved/phish360/phish360_tri_branch_cnn.pt"),
        "URL + HTML + Screenshot",
    ),
]


@dataclass(frozen=True)
class ExternalRecord:
    sample_id: str
    url: str
    html: str
    screenshot_file: Path
    label: int
    source: str
    domain: str


def load_external_records(root: Path, html_max_chars: int = 2000) -> list[ExternalRecord]:
    summary_path = root / "dataset_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing external dataset summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("dataset_role") != "external_test_only":
        raise ValueError("Dataset is not marked external_test_only.")
    if summary.get("complete") is not True:
        raise ValueError("External dataset is not marked complete.")

    index_path = root / "collection_results.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing external dataset index: {index_path}")
    records: list[ExternalRecord] = []
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = row["sample_id"]
            base_id = sample_id.split("_", 1)[0]
            sample_dir = root / sample_id
            url_path = sample_dir / "URL" / f"{base_id}.txt"
            html_path = sample_dir / "RAW-HTML" / f"{base_id}.html"
            screenshot_path = sample_dir / "SCREEN-SHOT" / f"{base_id}.png"
            label_path = sample_dir / "Label" / f"{base_id}.txt"
            required = [url_path, html_path, screenshot_path, label_path, sample_dir / "metadata.json"]
            if not all(path.is_file() for path in required):
                raise FileNotFoundError(f"Incomplete external sample: {sample_id}")
            label = int(label_path.read_text(encoding="utf-8-sig").strip())
            if label not in {0, 1} or label != int(row["label"]):
                raise ValueError(f"Invalid or inconsistent label for {sample_id}")
            html = html_path.read_text(encoding="utf-8", errors="replace")
            if html_max_chars > 0:
                html = html[:html_max_chars]
            records.append(
                ExternalRecord(
                    sample_id=sample_id,
                    url=url_path.read_text(encoding="utf-8-sig").strip(),
                    html=html,
                    screenshot_file=screenshot_path,
                    label=label,
                    source=row.get("source", ""),
                    domain=row.get("domain", ""),
                )
            )
    records.sort(key=lambda record: record.sample_id)
    if len(records) != 100 or {record.label for record in records} != {0, 1}:
        raise ValueError(f"Expected a balanced 100-sample external set; got {len(records)} rows.")
    return records


def build_torch_model(checkpoint: dict):
    model_type = checkpoint.get("model_type") or checkpoint.get("model_name")
    model_config = checkpoint.get("model_config", {})
    url_config = decode_config(checkpoint.get("url_tokenizer", {"max_len": 200, "vocab_size": 128}))
    html_config = decode_config(checkpoint.get("html_tokenizer", {"max_len": 2000, "vocab_size": 256}))
    embedding_dim = int(model_config.get("embedding_dim", 64))
    dropout_rate = float(model_config.get("dropout_rate", 0.5))
    image_feature_dim = int(model_config.get("image_feature_dim", 128))
    if model_type == "url_cnn":
        model = UrlCnnClassifier(url_config.vocab_size, embedding_dim, dropout_rate=dropout_rate)
    elif model_type == "url_lstm":
        model = UrlLstmClassifier(url_config.vocab_size, embedding_dim, dropout_rate=dropout_rate)
    elif model_type == "html_cnn":
        model = HtmlCnnClassifier(html_config.vocab_size, embedding_dim, dropout_rate=dropout_rate)
    elif model_type == "screenshot_cnn":
        model = ScreenshotCnnClassifier(dropout_rate=dropout_rate, feature_dim=image_feature_dim)
    elif model_type == "dual_branch_cnn":
        model = DualBranchCnnClassifier(
            url_vocab_size=url_config.vocab_size,
            html_vocab_size=html_config.vocab_size,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
        )
    elif model_type == "tri_branch_cnn":
        model = TriBranchCnnClassifier(
            url_vocab_size=url_config.vocab_size,
            html_vocab_size=html_config.vocab_size,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
            image_feature_dim=image_feature_dim,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return {
        "model": model,
        "model_type": model_type,
        "url_config": url_config,
        "html_config": html_config,
        "image_size": int(model_config.get("image_size", 128)),
        "threshold": float(checkpoint.get("threshold", 0.5)),
    }


def predict_torch(artifact: dict, records: list[ExternalRecord], batch_size: int) -> list[float]:
    model = artifact["model"]
    model_type = artifact["model_type"]
    probabilities: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            url_ids = torch.tensor(
                [
                    encode_char_sequence(normalize_url_text(record.url), artifact["url_config"])
                    for record in batch
                ],
                dtype=torch.long,
            )
            html_ids = torch.tensor(
                [
                    encode_char_sequence(html_to_visible_text(record.html), artifact["html_config"])
                    for record in batch
                ],
                dtype=torch.long,
            )
            if model_type == "url_cnn" or model_type == "url_lstm":
                logits = model(url_ids)
            elif model_type == "html_cnn":
                logits = model(html_ids)
            else:
                images = torch.stack(
                    [
                        load_image_tensor(record.screenshot_file, artifact["image_size"])
                        for record in batch
                    ]
                )
                if model_type == "screenshot_cnn":
                    logits = model(images)
                elif model_type == "dual_branch_cnn":
                    logits = model(url_ids, html_ids)
                else:
                    logits = model(url_ids, html_ids, images)
            probabilities.extend(float(value) for value in torch.sigmoid(logits).tolist())
    return probabilities


def predict_joblib(path: Path, records: list[ExternalRecord]) -> tuple[list[float], float]:
    artifact = joblib.load(path)
    model = artifact.get("model") if isinstance(artifact, dict) else artifact
    metadata = artifact.get("metadata", {}) if isinstance(artifact, dict) else {}
    if metadata.get("feature_names"):
        inputs = [handcrafted_feature_vector(record.url, record.html) for record in records]
    else:
        inputs = [f"{record.url} {record.html}" for record in records]
    return [float(value) for value in model.predict_proba(inputs)[:, 1]], 0.5


def read_internal_metrics() -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    paths = [
        Path("reports/results/mendeley/model_comparison.csv"),
        Path("reports/results/phish360/phish360_model_comparison.csv"),
    ]
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                metrics[row["model"]] = row
    return metrics


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def selected_model_specs(model_names: list[str]) -> list[tuple[str, str, Path, str]]:
    if not model_names or model_names == ["all"]:
        return MODEL_SPECS
    requested = set(model_names)
    available = {spec[1] for spec in MODEL_SPECS}
    unknown = requested - available
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(sorted(unknown))}")
    return [spec for spec in MODEL_SPECS if spec[1] in requested]


def evaluate(args: argparse.Namespace) -> list[dict]:
    records = load_external_records(Path(args.dataset), html_max_chars=args.html_max_chars)
    labels = [record.label for record in records]
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict] = []
    comparison: list[dict] = []
    internal = read_internal_metrics()

    for family, model_name, relative_path, model_input in selected_model_specs(args.models):
        model_path = PROJECT_ROOT / relative_path
        if not model_path.is_file():
            print(f"skip {model_name}: missing {model_path}")
            continue
        print(f"evaluating {model_name} ({model_input})")
        if model_path.suffix == ".joblib":
            probabilities, threshold = predict_joblib(model_path, records)
        else:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            artifact = build_torch_model(checkpoint)
            probabilities = predict_torch(artifact, records, args.batch_size)
            threshold = artifact["threshold"]
        metrics = evaluate_binary_classification(labels, probabilities, threshold=threshold)
        row = {
            "model": model_name,
            "family": family,
            "input": model_input,
            "threshold": threshold,
            **metrics,
            "model_file": str(relative_path),
        }
        internal_row = internal.get(model_name, {})
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            internal_value = internal_row.get(metric, "")
            row[f"internal_{metric}"] = internal_value
            row[f"delta_{metric}"] = (
                float(metrics[metric]) - float(internal_value) if internal_value != "" else ""
            )
        comparison.append(row)
        predictions_for_report = []
        for record, probability in zip(records, probabilities):
            prediction = 1 if probability >= threshold else 0
            predictions.append(
                {
                    "sample_id": record.sample_id,
                    "true_label": record.label,
                    "predicted_label": prediction,
                    "correct": int(prediction == record.label),
                    "phishing_probability": probability,
                    "threshold": threshold,
                    "model": model_name,
                    "family": family,
                    "input": model_input,
                    "source": record.source,
                    "domain": record.domain,
                    "url": record.url,
                }
            )
            predictions_for_report.append(prediction)
        report = classification_report(
            labels,
            predictions_for_report,
            labels=[0, 1],
            target_names=["legitimate", "phishing"],
            digits=4,
            zero_division=0,
        )
        (results_dir / f"classification_report_{model_name}.txt").write_text(report, encoding="utf-8")
        save_metric_figures(model_name, labels, probabilities, figures_dir, threshold=threshold)
        print(f"{model_name}: f1={metrics['f1']:.4f} recall={metrics['recall']:.4f} auc={metrics['roc_auc']:.4f}")

    comparison.sort(key=lambda row: float(row["f1"]), reverse=True)
    comparison_fields = [
        "model",
        "family",
        "input",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "false_positive_rate",
        "false_negative_rate",
        "confusion_matrix",
        "internal_accuracy",
        "delta_accuracy",
        "internal_precision",
        "delta_precision",
        "internal_recall",
        "delta_recall",
        "internal_f1",
        "delta_f1",
        "internal_roc_auc",
        "delta_roc_auc",
        "model_file",
    ]
    write_csv(results_dir / "model_comparison.csv", comparison, comparison_fields)
    write_csv(
        results_dir / "predictions.csv",
        predictions,
        [
            "sample_id",
            "true_label",
            "predicted_label",
            "correct",
            "phishing_probability",
            "threshold",
            "model",
            "family",
            "input",
            "source",
            "domain",
            "url",
        ],
    )
    report_rows: list[dict] = []
    for row in comparison:
        model_predictions = [
            prediction for prediction in predictions if prediction["model"] == row["model"]
        ]
        predicted = [int(item["predicted_label"]) for item in model_predictions]
        report = classification_report(
            labels,
            predicted,
            labels=[0, 1],
            target_names=["legitimate", "phishing"],
            output_dict=True,
            zero_division=0,
        )
        for label_name in ("legitimate", "phishing", "macro avg", "weighted avg"):
            values = report[label_name]
            report_rows.append(
                {
                    "model": row["model"],
                    "label": label_name,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1-score": values["f1-score"],
                    "support": int(values["support"]),
                }
            )
    write_csv(
        results_dir / "classification_reports.csv",
        report_rows,
        ["model", "label", "precision", "recall", "f1-score", "support"],
    )
    export_metric_chart(
        comparison,
        figures_dir / "external_model_comparison_metrics.png",
        title="External Test Model Comparison (100 samples)",
    )
    false_predictions = [row for row in predictions if not int(row["correct"])]
    write_csv(
        results_dir / "misclassifications.csv",
        false_predictions,
        [
            "sample_id",
            "true_label",
            "predicted_label",
            "correct",
            "phishing_probability",
            "threshold",
            "model",
            "family",
            "input",
            "source",
            "domain",
            "url",
        ],
    )
    summary = {
        "dataset": str(Path(args.dataset)),
        "dataset_role": "external_test_only",
        "samples": len(records),
        "label_counts": {"legitimate": labels.count(0), "phishing": labels.count(1)},
        "models_evaluated": len(comparison),
        "threshold_policy": "saved model threshold; 0.5 for joblib baselines",
        "training_performed": False,
        "threshold_tuning_performed": False,
        "best_external_model_by_f1": comparison[0]["model"] if comparison else None,
        "best_external_f1": comparison[0]["f1"] if comparison else None,
    }
    (results_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen models on an external-only URL + HTML + screenshot dataset."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES))
    parser.add_argument("--html-max-chars", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model names to evaluate, or 'all'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = evaluate(args)
    for row in rows:
        delta = row.get("delta_f1")
        delta_text = "" if delta == "" else f" delta_f1={float(delta):+.4f}"
        print(f"{row['model']}: external_f1={float(row['f1']):.4f}{delta_text}")


if __name__ == "__main__":
    main()
