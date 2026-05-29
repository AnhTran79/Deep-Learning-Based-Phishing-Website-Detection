from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.dataset_loader import UrlHtmlRecord, iter_text_labels, load_url_html_records
from src.data.splitting import split_train_val_test as stratified_split_train_val_test
from src.evaluation.metrics import evaluate_binary_classification, save_metric_figures, write_result_row
from src.features.handcrafted_features import FEATURE_NAMES, handcrafted_feature_vector


def split_train_val_test(
    records: list[UrlHtmlRecord],
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[list[UrlHtmlRecord], list[UrlHtmlRecord], list[UrlHtmlRecord]]:
    labels = [record.label for record in records]
    return stratified_split_train_val_test(records, labels, test_size, val_size, random_state)


def train_tfidf_logreg(train: list[UrlHtmlRecord], test: list[UrlHtmlRecord], max_features: int) -> tuple[Pipeline, dict]:
    train_texts, train_labels = iter_text_labels(train)
    test_texts, test_labels = iter_text_labels(test)
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=max_features, analyzer="char_wb", ngram_range=(3, 5))),
            ("classifier", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
    model.fit(train_texts, train_labels)
    probabilities = model.predict_proba(test_texts)[:, 1].tolist()
    return model, {"labels": test_labels, "probabilities": probabilities}


def train_random_forest(train: list[UrlHtmlRecord], test: list[UrlHtmlRecord], random_state: int) -> tuple[Pipeline, dict]:
    x_train = [handcrafted_feature_vector(record.url, record.html) for record in train]
    y_train = [record.label for record in train]
    x_test = [handcrafted_feature_vector(record.url, record.html) for record in test]
    y_test = [record.label for record in test]
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1].tolist()
    return model, {"labels": y_test, "probabilities": probabilities}


def save_baseline(
    name: str,
    model,
    payload: dict,
    model_input: str,
    output_dir: Path,
    results_path: Path,
    figures_dir: Path,
    threshold: float,
    extra_metadata: dict | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = output_dir / f"{name}.joblib"
    joblib.dump({"model": model, "metadata": extra_metadata or {}}, model_file)
    metrics = evaluate_binary_classification(payload["labels"], payload["probabilities"], threshold=threshold)
    save_metric_figures(name, payload["labels"], payload["probabilities"], figures_dir, threshold=threshold)
    row = {
        "model": name,
        "input": model_input,
        **metrics,
        "model_file": str(model_file),
    }
    write_result_row(results_path, row)
    (output_dir / f"{name}_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(f"{name} f1={metrics['f1']:.4f} recall={metrics['recall']:.4f} roc_auc={metrics['roc_auc']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline phishing detectors on URL + HTML data.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--html-root", default="")
    parser.add_argument("--out-dir", default="models/saved")
    parser.add_argument("--results-dir", default="reports/results")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--html-max-chars", type=int, default=2000)
    parser.add_argument("--tfidf-max-features", type=int, default=50000)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    html_root = Path(args.html_root) if args.html_root else None
    records = load_url_html_records(
        Path(args.data),
        html_root=html_root,
        max_rows=args.max_rows,
        html_max_chars=args.html_max_chars,
    )
    train, _, test = split_train_val_test(records, args.test_size, args.val_size, args.random_state)
    out_dir = Path(args.out_dir)
    results_path = Path(args.results_dir) / "model_comparison.csv"
    figures_dir = Path(args.figures_dir)

    tfidf_model, tfidf_payload = train_tfidf_logreg(train, test, args.tfidf_max_features)
    save_baseline(
        "baseline_tfidf_logreg",
        tfidf_model,
        tfidf_payload,
        "URL + HTML TF-IDF",
        out_dir,
        results_path,
        figures_dir,
        args.threshold,
    )

    rf_model, rf_payload = train_random_forest(train, test, args.random_state)
    save_baseline(
        "baseline_random_forest",
        rf_model,
        rf_payload,
        "URL + HTML handcrafted features",
        out_dir,
        results_path,
        figures_dir,
        args.threshold,
        extra_metadata={"feature_names": FEATURE_NAMES},
    )


if __name__ == "__main__":
    main()
