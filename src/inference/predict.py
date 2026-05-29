from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import torch

from src.features.handcrafted_features import handcrafted_feature_vector
from src.models.dual_branch_cnn import DualBranchCnnClassifier
from src.models.html_cnn import HtmlCnnClassifier
from src.models.url_cnn import UrlCnnClassifier
from src.models.url_lstm import UrlLstmClassifier
from src.preprocessing.html_processing import fetch_html
from src.preprocessing.text_cleaning import html_to_visible_text, normalize_url_text
from src.preprocessing.tokenizers import CharTokenizerConfig, decode_config, encode_char_sequence


def load_torch_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_type = checkpoint.get("model_type") or checkpoint.get("model_name")
    url_config = decode_config(checkpoint.get("url_tokenizer", {"max_len": 200, "vocab_size": 128}))
    html_config = decode_config(checkpoint.get("html_tokenizer", {"max_len": 2000, "vocab_size": 256}))
    model_config = checkpoint.get("model_config", {})
    embedding_dim = int(model_config.get("embedding_dim", 64))
    dropout_rate = float(model_config.get("dropout_rate", 0.5))
    if model_type == "url_cnn":
        model = UrlCnnClassifier(vocab_size=url_config.vocab_size, embedding_dim=embedding_dim, dropout_rate=dropout_rate)
    elif model_type == "url_lstm":
        model = UrlLstmClassifier(vocab_size=url_config.vocab_size, embedding_dim=embedding_dim, dropout_rate=dropout_rate)
    elif model_type == "html_cnn":
        model = HtmlCnnClassifier(vocab_size=html_config.vocab_size, embedding_dim=embedding_dim, dropout_rate=dropout_rate)
    elif model_type == "dual_branch_cnn":
        model = DualBranchCnnClassifier(
            url_vocab_size=url_config.vocab_size,
            html_vocab_size=html_config.vocab_size,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
        )
    else:
        raise ValueError(f"Unsupported torch model type: {model_type}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model_type, model, url_config, html_config, float(checkpoint.get("threshold", 0.5))


def predict_with_torch(model_type: str, model, url: str, html: str, url_config: CharTokenizerConfig, html_config: CharTokenizerConfig) -> float:
    url_ids = torch.tensor([encode_char_sequence(normalize_url_text(url), url_config)], dtype=torch.long)
    html_ids = torch.tensor([encode_char_sequence(html_to_visible_text(html), html_config)], dtype=torch.long)
    with torch.no_grad():
        if model_type in {"url_cnn", "url_lstm"}:
            logits = model(url_ids)
        elif model_type == "html_cnn":
            logits = model(html_ids)
        else:
            logits = model(url_ids, html_ids)
        return float(torch.sigmoid(logits)[0].item())


def predict_with_joblib(path: Path, url: str, html: str) -> tuple[float, str]:
    artifact = joblib.load(path)
    model = artifact.get("model") or artifact.get("pipeline") or artifact
    metadata = artifact.get("metadata", {}) if isinstance(artifact, dict) else {}
    if metadata.get("feature_names"):
        vector = [handcrafted_feature_vector(url, html)]
        probability = float(model.predict_proba(vector)[0][1])
        return probability, "baseline_random_forest"
    text = [f"{url} {html}"]
    probability = float(model.predict_proba(text)[0][1])
    return probability, "baseline_tfidf_logreg"


def predict(url: str, model_path: Path, timeout: float, threshold_override: float | None = None) -> dict:
    html, fetch_error = fetch_html(url, timeout=timeout)
    html = html or ""
    if model_path.suffix == ".joblib":
        probability, model_type = predict_with_joblib(model_path, url, html)
        threshold = threshold_override if threshold_override is not None else 0.5
    else:
        model_type, model, url_config, html_config, saved_threshold = load_torch_model(model_path)
        probability = predict_with_torch(model_type, model, url, html, url_config, html_config)
        threshold = threshold_override if threshold_override is not None else saved_threshold
    label = "phishing" if probability >= threshold else "legitimate"
    return {
        "url": url,
        "label": label,
        "phishing_probability": round(probability, 6),
        "model_type": model_type,
        "html_fetch": {
            "available": bool(html),
            "error": fetch_error,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict phishing probability for one URL.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="models/saved/dual_branch_cnn.pt")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict(args.url, Path(args.model), args.timeout, args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
