from __future__ import annotations

import csv
import os
from pathlib import Path

try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover
    joblib = None

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from app.features import extract_html_features, extract_url_features, fetch_html, heuristic_phishing_score

try:
    from src.features.handcrafted_features import handcrafted_feature_vector
    from src.models.dual_branch_cnn import DualBranchCnnClassifier
    from src.models.html_cnn import HtmlCnnClassifier
    from src.models.url_cnn import UrlCnnClassifier
    from src.models.url_lstm import UrlLstmClassifier
    from src.preprocessing.text_cleaning import html_to_visible_text, normalize_url_text
    from src.preprocessing.tokenizers import decode_config, encode_char_sequence
except ModuleNotFoundError:  # pragma: no cover
    DualBranchCnnClassifier = None
    HtmlCnnClassifier = None
    UrlCnnClassifier = None
    UrlLstmClassifier = None
    decode_config = None
    encode_char_sequence = None
    handcrafted_feature_vector = None
    html_to_visible_text = None
    normalize_url_text = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAVED_MODEL_DIR = PROJECT_ROOT / "models" / "saved"
MODEL_COMPARISON_PATH = PROJECT_ROOT / "reports" / "results" / "mendeley" / "model_comparison.csv"
LEGACY_MODEL_COMPARISON_PATH = PROJECT_ROOT / "reports" / "results" / "model_comparison.csv"
DUAL_BRANCH_CNN_PATH = SAVED_MODEL_DIR / "dual_branch_cnn.pt"
HTML_CNN_PATH = SAVED_MODEL_DIR / "html_cnn.pt"
URL_CNN_PATH = SAVED_MODEL_DIR / "url_cnn.pt"
URL_LSTM_PATH = SAVED_MODEL_DIR / "url_lstm.pt"
TFIDF_LOGREG_PATH = SAVED_MODEL_DIR / "baseline_tfidf_logreg.joblib"
RANDOM_FOREST_PATH = SAVED_MODEL_DIR / "baseline_random_forest.joblib"


class PhishingDetector:
    threshold = 0.50

    def __init__(self) -> None:
        self.fetch_html_at_runtime = os.environ.get("FETCH_HTML_AT_RUNTIME", "1") != "0"
        self.html_timeout = float(os.environ.get("HTML_FETCH_TIMEOUT", "8"))
        self.torch_artifacts = {
            "dual_branch_cnn": self._load_torch_artifact(DUAL_BRANCH_CNN_PATH),
            "html_cnn": self._load_torch_artifact(HTML_CNN_PATH),
            "url_cnn": self._load_torch_artifact(URL_CNN_PATH),
            "url_lstm": self._load_torch_artifact(URL_LSTM_PATH),
        }
        self.baseline_tfidf = self._load_joblib_model(TFIDF_LOGREG_PATH)
        self.baseline_rf = self._load_joblib_model(RANDOM_FOREST_PATH)
        self.model_scores = self._load_model_scores(MODEL_COMPARISON_PATH)
        if not self.model_scores:
            self.model_scores = self._load_model_scores(LEGACY_MODEL_COMPARISON_PATH)

    def _load_model_scores(self, path: Path) -> dict[str, float]:
        if not path.exists():
            return {}
        scores: dict[str, float] = {}
        with path.open("r", encoding="utf-8", newline="") as file:
            rows = csv.DictReader(file)
            for row in rows:
                if not row.get("model") or not row.get("f1"):
                    continue
                try:
                    scores[row["model"]] = float(row["f1"])
                except ValueError:
                    continue
        return scores

    def _load_joblib_model(self, path: Path):
        if joblib is None or not path.exists():
            return None
        artifact = joblib.load(path)
        return artifact.get("model") if isinstance(artifact, dict) else artifact

    def _load_torch_artifact(self, path: Path):
        if torch is None or not path.exists() or decode_config is None:
            return None
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model_type = checkpoint.get("model_type")
        url_config = decode_config(checkpoint.get("url_tokenizer", {"max_len": 200, "vocab_size": 128}))
        html_config = decode_config(checkpoint.get("html_tokenizer", {"max_len": 2000, "vocab_size": 256}))
        model_config = checkpoint.get("model_config", {})
        embedding_dim = int(model_config.get("embedding_dim", 64))
        dropout_rate = float(model_config.get("dropout_rate", 0.5))

        if model_type == "dual_branch_cnn" and DualBranchCnnClassifier is not None:
            model = DualBranchCnnClassifier(
                url_vocab_size=url_config.vocab_size,
                html_vocab_size=html_config.vocab_size,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
            )
        elif model_type == "html_cnn" and HtmlCnnClassifier is not None:
            model = HtmlCnnClassifier(
                vocab_size=html_config.vocab_size,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
            )
        elif model_type == "url_cnn" and UrlCnnClassifier is not None:
            model = UrlCnnClassifier(
                vocab_size=url_config.vocab_size,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
            )
        elif model_type == "url_lstm" and UrlLstmClassifier is not None:
            model = UrlLstmClassifier(
                vocab_size=url_config.vocab_size,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
            )
        else:
            return None

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return {
            "model": model,
            "model_type": model_type,
            "url_config": url_config,
            "html_config": html_config,
            "threshold": float(checkpoint.get("threshold", self.threshold)),
        }

    def _predict_torch(self, artifact: dict | None, url: str, html: str | None) -> float | None:
        if (
            torch is None
            or artifact is None
            or encode_char_sequence is None
            or normalize_url_text is None
            or html_to_visible_text is None
        ):
            return None
        model_type = artifact["model_type"]
        url_ids = torch.tensor(
            [encode_char_sequence(normalize_url_text(url), artifact["url_config"])],
            dtype=torch.long,
        )
        html_ids = torch.tensor(
            [encode_char_sequence(html_to_visible_text(html or ""), artifact["html_config"])],
            dtype=torch.long,
        )
        if model_type in {"dual_branch_cnn", "html_cnn"} and not html:
            return None
        with torch.no_grad():
            if model_type == "dual_branch_cnn":
                logits = artifact["model"](url_ids, html_ids)
            elif model_type == "html_cnn":
                logits = artifact["model"](html_ids)
            else:
                logits = artifact["model"](url_ids)
            return float(torch.sigmoid(logits)[0].item())

    def _predict_tfidf(self, url: str, html: str | None) -> float | None:
        if self.baseline_tfidf is None:
            return None
        return float(self.baseline_tfidf.predict_proba([f"{url} {html or ''}"])[0][1])

    def _predict_random_forest(self, url: str, html: str | None) -> float | None:
        if self.baseline_rf is None or handcrafted_feature_vector is None:
            return None
        return float(self.baseline_rf.predict_proba([handcrafted_feature_vector(url, html or "")])[0][1])

    def predict(self, url: str) -> dict:
        html = fetch_html(url, timeout=self.html_timeout) if self.fetch_html_at_runtime else None
        url_features = extract_url_features(url)
        html_features = extract_html_features(html, base_url=url)
        feature_values = {**url_features.to_dict(), **html_features.to_dict()}

        component_probabilities = {
            "dual_branch_cnn": self._predict_torch(self.torch_artifacts["dual_branch_cnn"], url, html),
            "html_cnn": self._predict_torch(self.torch_artifacts["html_cnn"], url, html),
            "url_cnn": self._predict_torch(self.torch_artifacts["url_cnn"], url, html),
            "url_lstm": self._predict_torch(self.torch_artifacts["url_lstm"], url, html),
            "baseline_tfidf_logreg": self._predict_tfidf(url, html),
            "baseline_random_forest": self._predict_random_forest(url, html),
            "heuristic_url_html_rules": heuristic_phishing_score(url_features, html_features),
        }

        priority = [
            "dual_branch_cnn",
            "html_cnn",
            "url_cnn",
            "url_lstm",
            "baseline_tfidf_logreg",
            "baseline_random_forest",
            "heuristic_url_html_rules",
        ]
        available_models = [name for name in priority if component_probabilities[name] is not None]
        if self.model_scores:
            model_source = max(
                available_models,
                key=lambda name: (self.model_scores.get(name, -1.0), -priority.index(name)),
            )
        else:
            model_source = available_models[0]
        phishing_probability = float(component_probabilities[model_source])
        threshold = self.threshold
        artifact = self.torch_artifacts.get(model_source)
        if artifact is not None:
            threshold = artifact["threshold"]

        label = "phishing" if phishing_probability >= threshold else "legitimate"
        confidence = phishing_probability if label == "phishing" else 1.0 - phishing_probability
        return {
            "url": url,
            "label": label,
            "confidence": round(confidence, 4),
            "phishing_probability": round(phishing_probability, 4),
            "model_source": model_source,
            "features": feature_values,
            "component_probabilities": {
                key: round(value, 4) if value is not None else None for key, value in component_probabilities.items()
            },
            "html_fetch": {
                "enabled": self.fetch_html_at_runtime,
                "available": bool(html_features.html_available),
            },
        }
