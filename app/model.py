from __future__ import annotations

import json
import os
from pathlib import Path

try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover
    joblib = None

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None

from app.features import (
    COMBINED_FEATURE_NAMES,
    extract_html_features,
    extract_url_features,
    fetch_html,
    heuristic_phishing_score,
    normalize_url,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "best_model.json"
TABULAR_MODEL_PATH = PROJECT_ROOT / "artifacts" / "deep_learning_model.joblib"
URL_CNN_PATH = PROJECT_ROOT / "artifacts" / "url_cnn.pt"

URL_CNN_MAX_LEN = 200
URL_CNN_VOCAB_SIZE = 128


def encode_url(url: str) -> list[int]:
    normalized = normalize_url(url)
    ids = [min(ord(char), URL_CNN_VOCAB_SIZE - 1) for char in normalized[:URL_CNN_MAX_LEN]]
    return ids + [0] * (URL_CNN_MAX_LEN - len(ids))


if nn is not None:

    class CharCnnUrlClassifier(nn.Module):
        def __init__(self, vocab_size: int = URL_CNN_VOCAB_SIZE, embedding_dim: int = 32) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.encoder = nn.Sequential(
                nn.Conv1d(embedding_dim, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(64, 1),
            )

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            embedded = self.embedding(token_ids).transpose(1, 2)
            encoded = self.encoder(embedded)
            return self.classifier(encoded).squeeze(-1)

else:

    class CharCnnUrlClassifier:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required to instantiate CharCnnUrlClassifier.")


class PhishingDetector:
    threshold = 0.40

    def __init__(self) -> None:
        self.artifact = self._load_json_artifact()
        self.tabular_artifact = self._load_tabular_artifact()
        self.url_cnn = self._load_url_cnn()
        self.fetch_html_at_runtime = os.environ.get("FETCH_HTML_AT_RUNTIME", "1") != "0"
        self.html_timeout = float(os.environ.get("HTML_FETCH_TIMEOUT", "4"))

    def _load_json_artifact(self) -> dict | None:
        if not ARTIFACT_PATH.exists():
            return None
        return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    def _load_tabular_artifact(self) -> dict | None:
        if joblib is None or not TABULAR_MODEL_PATH.exists():
            return None
        artifact = joblib.load(TABULAR_MODEL_PATH)
        if artifact.get("model_type") != "sklearn_mlp_url_html_features":
            return None
        return artifact

    def _load_url_cnn(self):
        if torch is None or not URL_CNN_PATH.exists():
            return None
        model = CharCnnUrlClassifier(vocab_size=URL_CNN_VOCAB_SIZE)
        state = torch.load(URL_CNN_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
        model.eval()
        return model

    def _predict_with_url_cnn(self, url: str) -> float | None:
        if torch is None or self.url_cnn is None:
            return None
        token_ids = torch.tensor([encode_url(url)], dtype=torch.long)
        with torch.no_grad():
            logits = self.url_cnn(token_ids)
            return float(torch.sigmoid(logits)[0].item())

    def _predict_with_tabular_model(self, feature_values: dict[str, int | float]) -> float | None:
        if self.tabular_artifact is None:
            return None
        pipeline = self.tabular_artifact.get("pipeline")
        feature_names = self.tabular_artifact.get("feature_names") or COMBINED_FEATURE_NAMES
        if pipeline is None or not hasattr(pipeline, "predict_proba"):
            return None
        vector = [[float(feature_values.get(name, 0.0)) for name in feature_names]]
        probabilities = pipeline.predict_proba(vector)[0]
        classes = list(getattr(pipeline, "classes_", []))
        if not classes and hasattr(pipeline, "steps"):
            classes = list(getattr(pipeline.steps[-1][1], "classes_", []))
        if 1 not in classes:
            return None
        return float(probabilities[classes.index(1)])

    def _combine_probabilities(
        self,
        heuristic_probability: float,
        cnn_probability: float | None,
        tabular_probability: float | None,
    ) -> tuple[float, str]:
        if cnn_probability is not None and tabular_probability is not None:
            return (cnn_probability * 0.65) + (tabular_probability * 0.35), "url_cnn_plus_url_html_mlp"
        if cnn_probability is not None:
            return cnn_probability, "url_cnn_deep_learning"
        if tabular_probability is not None:
            return max(tabular_probability, heuristic_probability), "url_html_mlp_deep_learning"
        return heuristic_probability, "heuristic_url_html_rules"

    def predict(self, url: str) -> dict:
        html = fetch_html(url, timeout=self.html_timeout) if self.fetch_html_at_runtime else None
        url_features = extract_url_features(url)
        html_features = extract_html_features(html, base_url=url)
        feature_values = {**url_features.to_dict(), **html_features.to_dict()}

        heuristic_probability = heuristic_phishing_score(url_features, html_features)
        cnn_probability = self._predict_with_url_cnn(url)
        tabular_probability = self._predict_with_tabular_model(feature_values)
        phishing_probability, model_source = self._combine_probabilities(
            heuristic_probability=heuristic_probability,
            cnn_probability=cnn_probability,
            tabular_probability=tabular_probability,
        )

        label = "phishing" if phishing_probability >= self.threshold else "legitimate"
        confidence = phishing_probability if label == "phishing" else 1.0 - phishing_probability
        return {
            "url": url,
            "label": label,
            "confidence": round(confidence, 4),
            "phishing_probability": round(phishing_probability, 4),
            "model_source": model_source,
            "features": feature_values,
            "component_probabilities": {
                "heuristic_url_html_rules": round(heuristic_probability, 4),
                "url_cnn_deep_learning": round(cnn_probability, 4) if cnn_probability is not None else None,
                "url_html_mlp_deep_learning": round(tabular_probability, 4) if tabular_probability is not None else None,
            },
            "html_fetch": {
                "enabled": self.fetch_html_at_runtime,
                "available": bool(html_features.html_available),
            },
        }
