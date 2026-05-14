from __future__ import annotations

import json
import math
from pathlib import Path

try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover - allows heuristic fallback without scikit-learn.
    joblib = None

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - allows the demo API to run before installing torch.
    torch = None
    nn = None

from app.features import UrlFeatures, extract_html_features, extract_url_features, heuristic_phishing_score, to_kaggle_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_model.json"
DEEP_MODEL_PATH = PROJECT_ROOT / "artifacts" / "deep_learning_model.joblib"
PHISHING_LABEL = "-1"
LEGITIMATE_LABEL = "1"


if nn is not None:

    class CharCnnUrlClassifier(nn.Module):
        """Skeleton model for thesis implementation."""

        def __init__(self, vocab_size: int = 128, embedding_dim: int = 32) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.encoder = nn.Sequential(
                nn.Conv1d(embedding_dim, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
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
    threshold = 0.35

    def __init__(self) -> None:
        self.model = None
        self.artifact = self._load_artifact()
        self.deep_learning_artifact = self._load_deep_learning_artifact()
        if torch is not None:
            self.model = CharCnnUrlClassifier()
            self.model.eval()

    def _load_artifact(self) -> dict | None:
        if not MODEL_PATH.exists():
            return None
        return json.loads(MODEL_PATH.read_text(encoding="utf-8"))

    def _load_deep_learning_artifact(self) -> dict | None:
        if joblib is None or not DEEP_MODEL_PATH.exists():
            return None
        return joblib.load(DEEP_MODEL_PATH)

    def _url_model(self) -> dict | None:
        if self.artifact is None:
            return None
        if "url_model" in self.artifact:
            return self.artifact["url_model"]
        if self.artifact.get("model_type") == "categorical_naive_bayes":
            return self.artifact
        return None

    def _html_model(self) -> dict | None:
        if self.artifact is None:
            return None
        return self.artifact.get("html_model")

    def _predict_with_categorical_model(self, model: dict, feature_values: dict[str, int]) -> float:
        log_scores = dict(model["priors"])
        for label in (PHISHING_LABEL, LEGITIMATE_LABEL):
            for name in model["feature_names"]:
                value = str(feature_values[name])
                likelihood = model["likelihoods"][label][name]
                if value in likelihood:
                    log_scores[label] += likelihood[value]
                else:
                    log_scores[label] += likelihood.get("__UNK__", min(likelihood.values()))

        max_log = max(log_scores.values())
        phishing_score = math.exp(log_scores[PHISHING_LABEL] - max_log)
        legitimate_score = math.exp(log_scores[LEGITIMATE_LABEL] - max_log)
        return phishing_score / (phishing_score + legitimate_score)

    def _predict_with_url_model(self, url: str, features: UrlFeatures) -> float | None:
        model = self._url_model()
        if model is None:
            return None
        kaggle_features = to_kaggle_features(url, features)
        return self._predict_with_categorical_model(model, kaggle_features)

    def _predict_with_deep_learning_model(self, url: str, features: UrlFeatures) -> float | None:
        if self.deep_learning_artifact is None:
            return None
        pipeline = self.deep_learning_artifact.get("pipeline")
        feature_names = self.deep_learning_artifact.get("feature_names") or []
        if pipeline is None or not feature_names:
            return None

        kaggle_features = to_kaggle_features(url, features)
        vector = [[kaggle_features[name] for name in feature_names]]
        if not hasattr(pipeline, "predict_proba"):
            return None

        probabilities = pipeline.predict_proba(vector)[0]
        classes = list(getattr(pipeline, "classes_", []))
        if not classes and hasattr(pipeline, "steps"):
            classes = list(getattr(pipeline.steps[-1][1], "classes_", []))
        positive_class = self.deep_learning_artifact.get("positive_class", 1)
        if positive_class not in classes:
            return None
        return float(probabilities[classes.index(positive_class)])

    def _choose_prediction_source(
        self,
        heuristic_probability: float,
        deep_learning_probability: float | None,
        url_probability: float | None,
        html_probability: float | None,
    ) -> tuple[float, str]:
        probabilities = [heuristic_probability]
        if deep_learning_probability is not None:
            probabilities.append(deep_learning_probability)
        if url_probability is not None:
            probabilities.append(url_probability)
        if html_probability is not None:
            probabilities.append(html_probability)

        if deep_learning_probability is not None and html_probability is not None:
            return max(probabilities), "sklearn_mlp_deep_learning_url_html_ensemble"
        if deep_learning_probability is not None:
            return max(heuristic_probability, deep_learning_probability), "sklearn_mlp_deep_learning_url_features"
        if url_probability is None and html_probability is None:
            return heuristic_probability, "heuristic_url_rules"
        if html_probability is None:
            return max(probabilities), "kaggle_naive_bayes_with_url_features"
        return max(probabilities), "url_html_ensemble"

    def _predict_with_html_model(self, html: str | None, url: str) -> tuple[float | None, dict | None]:
        if not html:
            return None, None
        model = self._html_model()
        if model is None:
            return None, None
        html_features = extract_html_features(html, base_url=url)
        feature_values = html_features.to_dict()
        probability = self._predict_with_categorical_model(model, feature_values)
        active_content_signals = (
            feature_values["form_count_bucket"]
            + feature_values["has_password_input"]
            + feature_values["has_email_input"]
            + feature_values["has_iframe"]
            + feature_values["has_meta_refresh"]
            + feature_values["blocks_right_click"]
            + feature_values["has_popup"]
            + feature_values["has_empty_form_action"]
            + feature_values["has_javascript_form_action"]
            + feature_values["has_external_form_action"]
        )
        if active_content_signals == 0:
            probability = min(probability, 0.25)
        return probability, feature_values

    def predict(self, url: str, html: str | None = None) -> dict:
        features = extract_url_features(url)

        heuristic_probability = heuristic_phishing_score(features)
        deep_learning_probability = self._predict_with_deep_learning_model(url, features)
        url_probability = self._predict_with_url_model(url, features)
        html_probability, html_features = self._predict_with_html_model(html, url)

        phishing_probability, model_source = self._choose_prediction_source(
            heuristic_probability=heuristic_probability,
            deep_learning_probability=deep_learning_probability,
            url_probability=url_probability,
            html_probability=html_probability,
        )
        label = "phishing" if phishing_probability >= self.threshold else "legitimate"
        confidence = phishing_probability if label == "phishing" else 1.0 - phishing_probability

        result = {
            "url": url,
            "label": label,
            "confidence": round(confidence, 4),
            "phishing_probability": round(phishing_probability, 4),
            "model_source": model_source,
            "features": features.to_dict(),
        }
        if html_features is not None:
            result["html_features"] = html_features
            result["component_probabilities"] = {
                "heuristic_url_rules": round(heuristic_probability, 4),
                "deep_learning_url_model": round(deep_learning_probability, 4)
                if deep_learning_probability is not None
                else None,
                "url_model": round(url_probability, 4) if url_probability is not None else None,
                "html_model": round(html_probability, 4) if html_probability is not None else None,
            }
        return result
