from __future__ import annotations

import os
import tempfile
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from app.features import extract_html_features, extract_url_features, fetch_html, heuristic_phishing_score, normalize_url

try:
    from src.models.tri_branch_cnn import TriBranchCnnClassifier
    from src.preprocessing.image_transforms import load_image_tensor
    from src.preprocessing.text_cleaning import html_to_visible_text, normalize_url_text
    from src.preprocessing.tokenizers import decode_config, encode_char_sequence
except ModuleNotFoundError:  # pragma: no cover
    TriBranchCnnClassifier = None
    decode_config = None
    encode_char_sequence = None
    html_to_visible_text = None
    load_image_tensor = None
    normalize_url_text = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHISH360_TRI_BRANCH_CNN_PATH = PROJECT_ROOT / "models" / "saved" / "phish360" / "phish360_tri_branch_cnn.pt"


class PhishingDetector:
    threshold = 0.50

    def __init__(self) -> None:
        self.fetch_html_at_runtime = os.environ.get("FETCH_HTML_AT_RUNTIME", "1") != "0"
        self.html_timeout = float(os.environ.get("HTML_FETCH_TIMEOUT", "8"))
        self.screenshot_timeout = float(os.environ.get("SCREENSHOT_TIMEOUT", "12"))
        self.tri_branch_artifact = self._load_torch_artifact(PHISH360_TRI_BRANCH_CNN_PATH)

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

        if model_type != "tri_branch_cnn" or TriBranchCnnClassifier is None:
            return None
        model = TriBranchCnnClassifier(
            url_vocab_size=url_config.vocab_size,
            html_vocab_size=html_config.vocab_size,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
            image_feature_dim=int(model_config.get("image_feature_dim", 128)),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return {
            "model": model,
            "model_type": model_type,
            "url_config": url_config,
            "html_config": html_config,
            "image_size": int(model_config.get("image_size", 128)),
            "threshold": float(checkpoint.get("threshold", self.threshold)),
        }

    def _capture_page(self, url: str) -> tuple[str | None, object | None, str | None]:
        if load_image_tensor is None:
            return None, None, "image transforms unavailable"
        try:
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
            from playwright.sync_api import sync_playwright
        except ModuleNotFoundError:
            return None, None, "playwright unavailable"

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                screenshot_path = Path(temp_dir) / "page.png"
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True)
                    try:
                        page = browser.new_page(viewport={"width": 1366, "height": 768})
                        page.goto(url, wait_until="domcontentloaded", timeout=int(self.screenshot_timeout * 1000))
                        try:
                            page.wait_for_load_state("networkidle", timeout=3000)
                        except PlaywrightTimeoutError:
                            pass
                        html = page.content()
                        page.screenshot(path=str(screenshot_path), full_page=False)
                    finally:
                        browser.close()
                image = load_image_tensor(screenshot_path, image_size=self.tri_branch_artifact["image_size"])
                return html, image, None
        except Exception as exc:
            return None, None, str(exc)

    def _predict_tri_branch(self, url: str, html: str | None, image) -> float | None:
        artifact = self.tri_branch_artifact
        if (
            torch is None
            or artifact is None
            or image is None
            or encode_char_sequence is None
            or normalize_url_text is None
            or html_to_visible_text is None
        ):
            return None
        url_ids = torch.tensor(
            [encode_char_sequence(normalize_url_text(url), artifact["url_config"])],
            dtype=torch.long,
        )
        html_ids = torch.tensor(
            [encode_char_sequence(html_to_visible_text(html or ""), artifact["html_config"])],
            dtype=torch.long,
        )
        with torch.no_grad():
            logits = artifact["model"](url_ids, html_ids, image.unsqueeze(0))
            return float(torch.sigmoid(logits)[0].item())

    def predict(self, url: str) -> dict:
        normalized_url = normalize_url(url)
        screenshot_error = None
        image = None
        html = None
        if self.tri_branch_artifact is not None:
            html, image, screenshot_error = self._capture_page(normalized_url)
        if html is None:
            html = fetch_html(normalized_url, timeout=self.html_timeout) if self.fetch_html_at_runtime else None
        url_features = extract_url_features(normalized_url)
        html_features = extract_html_features(html, base_url=normalized_url)
        feature_values = {**url_features.to_dict(), **html_features.to_dict()}

        component_probabilities = {
            "phish360_tri_branch_cnn": self._predict_tri_branch(normalized_url, html, image),
            "heuristic_url_html_rules": heuristic_phishing_score(url_features, html_features),
        }
        if component_probabilities["phish360_tri_branch_cnn"] is not None:
            model_source = "phish360_tri_branch_cnn"
        else:
            model_source = "heuristic_url_html_rules"
        phishing_probability = float(component_probabilities[model_source])
        threshold = (
            self.tri_branch_artifact["threshold"]
            if model_source == "phish360_tri_branch_cnn"
            else self.threshold
        )

        label = "phishing" if phishing_probability >= threshold else "legitimate"
        confidence = phishing_probability if label == "phishing" else 1.0 - phishing_probability
        return {
            "url": normalized_url,
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
            "screenshot_capture": {
                "enabled": self.tri_branch_artifact is not None,
                "available": image is not None,
                "error": screenshot_error,
            },
        }
