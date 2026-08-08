from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.model import PhishingDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "app" / "static"
INDEX_PATH = STATIC_DIR / "index.html"
HISTORY_PATH = PROJECT_ROOT / "reports" / "results" / "demo" / "prediction_history.csv"
PROJECT_NAME = "Multimodal Phishing Website Detection"

app = FastAPI(title=PROJECT_NAME, version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

detector = PhishingDetector()


class PredictRequest(BaseModel):
    url: str = Field(..., min_length=3, max_length=2048)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/predict")
def predict(payload: PredictRequest) -> dict:
    result = detector.predict(payload.url)
    _append_prediction_history(result)
    return result


@app.get("/", response_class=HTMLResponse)
def index(_: Request) -> str:
    with INDEX_PATH.open("r", encoding="utf-8") as file:
        return file.read()


def _append_prediction_history(result: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = HISTORY_PATH.exists()
    with HISTORY_PATH.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp_utc",
                "url",
                "risk_level",
                "label",
                "phishing_probability",
                "model_source",
                "html_available",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "url": result.get("url", ""),
                "risk_level": result.get("risk_level", ""),
                "label": result.get("label", ""),
                "phishing_probability": result.get("phishing_probability", ""),
                "model_source": result.get("model_source", ""),
                "html_available": (result.get("html_fetch") or {}).get("available", ""),
            }
        )
