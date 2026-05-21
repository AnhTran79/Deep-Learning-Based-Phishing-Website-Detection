from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.model import PhishingDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "app" / "static"
INDEX_PATH = STATIC_DIR / "index.html"

app = FastAPI(title="Phishing Website Detection Demo", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

detector = PhishingDetector()


class PredictRequest(BaseModel):
    url: str = Field(..., min_length=3, max_length=2048)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/predict")
def predict(payload: PredictRequest) -> dict:
    return detector.predict(payload.url)


@app.get("/", response_class=HTMLResponse)
def index(_: Request) -> str:
    with INDEX_PATH.open("r", encoding="utf-8") as file:
        return file.read()
