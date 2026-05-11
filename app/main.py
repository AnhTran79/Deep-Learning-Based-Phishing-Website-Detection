from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.features import normalize_url
from app.model import PhishingDetector


MAX_HTML_BYTES = 2_000_000
FETCH_TIMEOUT_SECONDS = 8
USER_AGENT = "phishing-detector-demo/0.1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "app" / "static"
INDEX_PATH = STATIC_DIR / "index.html"

app = FastAPI(title="Phishing Website Detection Demo", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

detector = PhishingDetector()


class PredictRequest(BaseModel):
    url: str = Field(..., min_length=3, max_length=2048)


def _is_blocked_address(hostname: str) -> bool:
    lowered = hostname.strip().lower().rstrip(".")
    if lowered in {"localhost", "localhost.localdomain"}:
        return True

    try:
        addresses = [ipaddress.ip_address(lowered)]
    except ValueError:
        try:
            infos = socket.getaddrinfo(lowered, None, type=socket.SOCK_STREAM)
        except socket.gaierror:
            return False
        addresses = []
        for info in infos:
            address = info[4][0]
            try:
                addresses.append(ipaddress.ip_address(address))
            except ValueError:
                continue

    return any(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
        for address in addresses
    )


def fetch_html(url: str) -> tuple[str | None, dict[str, str | int | bool | None]]:
    normalized = normalize_url(url)
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"}:
        return None, {"fetched": False, "error": "Only http and https URLs are supported."}
    if not parsed.hostname:
        return None, {"fetched": False, "error": "URL must include a valid hostname."}
    if _is_blocked_address(parsed.hostname):
        return None, {"fetched": False, "error": "Private, local, and reserved network addresses are blocked."}

    request = UrlRequest(
        normalized,
        headers={
            "Accept": "text/html,application/xhtml+xml",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            content_type = response.headers.get("content-type", "")
            raw = response.read(MAX_HTML_BYTES + 1)
    except HTTPError as exc:
        return None, {"fetched": False, "status_code": exc.code, "error": f"HTTP {exc.code}"}
    except URLError as exc:
        return None, {"fetched": False, "error": str(exc.reason)}
    except TimeoutError:
        return None, {"fetched": False, "error": "Request timed out."}
    except OSError as exc:
        return None, {"fetched": False, "error": str(exc)}

    truncated = len(raw) > MAX_HTML_BYTES
    raw = raw[:MAX_HTML_BYTES]
    charset = response.headers.get_content_charset() or "utf-8"
    html = raw.decode(charset, errors="ignore")
    return html, {
        "fetched": True,
        "content_type": content_type,
        "bytes_read": len(raw),
        "truncated": truncated,
        "error": None,
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/predict")
def predict(payload: PredictRequest) -> dict:
    html, fetch_status = fetch_html(payload.url)
    result = detector.predict(payload.url, html)
    result["html_fetch"] = fetch_status
    return result


@app.get("/", response_class=HTMLResponse)
def index(_: Request) -> str:
    with INDEX_PATH.open("r", encoding="utf-8") as file:
        return file.read()
