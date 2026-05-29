from __future__ import annotations

from pathlib import Path

import requests

from src.preprocessing.text_cleaning import trim_text


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 PhishingDetectionResearchBot/1.0",
    "Accept": "text/html,application/xhtml+xml",
}


def read_html_file(path: Path, max_chars: int = 0) -> str:
    html = path.read_text(encoding="utf-8", errors="replace")
    return trim_text(html, max_chars)


def fetch_html(url: str, timeout: float = 8.0, max_bytes: int = 1_000_000) -> tuple[str | None, str | None]:
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "html" not in content_type.lower():
            return None, f"Response content-type is not HTML: {content_type or 'unknown'}"
        body = response.content[:max_bytes]
        encoding = response.encoding or "utf-8"
        return body.decode(encoding, errors="replace"), None
    except requests.RequestException as exc:
        return None, str(exc)
