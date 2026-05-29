from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


@lru_cache(maxsize=8)
def _html_file_index(html_root: str) -> dict[str, str]:
    root = Path(html_root)
    if not root.exists():
        return {}
    return {path.name: str(path) for path in root.rglob("*.html") if path.is_file()}


@lru_cache(maxsize=200_000)
def _find_html_file(html_root: str, html_file: str) -> str:
    root = Path(html_root)
    candidate = root / html_file
    if candidate.exists() and candidate.is_file():
        return str(candidate)
    return _html_file_index(html_root).get(Path(html_file).name, "")


def read_html_file(html_root: str | Path, html_file: str) -> str:
    if not html_file:
        return ""
    path_text = _find_html_file(str(Path(html_root)), html_file)
    if not path_text:
        return ""
    try:
        return Path(path_text).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def attach_html(frame: pd.DataFrame, html_root: str | Path) -> pd.DataFrame:
    result = frame.copy()
    result["html"] = [read_html_file(html_root, value) for value in result["html_file"]]
    result["html_length"] = result["html"].str.len()
    return result
