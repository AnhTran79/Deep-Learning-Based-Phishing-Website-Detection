from __future__ import annotations

import csv
import gzip
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


@dataclass(frozen=True)
class UrlHtmlRecord:
    url: str
    html: str
    label: int
    html_file: str = ""
    rec_id: str = ""
    source: str = "mendeley"


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


@lru_cache(maxsize=200_000)
def _html_file_index(html_root: str) -> dict[str, str]:
    root = Path(html_root)
    if not root.exists():
        return {}
    return {path.name: str(path) for path in root.rglob("*.html") if path.is_file()}


@lru_cache(maxsize=200_000)
def _resolve_html_file(html_root: str, html_file: str) -> str:
    root = Path(html_root)
    candidate = root / html_file
    if candidate.exists() and candidate.is_file():
        return str(candidate)
    return _html_file_index(html_root).get(Path(html_file).name, "")


def load_url_html_records(
    data_path: Path,
    html_root: Path | None = None,
    max_rows: int = 0,
    html_max_chars: int = 0,
) -> list[UrlHtmlRecord]:
    _raise_csv_field_limit()
    records: list[UrlHtmlRecord] = []
    with _open_text(data_path) as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"No header found in dataset: {data_path}")
        required = {"url", "label"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Dataset missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            url = (row.get("url") or "").strip()
            label_text = (row.get("label") or "").strip()
            if not url or label_text == "":
                continue
            label = int(label_text)
            if label not in {0, 1}:
                raise ValueError(f"Invalid label {label}; expected 0 or 1.")
            html = row.get("html") or ""
            html_file = row.get("html_file") or ""
            if not html and html_root is not None and html_file:
                candidate = _resolve_html_file(str(html_root), html_file)
                if candidate:
                    html = Path(candidate).read_text(encoding="utf-8", errors="replace")
            if html_max_chars > 0:
                html = html[:html_max_chars]
            records.append(
                UrlHtmlRecord(
                    url=url,
                    html=html,
                    label=label,
                    html_file=html_file,
                    rec_id=row.get("rec_id") or "",
                    source=row.get("source") or "mendeley",
                )
            )
            if max_rows and len(records) >= max_rows:
                break
    if not records:
        raise ValueError(f"No usable rows found in dataset: {data_path}")
    return records


def iter_text_labels(records: Iterable[UrlHtmlRecord]) -> tuple[list[str], list[int]]:
    texts = [f"{record.url} {record.html}" for record in records]
    labels = [record.label for record in records]
    return texts, labels
