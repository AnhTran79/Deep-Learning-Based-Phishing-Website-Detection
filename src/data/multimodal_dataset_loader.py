from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MultimodalRecord:
    rec_id: str
    split: str
    url: str
    html: str
    screenshot_file: str
    label: int
    target_brand: str = ""
    source: str = "phish360"


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return PROJECT_ROOT / path


def load_multimodal_records(
    data_path: Path,
    split: str | None = None,
    max_rows: int = 0,
    html_max_chars: int = 0,
) -> list[MultimodalRecord]:
    records: list[MultimodalRecord] = []
    base_dir = data_path.resolve().parent
    with data_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"rec_id", "split", "url", "html_file", "screenshot_file", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            if split and row.get("split") != split:
                continue
            url = (row.get("url") or "").strip()
            label_text = (row.get("label") or "").strip()
            html_path = _resolve_path(row.get("html_file") or "", base_dir)
            screenshot_path = _resolve_path(row.get("screenshot_file") or "", base_dir)
            if not url or label_text == "" or not html_path.exists() or not screenshot_path.exists():
                continue
            label = int(label_text)
            if label not in {0, 1}:
                raise ValueError(f"Invalid label {label}; expected 0 or 1.")
            html = html_path.read_text(encoding="utf-8", errors="replace")
            if html_max_chars > 0:
                html = html[:html_max_chars]
            records.append(
                MultimodalRecord(
                    rec_id=row.get("rec_id") or "",
                    split=row.get("split") or "",
                    url=url,
                    html=html,
                    screenshot_file=str(screenshot_path),
                    label=label,
                    target_brand=row.get("target_brand") or "",
                    source=row.get("source") or "phish360",
                )
            )
            if max_rows and len(records) >= max_rows:
                break
    if not records:
        split_text = f" split={split}" if split else ""
        raise ValueError(f"No usable rows found in dataset: {data_path}{split_text}")
    return records
