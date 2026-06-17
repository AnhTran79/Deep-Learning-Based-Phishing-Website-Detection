from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_DATASET_ROOT = Path("data/external/Phish360")
DEFAULT_OUTPUT = Path("data/processed/phish360_url_html_screenshot.csv")


@dataclass(frozen=True)
class Phish360IndexRow:
    rec_id: str
    split: str
    url: str
    html_file: str
    screenshot_file: str
    label: int
    target_brand: str
    source: str = "phish360"


def _read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig", errors="replace").strip()
    return text.replace("\ufeff", "").replace("ï»¿", "")


def _label_from_sample_name(sample_name: str) -> int:
    if sample_name.startswith("L"):
        return 0
    if sample_name.startswith("P"):
        return 1
    raise ValueError(f"Cannot infer binary label from sample folder: {sample_name}")


def _portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def iter_phish360_rows(root: Path, require_html: bool = True) -> tuple[list[Phish360IndexRow], dict[str, int]]:
    rows: list[Phish360IndexRow] = []
    stats = {
        "samples": 0,
        "written": 0,
        "missing_url": 0,
        "missing_html": 0,
        "missing_screenshot": 0,
        "missing_label": 0,
        "invalid_sample_name": 0,
    }
    for split in ("trainval", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for sample_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            stats["samples"] += 1
            url_path = sample_dir / "URL" / "url.txt"
            html_path = sample_dir / "RAW-HTML" / "index.html"
            screenshot_path = sample_dir / "SCREEN-SHOT" / "screen_shoot.png"
            label_path = sample_dir / "Label" / "label.txt"

            if not url_path.exists():
                stats["missing_url"] += 1
                continue
            if not label_path.exists():
                stats["missing_label"] += 1
                continue
            if not screenshot_path.exists():
                stats["missing_screenshot"] += 1
                continue
            if require_html and not html_path.exists():
                stats["missing_html"] += 1
                continue

            try:
                label = _label_from_sample_name(sample_dir.name)
            except ValueError:
                stats["invalid_sample_name"] += 1
                continue

            rows.append(
                Phish360IndexRow(
                    rec_id=sample_dir.name,
                    split=split,
                    url=_read_text(url_path),
                    html_file=_portable_path(html_path),
                    screenshot_file=_portable_path(screenshot_path),
                    label=label,
                    target_brand=_read_text(label_path),
                )
            )
            stats["written"] += 1
    return rows, stats


def write_index(rows: list[Phish360IndexRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rec_id",
        "split",
        "url",
        "html_file",
        "screenshot_file",
        "label",
        "target_brand",
        "source",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a URL + HTML + screenshot index CSV for Phish360.")
    parser.add_argument("--root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--allow-missing-html",
        action="store_true",
        help="Keep samples without RAW-HTML/index.html. Use only for screenshot or URL-only experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Phish360 root not found: {root}")
    rows, stats = iter_phish360_rows(root, require_html=not args.allow_missing_html)
    write_index(rows, Path(args.out))
    print(f"wrote {args.out}")
    print(
        " ".join(
            f"{key}={value}"
            for key, value in stats.items()
        )
    )


if __name__ == "__main__":
    main()
