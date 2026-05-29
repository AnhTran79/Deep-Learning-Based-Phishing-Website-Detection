from __future__ import annotations

import argparse
import csv
import gzip
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_loader import UrlHtmlRecord, load_url_html_records


def write_records(path: Path, records: list[UrlHtmlRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["url", "html", "html_file", "label", "rec_id", "source"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "url": record.url,
                    "html": record.html,
                    "html_file": record.html_file,
                    "label": record.label,
                    "rec_id": record.rec_id,
                    "source": record.source,
                }
            )


def split_records(
    records: list[UrlHtmlRecord],
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[list[UrlHtmlRecord], list[UrlHtmlRecord], list[UrlHtmlRecord]]:
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be positive and leave room for training data.")

    rng = random.Random(random_state)
    by_label = {
        0: [record for record in records if record.label == 0],
        1: [record for record in records if record.label == 1],
    }
    if not by_label[0] or not by_label[1]:
        raise ValueError("Need both label 0 and label 1 for stratified splitting.")

    train: list[UrlHtmlRecord] = []
    val: list[UrlHtmlRecord] = []
    test: list[UrlHtmlRecord] = []
    for label_records in by_label.values():
        shuffled = label_records[:]
        rng.shuffle(shuffled)
        test_count = max(1, round(len(shuffled) * test_size))
        val_count = max(1, round(len(shuffled) * val_size))
        test.extend(shuffled[:test_count])
        val.extend(shuffled[test_count : test_count + val_count])
        train.extend(shuffled[test_count + val_count :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split URL + HTML dataset into train/validation/test files.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", default="data/processed/splits")
    parser.add_argument("--html-root", default="")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    html_root = Path(args.html_root) if args.html_root else None
    records = load_url_html_records(Path(args.data), html_root=html_root, max_rows=args.max_rows)
    train, val, test = split_records(records, args.test_size, args.val_size, args.random_state)
    out_dir = Path(args.out_dir)
    write_records(out_dir / "train.csv.gz", train)
    write_records(out_dir / "validation.csv.gz", val)
    write_records(out_dir / "test.csv.gz", test)
    print(f"train={len(train)} validation={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
