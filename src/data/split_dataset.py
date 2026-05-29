from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_loader import load_url_html_records
from src.data.split_data import split_records, write_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split prepared Mendeley dataset into train/val/test CSV files.")
    parser.add_argument("--input", default="data/processed/mendeley_url_html_label.csv.gz")
    parser.add_argument("--out-dir", default="data/splits")
    parser.add_argument("--html-root", default="", help="Optional HTML folder when input is clean metadata only.")
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = args.train_size + args.val_size + args.test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError("--train-size + --val-size + --test-size must equal 1.0")
    html_root = Path(args.html_root) if args.html_root else None
    records = load_url_html_records(Path(args.input), html_root=html_root, max_rows=args.max_rows)
    train, val, test = split_records(records, test_size=args.test_size, val_size=args.val_size, random_state=args.random_state)
    out_dir = Path(args.out_dir)
    write_records(out_dir / "train.csv", train)
    write_records(out_dir / "val.csv", val)
    write_records(out_dir / "test.csv", test)
    print(f"train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
