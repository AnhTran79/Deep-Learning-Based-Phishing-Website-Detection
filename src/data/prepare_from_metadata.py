from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.clean_dataset import clean_dataset
from src.data.html_loader import attach_html
from src.data.load_metadata import load_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Mendeley URL + HTML dataset from existing metadata CSV.")
    parser.add_argument("--metadata", default="output/mendeley_metadata.csv")
    parser.add_argument("--html-root", default="dataset")
    parser.add_argument("--out", default="data/processed/mendeley_url_html_label.csv.gz")
    parser.add_argument("--clean-metadata-out", default="data/processed/mendeley_clean_metadata.csv")
    parser.add_argument("--min-html-length", type=int, default=100)
    parser.add_argument("--metadata-only", action="store_true", help="Write clean metadata only without embedding HTML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    html_root = Path(args.html_root)
    if not args.metadata_only:
        if not html_root.exists():
            raise FileNotFoundError(f"HTML root does not exist: {html_root}")
        metadata = attach_html(metadata, html_root)
        cleaned = clean_dataset(metadata, min_html_length=args.min_html_length)
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(output, index=False, compression="gzip")
        print(f"wrote {output}")
    else:
        cleaned = metadata.copy()
        cleaned["source"] = "mendeley"

    metadata_output = Path(args.clean_metadata_out)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_columns = [column for column in ["rec_id", "url", "html_file", "label", "created_date", "source"] if column in cleaned.columns]
    cleaned[metadata_columns].to_csv(metadata_output, index=False)
    print(f"wrote {metadata_output}")


if __name__ == "__main__":
    main()
