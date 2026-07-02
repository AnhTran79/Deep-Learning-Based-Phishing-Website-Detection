from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
APPROVED_STATUSES = {"approved", "corrected"}


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_audit_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            status = (row.get("audit_status") or "").strip().lower()
            if status in APPROVED_STATUSES:
                rows[row["sample_id"]] = row
    return rows


def external_rows(dataset: Path, audit_rows: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    dataset_id = dataset.name
    with (dataset / "collection_results.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            audit = audit_rows.get(row["sample_id"])
            if not audit:
                continue
            sample_id = row["sample_id"]
            base_id = sample_id.split("_", 1)[0]
            label_text = (audit.get("audited_label") or row["label"]).strip()
            if label_text not in {"0", "1"}:
                raise ValueError(f"Invalid audited_label for {sample_id}: {label_text}")
            rows.append(
                {
                    "rec_id": f"audited_{dataset_id}_{sample_id}",
                    "split": "trainval",
                    "url": row["final_url"],
                    "html_file": portable(dataset / sample_id / "RAW-HTML" / f"{base_id}.html"),
                    "screenshot_file": portable(dataset / sample_id / "SCREEN-SHOT" / f"{base_id}.png"),
                    "label": int(label_text),
                    "target_brand": "audited_external",
                    "source": "audited_external",
                }
            )
    return rows


def read_base_rows(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {path}")
        return list(reader), list(reader.fieldnames)


def write_index(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build(args: argparse.Namespace) -> dict:
    base_rows, fieldnames = read_base_rows(Path(args.base_index))
    audit_rows = read_audit_rows(Path(args.audit_csv))
    new_rows = external_rows(Path(args.dataset), audit_rows)
    existing_ids = {row["rec_id"] for row in base_rows}
    deduped_new_rows = [row for row in new_rows if row["rec_id"] not in existing_ids]
    combined = [*base_rows, *deduped_new_rows]
    write_index(Path(args.out), combined, fieldnames)
    summary = {
        "base_index": args.base_index,
        "dataset": args.dataset,
        "audit_csv": args.audit_csv,
        "out": args.out,
        "base_rows": len(base_rows),
        "approved_external_rows": len(deduped_new_rows),
        "combined_rows": len(combined),
    }
    Path(args.out).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append audited external samples to a Phish360-compatible training index.")
    parser.add_argument("--base-index", default="data/processed/phish360_url_html_screenshot.csv")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--out", default="data/processed/phish360_plus_audited.csv")
    return parser.parse_args()


def main() -> None:
    summary = build(parse_args())
    print(
        f"wrote {summary['out']} base={summary['base_rows']} "
        f"approved_external={summary['approved_external_rows']} combined={summary['combined_rows']}"
    )


if __name__ == "__main__":
    main()
