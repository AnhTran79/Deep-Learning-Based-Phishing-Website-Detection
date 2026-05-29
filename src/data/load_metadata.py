from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"url", "html_file", "label"}
COLUMN_ALIASES = {
    "website": "html_file",
    "result": "label",
    "class": "label",
    "target": "label",
    "id": "rec_id",
}


def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = frame.rename(columns={column: COLUMN_ALIASES.get(column.strip(), column.strip()) for column in frame.columns})
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Metadata CSV is missing required columns: {', '.join(sorted(missing))}")

    frame["url"] = frame["url"].fillna("").astype(str).str.strip()
    frame["html_file"] = frame["html_file"].fillna("").astype(str).str.strip()
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame = frame[frame["label"].isin([0, 1])].copy()
    frame["label"] = frame["label"].astype(int)
    frame = frame[(frame["url"] != "") & (frame["html_file"] != "")]
    frame = frame.drop_duplicates(subset=["url"], keep="first")

    print(f"metadata_rows={len(frame)}")
    print("label_counts=" + str(frame["label"].value_counts().sort_index().to_dict()))
    return frame
