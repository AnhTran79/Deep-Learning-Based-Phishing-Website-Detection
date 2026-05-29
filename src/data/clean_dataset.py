from __future__ import annotations

import hashlib

import pandas as pd


def clean_dataset(frame: pd.DataFrame, min_html_length: int = 100, min_url_length: int = 4) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned["url"] = cleaned["url"].fillna("").astype(str).str.strip()
    cleaned["html_file"] = cleaned["html_file"].fillna("").astype(str).str.strip()
    if "html" not in cleaned.columns:
        cleaned["html"] = ""
    cleaned["html"] = cleaned["html"].fillna("").astype(str)
    cleaned["html_length"] = cleaned["html"].str.len()

    cleaned = cleaned[cleaned["url"].str.len() >= min_url_length]
    cleaned = cleaned[cleaned["html"].str.len() >= min_html_length]
    cleaned = cleaned.drop_duplicates(subset=["url"], keep="first")

    html_hashes = cleaned["html"].map(lambda value: hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest())
    cleaned = cleaned.loc[~html_hashes.duplicated()].copy()
    cleaned["source"] = "mendeley"

    preferred_columns = ["rec_id", "url", "html_file", "html", "label", "created_date", "html_length", "source"]
    existing_columns = [column for column in preferred_columns if column in cleaned.columns]
    remaining_columns = [column for column in cleaned.columns if column not in existing_columns]
    cleaned = cleaned[existing_columns + remaining_columns]

    print(f"clean_rows={len(cleaned)}")
    print("clean_label_counts=" + str(cleaned["label"].value_counts().sort_index().to_dict()))
    return cleaned
