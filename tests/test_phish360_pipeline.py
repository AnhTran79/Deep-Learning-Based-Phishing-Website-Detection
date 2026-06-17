from pathlib import Path

from src.data.prepare_phish360 import iter_phish360_rows, write_index
from src.data.multimodal_dataset_loader import load_multimodal_records


def _write_sample(root: Path, split: str, name: str, raw_label: str) -> None:
    sample = root / split / name
    (sample / "URL").mkdir(parents=True)
    (sample / "RAW-HTML").mkdir()
    (sample / "SCREEN-SHOT").mkdir()
    (sample / "Label").mkdir()
    (sample / "URL" / "url.txt").write_text("https://example.test/login", encoding="utf-8")
    (sample / "RAW-HTML" / "index.html").write_text("<html><form></form></html>", encoding="utf-8")
    (sample / "SCREEN-SHOT" / "screen_shoot.png").write_bytes(b"not-used-by-this-test")
    (sample / "Label" / "label.txt").write_text(raw_label, encoding="utf-8")


def test_phish360_index_maps_binary_label_from_folder_name(tmp_path):
    _write_sample(tmp_path, "trainval", "L00001_legitimate", "legitimate")
    _write_sample(tmp_path, "test", "P10001_santander", "santander")

    rows, stats = iter_phish360_rows(tmp_path)

    assert stats["written"] == 2
    assert {row.rec_id: row.label for row in rows} == {
        "L00001_legitimate": 0,
        "P10001_santander": 1,
    }
    assert {row.rec_id: row.target_brand for row in rows}["P10001_santander"] == "santander"


def test_multimodal_loader_reads_prepared_phish360_index(tmp_path):
    _write_sample(tmp_path, "trainval", "P10001_santander", "santander")
    rows, _ = iter_phish360_rows(tmp_path)
    index_path = tmp_path / "phish360.csv"
    write_index(rows, index_path)

    records = load_multimodal_records(index_path, split="trainval")

    assert len(records) == 1
    assert records[0].label == 1
    assert records[0].target_brand == "santander"
    assert records[0].html == "<html><form></form></html>"
