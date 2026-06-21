from pathlib import Path

from evaluate_external_dataset import load_external_records, selected_model_specs


def test_load_external_records_reads_frozen_dataset():
    root = Path("data/external_test/dataset_100_clean")

    records = load_external_records(root)

    assert len(records) == 100
    assert sum(record.label == 0 for record in records) == 50
    assert sum(record.label == 1 for record in records) == 50
    assert all(record.html and record.screenshot_file.is_file() for record in records)


def test_select_only_tri_branch_multimodal_model():
    specs = selected_model_specs(["phish360_tri_branch_cnn"])

    assert len(specs) == 1
    assert specs[0][1] == "phish360_tri_branch_cnn"
    assert specs[0][3] == "URL + HTML + Screenshot"
