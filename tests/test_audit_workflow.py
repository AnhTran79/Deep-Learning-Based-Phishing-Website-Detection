import csv
from argparse import Namespace

from build_audited_training_index import build
from create_audit_queue import create_queue


def _write_sample(root, sample_id, label):
    base_id = sample_id.split("_", 1)[0]
    sample = root / sample_id
    for folder in ("URL", "RAW-HTML", "SCREEN-SHOT", "Label"):
        (sample / folder).mkdir(parents=True, exist_ok=True)
    (sample / "URL" / f"{base_id}.txt").write_text("https://example.test/", encoding="utf-8")
    (sample / "RAW-HTML" / f"{base_id}.html").write_text("<html>ok</html>", encoding="utf-8")
    (sample / "SCREEN-SHOT" / f"{base_id}.png").write_bytes(b"png")
    (sample / "Label" / f"{base_id}.txt").write_text(str(label), encoding="utf-8")


def test_create_audit_queue_prioritizes_model_errors(tmp_path):
    dataset = tmp_path / "batch"
    dataset.mkdir()
    _write_sample(dataset, "L00001_legitimate", 0)
    with (dataset / "collection_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "label", "original_url", "final_url", "domain", "html_length"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "L00001_legitimate",
                "label": "0",
                "original_url": "https://example.test/",
                "final_url": "https://example.test/",
                "domain": "example.test",
                "html_length": "5000",
            }
        )
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "model", "true_label", "predicted_label", "correct", "phishing_probability", "threshold"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "L00001_legitimate",
                "model": "phish360_tri_branch_cnn",
                "true_label": "0",
                "predicted_label": "1",
                "correct": "0",
                "phishing_probability": "0.9",
                "threshold": "0.5",
            }
        )

    rows = create_queue(
        Namespace(dataset=str(dataset), predictions=str(predictions), focus_model="phish360_tri_branch_cnn", limit=0)
    )

    assert rows[0]["flags"] == "false_positive"
    assert rows[0]["priority"] == 2


def test_build_audited_training_index_appends_only_approved_rows(tmp_path):
    dataset = tmp_path / "batch"
    dataset.mkdir()
    _write_sample(dataset, "P00001_phishing", 1)
    with (dataset / "collection_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "label", "final_url"],
        )
        writer.writeheader()
        writer.writerow({"sample_id": "P00001_phishing", "label": "1", "final_url": "https://phish.test/"})
    audit = tmp_path / "audit.csv"
    audit.write_text(
        "sample_id,audit_status,audited_label\n"
        "P00001_phishing,approved,1\n",
        encoding="utf-8",
    )
    base = tmp_path / "base.csv"
    base.write_text(
        "rec_id,split,url,html_file,screenshot_file,label,target_brand,source\n"
        "base1,trainval,https://base.test/,base.html,base.png,0,,phish360\n",
        encoding="utf-8",
    )
    out = tmp_path / "combined.csv"

    summary = build(Namespace(base_index=str(base), dataset=str(dataset), audit_csv=str(audit), out=str(out)))

    assert summary["approved_external_rows"] == 1
    assert "audited_batch_P00001_phishing" in out.read_text(encoding="utf-8")
