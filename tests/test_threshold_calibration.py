import csv

from calibrate_thresholds import calibrate, parse_args
from evaluate_external_dataset import load_threshold_overrides


def test_calibrate_thresholds_writes_json_and_csv(tmp_path):
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "true_label",
                "predicted_label",
                "correct",
                "phishing_probability",
                "threshold",
                "model",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"sample_id": "L1", "true_label": 0, "predicted_label": 0, "correct": 1, "phishing_probability": 0.10, "threshold": 0.5, "model": "m1"},
                {"sample_id": "L2", "true_label": 0, "predicted_label": 1, "correct": 0, "phishing_probability": 0.55, "threshold": 0.5, "model": "m1"},
                {"sample_id": "P1", "true_label": 1, "predicted_label": 1, "correct": 1, "phishing_probability": 0.70, "threshold": 0.5, "model": "m1"},
                {"sample_id": "P2", "true_label": 1, "predicted_label": 1, "correct": 1, "phishing_probability": 0.95, "threshold": 0.5, "model": "m1"},
            ]
        )
    output = tmp_path / "thresholds.json"
    args = parse_args(["--predictions", str(predictions), "--out", str(output), "--objective", "f1"])

    result = calibrate(args)

    assert output.is_file()
    assert output.with_suffix(".csv").is_file()
    assert "m1" in result["models"]
    assert 0.55 < result["models"]["m1"]["threshold"] <= 0.70


def test_load_threshold_overrides_reads_calibration_json(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(
        '{"models": {"phish360_tri_branch_cnn": {"threshold": 0.73}}}',
        encoding="utf-8",
    )

    assert load_threshold_overrides(str(path)) == {"phish360_tri_branch_cnn": 0.73}
