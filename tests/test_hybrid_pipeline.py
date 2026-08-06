from hybrid_pipeline import batch_name, paths


def test_batch_name_builds_pipeline_paths():
    batch = batch_name("2026_07_25")

    assert batch == "week_2026_07_25"
    assert paths(batch)["dataset"].as_posix() == "data/external_test/week_2026_07_25"
    assert paths(batch)["audit"].name == "week_2026_07_25_audit_queue.csv"
