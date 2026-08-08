import csv
import io
import json
from argparse import Namespace
from pathlib import Path

from PIL import Image

import collect_external_dataset as collector_module
from collect_external_dataset import (
    RESULT_FIELDS,
    build_leakage_index,
    canonicalize_html,
    detect_error_page,
    hash_distance,
    is_near_duplicate,
    load_candidates,
    load_resume_state,
    map_label,
    merge_rejections,
    normalize_url,
    next_sample_id,
    PageCapture,
    parse_args,
    publish_directory,
    run,
    sha256_text,
    screenshot_dhash,
    validate_screenshot,
    write_csv_atomic,
)


def test_url_normalization_removes_fragment_default_port_and_trailing_slash():
    assert normalize_url("HTTPS://Example.COM:443/login/#section") == "https://example.com/login"
    assert normalize_url("example.com") == "https://example.com/"


def test_label_mapping_supports_binary_and_kaggle_good_bad():
    assert map_label("0") == 0
    assert map_label("good") == 0
    assert map_label("1") == 1
    assert map_label("BAD") == 1


def test_perceptual_hash_detects_identical_and_near_duplicate_images():
    first = Image.new("RGB", (80, 80), "black")
    second = first.copy()
    second.putpixel((0, 0), (255, 255, 255))
    first_hash = screenshot_dhash(first)
    second_hash = screenshot_dhash(second)

    assert hash_distance(first_hash, first_hash) == 0
    assert is_near_duplicate(second_hash, {first_hash}, threshold=2)


def test_white_screenshot_is_rejected(tmp_path):
    path = tmp_path / "white.png"
    Image.new("RGB", (200, 100), "white").save(path)

    try:
        validate_screenshot(path.read_bytes())
    except ValueError as exc:
        assert str(exc) == "blank_or_monochrome_screenshot"
    else:
        raise AssertionError("white screenshot should be rejected")


def test_gateway_error_page_is_detected():
    html = "<html><h1>Installation Error</h1><p>Service Worker Gateway</p></html>"

    assert detect_error_page(html) == "installation error"


def test_html_hash_is_stable_across_line_endings():
    assert canonicalize_html("a\r\nb\rc") == "a\nb\nc"
    assert sha256_text("a\r\nb") == sha256_text("a\nb")


def test_publish_directory_preserves_files(tmp_path):
    source = tmp_path / ".sample.tmp"
    destination = tmp_path / "P00001_phishing"
    (source / "URL").mkdir(parents=True)
    (source / "URL" / "P00001.txt").write_text("https://example.test/", encoding="utf-8")

    publish_directory(source, destination)

    assert not source.exists()
    assert (destination / "URL" / "P00001.txt").read_text(encoding="utf-8") == "https://example.test/"


def test_resume_loads_only_complete_samples(tmp_path):
    output = tmp_path / "dataset_50"
    sample_id = "L00001_legitimate"
    base_id = "L00001"
    sample = output / sample_id
    for folder in ("URL", "RAW-HTML", "SCREEN-SHOT", "Label"):
        (sample / folder).mkdir(parents=True, exist_ok=True)
    (sample / "URL" / f"{base_id}.txt").write_text("https://example.com/\n", encoding="utf-8")
    (sample / "RAW-HTML" / f"{base_id}.html").write_text("<html>valid</html>", encoding="utf-8")
    Image.new("RGB", (80, 80), "blue").save(sample / "SCREEN-SHOT" / f"{base_id}.png")
    (sample / "Label" / f"{base_id}.txt").write_text("0\n", encoding="utf-8")
    (sample / "metadata.json").write_text(json.dumps({"sample_id": sample_id}), encoding="utf-8")
    row = {field: "" for field in RESULT_FIELDS}
    row.update(
        {
            "sample_id": sample_id,
            "label": "0",
            "normalized_original_url": "https://example.com/",
            "normalized_final_url": "https://example.com/",
            "html_sha256": "a" * 64,
            "screenshot_dhash": "b" * 16,
        }
    )
    write_csv_atomic(output / "collection_results.csv", [row], RESULT_FIELDS)

    state = load_resume_state(output)

    assert state.counts[0] == 1
    assert "https://example.com/" in state.urls
    assert "a" * 64 in state.html_hashes


def test_resume_does_not_duplicate_the_same_rejection():
    existing = [
        {
            "input_row": "2",
            "normalized_url": "",
            "raw_label": "unknown",
            "reason": "unmapped_label",
        }
    ]

    merge_rejections(existing, [dict(existing[0])])

    assert len(existing) == 1


def test_next_sample_id_fills_resume_gap():
    base_id, sample_id = next_sample_id(1, {0: set(), 1: {1, 2, 4}})

    assert base_id == "P00003"
    assert sample_id == "P00003_phishing"


def test_input_duplicate_is_rejected_even_when_labels_conflict(tmp_path):
    input_path = tmp_path / "urls.csv"
    input_path.write_text(
        "url,label\n"
        "example.com,0\n"
        "https://example.com/,1\n",
        encoding="utf-8",
    )
    args = Namespace(
        input=str(input_path),
        encoding="utf-8",
        url_column="url",
        label_column="label",
        legitimate_values="0",
        phishing_values="1",
        random_state=42,
        source="test",
        input_format="labeled-csv",
        no_header=False,
        url_column_index=0,
        label_column_index=1,
        fixed_label=None,
        max_input_rows=0,
    )

    candidates, rejected = load_candidates(args)

    assert len(candidates) == 1
    assert rejected[0]["reason"] == "conflicting_input_label"


def test_tranco_input_maps_headerless_domains_to_legitimate(tmp_path):
    input_path = tmp_path / "top-1m.csv"
    input_path.write_text("1,google.com\n2,example.com\n", encoding="utf-8")
    args = parse_args(
        [
            "--input",
            str(input_path),
            "--input-format",
            "tranco",
            "--max-input-rows",
            "2",
        ]
    )

    candidates, rejected = load_candidates(args)

    assert rejected == []
    assert {candidate.normalized_url for candidate in candidates} == {
        "https://google.com/",
        "https://example.com/",
    }
    assert {candidate.label for candidate in candidates} == {0}


def test_leakage_index_reads_phish360_hashes(tmp_path):
    phish_html = tmp_path / "p.html"
    phish_html.write_text("<html>phish360</html>", encoding="utf-8")
    phish_image = tmp_path / "p.png"
    image = Image.new("RGB", (80, 80), "white")
    for position in range(80):
        image.putpixel((position, position), (0, 0, 0))
    image.save(phish_image)
    phish360 = tmp_path / "phish360.csv"
    with phish360.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["url", "html_file", "screenshot_file", "label", "split"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "url": "https://phish.test/login",
                "html_file": str(phish_html),
                "screenshot_file": str(phish_image),
                "label": 1,
                "split": "trainval",
            }
        )

    index = build_leakage_index(phish360, project_root=tmp_path)

    assert "https://phish.test/login" in index.urls
    assert "phish.test" in index.phishing_training_domains
    assert len(index.html_hashes) == 1
    assert len(index.screenshot_hashes) == 1


def test_end_to_end_smoke_writes_two_samples_and_summary(tmp_path):
    input_path = tmp_path / "urls.csv"
    input_path.write_text(
        "url,label\n"
        "legitimate.test,0\n"
        "phishing.test/login,1\n",
        encoding="utf-8",
    )
    output = tmp_path / "dataset_50"
    cache = tmp_path / "leakage.json"
    cache.write_text(
        json.dumps(
            {
                "version": 1,
                "urls": [],
                "html_hashes": [],
                "screenshot_hashes": [],
                "phishing_training_domains": [],
                "stats": {},
            }
        ),
        encoding="utf-8",
    )

    def png_bytes(vertical):
        image = Image.new("RGB", (120, 80), "white")
        if vertical:
            for x in range(0, 120, 4):
                for y in range(80):
                    image.putpixel((x, y), (0, 0, 0))
        else:
            for y in range(0, 80, 4):
                for x in range(120):
                    image.putpixel((x, y), (0, 0, 0))
        content = io.BytesIO()
        image.save(content, format="PNG")
        return content.getvalue()

    class FakeCollector:
        def __init__(self, args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def capture(self, url):
            phishing = "phishing" in url
            return PageCapture(
                final_url=url,
                html="<html>" + ("phishing" if phishing else "legitimate") * 20 + "</html>",
                screenshot=png_bytes(vertical=phishing),
                status=200,
            )

    args = parse_args(
        [
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--legitimate-count",
            "1",
            "--phishing-count",
            "1",
            "--leakage-cache",
            str(cache),
            "--random-state",
            "1",
        ]
    )
    original_collector = collector_module.PlaywrightCollector
    collector_module.PlaywrightCollector = FakeCollector
    try:
        exit_code = run(args)
    finally:
        collector_module.PlaywrightCollector = original_collector

    assert exit_code == 0
    assert (output / "L00001_legitimate" / "RAW-HTML" / "L00001.html").is_file()
    assert (output / "P00001_phishing" / "SCREEN-SHOT" / "P00001.png").is_file()
    summary = json.loads((output / "dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["complete"] is True
    assert summary["dataset_role"] == "external_test_only"
