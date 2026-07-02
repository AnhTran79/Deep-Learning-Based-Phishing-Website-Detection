from pathlib import Path

from fetch_weekly_candidates import (
    build_candidates,
    domain_to_url,
    normalize_url,
    parse_args,
    read_seen_urls,
)


def test_normalize_url_removes_fragment_and_default_port():
    assert normalize_url("HTTPS://Example.COM:443/login/#top") == "https://example.com/login"
    assert domain_to_url("example.com") == "https://example.com/"


def test_build_candidates_from_offline_openphish_and_tranco(tmp_path):
    openphish = tmp_path / "openphish.txt"
    openphish.write_text(
        "https://fake-login.test/path\n"
        "https://fake-login.test/path\n"
        "# ignored\n",
        encoding="utf-8",
    )
    tranco = tmp_path / "top-1m.csv"
    tranco.write_text("1,example.com\n2,example.org\n", encoding="utf-8")
    seen = tmp_path / "seen_urls.csv"
    seen.write_text("url,label,source,added_date\nhttps://example.org/,0,tranco,2026-01-01\n", encoding="utf-8")

    args = parse_args(
        [
            "--openphish-offline-file",
            str(openphish),
            "--tranco-file",
            str(tranco),
            "--seen-urls",
            str(seen),
            "--phishing-count",
            "5",
            "--legitimate-count",
            "5",
            "--random-state",
            "1",
        ]
    )

    rows, stats = build_candidates(args)

    assert {(row.url, row.label) for row in rows} == {
        ("https://fake-login.test/path", 1),
        ("https://example.com/", 0),
    }
    assert stats["duplicate_batch"] == 1
    assert stats["duplicate_seen"] == 1


def test_read_seen_urls_supports_plain_line_file(tmp_path):
    path = tmp_path / "seen.txt"
    path.write_text("https://example.com/\n", encoding="utf-8")

    assert read_seen_urls(path) == {"https://example.com/"}
