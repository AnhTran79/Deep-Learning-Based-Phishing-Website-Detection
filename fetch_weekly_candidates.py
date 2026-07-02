from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

DEFAULT_OPENPHISH_FEED = "https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt"
DEFAULT_OUT_DIR = Path("data/candidates")
DEFAULT_SEEN_URLS = DEFAULT_OUT_DIR / "seen_urls.csv"


@dataclass(frozen=True)
class Candidate:
    url: str
    label: int
    source: str


def normalize_url(value: str) -> str:
    raw = (value or "").replace("\ufeff", "").strip()
    if not raw:
        raise ValueError("empty_url")
    if "://" not in raw:
        raw = "https://" + raw
    parsed = urlsplit(raw)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("unsupported_scheme")
    if not parsed.hostname:
        raise ValueError("missing_hostname")
    host = parsed.hostname.encode("idna").decode("ascii").lower().rstrip(".")
    netloc = host
    if parsed.port and not ((parsed.scheme == "http" and parsed.port == 80) or (parsed.scheme == "https" and parsed.port == 443)):
        netloc = f"{host}:{parsed.port}"
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.scheme.lower()}://{netloc}{path}{query}"


def domain_to_url(value: str) -> str:
    raw = (value or "").replace("\ufeff", "").strip()
    if not raw:
        raise ValueError("empty_domain")
    if "://" in raw:
        return normalize_url(raw)
    domain = raw.split(",", 1)[0].strip()
    if re.search(r"\s", domain):
        raise ValueError("invalid_domain")
    return normalize_url(domain)


def fetch_text(url: str, timeout: float) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 PhishingDetectionResearchBot/1.0",
            "Accept": "text/plain,text/csv,*/*",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def load_openphish_urls(feed_url: str, timeout: float, offline_file: Path | None = None) -> list[str]:
    if offline_file:
        text = offline_file.read_text(encoding="utf-8", errors="replace")
    else:
        text = fetch_text(feed_url, timeout)
    urls = []
    for line in text.splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            urls.append(value)
    return urls


def load_tranco_urls(path: Path, limit: int) -> list[str]:
    urls: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            domain = row[1] if len(row) > 1 and row[0].strip().isdigit() else row[0]
            try:
                urls.append(domain_to_url(domain))
            except ValueError:
                continue
            if limit and len(urls) >= limit:
                break
    return urls


def load_labeled_urls(path: Path, url_column: str, label_column: str, limit: int = 0) -> list[Candidate]:
    rows: list[Candidate] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {path}")
        missing = {url_column, label_column} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            label = int(str(row[label_column]).strip())
            if label not in {0, 1}:
                continue
            rows.append(Candidate(normalize_url(row[url_column]), label, path.stem))
            if limit and len(rows) >= limit:
                break
    return rows


def read_seen_urls(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and "url" in reader.fieldnames:
            return {row["url"].strip() for row in reader if row.get("url")}
        handle.seek(0)
        return {line.strip() for line in handle if line.strip()}


def append_seen_urls(path: Path, rows: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["url", "label", "source", "added_date"])
        if not exists:
            writer.writeheader()
        today = date.today().isoformat()
        for row in rows:
            writer.writerow({"url": row.url, "label": row.label, "source": row.source, "added_date": today})


def deduplicate(rows: list[Candidate], seen_urls: set[str]) -> tuple[list[Candidate], dict[str, int]]:
    accepted: list[Candidate] = []
    in_batch: set[str] = set()
    stats = {"duplicate_seen": 0, "duplicate_batch": 0}
    for row in rows:
        if row.url in seen_urls:
            stats["duplicate_seen"] += 1
            continue
        if row.url in in_batch:
            stats["duplicate_batch"] += 1
            continue
        accepted.append(row)
        in_batch.add(row.url)
    return accepted, stats


def write_candidates(path: Path, rows: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["url", "label", "source"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"url": row.url, "label": row.label, "source": row.source})


def build_candidates(args: argparse.Namespace) -> tuple[list[Candidate], dict]:
    rng = random.Random(args.random_state)
    rows: list[Candidate] = []
    stats: dict[str, int | str] = {}

    phishing_rows: list[Candidate] = []
    if args.phishing_file:
        phishing_rows.extend(
            candidate
            for candidate in load_labeled_urls(Path(args.phishing_file), args.url_column, args.label_column)
            if candidate.label == 1
        )
    if not args.skip_openphish:
        try:
            phishing_rows.extend(
                Candidate(normalize_url(url), 1, "openphish")
                for url in load_openphish_urls(
                    args.openphish_feed,
                    args.timeout,
                    Path(args.openphish_offline_file) if args.openphish_offline_file else None,
                )
            )
        except (OSError, URLError, ValueError) as exc:
            if not args.allow_feed_failure:
                raise
            stats["openphish_error"] = str(exc)

    rng.shuffle(phishing_rows)
    rows.extend(phishing_rows[: args.phishing_count])

    legitimate_urls: list[str] = []
    if args.tranco_file:
        legitimate_urls.extend(load_tranco_urls(Path(args.tranco_file), args.tranco_max_rows))
    if args.legitimate_file:
        legitimate_urls.extend(candidate.url for candidate in load_labeled_urls(Path(args.legitimate_file), args.url_column, args.label_column) if candidate.label == 0)
    rng.shuffle(legitimate_urls)
    for url in legitimate_urls[: args.legitimate_count]:
        rows.append(Candidate(url, 0, "tranco" if args.tranco_file else "legitimate_file"))

    seen_urls = read_seen_urls(Path(args.seen_urls))
    rows, duplicate_stats = deduplicate(rows, seen_urls)
    rng.shuffle(rows)
    stats.update(duplicate_stats)
    stats.update(
        {
            "requested_phishing": args.phishing_count,
            "requested_legitimate": args.legitimate_count,
            "written": len(rows),
            "written_phishing": sum(row.label == 1 for row in rows),
            "written_legitimate": sum(row.label == 0 for row in rows),
        }
    )
    return rows, stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    today = date.today().isoformat().replace("-", "_")
    parser = argparse.ArgumentParser(description="Build weekly URL candidates for external collection.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR / f"week_{today}.csv"))
    parser.add_argument("--seen-urls", default=str(DEFAULT_SEEN_URLS))
    parser.add_argument("--openphish-feed", default=DEFAULT_OPENPHISH_FEED)
    parser.add_argument("--openphish-offline-file", default="")
    parser.add_argument("--skip-openphish", action="store_true")
    parser.add_argument("--allow-feed-failure", action="store_true")
    parser.add_argument("--phishing-file", default="", help="Optional CSV with url,label; label 1 rows are used.")
    parser.add_argument("--legitimate-file", default="", help="Optional CSV with url,label; label 0 rows are used.")
    parser.add_argument("--tranco-file", default="", help="Optional Tranco CSV, usually rank,domain.")
    parser.add_argument("--url-column", default="url")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--phishing-count", type=int, default=50)
    parser.add_argument("--legitimate-count", type=int, default=50)
    parser.add_argument("--tranco-max-rows", type=int, default=50000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--update-seen", action="store_true", help="Append written candidates to seen_urls.csv.")
    parser.add_argument("--summary", default="", help="Optional JSON summary output path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows, stats = build_candidates(args)
    if args.phishing_count and stats["written_phishing"] < args.phishing_count:
        print(f"warning: wrote only {stats['written_phishing']} phishing candidates", file=sys.stderr)
    if args.legitimate_count and stats["written_legitimate"] < args.legitimate_count:
        print(f"warning: wrote only {stats['written_legitimate']} legitimate candidates", file=sys.stderr)
    output = Path(args.out)
    write_candidates(output, rows)
    if args.update_seen:
        append_seen_urls(Path(args.seen_urls), rows)
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    print(" ".join(f"{key}={value}" for key, value in stats.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
