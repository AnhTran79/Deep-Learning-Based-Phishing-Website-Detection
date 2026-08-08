from __future__ import annotations

import argparse
import csv
import hashlib
import io
import ipaddress
import json
import random
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import SplitResult, urlsplit, urlunsplit

from PIL import Image, ImageStat, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path("data/external_test/dataset_50")
DEFAULT_PHISH360_INDEX = Path("data/processed/phish360_url_html_screenshot.csv")
DEFAULT_PHISH360_ROOT = Path("data/external/Phish360")

RESULT_FIELDS = [
    "sample_id",
    "label",
    "label_name",
    "source",
    "input_row",
    "original_url",
    "final_url",
    "normalized_original_url",
    "normalized_final_url",
    "domain",
    "html_sha256",
    "screenshot_dhash",
    "html_length",
    "screenshot_width",
    "screenshot_height",
    "collected_at",
]
REJECTED_FIELDS = [
    "input_row",
    "source",
    "original_url",
    "normalized_url",
    "raw_label",
    "mapped_label",
    "reason",
    "detail",
    "attempts",
    "rejected_at",
]
LABEL_NAMES = {0: "legitimate", 1: "phishing"}
ERROR_PAGE_SIGNATURES = (
    "this site can't be reached",
    "this page isn’t working",
    "this page isn't working",
    "err_name_not_resolved",
    "err_connection_",
    "installation error",
    "service worker gateway",
    "an error occurred while trying to install the service worker",
    "access denied",
    "error 404",
    "404 not found",
    "domain is parked",
    "domain parking",
    "verify you are human",
    "checking if the site connection is secure",
    "just a moment",
    "this account has been suspended",
    "account suspended",
    "domain suspended",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_url(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        raise ValueError("empty_url")
    if "://" not in raw:
        raw = "https://" + raw
    try:
        parsed = urlsplit(raw)
        if parsed.scheme.lower() not in {"http", "https"}:
            raise ValueError("unsupported_scheme")
        if not parsed.hostname:
            raise ValueError("missing_hostname")
        host = parsed.hostname.encode("idna").decode("ascii").lower().rstrip(".")
        port = parsed.port
    except (UnicodeError, ValueError) as exc:
        if str(exc) in {"unsupported_scheme", "missing_hostname"}:
            raise
        raise ValueError("invalid_url") from exc

    if is_unsafe_host(host):
        raise ValueError("unsafe_hostname")

    default_port = (parsed.scheme.lower() == "http" and port == 80) or (
        parsed.scheme.lower() == "https" and port == 443
    )
    netloc = host if port is None or default_port else f"{host}:{port}"
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    return urlunsplit(SplitResult(parsed.scheme.lower(), netloc, path, parsed.query, ""))


def is_unsafe_host(host: str) -> bool:
    lowered = host.lower().rstrip(".")
    if lowered in {"localhost", "localhost.localdomain"} or lowered.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(lowered)
    except ValueError:
        return False
    return not address.is_global


def domain_from_url(url: str) -> str:
    try:
        return (urlsplit(normalize_url(url)).hostname or "").lower()
    except ValueError:
        return ""


def map_label(
    value: object,
    legitimate_values: set[str] | None = None,
    phishing_values: set[str] | None = None,
) -> int:
    legitimate = legitimate_values or {"0", "legitimate", "legit", "good", "benign", "safe"}
    phishing = phishing_values or {"1", "phishing", "phish", "bad", "malicious"}
    normalized = str(value).strip().lower()
    if normalized in legitimate:
        return 0
    if normalized in phishing:
        return 1
    raise ValueError(f"unmapped_label:{value}")


def sha256_text(text: str) -> str:
    return hashlib.sha256(canonicalize_html(text).encode("utf-8", errors="ignore")).hexdigest()


def canonicalize_html(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def detect_error_page(html: str) -> str:
    normalized = " ".join((html or "").lower().split())
    for signature in ERROR_PAGE_SIGNATURES:
        if signature in normalized:
            return signature
    return ""


def screenshot_dhash(image: Image.Image, hash_size: int = 8) -> str:
    grayscale = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    if hasattr(grayscale, "get_flattened_data"):
        pixels = list(grayscale.get_flattened_data())
    else:  # Pillow < 12
        pixels = list(grayscale.getdata())
    bits = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for column in range(hash_size):
            bits = (bits << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return f"{bits:0{hash_size * hash_size // 4}x}"


def hash_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("hashes must have the same length")
    return (int(left, 16) ^ int(right, 16)).bit_count()


def is_near_duplicate(image_hash: str, known_hashes: Iterable[str], threshold: int) -> bool:
    return any(hash_distance(image_hash, known) <= threshold for known in known_hashes if len(known) == len(image_hash))


@dataclass(frozen=True)
class ImageInfo:
    dhash: str
    width: int
    height: int
    grayscale_stddev: float


def validate_screenshot(
    content: bytes,
    min_width: int = 64,
    min_height: int = 64,
    min_stddev: float = 2.0,
) -> ImageInfo:
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            width, height = image.size
            if width < min_width or height < min_height:
                raise ValueError("screenshot_too_small")
            grayscale = image.convert("L")
            extrema = grayscale.getextrema()
            stddev = float(ImageStat.Stat(grayscale).stddev[0])
            if extrema[1] - extrema[0] <= 2 or stddev < min_stddev:
                raise ValueError("blank_or_monochrome_screenshot")
            return ImageInfo(
                dhash=screenshot_dhash(image),
                width=width,
                height=height,
                grayscale_stddev=stddev,
            )
    except UnidentifiedImageError as exc:
        raise ValueError("invalid_screenshot") from exc


@dataclass
class LeakageIndex:
    urls: set[str] = field(default_factory=set)
    html_hashes: set[str] = field(default_factory=set)
    screenshot_hashes: set[str] = field(default_factory=set)
    phishing_training_domains: set[str] = field(default_factory=set)
    stats: Counter = field(default_factory=Counter)

    def to_json(self) -> dict:
        return {
            "version": 1,
            "created_at": utc_now(),
            "urls": sorted(self.urls),
            "html_hashes": sorted(self.html_hashes),
            "screenshot_hashes": sorted(self.screenshot_hashes),
            "phishing_training_domains": sorted(self.phishing_training_domains),
            "stats": dict(self.stats),
        }

    @classmethod
    def from_json(cls, payload: dict) -> "LeakageIndex":
        if payload.get("version") != 1:
            raise ValueError("unsupported leakage index version")
        return cls(
            urls=set(payload.get("urls", [])),
            html_hashes=set(payload.get("html_hashes", [])),
            screenshot_hashes=set(payload.get("screenshot_hashes", [])),
            phishing_training_domains=set(payload.get("phishing_training_domains", [])),
            stats=Counter(payload.get("stats", {})),
        )


def _add_url(index: LeakageIndex, value: str, stat_prefix: str) -> None:
    try:
        index.urls.add(normalize_url(value))
        index.stats[f"{stat_prefix}_urls"] += 1
    except ValueError:
        index.stats[f"{stat_prefix}_invalid_urls"] += 1


def _hash_html_file(index: LeakageIndex, path: Path, stat_prefix: str) -> None:
    if not path.is_file():
        index.stats[f"{stat_prefix}_missing_html"] += 1
        return
    try:
        html = path.read_text(encoding="utf-8", errors="replace")
        index.html_hashes.add(sha256_text(html))
        index.stats[f"{stat_prefix}_html"] += 1
    except OSError:
        index.stats[f"{stat_prefix}_html_errors"] += 1


def build_leakage_index(
    phish360_index: Path,
    project_root: Path = PROJECT_ROOT,
) -> LeakageIndex:
    index = LeakageIndex()
    if phish360_index.is_file():
        with phish360_index.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                url = row.get("url", "")
                _add_url(index, url, "phish360")
                label = (row.get("label") or "").strip()
                split = (row.get("split") or "").strip().lower()
                if label == "1" and split in {"train", "trainval", "training"}:
                    domain = domain_from_url(url)
                    if domain:
                        index.phishing_training_domains.add(domain)
                html_path = resolve_project_path(row.get("html_file", ""), project_root)
                screenshot_path = resolve_project_path(row.get("screenshot_file", ""), project_root)
                _hash_html_file(index, html_path, "phish360")
                if screenshot_path.is_file():
                    try:
                        with Image.open(screenshot_path) as image:
                            image.load()
                            index.screenshot_hashes.add(screenshot_dhash(image))
                            index.stats["phish360_screenshots"] += 1
                    except (OSError, UnidentifiedImageError):
                        index.stats["phish360_screenshot_errors"] += 1
                else:
                    index.stats["phish360_missing_screenshots"] += 1
    return index


def resolve_project_path(value: str, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path((value or "").strip())
    return path if path.is_absolute() else project_root / path


def load_or_build_leakage_index(args: argparse.Namespace) -> LeakageIndex:
    cache_path = Path(args.leakage_cache)
    if cache_path.is_file() and not args.refresh_leakage_index:
        return LeakageIndex.from_json(json.loads(cache_path.read_text(encoding="utf-8")))
    index = build_leakage_index(
        Path(args.phish360_index),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(cache_path, index.to_json())
    return index


@dataclass(frozen=True)
class Candidate:
    input_row: int
    original_url: str
    normalized_url: str
    raw_label: str
    label: int


def load_candidates(args: argparse.Namespace) -> tuple[list[Candidate], list[dict]]:
    candidates: list[Candidate] = []
    rejected: list[dict] = []
    seen: dict[str, int] = {}
    legitimate_values = split_values(args.legitimate_values)
    phishing_values = split_values(args.phishing_values)
    with Path(args.input).open("r", encoding=args.encoding, errors="replace", newline="") as handle:
        if args.input_format == "tranco":
            rows = (
                (row_number, row[1] if len(row) > 1 else "", "0")
                for row_number, row in enumerate(csv.reader(handle), start=1)
            )
        elif args.no_header:
            rows = (
                (
                    row_number,
                    row[args.url_column_index] if len(row) > args.url_column_index else "",
                    str(args.fixed_label) if args.fixed_label is not None else (
                        row[args.label_column_index] if len(row) > args.label_column_index else ""
                    ),
                )
                for row_number, row in enumerate(csv.reader(handle), start=1)
            )
        else:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            required = {args.url_column}
            if args.fixed_label is None:
                required.add(args.label_column)
            missing = required - fields
            if missing:
                raise ValueError(f"Input CSV missing columns: {', '.join(sorted(missing))}")
            rows = (
                (
                    row_number,
                    row.get(args.url_column, ""),
                    str(args.fixed_label) if args.fixed_label is not None else row.get(args.label_column, ""),
                )
                for row_number, row in enumerate(reader, start=2)
            )

        for processed_rows, (row_number, raw_url, raw_label) in enumerate(rows, start=1):
            if args.max_input_rows and processed_rows > args.max_input_rows:
                break
            try:
                label = map_label(raw_label, legitimate_values, phishing_values)
                normalized = normalize_url(raw_url)
                if normalized in seen:
                    reason = "duplicate_input_url" if seen[normalized] == label else "conflicting_input_label"
                    raise ValueError(reason)
                seen[normalized] = label
                candidates.append(Candidate(row_number, str(raw_url), normalized, str(raw_label), label))
            except ValueError as exc:
                rejected.append(
                    rejection_row(
                        input_row=row_number,
                        source=args.source,
                        original_url=str(raw_url),
                        normalized_url="",
                        raw_label=str(raw_label),
                        mapped_label="",
                        reason=str(exc).split(":", 1)[0],
                        detail=str(exc),
                    )
                )
    random.Random(args.random_state).shuffle(candidates)
    return candidates, rejected


def split_values(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def rejection_row(
    *,
    input_row: int | str,
    source: str,
    original_url: str,
    normalized_url: str,
    raw_label: str,
    mapped_label: int | str,
    reason: str,
    detail: str = "",
    attempts: int = 0,
) -> dict:
    return {
        "input_row": input_row,
        "source": source,
        "original_url": original_url,
        "normalized_url": normalized_url,
        "raw_label": raw_label,
        "mapped_label": mapped_label,
        "reason": reason,
        "detail": detail,
        "attempts": attempts,
        "rejected_at": utc_now(),
    }


@dataclass
class ResumeState:
    accepted_rows: list[dict] = field(default_factory=list)
    rejected_rows: list[dict] = field(default_factory=list)
    attempted_keys: set[tuple[str, int]] = field(default_factory=set)
    urls: set[str] = field(default_factory=set)
    html_hashes: set[str] = field(default_factory=set)
    screenshot_hashes: set[str] = field(default_factory=set)
    counts: Counter = field(default_factory=Counter)
    max_ids: Counter = field(default_factory=Counter)
    used_ids: dict[int, set[int]] = field(default_factory=lambda: {0: set(), 1: set()})


def load_resume_state(output_dir: Path, resume: bool = True) -> ResumeState:
    state = ResumeState()
    if not resume:
        return state
    results_path = output_dir / "collection_results.csv"
    rejected_path = output_dir / "rejected_samples.csv"
    if results_path.is_file():
        with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                sample_dir = output_dir / (row.get("sample_id") or "")
                sample_id = row.get("sample_id") or ""
                required = [
                    sample_dir / "URL" / f"{sample_id.split('_', 1)[0]}.txt",
                    sample_dir / "RAW-HTML" / f"{sample_id.split('_', 1)[0]}.html",
                    sample_dir / "SCREEN-SHOT" / f"{sample_id.split('_', 1)[0]}.png",
                    sample_dir / "Label" / f"{sample_id.split('_', 1)[0]}.txt",
                    sample_dir / "metadata.json",
                ]
                if not sample_id or not all(path.is_file() for path in required):
                    continue
                state.accepted_rows.append(row)
                label = int(row["label"])
                state.counts[label] += 1
                prefix = "L" if label == 0 else "P"
                numeric_id = sample_id.split("_", 1)[0].removeprefix(prefix)
                if numeric_id.isdigit():
                    numeric_value = int(numeric_id)
                    state.max_ids[label] = max(state.max_ids[label], numeric_value)
                    state.used_ids[label].add(numeric_value)
                for key in ("normalized_original_url", "normalized_final_url"):
                    if row.get(key):
                        state.urls.add(row[key])
                if row.get("html_sha256"):
                    state.html_hashes.add(row["html_sha256"])
                if row.get("screenshot_dhash"):
                    state.screenshot_hashes.add(row["screenshot_dhash"])
                state.attempted_keys.add((row.get("normalized_original_url", ""), label))
    if rejected_path.is_file():
        with rejected_path.open("r", encoding="utf-8-sig", newline="") as handle:
            state.rejected_rows.extend(csv.DictReader(handle))
        for row in state.rejected_rows:
            normalized = row.get("normalized_url", "")
            mapped = row.get("mapped_label", "")
            if normalized and str(mapped) in {"0", "1"}:
                state.attempted_keys.add((normalized, int(mapped)))
    return state


def merge_rejections(existing: list[dict], incoming: Iterable[dict]) -> None:
    keys = {
        (
            str(row.get("input_row", "")),
            row.get("normalized_url", ""),
            row.get("raw_label", ""),
            row.get("reason", ""),
        )
        for row in existing
    }
    for row in incoming:
        key = (
            str(row.get("input_row", "")),
            row.get("normalized_url", ""),
            row.get("raw_label", ""),
            row.get("reason", ""),
        )
        if key not in keys:
            existing.append(row)
            keys.add(key)


@dataclass(frozen=True)
class PageCapture:
    final_url: str
    html: str
    screenshot: bytes
    status: int | None
    content_type: str = ""


class PlaywrightCollector:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._playwright = None
        self._browser = None
        self._context = None

    def __enter__(self) -> "PlaywrightCollector":
        try:
            from playwright.sync_api import sync_playwright
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Playwright is required. Run: pip install -r requirements.txt && playwright install chromium"
            ) from exc
        self._playwright = sync_playwright().start()
        launch_options = {"headless": self.args.headless}
        if self.args.browser_channel != "chromium":
            launch_options["channel"] = self.args.browser_channel
        self._browser = self._playwright.chromium.launch(**launch_options)
        self._context = self._browser.new_context(
            accept_downloads=False,
            ignore_https_errors=self.args.ignore_https_errors,
            java_script_enabled=True,
            permissions=[],
            service_workers=self.args.service_workers,
            viewport={"width": self.args.viewport_width, "height": self.args.viewport_height},
        )
        self._context.set_default_timeout(self.args.timeout_ms)
        self._context.set_default_navigation_timeout(self.args.timeout_ms)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    def capture(self, url: str) -> PageCapture:
        page = self._context.new_page()
        navigation_responses = []
        page.on("dialog", lambda dialog: dialog.dismiss())
        page.on("popup", lambda popup: popup.close())
        page.on("download", lambda download: download.cancel())
        page.on(
            "response",
            lambda response: navigation_responses.append(response)
            if response.request.is_navigation_request() and response.frame == page.main_frame
            else None,
        )

        def route_request(route) -> None:
            request_url = route.request.url
            try:
                parsed = urlsplit(request_url)
                if parsed.scheme in {"http", "https"} and parsed.hostname and is_unsafe_host(parsed.hostname):
                    route.abort()
                    return
            except ValueError:
                route.abort()
                return
            if route.request.resource_type in {"media"}:
                route.abort()
                return
            route.continue_()

        page.route("**/*", route_request)
        try:
            response = page.goto(url, wait_until="domcontentloaded", timeout=self.args.timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=min(self.args.timeout_ms, 5000))
            except Exception:
                pass
            final_url = page.url
            final_response = navigation_responses[-1] if navigation_responses else response
            try:
                html = final_response.text() if final_response else page.content()
            except Exception:
                html = page.content()
            screenshot = page.screenshot(type="png", full_page=self.args.full_page)
            return PageCapture(
                final_url=final_url,
                html=html,
                screenshot=screenshot,
                status=final_response.status if final_response else None,
                content_type=(final_response.headers.get("content-type", "") if final_response else ""),
            )
        finally:
            page.close()


def collect_with_retry(collector: PlaywrightCollector, candidate: Candidate, args: argparse.Namespace) -> tuple[PageCapture, int]:
    last_error: Exception | None = None
    for attempt in range(1, args.retries + 2):
        try:
            return collector.capture(candidate.normalized_url), attempt
        except Exception as exc:
            last_error = exc
            if attempt <= args.retries:
                time.sleep(args.retry_delay_seconds * attempt)
    raise RuntimeError(f"{type(last_error).__name__}: {last_error}")


def next_sample_id(label: int, used_ids: dict[int, set[int]]) -> tuple[str, str]:
    prefix = "L" if label == 0 else "P"
    name = LABEL_NAMES[label]
    numeric_id = 1
    while numeric_id in used_ids[label]:
        numeric_id += 1
    base_id = f"{prefix}{numeric_id:05d}"
    return base_id, f"{base_id}_{name}"


def write_sample(
    output_dir: Path,
    base_id: str,
    sample_id: str,
    candidate: Candidate,
    capture: PageCapture,
    result: dict,
) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{sample_id}.", dir=output_dir))
    try:
        for folder in ("URL", "RAW-HTML", "SCREEN-SHOT", "Label"):
            (temporary / folder).mkdir()
        (temporary / "URL" / f"{base_id}.txt").write_text(capture.final_url + "\n", encoding="utf-8")
        with (temporary / "RAW-HTML" / f"{base_id}.html").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            handle.write(canonicalize_html(capture.html))
        (temporary / "SCREEN-SHOT" / f"{base_id}.png").write_bytes(capture.screenshot)
        (temporary / "Label" / f"{base_id}.txt").write_text(str(candidate.label) + "\n", encoding="utf-8")
        metadata = dict(result)
        metadata["external_test_only"] = True
        metadata["http_status"] = capture.status
        metadata["content_type"] = capture.content_type
        metadata["input_label"] = candidate.raw_label
        write_json_atomic(temporary / "metadata.json", metadata)
        destination = output_dir / sample_id
        if destination.exists():
            raise FileExistsError(f"Sample directory already exists: {destination}")
        publish_directory(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def publish_directory(source: Path, destination: Path, attempts: int = 5) -> None:
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            source.replace(destination)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.25 * (attempt + 1))

    try:
        shutil.copytree(source, destination)
        source_files = sorted(
            path.relative_to(source) for path in source.rglob("*") if path.is_file()
        )
        destination_files = sorted(
            path.relative_to(destination) for path in destination.rglob("*") if path.is_file()
        )
        if source_files != destination_files:
            raise OSError("published sample file list does not match temporary sample")
        shutil.rmtree(source)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        if last_error is not None:
            raise last_error
        raise


def write_csv_atomic(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def persist_state(output_dir: Path, state: ResumeState, args: argparse.Namespace, leakage: LeakageIndex) -> None:
    write_csv_atomic(output_dir / "collection_results.csv", state.accepted_rows, RESULT_FIELDS)
    write_csv_atomic(output_dir / "rejected_samples.csv", state.rejected_rows, REJECTED_FIELDS)
    summary = {
        "dataset_role": "external_test_only",
        "prohibited_uses": ["training", "validation", "model_selection", "threshold_tuning"],
        "source": args.source,
        "source_counts": dict(Counter(row.get("source", "") for row in state.accepted_rows)),
        "input": str(Path(args.input).resolve()),
        "input_format": args.input_format,
        "fixed_label": args.fixed_label,
        "requested": {"legitimate": args.legitimate_count, "phishing": args.phishing_count},
        "accepted": {"legitimate": state.counts[0], "phishing": state.counts[1]},
        "rejected": len(state.rejected_rows),
        "complete": state.counts[0] >= args.legitimate_count and state.counts[1] >= args.phishing_count,
        "random_state": args.random_state,
        "min_html_length": args.min_html_length,
        "screenshot_hash": {"algorithm": "dhash64", "max_hamming_distance": args.phash_threshold},
        "leakage_index_stats": dict(leakage.stats),
        "updated_at": utc_now(),
    }
    write_json_atomic(output_dir / "dataset_summary.json", summary)


def reject_candidate(state: ResumeState, candidate: Candidate, args: argparse.Namespace, reason: str, detail: str = "", attempts: int = 0) -> None:
    state.rejected_rows.append(
        rejection_row(
            input_row=candidate.input_row,
            source=args.source,
            original_url=candidate.original_url,
            normalized_url=candidate.normalized_url,
            raw_label=candidate.raw_label,
            mapped_label=candidate.label,
            reason=reason,
            detail=detail,
            attempts=attempts,
        )
    )
    state.attempted_keys.add((candidate.normalized_url, candidate.label))


def create_zip(output_dir: Path) -> Path:
    archive = output_dir.with_suffix(".zip")
    temporary_base = output_dir.parent / f".{output_dir.name}"
    temporary_zip = Path(shutil.make_archive(str(temporary_base), "zip", root_dir=output_dir.parent, base_dir=output_dir.name))
    temporary_zip.replace(archive)
    return archive


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output).resolve()
    if not args.resume and (output_dir / "collection_results.csv").exists():
        raise FileExistsError(
            f"Output already contains collection state: {output_dir}. "
            "Use --resume or choose an empty output directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    leakage = load_or_build_leakage_index(args)
    state = load_resume_state(output_dir, resume=args.resume)
    candidates, initial_rejections = load_candidates(args)
    if not args.resume:
        state.rejected_rows = []
    merge_rejections(state.rejected_rows, initial_rejections)

    processed = 0
    with PlaywrightCollector(args) as collector:
        for candidate in candidates:
            if state.counts[0] >= args.legitimate_count and state.counts[1] >= args.phishing_count:
                break
            if args.max_candidates and processed >= args.max_candidates:
                break
            if state.counts[candidate.label] >= (
                args.legitimate_count if candidate.label == 0 else args.phishing_count
            ):
                continue
            key = (candidate.normalized_url, candidate.label)
            if key in state.attempted_keys:
                continue
            processed += 1

            if candidate.normalized_url in leakage.urls or candidate.normalized_url in state.urls:
                reject_candidate(state, candidate, args, "duplicate_original_url")
                persist_state(output_dir, state, args, leakage)
                continue
            domain = domain_from_url(candidate.normalized_url)
            if (
                candidate.label == 1
                and args.exclude_phishing_training_domains
                and domain in leakage.phishing_training_domains
            ):
                reject_candidate(state, candidate, args, "phishing_training_domain")
                persist_state(output_dir, state, args, leakage)
                continue

            try:
                capture, attempts = collect_with_retry(collector, candidate, args)
            except Exception as exc:
                reject_candidate(state, candidate, args, "crawl_error", str(exc), args.retries + 1)
                persist_state(output_dir, state, args, leakage)
                continue

            try:
                final_url = normalize_url(capture.final_url)
                if capture.status is not None and capture.status >= 400 and not args.accept_http_errors:
                    raise ValueError(f"http_status_{capture.status}")
                if capture.content_type and "html" not in capture.content_type.lower():
                    raise ValueError("non_html_content_type")
                if final_url in leakage.urls or final_url in state.urls:
                    raise ValueError("duplicate_final_url")
                if len(capture.html) < args.min_html_length:
                    raise ValueError("html_too_short")
                error_signature = detect_error_page(capture.html)
                if error_signature:
                    raise ValueError(f"error_page:{error_signature}")
                html_hash = sha256_text(capture.html)
                if html_hash in leakage.html_hashes or html_hash in state.html_hashes:
                    raise ValueError("duplicate_html")
                image_info = validate_screenshot(
                    capture.screenshot,
                    min_width=args.min_screenshot_width,
                    min_height=args.min_screenshot_height,
                    min_stddev=args.min_screenshot_stddev,
                )
                known_image_hashes = leakage.screenshot_hashes | state.screenshot_hashes
                if is_near_duplicate(image_info.dhash, known_image_hashes, args.phash_threshold):
                    raise ValueError("duplicate_screenshot")
                final_domain = domain_from_url(final_url)
                if (
                    candidate.label == 1
                    and args.exclude_phishing_training_domains
                    and final_domain in leakage.phishing_training_domains
                ):
                    raise ValueError("phishing_training_domain_after_redirect")
            except ValueError as exc:
                reject_candidate(state, candidate, args, str(exc), attempts=attempts)
                persist_state(output_dir, state, args, leakage)
                continue

            base_id, sample_id = next_sample_id(candidate.label, state.used_ids)
            result = {
                "sample_id": sample_id,
                "label": candidate.label,
                "label_name": LABEL_NAMES[candidate.label],
                "source": args.source,
                "input_row": candidate.input_row,
                "original_url": candidate.original_url,
                "final_url": final_url,
                "normalized_original_url": candidate.normalized_url,
                "normalized_final_url": final_url,
                "domain": domain_from_url(final_url),
                "html_sha256": html_hash,
                "screenshot_dhash": image_info.dhash,
                "html_length": len(capture.html),
                "screenshot_width": image_info.width,
                "screenshot_height": image_info.height,
                "collected_at": utc_now(),
            }
            write_sample(output_dir, base_id, sample_id, candidate, capture, result)
            state.accepted_rows.append(result)
            state.counts[candidate.label] += 1
            numeric_id = int(base_id[1:])
            state.max_ids[candidate.label] = max(state.max_ids[candidate.label], numeric_id)
            state.used_ids[candidate.label].add(numeric_id)
            state.urls.update({candidate.normalized_url, final_url})
            state.html_hashes.add(html_hash)
            state.screenshot_hashes.add(image_info.dhash)
            state.attempted_keys.add(key)
            persist_state(output_dir, state, args, leakage)
            print(
                f"accepted {sample_id} "
                f"legitimate={state.counts[0]}/{args.legitimate_count} "
                f"phishing={state.counts[1]}/{args.phishing_count}"
            )

    persist_state(output_dir, state, args, leakage)
    complete = state.counts[0] >= args.legitimate_count and state.counts[1] >= args.phishing_count
    if args.zip and complete:
        archive = create_zip(output_dir)
        print(f"wrote {archive}")
    print(f"complete={complete} legitimate={state.counts[0]} phishing={state.counts[1]}")
    return 0 if complete else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect a leakage-resistant URL + HTML + screenshot external test dataset."
    )
    parser.add_argument("--input", required=True, help="Input CSV containing URL and label columns.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--input-format",
        choices=("labeled-csv", "tranco"),
        default="labeled-csv",
        help="Use 'tranco' for headerless rank,domain files; all rows map to legitimate.",
    )
    parser.add_argument("--url-column", default="url")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--url-column-index", type=int, default=0)
    parser.add_argument("--label-column-index", type=int, default=1)
    parser.add_argument("--fixed-label", type=int, choices=(0, 1))
    parser.add_argument(
        "--max-input-rows",
        type=int,
        default=0,
        help="Read at most this many source rows before shuffling; 0 reads all rows.",
    )
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--legitimate-values", default="0,legitimate,legit,good,benign,safe")
    parser.add_argument("--phishing-values", default="1,phishing,phish,bad,malicious")
    parser.add_argument("--legitimate-count", type=int, default=25)
    parser.add_argument("--phishing-count", type=int, default=25)
    parser.add_argument("--source", default="external")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=1.0)
    parser.add_argument("--min-html-length", type=int, default=100)
    parser.add_argument("--min-screenshot-width", type=int, default=64)
    parser.add_argument("--min-screenshot-height", type=int, default=64)
    parser.add_argument("--min-screenshot-stddev", type=float, default=2.0)
    parser.add_argument("--phash-threshold", type=int, default=5)
    parser.add_argument("--viewport-width", type=int, default=1366)
    parser.add_argument("--viewport-height", type=int, default=768)
    parser.add_argument("--full-page", action="store_true")
    parser.add_argument(
        "--browser-channel",
        choices=("chromium", "chrome", "msedge"),
        default="chromium",
        help="Use bundled Chromium or an installed Chrome/Edge channel.",
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore-https-errors", action="store_true")
    parser.add_argument(
        "--service-workers",
        choices=("allow", "block"),
        default="allow",
        help="Allow by default for faithful rendering; each browser context is temporary.",
    )
    parser.add_argument(
        "--accept-http-errors",
        action="store_true",
        help="Allow final navigation responses with HTTP status >= 400.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument(
        "--exclude-phishing-training-domains",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--phish360-index", default=str(DEFAULT_PHISH360_INDEX))
    parser.add_argument(
        "--leakage-cache",
        default="data/external_test/phish360_leakage_index.json",
        help="Cached hashes and URLs derived from Phish360.",
    )
    parser.add_argument("--refresh-leakage-index", action="store_true")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Limit attempted candidates for smoke tests; 0 means unlimited.",
    )
    args = parser.parse_args(argv)
    if args.input_format == "tranco":
        args.no_header = True
        args.url_column_index = 1
        args.fixed_label = 0
    if args.legitimate_count < 0 or args.phishing_count < 0:
        parser.error("requested counts must be non-negative")
    if args.retries < 0 or args.timeout_ms <= 0:
        parser.error("retries must be non-negative and timeout must be positive")
    return args


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except KeyboardInterrupt:
        print("collection interrupted; progress already persisted for --resume", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
