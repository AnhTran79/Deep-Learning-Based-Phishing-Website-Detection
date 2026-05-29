from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from urllib.parse import urlparse

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:  # pragma: no cover
    BeautifulSoup = None


IP_PATTERN = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
SCHEME_PATTERN = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
HREF_PATTERN = re.compile(r"""href\s*=\s*["']([^"']*)["']""", re.IGNORECASE)
FORM_ACTION_PATTERN = re.compile(r"""<form\b[^>]*action\s*=\s*["']([^"']*)["']""", re.IGNORECASE)


@dataclass(frozen=True)
class HandcraftedFeatures:
    url_length: int
    num_dots: int
    num_hyphens: int
    num_digits: int
    num_special_chars: int
    has_https: int
    has_ip_address: int
    num_subdomains: int
    contains_login: int
    contains_verify: int
    contains_secure: int
    contains_update: int
    html_length: int
    num_forms: int
    num_password_inputs: int
    num_iframes: int
    num_scripts: int
    num_links: int
    num_external_links: int
    num_empty_links: int
    num_form_actions: int
    num_suspicious_form_actions: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


FEATURE_NAMES = list(HandcraftedFeatures.__dataclass_fields__.keys())


def normalize_url(url: str) -> str:
    value = (url or "").strip()
    if value and not SCHEME_PATTERN.match(value):
        value = "https://" + value
    return value


def _host(url: str) -> str:
    try:
        return (urlparse(normalize_url(url)).hostname or "").lower()
    except ValueError:
        return ""


def _html_counts_with_bs4(html: str, base_host: str) -> dict[str, int]:
    soup = BeautifulSoup(html, "html.parser")
    forms = soup.find_all("form")
    links = soup.find_all("a")
    actions = [(form.get("action") or "").strip() for form in forms if form.get("action") is not None]
    external_links = 0
    empty_links = 0
    for link in links:
        href = (link.get("href") or "").strip()
        if not href or href in {"#", "javascript:void(0)", "javascript:;"}:
            empty_links += 1
            continue
        href_host = _host(href)
        if href_host and base_host and href_host != base_host:
            external_links += 1
    suspicious_actions = sum(1 for action in actions if _is_suspicious_action(action, base_host))
    return {
        "num_forms": len(forms),
        "num_password_inputs": len(soup.find_all("input", attrs={"type": re.compile("password", re.I)})),
        "num_iframes": len(soup.find_all("iframe")),
        "num_scripts": len(soup.find_all("script")),
        "num_links": len(links),
        "num_external_links": external_links,
        "num_empty_links": empty_links,
        "num_form_actions": len(actions),
        "num_suspicious_form_actions": suspicious_actions,
    }


def _html_counts_with_regex(html: str, base_host: str) -> dict[str, int]:
    lowered = html.lower()
    hrefs = HREF_PATTERN.findall(html)
    actions = FORM_ACTION_PATTERN.findall(html)
    external_links = 0
    empty_links = 0
    for href in hrefs:
        href = href.strip()
        if not href or href in {"#", "javascript:void(0)", "javascript:;"}:
            empty_links += 1
            continue
        href_host = _host(href)
        if href_host and base_host and href_host != base_host:
            external_links += 1
    return {
        "num_forms": lowered.count("<form"),
        "num_password_inputs": lowered.count('type="password"') + lowered.count("type='password'"),
        "num_iframes": lowered.count("<iframe"),
        "num_scripts": lowered.count("<script"),
        "num_links": len(hrefs),
        "num_external_links": external_links,
        "num_empty_links": empty_links,
        "num_form_actions": len(actions),
        "num_suspicious_form_actions": sum(1 for action in actions if _is_suspicious_action(action, base_host)),
    }


def _is_suspicious_action(action: str, base_host: str) -> bool:
    lowered = action.lower().strip()
    if not lowered or lowered in {"#", "javascript:void(0)", "javascript:;"}:
        return True
    action_host = _host(lowered)
    return bool(action_host and base_host and action_host != base_host)


def extract_handcrafted_features(url: str, html: str | None = None) -> HandcraftedFeatures:
    normalized = normalize_url(url)
    parsed = urlparse(normalized)
    host = (parsed.hostname or "").lower()
    labels = [label for label in host.split(".") if label]
    lowered_url = normalized.lower()
    html_text = html or ""
    html_counts = (
        _html_counts_with_bs4(html_text, host)
        if html_text and BeautifulSoup is not None
        else _html_counts_with_regex(html_text, host)
    )
    return HandcraftedFeatures(
        url_length=len(normalized),
        num_dots=normalized.count("."),
        num_hyphens=normalized.count("-"),
        num_digits=sum(char.isdigit() for char in normalized),
        num_special_chars=sum(1 for char in normalized if not char.isalnum()),
        has_https=1 if parsed.scheme == "https" else 0,
        has_ip_address=1 if IP_PATTERN.match(host.split(":")[0] or "") else 0,
        num_subdomains=max(len(labels) - 2, 0),
        contains_login=1 if "login" in lowered_url else 0,
        contains_verify=1 if "verify" in lowered_url else 0,
        contains_secure=1 if "secure" in lowered_url else 0,
        contains_update=1 if "update" in lowered_url else 0,
        html_length=len(html_text),
        **html_counts,
    )


def handcrafted_feature_vector(url: str, html: str | None = None) -> list[float]:
    values = extract_handcrafted_features(url, html).to_dict()
    return [float(values[name]) for name in FEATURE_NAMES]
