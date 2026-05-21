from __future__ import annotations

import math
import re
import socket
from collections import Counter
from dataclasses import asdict, dataclass
from html import unescape
from urllib.parse import urlparse
from urllib.request import Request, urlopen


SUSPICIOUS_WORDS = {
    "account",
    "admin",
    "amazon",
    "apple",
    "bank",
    "binance",
    "bonus",
    "coinbase",
    "confirm",
    "credential",
    "delivery",
    "dhl",
    "dropbox",
    "facebook",
    "gift",
    "google",
    "invoice",
    "login",
    "metamask",
    "microsoft",
    "netflix",
    "office",
    "onedrive",
    "outlook",
    "paypal",
    "password",
    "recover",
    "secure",
    "security",
    "signin",
    "suspended",
    "unlock",
    "update",
    "usps",
    "verify",
    "wallet",
}

IMPERSONATED_BRAND_WORDS = {
    "adobe",
    "airdrop",
    "apple",
    "binance",
    "coinbase",
    "docusign",
    "ledger",
    "metamask",
    "microsoft",
    "office",
    "onedrive",
    "paypal",
    "seed",
    "treezo",
    "trezo",
    "trezor",
    "trezr",
    "wallet",
}

PUBLIC_HOSTING_DOMAINS = {
    "blogspot.com",
    "firebaseapp.com",
    "github.io",
    "glitch.me",
    "netlify.app",
    "pages.dev",
    "replit.app",
    "sites.google.com",
    "vercel.app",
    "web.app",
    "weebly.com",
    "wixsite.com",
    "wixstudio.com",
    "wordpress.com",
}

URL_SHORTENER_DOMAINS = {
    "bit.ly",
    "cutt.ly",
    "is.gd",
    "ow.ly",
    "rebrand.ly",
    "shorturl.at",
    "t.co",
    "tiny.cc",
    "tinyurl.com",
}

SUSPICIOUS_TLDS = {
    "cam",
    "cfd",
    "click",
    "club",
    "cyou",
    "fit",
    "gq",
    "icu",
    "live",
    "ml",
    "mom",
    "quest",
    "rest",
    "sbs",
    "shop",
    "tk",
    "top",
    "work",
    "xyz",
    "zip",
}

REDIRECT_PARAM_NAMES = {
    "continue",
    "dest",
    "destination",
    "next",
    "redirect",
    "redirect_uri",
    "return",
    "returnurl",
    "target",
    "url",
}

HTML_SUSPICIOUS_WORDS = {
    "confirm",
    "credential",
    "login",
    "password",
    "recover",
    "secure",
    "security",
    "signin",
    "suspended",
    "unlock",
    "update",
    "verify",
    "wallet",
}

IP_PATTERN = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
LONG_TOKEN_PATTERN = re.compile(r"[a-z0-9]{18,}", re.IGNORECASE)
TAG_PATTERN = re.compile(r"<\s*([a-z0-9]+)\b", re.IGNORECASE)
HREF_PATTERN = re.compile(r"""href\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
TITLE_PATTERN = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class UrlFeatures:
    url_length: int
    domain_length: int
    path_length: int
    query_length: int
    num_dots: int
    num_hyphens: int
    num_digits: int
    num_special_chars: int
    has_https: int
    has_ip: int
    has_at: int
    num_subdomains: int
    has_suspicious_tld: int
    has_url_shortener: int
    has_punycode: int
    has_encoded_chars: int
    has_redirect_param: int
    has_embedded_url: int
    has_long_token: int
    has_public_hosting_platform: int
    has_brand_impersonation: int
    suspicious_word_count: int
    entropy: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class HtmlFeatures:
    html_available: int
    html_length: int
    title_length: int
    num_forms: int
    num_password_inputs: int
    num_iframes: int
    num_scripts: int
    num_external_links: int
    has_login_form: int
    has_meta_refresh: int
    has_javascript_redirect: int
    suspicious_html_word_count: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


URL_FEATURE_NAMES = list(UrlFeatures.__dataclass_fields__.keys())
HTML_FEATURE_NAMES = list(HtmlFeatures.__dataclass_fields__.keys())
COMBINED_FEATURE_NAMES = [*URL_FEATURE_NAMES, *HTML_FEATURE_NAMES]


def normalize_url(url: str) -> str:
    value = url.strip()
    if not value:
        return value
    if "://" not in value:
        value = "https://" + value
    return value


def safe_urlparse(url: str):
    normalized = normalize_url(url)
    try:
        return urlparse(normalized)
    except ValueError:
        cleaned = normalized.replace("[", "").replace("]", "")
        try:
            return urlparse(cleaned)
        except ValueError:
            return urlparse("https://invalid.local/")


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    total = len(value)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _hostname_matches(hostname: str, domain: str) -> bool:
    return hostname == domain or hostname.endswith("." + domain)


def extract_url_features(url: str) -> UrlFeatures:
    normalized = normalize_url(url)
    parsed = safe_urlparse(normalized)
    domain = parsed.netloc.lower()
    hostname = (parsed.hostname or domain).lower()
    path = parsed.path or ""
    query = parsed.query or ""
    lowered = normalized.lower()
    labels = [label for label in hostname.split(".") if label]
    tld = labels[-1] if labels else ""
    is_ip = IP_PATTERN.match(hostname.split(":")[0] or "") is not None

    special_chars = sum(1 for char in normalized if not char.isalnum())
    suspicious_words = sum(1 for word in SUSPICIOUS_WORDS if word in lowered)
    has_public_hosting = any(_hostname_matches(hostname, domain) for domain in PUBLIC_HOSTING_DOMAINS)
    has_brand_impersonation = any(word in lowered for word in IMPERSONATED_BRAND_WORDS)
    redirect_param = any(f"{name}=" in query.lower() for name in REDIRECT_PARAM_NAMES)
    embedded_url = lowered.count("http://") + lowered.count("https://") > 1
    long_token_source = f"{hostname}{path}{query}"

    return UrlFeatures(
        url_length=len(normalized),
        domain_length=len(domain),
        path_length=len(path),
        query_length=len(query),
        num_dots=normalized.count("."),
        num_hyphens=normalized.count("-"),
        num_digits=sum(char.isdigit() for char in normalized),
        num_special_chars=special_chars,
        has_https=1 if parsed.scheme == "https" else 0,
        has_ip=1 if is_ip else 0,
        has_at=1 if "@" in normalized else 0,
        num_subdomains=0 if is_ip else max(len(labels) - 2, 0),
        has_suspicious_tld=1 if tld in SUSPICIOUS_TLDS else 0,
        has_url_shortener=1 if hostname in URL_SHORTENER_DOMAINS else 0,
        has_punycode=1 if "xn--" in hostname else 0,
        has_encoded_chars=1 if "%" in normalized else 0,
        has_redirect_param=1 if redirect_param else 0,
        has_embedded_url=1 if embedded_url else 0,
        has_long_token=1 if LONG_TOKEN_PATTERN.search(long_token_source) else 0,
        has_public_hosting_platform=1 if has_public_hosting else 0,
        has_brand_impersonation=1 if has_brand_impersonation else 0,
        suspicious_word_count=suspicious_words,
        entropy=round(shannon_entropy(normalized), 4),
    )


def fetch_html(url: str, timeout: float = 4.0, max_bytes: int = 500_000) -> str | None:
    normalized = normalize_url(url)
    request = Request(
        normalized,
        headers={
            "User-Agent": "Mozilla/5.0 PhishingDetectionResearchBot/1.0",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("content-type", "")
            if "html" not in content_type.lower():
                return None
            body = response.read(max_bytes)
            charset = response.headers.get_content_charset() or "utf-8"
            return body.decode(charset, errors="replace")
    except Exception:
        return None
    finally:
        socket.setdefaulttimeout(previous_timeout)


def extract_html_features(html: str | None, base_url: str = "") -> HtmlFeatures:
    if not html:
        return HtmlFeatures(
            html_available=0,
            html_length=0,
            title_length=0,
            num_forms=0,
            num_password_inputs=0,
            num_iframes=0,
            num_scripts=0,
            num_external_links=0,
            has_login_form=0,
            has_meta_refresh=0,
            has_javascript_redirect=0,
            suspicious_html_word_count=0,
        )

    lowered = html.lower()
    text = unescape(re.sub(r"<[^>]+>", " ", lowered))
    tags = Counter(tag.lower() for tag in TAG_PATTERN.findall(html))
    title_match = TITLE_PATTERN.search(html)
    title = unescape(re.sub(r"\s+", " ", title_match.group(1))).strip() if title_match else ""
    hrefs = HREF_PATTERN.findall(html)
    base_host = (safe_urlparse(base_url).hostname or "").lower()
    external_links = 0
    for href in hrefs:
        try:
            parsed = urlparse(href)
        except ValueError:
            continue
        host = (parsed.hostname or "").lower()
        if host and base_host and host != base_host:
            external_links += 1

    has_password = "type=\"password\"" in lowered or "type='password'" in lowered
    has_login_word = any(word in text for word in ("login", "signin", "password", "account"))
    suspicious_words = sum(1 for word in HTML_SUSPICIOUS_WORDS if word in text)

    return HtmlFeatures(
        html_available=1,
        html_length=len(html),
        title_length=len(title),
        num_forms=tags["form"],
        num_password_inputs=lowered.count("type=\"password\"") + lowered.count("type='password'"),
        num_iframes=tags["iframe"],
        num_scripts=tags["script"],
        num_external_links=external_links,
        has_login_form=1 if has_password or (tags["form"] and has_login_word) else 0,
        has_meta_refresh=1 if "http-equiv=\"refresh\"" in lowered or "http-equiv='refresh'" in lowered else 0,
        has_javascript_redirect=1 if "window.location" in lowered or "location.href" in lowered else 0,
        suspicious_html_word_count=suspicious_words,
    )


def combined_feature_dict(url: str, html: str | None = None) -> dict[str, int | float]:
    url_features = extract_url_features(url).to_dict()
    html_features = extract_html_features(html, base_url=url).to_dict()
    return {**url_features, **html_features}


def combined_feature_vector(url: str, html: str | None = None) -> list[float]:
    values = combined_feature_dict(url, html)
    return [float(values[name]) for name in COMBINED_FEATURE_NAMES]


def heuristic_phishing_score(features: UrlFeatures, html_features: HtmlFeatures | None = None) -> float:
    score = 0.02
    score += min(features.url_length / 160, 1.0) * 0.12
    score += min((features.path_length + features.query_length) / 90, 1.0) * 0.08
    score += min(features.num_special_chars / 28, 1.0) * 0.10
    score += min(features.num_digits / 10, 1.0) * 0.10
    score += min(features.num_dots / 5, 1.0) * 0.06
    score += min(features.num_hyphens / 4, 1.0) * 0.06
    score += min(features.num_subdomains / 3, 1.0) * 0.08
    score += min(features.suspicious_word_count / 2, 1.0) * 0.24
    score += 0.18 if features.has_ip else 0.0
    score += 0.18 if features.has_at else 0.0
    score += 0.12 if not features.has_https else 0.0
    score += 0.20 if features.has_url_shortener else 0.0
    score += 0.14 if features.has_redirect_param else 0.0
    score += 0.22 if features.has_embedded_url else 0.0
    score += 0.14 if features.has_long_token else 0.0
    score += 0.18 if features.has_brand_impersonation else 0.0
    score += 0.12 if features.has_public_hosting_platform and features.num_subdomains else 0.0
    score += 0.34 if features.has_public_hosting_platform and features.has_brand_impersonation else 0.0
    score += 0.12 if features.has_punycode else 0.0
    score += 0.10 if features.has_encoded_chars else 0.0
    score += 0.10 if features.has_suspicious_tld else 0.0
    score += min(max(features.entropy - 3.4, 0.0) / 2.4, 1.0) * 0.06

    if html_features is not None and html_features.html_available:
        score += min(html_features.num_forms / 3, 1.0) * 0.06
        score += min(html_features.num_password_inputs / 2, 1.0) * 0.12
        score += min(html_features.num_iframes / 2, 1.0) * 0.06
        score += min(html_features.num_external_links / 20, 1.0) * 0.06
        score += 0.12 if html_features.has_login_form else 0.0
        score += 0.10 if html_features.has_meta_refresh else 0.0
        score += 0.10 if html_features.has_javascript_redirect else 0.0
        score += min(html_features.suspicious_html_word_count / 4, 1.0) * 0.12

    return max(0.01, min(score, 0.99))
