from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from urllib.parse import urlparse


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


IP_PATTERN = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
LONG_TOKEN_PATTERN = re.compile(r"[a-z0-9]{18,}", re.IGNORECASE)

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

KAGGLE_FEATURE_NAMES = [
    "having_IPhaving_IP_Address",
    "URLURL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "SSLfinal_State",
    "Domain_registeration_length",
    "Favicon",
    "port",
    "HTTPS_token",
    "Request_URL",
    "URL_of_Anchor",
    "Links_in_tags",
    "SFH",
    "Submitting_to_email",
    "Abnormal_URL",
    "Redirect",
    "on_mouseover",
    "RightClick",
    "popUpWidnow",
    "Iframe",
    "age_of_domain",
    "DNSRecord",
    "web_traffic",
    "Page_Rank",
    "Google_Index",
    "Links_pointing_to_page",
    "Statistical_report",
]

HTML_FEATURE_NAMES = [
    "html_length_bucket",
    "form_count_bucket",
    "input_count_bucket",
    "password_input_count_bucket",
    "hidden_input_count_bucket",
    "script_count_bucket",
    "external_link_count_bucket",
    "external_resource_ratio_bucket",
    "suspicious_word_count_bucket",
    "has_password_input",
    "has_email_input",
    "has_iframe",
    "has_meta_refresh",
    "has_mailto",
    "has_onmouseover",
    "blocks_right_click",
    "has_popup",
    "has_empty_form_action",
    "has_javascript_form_action",
    "has_external_form_action",
]

HTML_URL_ATTRS = {
    "a": ("href",),
    "form": ("action",),
    "iframe": ("src",),
    "img": ("src",),
    "link": ("href",),
    "script": ("src",),
}


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
    html_length_bucket: int
    form_count_bucket: int
    input_count_bucket: int
    password_input_count_bucket: int
    hidden_input_count_bucket: int
    script_count_bucket: int
    external_link_count_bucket: int
    external_resource_ratio_bucket: int
    suspicious_word_count_bucket: int
    has_password_input: int
    has_email_input: int
    has_iframe: int
    has_meta_refresh: int
    has_mailto: int
    has_onmouseover: int
    blocks_right_click: int
    has_popup: int
    has_empty_form_action: int
    has_javascript_form_action: int
    has_external_form_action: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class _HtmlSignalParser(HTMLParser):
    def __init__(self, base_hostname: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self.base_hostname = base_hostname.lower()
        self.form_count = 0
        self.input_count = 0
        self.password_input_count = 0
        self.hidden_input_count = 0
        self.email_input_count = 0
        self.script_count = 0
        self.iframe_count = 0
        self.meta_refresh_count = 0
        self.mailto_count = 0
        self.onmouseover_count = 0
        self.right_click_block_count = 0
        self.popup_count = 0
        self.empty_form_action_count = 0
        self.javascript_form_action_count = 0
        self.external_form_action_count = 0
        self.total_url_count = 0
        self.external_url_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {name.lower(): (value or "") for name, value in attrs}
        tag = tag.lower()

        if tag == "form":
            self.form_count += 1
            action = attr_map.get("action", "").strip()
            lowered_action = action.lower()
            if not action or action == "#":
                self.empty_form_action_count += 1
            if lowered_action.startswith("javascript:"):
                self.javascript_form_action_count += 1
            if self._is_external_url(action):
                self.external_form_action_count += 1

        if tag == "input":
            self.input_count += 1
            input_type = attr_map.get("type", "").lower()
            if input_type == "password":
                self.password_input_count += 1
            if input_type == "hidden":
                self.hidden_input_count += 1
            if input_type == "email":
                self.email_input_count += 1

        if tag == "script":
            self.script_count += 1
        if tag == "iframe":
            self.iframe_count += 1
        if tag == "meta" and attr_map.get("http-equiv", "").lower() == "refresh":
            self.meta_refresh_count += 1

        inline_handlers = " ".join(f"{name}={value}" for name, value in attr_map.items()).lower()
        if "onmouseover" in attr_map or "onmouseover" in inline_handlers:
            self.onmouseover_count += 1
        if "oncontextmenu" in attr_map or "event.button==2" in inline_handlers or "return false" in inline_handlers:
            self.right_click_block_count += 1
        if "window.open" in inline_handlers or "alert(" in inline_handlers:
            self.popup_count += 1

        for attr_name in HTML_URL_ATTRS.get(tag, ()):
            url = attr_map.get(attr_name, "").strip()
            if not url:
                continue
            if url.lower().startswith("mailto:"):
                self.mailto_count += 1
            if url.startswith(("http://", "https://", "//")):
                self.total_url_count += 1
                if self._is_external_url(url):
                    self.external_url_count += 1

    def _is_external_url(self, value: str) -> bool:
        if not self.base_hostname:
            return value.startswith(("http://", "https://", "//"))
        if value.startswith("//"):
            value = "https:" + value
        parsed = urlparse(value)
        hostname = (parsed.hostname or "").lower()
        return bool(hostname and hostname != self.base_hostname and not hostname.endswith("." + self.base_hostname))


def normalize_url(url: str) -> str:
    value = url.strip()
    if not value:
        return value
    if "://" not in value:
        value = "https://" + value
    return value


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    total = len(value)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _bucket(value: int, low: int, high: int) -> int:
    if value <= low:
        return 0
    if value <= high:
        return 1
    return 2


def _filename_to_hostname(filename: str) -> str:
    stem = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = re.sub(r"^\d+_", "", stem)
    stem = re.sub(r"\.html?$", "", stem, flags=re.IGNORECASE)
    if "." not in stem and "_" in stem:
        stem = stem.replace("_", ".")
    return stem.lower()


def _hostname_matches(hostname: str, domain: str) -> bool:
    return hostname == domain or hostname.endswith("." + domain)


def extract_html_features(html: str, source_name: str = "", base_url: str = "") -> HtmlFeatures:
    base_hostname = ""
    if base_url:
        parsed = urlparse(normalize_url(base_url))
        base_hostname = (parsed.hostname or "").lower()
    if not base_hostname and source_name:
        base_hostname = _filename_to_hostname(source_name)

    parser = _HtmlSignalParser(base_hostname=base_hostname)
    try:
        parser.feed(html)
    except Exception:
        pass

    lowered = html.lower()
    suspicious_words = sum(lowered.count(word) for word in SUSPICIOUS_WORDS)
    external_ratio = 0
    if parser.total_url_count:
        external_ratio = round((parser.external_url_count / parser.total_url_count) * 100)

    return HtmlFeatures(
        html_length_bucket=_bucket(len(html), 20_000, 150_000),
        form_count_bucket=_bucket(parser.form_count, 0, 2),
        input_count_bucket=_bucket(parser.input_count, 3, 12),
        password_input_count_bucket=_bucket(parser.password_input_count, 0, 1),
        hidden_input_count_bucket=_bucket(parser.hidden_input_count, 2, 10),
        script_count_bucket=_bucket(parser.script_count, 5, 30),
        external_link_count_bucket=_bucket(parser.external_url_count, 3, 25),
        external_resource_ratio_bucket=_bucket(external_ratio, 25, 70),
        suspicious_word_count_bucket=_bucket(suspicious_words, 3, 20),
        has_password_input=1 if parser.password_input_count else 0,
        has_email_input=1 if parser.email_input_count else 0,
        has_iframe=1 if parser.iframe_count else 0,
        has_meta_refresh=1 if parser.meta_refresh_count else 0,
        has_mailto=1 if parser.mailto_count else 0,
        has_onmouseover=1 if parser.onmouseover_count else 0,
        blocks_right_click=1 if parser.right_click_block_count else 0,
        has_popup=1 if parser.popup_count else 0,
        has_empty_form_action=1 if parser.empty_form_action_count else 0,
        has_javascript_form_action=1 if parser.javascript_form_action_count else 0,
        has_external_form_action=1 if parser.external_form_action_count else 0,
    )


def extract_url_features(url: str) -> UrlFeatures:
    normalized = normalize_url(url)
    parsed = urlparse(normalized)
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


def to_kaggle_features(url: str, features: UrlFeatures) -> dict[str, int]:
    normalized = normalize_url(url)
    parsed = urlparse(normalized)
    hostname = (parsed.hostname or parsed.netloc).lower()
    after_scheme = normalized.split("://", 1)[1] if "://" in normalized else normalized
    has_extra_double_slash = "//" in after_scheme
    try:
        port = parsed.port
    except ValueError:
        port = -1
    has_nonstandard_port = port not in (None, 80, 443)

    if features.url_length < 54:
        url_length = 1
    elif features.url_length <= 75:
        url_length = 0
    else:
        url_length = -1

    if features.num_subdomains <= 1:
        subdomain_state = 1
    elif features.num_subdomains == 2:
        subdomain_state = 0
    else:
        subdomain_state = -1

    return {
        "having_IPhaving_IP_Address": -1 if features.has_ip else 1,
        "URLURL_Length": url_length,
        "Shortining_Service": -1 if features.has_url_shortener else 1,
        "having_At_Symbol": -1 if features.has_at else 1,
        "double_slash_redirecting": -1 if has_extra_double_slash else 1,
        "Prefix_Suffix": -1 if "-" in hostname else 1,
        "having_Sub_Domain": subdomain_state,
        "SSLfinal_State": 1 if features.has_https else -1,
        "Domain_registeration_length": 0,
        "Favicon": 0,
        "port": -1 if has_nonstandard_port else 1,
        "HTTPS_token": -1 if "https" in hostname else 1,
        "Request_URL": 0,
        "URL_of_Anchor": 0,
        "Links_in_tags": 0,
        "SFH": 0,
        "Submitting_to_email": -1 if "mailto:" in normalized.lower() else 1,
        "Abnormal_URL": -1 if features.has_embedded_url or features.has_redirect_param else 1,
        "Redirect": 0 if features.has_redirect_param else 1,
        "on_mouseover": 0,
        "RightClick": 0,
        "popUpWidnow": 0,
        "Iframe": 0,
        "age_of_domain": 0,
        "DNSRecord": 0,
        "web_traffic": 0,
        "Page_Rank": 0,
        "Google_Index": 0,
        "Links_pointing_to_page": 0,
        "Statistical_report": 0,
    }


def heuristic_phishing_score(features: UrlFeatures) -> float:
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
    return max(0.01, min(score, 0.99))
