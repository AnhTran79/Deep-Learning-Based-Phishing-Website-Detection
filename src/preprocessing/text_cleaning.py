from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urlparse


SCRIPT_STYLE_PATTERN = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
TOKEN_CHARS_PATTERN = re.compile(r"[^a-z0-9._/-]+")
SIGNAL_ATTRS = {"action", "alt", "aria-label", "content", "href", "id", "name", "placeholder", "property", "src", "type", "value"}
SIGNAL_TAGS = {"a", "button", "form", "iframe", "img", "input", "label", "meta", "select", "textarea", "title"}


def _token_value(value: str | None, max_len: int = 80) -> str:
    text = normalize_text(value, lowercase=True)
    text = TOKEN_CHARS_PATTERN.sub("_", text).strip("_")
    return text[:max_len]


class _HtmlSignalParser(HTMLParser):
    def __init__(self, keep_markup_tokens: bool) -> None:
        super().__init__(convert_charrefs=True)
        self.keep_markup_tokens = keep_markup_tokens
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if self.keep_markup_tokens:
            self.parts.append(f"tag_{tag}")
        if not self.keep_markup_tokens and tag not in SIGNAL_TAGS:
            return
        for name, value in attrs:
            name = name.lower()
            if name not in SIGNAL_ATTRS:
                continue
            token = _token_value(value)
            if token:
                self.parts.append(f"attr_{name}_{token}")
            if name in {"action", "href", "src"}:
                domain = _token_value(urlparse(value or "").netloc)
                if domain:
                    self.parts.append(f"{name}_domain_{domain}")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def normalize_text(value: str | None, lowercase: bool = True) -> str:
    text = value or ""
    text = CONTROL_PATTERN.sub(" ", text)
    text = unescape(text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text.lower() if lowercase else text


def normalize_url_text(url: str | None) -> str:
    return normalize_text(url, lowercase=True)


def html_to_visible_text(html: str | None, keep_markup_tokens: bool = True) -> str:
    if not html:
        return ""
    parser = _HtmlSignalParser(keep_markup_tokens=keep_markup_tokens)
    try:
        parser.feed(html)
        return normalize_text(" ".join(parser.parts), lowercase=True)
    except Exception:
        cleaned = SCRIPT_STYLE_PATTERN.sub(" ", html)
        cleaned = TAG_PATTERN.sub(" ", cleaned)
        return normalize_text(cleaned, lowercase=True)


def trim_text(value: str | None, max_chars: int) -> str:
    text = value or ""
    if max_chars <= 0:
        return text
    return text[:max_chars]
