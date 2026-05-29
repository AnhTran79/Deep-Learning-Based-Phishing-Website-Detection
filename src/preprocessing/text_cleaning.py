from __future__ import annotations

import re
from html import unescape


SCRIPT_STYLE_PATTERN = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


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
    cleaned = SCRIPT_STYLE_PATTERN.sub(" ", html)
    if keep_markup_tokens:
        cleaned = re.sub(r"<\s*/?\s*([a-z0-9]+)[^>]*>", r" tag_\1 ", cleaned, flags=re.IGNORECASE)
    else:
        cleaned = TAG_PATTERN.sub(" ", cleaned)
    return normalize_text(cleaned, lowercase=True)


def trim_text(value: str | None, max_chars: int) -> str:
    text = value or ""
    if max_chars <= 0:
        return text
    return text[:max_chars]
