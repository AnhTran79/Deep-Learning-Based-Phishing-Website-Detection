from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CharTokenizerConfig:
    max_len: int
    vocab_size: int
    lowercase: bool = True


def encode_char_sequence(value: str | None, config: CharTokenizerConfig) -> list[int]:
    text = value or ""
    if config.lowercase:
        text = text.lower()
    ids = [min(ord(char), config.vocab_size - 1) for char in text[: config.max_len]]
    return ids + [0] * (config.max_len - len(ids))


def decode_config(config: dict) -> CharTokenizerConfig:
    return CharTokenizerConfig(
        max_len=int(config["max_len"]),
        vocab_size=int(config["vocab_size"]),
        lowercase=bool(config.get("lowercase", True)),
    )
