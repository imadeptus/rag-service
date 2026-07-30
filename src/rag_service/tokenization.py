"""Shared tokenization for lexical and deterministic vector retrieval."""

import re

TOKEN_PATTERN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())
