"""Utilities for detecting and redacting PII from text."""
from __future__ import annotations

import re
from typing import Pattern

EMAIL_RE: Pattern[str] = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE: Pattern[str] = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
SSN_RE: Pattern[str] = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

REPLACEMENTS = {
    EMAIL_RE: "[EMAIL]",
    PHONE_RE: "[PHONE]",
    SSN_RE: "[SSN]",
}

def redact(text: str) -> str:
    """Return text with simple PII patterns replaced."""
    for pattern, token in REPLACEMENTS.items():
        text = pattern.sub(token, text)
    return text
