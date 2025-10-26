"""Rate limiting with basic prompt injection detection."""
from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Pattern


def _compile(patterns: Iterable[str]) -> list[Pattern[str]]:  # type: ignore[name-defined]
    return [re.compile(p, re.IGNORECASE) for p in patterns]


class RateLimiter:
    """Simple sliding-window rate limiter with injection detection."""

    def __init__(
        self,
        max_per_minute: int,
        injection_patterns: Iterable[str] | None = None,
    ) -> None:
        self.max_per_minute = max_per_minute
        self.history: Dict[str, Deque[float]] = defaultdict(deque)
        self.patterns = _compile(injection_patterns or [])

    def allow(self, user_id: str, content: str) -> bool:
        now = time.time()
        window = self.history[user_id]
        while window and now - window[0] > 60:
            window.popleft()

        if any(p.search(content) for p in self.patterns):
            return False

        if len(window) >= self.max_per_minute:
            return False

        window.append(now)
        return True
