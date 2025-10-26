"""Toy homomorphic encryption utilities for routing decisions."""
from __future__ import annotations

from typing import Iterable, List


class HomomorphicRouter:
    """Route based on encrypted numeric features using additive homomorphism."""

    def __init__(self, secret: float) -> None:
        self.secret = secret

    def encrypt(self, value: float) -> float:
        return value + self.secret

    def route(self, scores: Iterable[float]) -> int:
        """Return index of maximum score without decrypting."""
        enc_scores: List[float] = [self.encrypt(v) for v in scores]
        max_idx = 0
        for i in range(1, len(enc_scores)):
            if enc_scores[i] > enc_scores[max_idx]:
                max_idx = i
        return max_idx
