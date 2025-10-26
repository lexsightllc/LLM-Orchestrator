"""Basic differential privacy utilities for shared memory statistics."""
from __future__ import annotations

import math
import random
from typing import Dict

class DifferentialPrivacyStore:
    """Store counts with Laplace noise to provide differential privacy."""

    def __init__(self, epsilon: float = 1.0) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon
        self.counts: Dict[str, int] = {}

    def increment(self, key: str, value: int = 1) -> None:
        """Record an occurrence for the given key."""
        self.counts[key] = self.counts.get(key, 0) + value

    def _laplace_noise(self, scale: float) -> float:
        """Sample Laplace noise with given scale."""
        u = random.random() - 0.5
        return -scale * math.copysign(math.log(1 - 2 * abs(u)), u)

    def noisy_count(self, key: str) -> float:
        """Return count for key with Laplace noise applied."""
        true_count = self.counts.get(key, 0)
        noise = self._laplace_noise(1.0 / self.epsilon)
        return true_count + noise
