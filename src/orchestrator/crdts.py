# SPDX-License-Identifier: MPL-2.0
"""Domain specific CRDT implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class ReducibleMap:
    """Minimal map that can be reduced using a merge function."""
    merge_fn: Callable[[List[Any], List[Any]], List[Any]]
    identity: List[Any]

    def merge(self, a: List[Any], b: List[Any]) -> List[Any]:
        return self.merge_fn(a, b)


@dataclass
class AssociativeCounter:
    """Commutative aggregation of numerical scores."""
    combine_fn: Callable[[List[float]], Dict[str, float]]

    def combine(self, scores: List[float]) -> Dict[str, float]:
        return self.combine_fn(scores)


class DomainSpecificCRDTs:
    """CRDTs tailored to orchestration patterns."""

    def top_k_list(self, k: int) -> ReducibleMap:
        """Mergeable top-k results across agents."""
        return ReducibleMap(
            merge_fn=lambda a, b: sorted(a + b, key=lambda x: x.score)[:k],
            identity=[],
        )

    def score_aggregator(self) -> AssociativeCounter:
        """Commutative score aggregation."""
        return AssociativeCounter(
            combine_fn=lambda scores: {
                "min": min(scores) if scores else 0.0,
                "max": max(scores) if scores else 0.0,
                "mean": sum(scores) / len(scores) if scores else 0.0,
                "consensus": self._weighted_consensus(scores),
            }
        )

    def _weighted_consensus(self, scores: List[float]) -> float:
        """Simple weighted consensus calculation.

        Currently uses the arithmetic mean as a placeholder.
        """
        if not scores:
            return 0.0
        return sum(scores) / len(scores)
