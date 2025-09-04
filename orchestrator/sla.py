"""Runtime SLA registry with dynamic renegotiation."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict


@dataclass
class SLA:
    """Simple service-level agreement parameters."""
    latency_ms: int
    cost_per_token: float


class SLARegistry:
    """Track provider SLAs and support renegotiation."""

    def __init__(self) -> None:
        self._slas: Dict[str, SLA] = {}

    def register(self, provider: str, sla: SLA) -> None:
        self._slas[provider] = sla

    def get(self, provider: str) -> SLA:
        return self._slas[provider]

    def renegotiate(self, provider: str, proposal: Callable[[SLA], SLA]) -> SLA:
        """Apply a renegotiation proposal and update SLA atomically."""
        current = self.get(provider)
        updated = proposal(current)
        # Use `replace` to allow partial updates while keeping dataclass semantics
        self._slas[provider] = replace(current, **updated.__dict__)
        return self._slas[provider]
