# SPDX-License-Identifier: MPL-2.0
"""Command line interface for the LLM orchestrator."""
from __future__ import annotations

import sys
from typing import List, Any


class OrchestratorCLI:
    """Terminal interface with predictable contracts."""

    def __init__(self, orchestrator: Any) -> None:
        """Create a new CLI wrapper.

        Args:
            orchestrator: The underlying orchestrator instance.
        """
        self.orchestrator = orchestrator
        self.is_tty = sys.stdout.isatty()

    def execute(self, args: List[str]) -> int:
        """Single entry point with semantic exit codes."""
        if self.is_tty:
            return self._run_tui()
        return self._run_pipeline()

    def replay(self, execution_id: str) -> int:
        """Bit-perfect replay from event log."""
        events = self.orchestrator.event_log.get_events(execution_id)
        return self._replay_deterministic(events)

    # Internal helpers -----------------------------------------------------
    def _run_tui(self) -> int:
        """Interactive TUI mode placeholder."""
        # Real implementation would start an interactive interface
        return 0

    def _run_pipeline(self) -> int:
        """Pipeline mode producing JSON lines."""
        # Real implementation would parse ``args`` and emit JSON
        return 0

    def _replay_deterministic(self, events: List[Any]) -> int:
        """Deterministically replay a list of events."""
        # Real implementation would re-create state from events
        return 0
