# SPDX-License-Identifier: MPL-2.0
"""Placeholder for confidential computing sandbox."""
from __future__ import annotations

import os
from typing import Any

from .sandbox import Sandbox


class ConfidentialSandbox(Sandbox):
    """Sandbox that simulates encrypted in-memory execution."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._encryption_key = os.urandom(32)

    async def run_code(self, code: str, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        """Execute code after 'decrypting' it."""
        # In real implementation, code would execute inside an enclave.
        # Here we simply forward to base class.
        return await super().run_code(code, *args, **kwargs)
