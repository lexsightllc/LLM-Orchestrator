"""Simple placeholder interface for an HSM signer."""
from __future__ import annotations

import hashlib
import hmac
import os
from typing import Callable

class HSMClient:
    """Asynchronous-friendly signing client using HMAC as a stand-in for HSM."""

    def __init__(self, key: bytes | None = None) -> None:
        self._key = key or os.urandom(32)

    def sign(self, message: bytes) -> bytes:
        """Return a signature for the given message."""
        return hmac.new(self._key, message, hashlib.sha256).digest()

    def verifier(self) -> Callable[[bytes, bytes], bool]:
        """Return a function verifying signatures from this HSM."""
        def _verify(message: bytes, signature: bytes) -> bool:
            expected = hmac.new(self._key, message, hashlib.sha256).digest()
            return hmac.compare_digest(expected, signature)

        return _verify
