# SPDX-License-Identifier: MPL-2.0
"""Simple key management supporting a right-to-be-forgotten mechanism."""
from __future__ import annotations

import os
from typing import Dict


def xor_encrypt(data: bytes, key: bytes) -> bytes:
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


class ForgetfulStorage:
    """Manage per-user encryption keys that can be discarded."""

    def __init__(self) -> None:
        self._keys: Dict[str, bytes] = {}

    def ensure_key(self, user_id: str) -> bytes:
        return self._keys.setdefault(user_id, os.urandom(32))

    def encrypt(self, user_id: str, data: bytes) -> bytes:
        key = self.ensure_key(user_id)
        return xor_encrypt(data, key)

    def decrypt(self, user_id: str, data: bytes) -> bytes:
        key = self._keys[user_id]
        return xor_encrypt(data, key)

    def forget(self, user_id: str) -> None:
        """Forget user data by deleting the key."""
        self._keys.pop(user_id, None)
