# SPDX-License-Identifier: MPL-2.0
"""Sharded event log with cryptographic hash chains and Merkle anchoring."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Dict, List, Optional, Tuple

from .security.pii import redact
from .security.hsm import HSMClient
from .security.forget import ForgetfulStorage


def _hash(data: bytes) -> str:
    return sha256(data).hexdigest()


@dataclass
class Event:
    """Single event in a hash chain."""
    data: bytes
    prev_hash: str
    hash: str


class EventLogShard:
    """Append-only log shard maintaining its own hash chain."""

    def __init__(self, segment_size: int = 1000) -> None:
        self.events: List[Event] = []
        self.tip: str = "0" * 64
        # Pre-computed segment tips for O(1) lookup.
        self.segment_size = segment_size
        self.segments: List[str] = []

    def append(self, data: bytes | str) -> Event:
        if isinstance(data, str):
            data = data.encode()
        prev = self.tip
        h = _hash(prev.encode() + data)
        event = Event(data=data, prev_hash=prev, hash=h)
        self.events.append(event)
        self.tip = h
        if len(self.events) % self.segment_size == 0:
            # Persist the tip to allow fast verification later on.
            self.segments.append(self.tip)
        return event

    def verify(self, index: int) -> bool:
        if index >= len(self.events):
            return False
        segment_idx = index // self.segment_size
        start = segment_idx * self.segment_size
        prev = "0" * 64 if segment_idx == 0 else self.segments[segment_idx - 1]
        for e in self.events[start : index + 1]:
            if _hash(prev.encode() + e.data) != e.hash:
                return False
            prev = e.hash
        return True


def merkle_root(leaves: List[str]) -> str:
    """Compute Merkle root of hex-encoded leaf hashes."""
    if not leaves:
        return "0" * 64
    nodes = [bytes.fromhex(h) for h in leaves]
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])
        nodes = [
            sha256(nodes[i] + nodes[i + 1]).digest()
            for i in range(0, len(nodes), 2)
        ]
    return nodes[0].hex()


class ShardedEventLog:
    """Manage multiple event log shards with optional privacy features."""

    def __init__(
        self,
        shard_func: Callable[[bytes], str],
        *,
        pii_redactor: Callable[[str], str] | None = redact,
        signer: Optional[HSMClient] = None,
        forgetful_storage: Optional[ForgetfulStorage] = None,
    ) -> None:
        self.shard_func = shard_func
        self.shards: Dict[str, EventLogShard] = {}
        self.redactor = pii_redactor
        self.signer = signer
        self.forgetful = forgetful_storage
        self.signatures: Dict[Tuple[str, int], bytes] = {}
        self.user_event_map: Dict[str, List[Tuple[str, int]]] = {}

    def _get_shard(self, key: str) -> EventLogShard:
        return self.shards.setdefault(key, EventLogShard())

    def append(
        self,
        data: bytes | str,
        user_id: str | None = None,
        sign: bool = False,
    ) -> Event:
        if isinstance(data, str):
            data = data.encode()

        # PII redaction
        if self.redactor:
            data = self.redactor(data.decode("utf-8", errors="ignore")).encode()

        # Determine shard key before encryption for stability
        key = self.shard_func(data)
        shard = self._get_shard(key)
        index = len(shard.events)

        # Encrypt if forgetful storage provided and user_id given
        if user_id and self.forgetful:
            data = self.forgetful.encrypt(user_id, data)
            self.user_event_map.setdefault(user_id, []).append((key, index))

        event = shard.append(data)

        # Optional signing
        if sign and self.signer:
            self.signatures[(key, len(shard.events) - 1)] = self.signer.sign(
                event.hash.encode()
            )

        return event

    def forget_user(self, user_id: str) -> None:
        if self.forgetful:
            self.forgetful.forget(user_id)
        # Record a tombstone event to indicate deletion
        self.append(f"FORGET:{user_id}")

    def anchor(self) -> str:
        leaves = [self.shards[k].tip for k in sorted(self.shards)]
        return merkle_root(leaves)

    def verify_event(self, key: str, index: int, anchor_root: str) -> bool:
        shard = self.shards.get(key)
        if not shard:
            return False
        if not shard.verify(index):
            return False
        leaves = [self.shards[k].tip for k in sorted(self.shards)]
        return merkle_root(leaves) == anchor_root

