"""Sharded event log with cryptographic hash chains and Merkle anchoring."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Dict, List


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

    def __init__(self) -> None:
        self.events: List[Event] = []
        self.tip: str = "0" * 64

    def append(self, data: bytes | str) -> Event:
        if isinstance(data, str):
            data = data.encode()
        prev = self.tip
        h = _hash(prev.encode() + data)
        event = Event(data=data, prev_hash=prev, hash=h)
        self.events.append(event)
        self.tip = h
        return event

    def verify(self, index: int) -> bool:
        if index >= len(self.events):
            return False
        prev = "0" * 64
        for e in self.events[: index + 1]:
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
    """Manage multiple event log shards with Merkle anchoring."""

    def __init__(self, shard_func: Callable[[bytes], str]):
        self.shard_func = shard_func
        self.shards: Dict[str, EventLogShard] = {}

    def _get_shard(self, key: str) -> EventLogShard:
        return self.shards.setdefault(key, EventLogShard())

    def append(self, data: bytes | str) -> Event:
        if isinstance(data, str):
            data = data.encode()
        key = self.shard_func(data)
        shard = self._get_shard(key)
        return shard.append(data)

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

