# SPDX-License-Identifier: MPL-2.0
"""Tests for sharded event log with Merkle anchoring."""
from orchestrator.event_log import EventLogShard, ShardedEventLog


def test_shard_append_and_verify():
    shard = EventLogShard()
    shard.append(b"a")
    shard.append(b"b")
    assert shard.verify(0)
    assert shard.verify(1)

    # Tamper with second event
    shard.events[1].data = b"x"
    assert not shard.verify(1)


def test_segmented_verification():
    shard = EventLogShard(segment_size=2)
    for i in range(5):
        shard.append(str(i))

    # two full segments should be stored
    assert len(shard.segments) == 2
    assert shard.verify(4)

    shard.events[4].data = b"tamper"
    assert not shard.verify(4)


def test_sharded_anchor_and_verify():
    log = ShardedEventLog(lambda data: str(data[0] % 2))
    log.append(b"a")  # shard '1'
    log.append(b"b")  # shard '0'

    anchor = log.anchor()
    assert len(anchor) == 64
    assert log.verify_event("1", 0, anchor)

    # Tampering breaks verification
    log.shards["1"].events[0].data = b"y"
    assert not log.verify_event("1", 0, anchor)

