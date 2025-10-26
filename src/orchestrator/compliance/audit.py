# SPDX-License-Identifier: MPL-2.0
"""Generate simple audit reports from the event log."""
from __future__ import annotations

from typing import Dict

from ..event_log import ShardedEventLog


def generate_audit_report(log: ShardedEventLog) -> Dict[str, int]:
    """Return a dictionary summarizing event counts per shard."""
    report: Dict[str, int] = {}
    for key, shard in log.shards.items():
        report[key] = len(shard.events)
    return report
