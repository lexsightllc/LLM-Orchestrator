# SPDX-License-Identifier: MPL-2.0
"""Minimal multi-party computation utilities using additive secret sharing."""
from __future__ import annotations

import os
from typing import List, Tuple


def split_secret(value: int, parties: int) -> List[int]:
    shares = [int.from_bytes(os.urandom(4), "little") for _ in range(parties - 1)]
    final = value - sum(shares)
    shares.append(final)
    return shares


def combine_shares(shares: List[int]) -> int:
    return sum(shares)


def secure_add(values: List[int]) -> Tuple[List[int], int]:
    """Return shares for each value and their combined result."""
    all_shares = [split_secret(v, len(values)) for v in values]
    # sum across parties
    combined = [sum(share[i] for share in all_shares) for i in range(len(values))]
    return combined, sum(values)
