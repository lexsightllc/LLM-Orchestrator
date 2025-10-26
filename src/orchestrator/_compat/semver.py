# SPDX-License-Identifier: MPL-2.0
"""Minimal semver-compatible utilities for internal use.

Provides a lightweight replacement for the external ``semver`` dependency
with just the features required by the test-suite. It supports parsing
semantic version strings and checking them against simple constraints such
as ``^1.0.0`` or range specifiers like ``>=1.0.0 <2.0.0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from packaging.version import Version
from packaging.specifiers import SpecifierSet


@dataclass(frozen=True, order=True)
class VersionInfo:
    """Represents a semantic version.

    The implementation is intentionally small; it only implements the pieces
    of the public API that are exercised by the tests. It supports parsing
    and matching against constraint expressions.
    """

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, version: str) -> "VersionInfo":
        parts = version.split(".")
        if len(parts) < 3:
            raise ValueError("Invalid version string")
        major, minor, patch = (int(parts[0]), int(parts[1]), int(parts[2]))
        return cls(major, minor, patch)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.major}.{self.minor}.{self.patch}"

    def match(self, constraint: str) -> bool:
        """Return True if this version satisfies the given constraint.

        The constraint syntax is a small subset of `semver`'s, sufficient for
        the tests:
        * ``^X.Y.Z`` - compatible with the given version.
        * standard comparison operators combined with spaces, e.g.
          ``">=1.0.0 <2.0.0"``.
        """

        constraint = constraint.strip()
        if constraint.startswith("^"):
            base = Version(constraint[1:])
            major, minor, patch = base.release[:3]
            if major > 0:
                upper = Version(f"{major + 1}.0.0")
            elif minor > 0:
                upper = Version(f"0.{minor + 1}.0")
            else:
                upper = Version(f"0.0.{patch + 1}")
            spec = SpecifierSet(f">={base},<{upper}")
        else:
            # Replace whitespace separators with commas for SpecifierSet
            normalized = constraint.replace(" ", ",")
            spec = SpecifierSet(normalized)

        return Version(str(self)) in spec
