"""SBOM generation and simple vulnerability tracking."""
from __future__ import annotations

from importlib.metadata import distributions
from typing import Dict


def generate_sbom() -> Dict[str, str]:
    """Return a mapping of installed packages to versions."""
    sbom: Dict[str, str] = {}
    for dist in distributions():
        name = dist.metadata.get("Name") or dist.metadata.get("Summary")
        if name:
            sbom[name] = dist.version
    return sbom
