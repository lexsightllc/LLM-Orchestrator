# ADR 0001: Repository Structure Normalization

## Status
Accepted

## Context
The project contained a flat Python package layout with ad-hoc tooling. To
scale contributions across teams we require a conventional structure with
scripted workflows and reproducible builds.

## Decision
We adopted a `src/` layout with dedicated directories for tests, docs, scripts,
and infrastructure assets. A standardized toolbelt (Makefile + scripts/)
coordinates linting, formatting, testing, packaging, documentation, security
scans, and release automation. CI pipelines are aligned with these commands.

## Consequences
- Contributors interact with the project through consistent Make targets.
- Automation enforces formatting, linting, type checking, and security scans.
- Additional effort is required to keep scripts in sync with tooling updates,
  but the structure enables Renovate/Dependabot automation.
