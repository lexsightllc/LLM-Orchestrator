# Contributing Guide

Thank you for considering a contribution to LLM Orchestrator! This project follows a
standardized workflow so that every change is reproducible and reviewable.

## Development Workflow

1. **Bootstrap** your environment:
   ```bash
   make bootstrap
   ```
2. **Create a feature branch** using Conventional Commit prefixes, e.g. `feat/structured-context`.
3. **Run checks locally** before opening a pull request:
   ```bash
   make check
   ```
4. **Open a Pull Request** that describes the change, links to any issues, and
   includes screenshots or logs when applicable.

## Commit Standards

- Use [Conventional Commits](https://www.conventionalcommits.org/) for all commit messages.
- Each commit should be focused and include accompanying tests when practical.
- Pre-commit hooks enforce formatting, linting, type checking, and commit linting.

## Code Review

- All changes require review from at least one CODEOWNER.
- Pull requests must pass the full CI suite and maintain code coverage thresholds.
- Address review feedback promptly and document significant design decisions via ADRs in `docs/adr/`.

## Reporting Issues

- Use the issue templates provided in the repository.
- Include reproduction steps, expected vs actual behavior, and environment details.

We appreciate every contribution that makes LLM Orchestrator more reliable and maintainable!
