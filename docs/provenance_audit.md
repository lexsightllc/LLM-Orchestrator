<!-- SPDX-License-Identifier: MPL-2.0 -->
# Provenance Audit Summary

Date: 2025-01-01

## Contributors

- Augusto Ochoa Ughini <lexsightllc@lexsightllc.com>
- Your Name <you@example.com>

## Repository Inventory

```
.editorconfig
.env.example
.gitattributes
.github/CODEOWNERS
.github/workflows/ci.yml
.github/workflows/release.yml
.gitignore
.gitmessage
.pre-commit-config.yaml
.tool-versions
CHANGELOG.md
CODE_OF_CONDUCT.md
CONTRIBUTING.md
LICENSE
Makefile
README.md
assets/.gitkeep
ci/.gitkeep
commitlint.config.cjs
configs/.gitkeep
conftest.py
data/.gitkeep
docker-compose.yml
docs/adr/0001-repository-structure.md
docs/advanced_architecture.md
docs/architecture_overview.md
examples/demo_artifacts.py
examples/demo_repl.py
examples/demo_tool_normalization.py
examples/demo_tool_versioning.py
examples/tools/calculator/1.0.0/impl.py
examples/tools/calculator/1.0.0/migrations/1.0.0_to_1.1.0.py
examples/tools/calculator/1.0.0/tool.json
examples/visualization_demo.py
infra/docker/Dockerfile
jsonschema/__init__.py
mkdocs.yml
project.yaml
pyproject.toml
requirements-dev.in
requirements-dev.txt
requirements.in
requirements.txt
scripts/_common.sh
scripts/bootstrap
scripts/build
scripts/check
scripts/clean
scripts/coverage
scripts/dev
scripts/e2e
scripts/fmt
scripts/gen-docs
scripts/lint
scripts/migrate
scripts/package
scripts/powershell/bootstrap.ps1
scripts/powershell/build.ps1
scripts/powershell/check.ps1
scripts/powershell/clean.ps1
scripts/powershell/common.ps1
scripts/powershell/coverage.ps1
scripts/powershell/dev.ps1
scripts/powershell/e2e.ps1
scripts/powershell/fmt.ps1
scripts/powershell/gen-docs.ps1
scripts/powershell/lint.ps1
scripts/powershell/migrate.ps1
scripts/powershell/package.ps1
scripts/powershell/release.ps1
scripts/powershell/sbom.ps1
scripts/powershell/security-scan.ps1
scripts/powershell/test.ps1
scripts/powershell/typecheck.ps1
scripts/powershell/update-deps.ps1
scripts/release
scripts/sbom
scripts/security-scan
scripts/test
scripts/typecheck
scripts/update-deps
src/orchestrator/__init__.py
src/orchestrator/_compat/__init__.py
src/orchestrator/_compat/semver.py
src/orchestrator/artifacts/__init__.py
src/orchestrator/cli.py
src/orchestrator/compliance/__init__.py
src/orchestrator/compliance/audit.py
src/orchestrator/context/assembler.py
src/orchestrator/crdts.py
src/orchestrator/event_log.py
src/orchestrator/governance.py
src/orchestrator/model_pool.py
src/orchestrator/repl/__init__.py
src/orchestrator/sandbox/__init__.py
src/orchestrator/sandbox/confidential.py
src/orchestrator/sandbox/sandbox.py
src/orchestrator/sandbox/tool_worker.py
src/orchestrator/sandbox/types.py
src/orchestrator/sandbox/workers.py
src/orchestrator/security/__init__.py
src/orchestrator/security/differential_privacy.py
src/orchestrator/security/forget.py
src/orchestrator/security/homomorphic.py
src/orchestrator/security/hsm.py
src/orchestrator/security/mpc.py
src/orchestrator/security/pii.py
src/orchestrator/security/rate_limiter.py
src/orchestrator/security/sbom.py
src/orchestrator/sla.py
src/orchestrator/tools/__init__.py
src/orchestrator/tools/normalization.py
src/orchestrator/tools/visualization.py
tests/conftest.py
tests/e2e/test_tool_worker_flow.py
tests/integration/test_sandbox_tools.py
tests/unit/orchestrator/artifacts/test_artifacts.py
tests/unit/orchestrator/context/test_context_assembler.py
tests/unit/orchestrator/event_log/test_event_log.py
tests/unit/orchestrator/repl/test_repl.py
tests/unit/orchestrator/sandbox/test_sandbox.py
tests/unit/orchestrator/sandbox/test_sandbox_fixed.py
tests/unit/orchestrator/security/test_security_features.py
tests/unit/orchestrator/sla/test_sla.py
tests/unit/orchestrator/tools/test_tool_normalization.py
tests/unit/orchestrator/tools/test_tool_versioning.py
tests/unit/orchestrator/tools/test_tool_worker.py
tests/unit/orchestrator/tools/test_tool_worker_fixed.py
tests/unit/orchestrator/workers/test_workers.py
```

## License Compatibility Review

- No files reference GPL-only licensing terms.
- Third-party packages are consumed via package managers and documented in
  `THIRD_PARTY_NOTICES` with their original license text.
- No vendored GPL code was identified.

## Actions

- Replaced the proprietary license with the Mozilla Public License 2.0.
- Added SPDX license identifiers to all source and documentation files.
- Documented third-party attributions in `NOTICE` and `THIRD_PARTY_NOTICES`.
