# Architecture Overview

LLM Orchestrator is organized into modular subsystems that collaborate to run multi‑agent workflows.

## Core Components

- **Context Assembler** (`orchestrator/context/assembler.py`): builds prompt context with token budgeting, deduplication and summarization.
- **Sharded Event Log** (`orchestrator/event_log.py`): append‑only log with cryptographic hash chains, optional PII redaction, signing and Merkle anchoring.
- **Artifact Storage** (`orchestrator/artifacts`): content‑addressable store for JSON, text, binary and other artifacts with rich metadata.
- **Sandbox** (`orchestrator/sandbox`): isolated execution environment for tools and confidential workloads.
- **Security Modules** (`orchestrator/security`): primitives for HSM signing, rate limiting, differential privacy, secure deletion and more.
- **Real‑Time Governance** (`orchestrator/governance.py`): telemetry‑driven policy adjustment via PID control and anomaly detection.
- **Model Pool Manager** (`orchestrator/model_pool.py`): predictive scaling of model instances using time‑series forecasts.

## Extensibility

The orchestrator exposes a type‑safe artifact API and a versioned tool registry, enabling teams to integrate custom models, tools and governance policies while preserving reproducibility.

