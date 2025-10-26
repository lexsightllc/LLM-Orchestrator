# SPDX-License-Identifier: MPL-2.0
"""LLM Orchestrator - A framework for managing multi-agent LLM workflows."""

__version__ = "0.1.0"

# Import key components for easier access
from .context.assembler import ContextAssembler, ContextBudget
from .cli import OrchestratorCLI
from .crdts import DomainSpecificCRDTs
from .model_pool import ModelPoolManager
from .governance import RealTimeGovernance

__all__ = [
    "ContextAssembler",
    "ContextBudget",
    "OrchestratorCLI",
    "DomainSpecificCRDTs",
    "ModelPoolManager",
    "RealTimeGovernance",
]
