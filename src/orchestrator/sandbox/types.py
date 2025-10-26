# SPDX-License-Identifier: MPL-2.0
"""Shared types and constants for the sandbox module."""
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import sys
from pydantic import BaseModel, Field

class SandboxState(str, Enum):
    """State of a sandboxed process."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    KILLED = "killed"

class ResourceLimits(BaseModel):
    """Resource limits for sandboxed execution."""
    cpu_percent: float = Field(100.0, description="Maximum CPU percentage (0-100)")
    memory_mb: int = Field(1024, description="Maximum memory in MB")
    timeout_seconds: float = Field(300.0, description="Maximum execution time in seconds")
    max_output_bytes: int = Field(10 * 1024 * 1024, description="Maximum output size in bytes")
    max_processes: int = Field(10, description="Maximum number of processes")
    max_files: int = Field(100, description="Maximum number of open files")
    allow_network: bool = Field(False, description="Allow network access")
    allowed_hosts: List[str] = Field(default_factory=list, description="List of allowed hosts for network access")
    env_vars: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    workdir: Optional[str] = Field(None, description="Working directory")
    python_path: str = Field(sys.executable, description="Path to Python interpreter")

class SandboxResult(BaseModel):
    """Result of a sandboxed execution."""
    success: bool = Field(..., description="Whether the execution was successful")
    state: SandboxState = Field(..., description="Final state of the execution")
    return_code: Optional[int] = Field(None, description="Process return code")
    stdout: str = Field("", description="Standard output")
    stderr: str = Field("", description="Standard error")
    cpu_time: float = Field(0.0, description="CPU time used in seconds")
    memory_used_mb: float = Field(0.0, description="Peak memory used in MB")
    execution_time: float = Field(0.0, description="Wall-clock execution time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    artifacts: Dict[str, Any] = Field(default_factory=dict, description="Generated artifacts")

class WorkerType(str, Enum):
    """Types of workers available in the sandbox."""
    CPU = "cpu"
    IO = "io"
    SHELL = "shell"
    PYTHON = "python"
    TOOL = "tool"

class WorkerConfig(BaseModel):
    """Configuration for a worker."""
    worker_type: WorkerType = Field(..., description="Type of worker")
    limits: Optional[ResourceLimits] = Field(default=None, description="Resource limits")
    env_vars: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    workdir: Optional[str] = Field(None, description="Working directory")
    cleanup: bool = Field(True, description="Whether to clean up after execution")

class WorkerResult(BaseModel):
    """Result of a worker execution."""
    success: bool = Field(..., description="Whether the execution was successful")
    output: Any = Field(None, description="Output of the execution")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    artifacts: Dict[str, Any] = Field(default_factory=dict, description="Generated artifacts")
