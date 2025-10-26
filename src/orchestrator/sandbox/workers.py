# SPDX-License-Identifier: MPL-2.0
"""Worker implementations for the sandbox environment."""
from __future__ import annotations

import asyncio
import logging
import shlex
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union

from ..artifacts import Artifact, ArtifactStorage
from .types import ResourceLimits, SandboxResult, SandboxState

if TYPE_CHECKING:  # pragma: no cover - imported only for typing
    from . import Sandbox

logger = logging.getLogger(__name__)


class WorkerType(str, Enum):
    """Types of workers supported by the orchestrator."""

    CPU = "cpu"
    IO = "io"
    SHELL = "shell"
    PYTHON = "python"
    TOOL = "tool"


@dataclass
class WorkerConfig:
    """Configuration options shared across workers."""

    worker_type: WorkerType
    limits: Optional[ResourceLimits] = None
    env_vars: Dict[str, str] = field(default_factory=dict)
    workdir: Optional[str] = None
    cleanup: bool = True

    def copy(self) -> "WorkerConfig":
        return WorkerConfig(
            worker_type=self.worker_type,
            limits=self.limits.copy() if self.limits is not None else None,
            env_vars=dict(self.env_vars),
            workdir=self.workdir,
            cleanup=self.cleanup,
        )


@dataclass
class WorkerResult:
    """Structured response produced by workers."""

    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Artifact] = field(default_factory=dict)

    @classmethod
    def from_sandbox_result(
        cls,
        sandbox_result: SandboxResult,
        output_parser: Optional[Callable[[str], Any]] = None,
    ) -> "WorkerResult":
        output: Any = sandbox_result.stdout
        if output_parser is not None:
            try:
                output = output_parser(output)
            except Exception as exc:  # pragma: no cover - defensive
                return cls(
                    success=False,
                    output=None,
                    error=f"Failed to parse sandbox output: {exc}",
                    metadata={"sandbox_result": sandbox_result.dict()},
                    artifacts=sandbox_result.artifacts,
                )

        metadata = {
            "sandbox_result": sandbox_result.dict(),
        }
        if sandbox_result.stderr:
            metadata["stderr"] = sandbox_result.stderr

        return cls(
            success=sandbox_result.success,
            output=output,
            error=None if sandbox_result.success else sandbox_result.stderr,
            metadata=metadata,
            artifacts=sandbox_result.artifacts,
        )


def worker_result_from_sandbox_result(
    sandbox_result: SandboxResult,
    output_parser: Optional[Callable[[str], Any]] = None,
) -> WorkerResult:
    """Convenience wrapper mirroring the classmethod for legacy callers."""

    return WorkerResult.from_sandbox_result(sandbox_result, output_parser=output_parser)


class BaseWorker(ABC):
    """Base class for all workers that orchestrate sandbox operations."""

    def __init__(
        self,
        config: Optional[WorkerConfig] = None,
        storage: Optional[ArtifactStorage] = None,
    ) -> None:
        if config is None:
            config = WorkerConfig(worker_type=WorkerType.CPU, limits=ResourceLimits())
        if config.limits is None:
            config.limits = ResourceLimits()
        self.config = config
        self.storage = storage
        self._sandbox: Optional[Sandbox] = None
        self._is_initialized = False

    async def __aenter__(self) -> "BaseWorker":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def initialize(self) -> None:
        if self._is_initialized:
            return
        limits = self.config.limits.copy() if self.config.limits else ResourceLimits()
        limits.env_vars.update(self.config.env_vars)
        from . import Sandbox as SandboxImpl

        self._sandbox = SandboxImpl(
            limits=limits,
            workdir=self.config.workdir,
            cleanup=self.config.cleanup,
            storage=self.storage,
        )
        self._is_initialized = True

    async def close(self) -> None:
        if self._sandbox is not None:
            await self._sandbox.cleanup_resources()
        self._is_initialized = False

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> WorkerResult:
        """Execute a unit of work and return a structured result."""


class CPUWorker(BaseWorker):
    """Executes CPU-bound Python callables in a thread pool."""

    async def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> WorkerResult:
        await self.initialize()
        start = time.perf_counter()
        try:
            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
            duration = time.perf_counter() - start
            metadata = {
                "sandbox_result": {
                    "state": SandboxState.COMPLETED.value,
                    "execution_time": duration,
                }
            }
            return WorkerResult(success=True, output=output, metadata=metadata)
        except Exception as exc:  # pragma: no cover - errors tested separately
            duration = time.perf_counter() - start
            metadata = {
                "sandbox_result": {
                    "state": SandboxState.FAILED.value,
                    "execution_time": duration,
                }
            }
            logger.exception("CPU worker execution failed")
            return WorkerResult(success=False, error=str(exc), metadata=metadata)


class IOWorker(BaseWorker):
    """Performs deterministic file-system operations within the sandbox."""

    def _resolve_path(self, relative: Union[str, Path]) -> Path:
        if self._sandbox is None:
            raise RuntimeError("Worker not initialized")
        root = self._sandbox.workdir
        candidate = (root / relative).resolve()
        if root not in candidate.parents and candidate != root:
            raise ValueError("Path escapes sandbox root")
        return candidate

    async def execute(self, operation: str, *args: Any) -> WorkerResult:
        await self.initialize()
        assert self._sandbox is not None

        try:
            op = operation.lower()
            if op == "write":
                path, content = args
                target = self._resolve_path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                data = content.encode("utf-8") if isinstance(content, str) else content
                target.write_bytes(data)
                output = str(target)
            elif op == "read":
                (path,) = args
                target = self._resolve_path(path)
                output = target.read_text(encoding="utf-8")
            elif op == "copy":
                src, dest = args
                source_path = self._resolve_path(src)
                dest_path = self._resolve_path(dest)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_bytes(source_path.read_bytes())
                output = str(dest_path)
            elif op == "delete":
                (path,) = args
                target = self._resolve_path(path)
                if target.exists():
                    target.unlink()
                output = True
            else:
                raise ValueError(f"Unsupported IO operation: {operation}")

            metadata = {
                "sandbox_result": {
                    "state": SandboxState.COMPLETED.value,
                }
            }
            return WorkerResult(success=True, output=output, metadata=metadata)
        except Exception as exc:
            logger.exception("IO worker failed")
            metadata = {
                "sandbox_result": {
                    "state": SandboxState.FAILED.value,
                }
            }
            return WorkerResult(success=False, error=str(exc), metadata=metadata)


class ShellWorker(BaseWorker):
    """Runs shell commands in the sandbox."""

    async def execute(
        self,
        command: Union[str, Iterable[str]],
        args: Optional[Iterable[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        input_data: Optional[Union[str, bytes]] = None,
    ) -> WorkerResult:
        await self.initialize()
        assert self._sandbox is not None

        if isinstance(command, str) and args:
            cmd = [command, *list(args)]
        elif isinstance(command, Iterable) and not isinstance(command, str):
            cmd = list(command)
        else:
            cmd = command  # type: ignore[assignment]

        if isinstance(cmd, list) and len(cmd) > 1:
            cmd_str = " ".join(shlex.quote(part) for part in cmd)
        else:
            cmd_str = cmd

        result = await self._sandbox.run_shell_command(cmd_str, env=env, cwd=cwd, input_data=input_data)
        return worker_result_from_sandbox_result(result)


class PythonWorker(BaseWorker):
    """Executes Python snippets within the sandbox."""

    async def execute(
        self,
        code: str,
        args: Optional[Iterable[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> WorkerResult:
        await self.initialize()
        assert self._sandbox is not None
        result = await self._sandbox.run_python_code(code, args=list(args or []), env=env, cwd=cwd)
        return worker_result_from_sandbox_result(result)


class ToolWorker(BaseWorker):
    """Placeholder subclass allowing factories to return tool workers."""

    async def execute(self, *args: Any, **kwargs: Any) -> WorkerResult:  # pragma: no cover - actual implementation lives elsewhere
        raise NotImplementedError("ToolWorker is implemented in sandbox.tool_worker")


def create_worker(
    worker_type: Union[str, WorkerType],
    config: Optional[WorkerConfig] = None,
    storage: Optional[ArtifactStorage] = None,
    **kwargs: Any,
) -> BaseWorker:
    """Factory that constructs a worker based on type."""

    if isinstance(worker_type, str):
        worker_type = WorkerType(worker_type.lower())

    config = config.copy() if config is not None else WorkerConfig(worker_type=worker_type, limits=ResourceLimits())
    config.worker_type = worker_type

    mapping: Dict[WorkerType, Type[BaseWorker]] = {
        WorkerType.CPU: CPUWorker,
        WorkerType.IO: IOWorker,
        WorkerType.SHELL: ShellWorker,
        WorkerType.PYTHON: PythonWorker,
        WorkerType.TOOL: ToolWorker,
    }

    worker_cls = mapping.get(worker_type)
    if worker_cls is None:
        raise ValueError(f"Unknown worker type: {worker_type}")

    return worker_cls(config=config, storage=storage, **kwargs)
