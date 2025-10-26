# SPDX-License-Identifier: MPL-2.0
"""Sandbox execution environment for the LLM Orchestrator."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import resource
except ImportError:  # pragma: no cover - Windows compatibility
    resource = None

from ..artifacts import (
    Artifact,
    ArtifactStorage,
    create_binary_artifact,
    create_json_artifact,
)
from ..tools import ToolResult, tool_registry
from .types import ResourceLimits, SandboxResult, SandboxState

logger = logging.getLogger(__name__)

__all__ = [
    "Sandbox",
    "SandboxError",
    "SandboxTimeoutError",
    "ResourceLimitExceeded",
    "SandboxManager",
    "create_sandbox",
    "get_sandbox_manager",
    # Worker exports are appended at the bottom once workers are imported.
]


class SandboxError(Exception):
    """Base exception for sandbox-related failures."""


class SandboxTimeoutError(SandboxError):
    """Raised when a sandboxed command exceeds its timeout."""


class ResourceLimitExceeded(SandboxError):
    """Raised when a sandbox process exceeds configured resource limits."""


class Sandbox:
    """Isolated execution environment with resource enforcement."""

    def __init__(
        self,
        limits: Optional[ResourceLimits] = None,
        workdir: Optional[Union[str, Path]] = None,
        cleanup: bool = False,
        storage: Optional[ArtifactStorage] = None,
    ) -> None:
        self.limits = limits.copy() if limits is not None else ResourceLimits()
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="sandbox-"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.cleanup = cleanup
        self.storage = storage
        self._state = SandboxState.PENDING
        self._process: Optional[asyncio.subprocess.Process] = None
        self._known_artifacts: Dict[Path, float] = {}

    async def __aenter__(self) -> "Sandbox":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.cleanup_resources()

    async def cleanup_resources(self) -> None:
        """Terminate running processes and optionally remove the workdir."""

        if self._process and self._process.returncode is None:
            self._process.kill()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._process.wait(), timeout=5)

        if self.cleanup and self.workdir.exists():
            shutil.rmtree(self.workdir, ignore_errors=True)
        self._state = SandboxState.PENDING

    async def run_shell_command(
        self,
        command: Union[str, List[str]],
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[Union[str, Path]] = None,
        input_data: Optional[Union[str, bytes]] = None,
    ) -> SandboxResult:
        """Execute a shell command within the sandbox."""

        if isinstance(command, str):
            if sys.platform == "win32":
                cmd = ["cmd", "/c", command]
            else:
                cmd = ["/bin/sh", "-c", command]
        else:
            cmd = command
        return await self._run_command(cmd, env=env, cwd=cwd, input_data=input_data)

    async def run_python_code(
        self,
        code: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[Union[str, Path]] = None,
    ) -> SandboxResult:
        """Execute Python code using the sandbox interpreter."""

        script_path = self.workdir / "script.py"
        script_path.write_text(code, encoding="utf-8")
        cmd = [self.limits.python_path, str(script_path)]
        if args:
            cmd.extend(args)
        return await self._run_command(cmd, env=env, cwd=cwd)

    async def execute_tool(
        self,
        tool_name: str,
        tool_version: str = "latest",
        args: Optional[Dict[str, Any]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ToolResult:
        """Execute a registered tool inside the sandbox."""

        tool = tool_registry.get_tool(tool_name, tool_version)
        impl = tool.implementation
        kwargs = args or {}

        start = time.perf_counter()

        async def _invoke(payload: Dict[str, Any]) -> ToolResult:
            if asyncio.iscoroutinefunction(impl):
                try:
                    return await impl(**payload)
                except TypeError:
                    return await impl(payload)
            loop = asyncio.get_running_loop()

            def _call() -> Any:
                try:
                    return impl(**payload)
                except TypeError:
                    return impl(payload)

            return await loop.run_in_executor(None, _call)

        result = await _invoke(kwargs or {})
        duration = time.perf_counter() - start

        if isinstance(result, ToolResult):
            result.metadata.setdefault("execution_time", duration)
            return result
        return ToolResult.from_success(result, execution_time=duration)

    async def _run_command(
        self,
        cmd: List[str],
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[Union[str, Path]] = None,
        input_data: Optional[Union[str, bytes]] = None,
    ) -> SandboxResult:
        env_vars = os.environ.copy()
        env_vars.update(self.limits.env_vars)
        if env:
            env_vars.update(env)

        cwd_path = Path(cwd) if cwd else self.workdir
        cwd_path.mkdir(parents=True, exist_ok=True)

        self._state = SandboxState.RUNNING
        start = time.monotonic()
        preexec = self._build_preexec()

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd_path),
            env=env_vars,
            stdin=asyncio.subprocess.PIPE if input_data else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=preexec,
        )
        self._process = process
        signal: Dict[str, Optional[str]] = {"reason": None}
        if input_data is not None and process.stdin is not None:
            payload = input_data.encode("utf-8") if isinstance(input_data, str) else input_data
            process.stdin.write(payload)
            await process.stdin.drain()
            process.stdin.close()

        ru_before = self._get_rusage()

        stdout_task = asyncio.create_task(self._read_stream(process, process.stdout))
        stderr_task = asyncio.create_task(self._read_stream(process, process.stderr))

        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=self.limits.timeout_seconds)
        except asyncio.TimeoutError:
            timed_out = True
            signal.setdefault("reason", "timeout")
            process.kill()
            await process.wait()

        stdout_bytes, stdout_truncated = await stdout_task
        stderr_bytes, _ = await stderr_task
        ru_after = self._get_rusage()
        cpu_time = self._calculate_cpu_time(ru_before, ru_after)
        peak_memory = self._calculate_memory_usage(ru_before, ru_after)

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        execution_time = time.monotonic() - start

        reason = signal.get("reason")
        if timed_out:
            reason = "timeout"

        state = SandboxState.COMPLETED
        success = process.returncode == 0 and not stdout_truncated and reason is None
        if timed_out:
            state = SandboxState.TIMED_OUT
        elif reason == "memory":
            state = SandboxState.KILLED
        elif process.returncode != 0 or stdout_truncated:
            state = SandboxState.FAILED

        metadata: Dict[str, Any] = {
            "return_code": process.returncode,
            "execution_time": execution_time,
            "memory_used_mb": peak_memory,
            "cpu_time": cpu_time,
        }
        if stdout_truncated:
            metadata["output_truncated"] = True
        if reason:
            metadata["termination_reason"] = reason
        if timed_out:
            metadata["timeout_seconds"] = self.limits.timeout_seconds

        artifacts = self._capture_artifacts()

        result = SandboxResult(
            success=success,
            state=state,
            return_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            cpu_time=cpu_time,
            memory_used_mb=peak_memory,
            execution_time=execution_time,
            metadata=metadata,
            artifacts=artifacts,
        )
        self._state = state
        self._process = None
        return result

    async def _read_stream(
        self,
        process: asyncio.subprocess.Process,
        stream: Optional[asyncio.StreamReader],
    ) -> Tuple[bytes, bool]:
        if stream is None:
            return b"", False

        buffer = bytearray()
        truncated = False
        limit = self.limits.max_output_bytes
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            if limit and len(buffer) + len(chunk) > limit:
                truncated = True
                allowed = max(0, limit - len(buffer))
                buffer.extend(chunk[:allowed])
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                break
            buffer.extend(chunk)
        return bytes(buffer), truncated

    def _get_rusage(self) -> Optional[resource.struct_rusage]:
        if resource is None:  # pragma: no cover - Windows fallback
            return None
        return resource.getrusage(resource.RUSAGE_CHILDREN)

    @staticmethod
    def _calculate_cpu_time(
        before: Optional[resource.struct_rusage],
        after: Optional[resource.struct_rusage],
    ) -> float:
        if before is None or after is None:
            return 0.0
        return (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)

    def _calculate_memory_usage(
        self,
        before: Optional[resource.struct_rusage],
        after: Optional[resource.struct_rusage],
    ) -> float:
        if before is None or after is None:
            return 0.0
        # ru_maxrss is kilobytes on Linux, bytes on macOS. Normalize to MB.
        scale = 1024.0 if sys.platform != "darwin" else 1.0
        peak_kb = max(after.ru_maxrss, before.ru_maxrss)
        return peak_kb / scale / 1024.0

    def _build_preexec(self) -> Optional[Callable[[], None]]:
        if resource is None or sys.platform == "win32":  # pragma: no cover - Windows fallback
            return None

        limits = self.limits

        def _preexec() -> None:
            if limits.memory_mb:
                bytes_limit = int(limits.memory_mb * 1024 * 1024)
                resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
            if limits.max_processes:
                resource.setrlimit(resource.RLIMIT_NPROC, (limits.max_processes, limits.max_processes))
            if limits.max_files:
                resource.setrlimit(resource.RLIMIT_NOFILE, (limits.max_files, limits.max_files))

        return _preexec

    def _capture_artifacts(self) -> Dict[str, Artifact]:
        if self.storage is None:
            return {}

        captured: Dict[str, Artifact] = {}
        for path in self.workdir.rglob("*"):
            if not path.is_file():
                continue
            mtime = path.stat().st_mtime
            if self._known_artifacts.get(path) == mtime:
                continue
            rel = path.relative_to(self.workdir)
            name = rel.as_posix()
            try:
                if path.suffix.lower() == ".json":
                    data = json.loads(path.read_text(encoding="utf-8"))
                    artifact = create_json_artifact(data, name)
                else:
                    artifact = create_binary_artifact(path.read_bytes(), name)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to capture artifact %s: %s", path, exc)
                continue

            ref_name = rel.as_posix().replace("/", "_")
            self.storage.store(artifact, ref_name)
            captured[name] = artifact
            self._known_artifacts[path] = mtime
        return captured


class SandboxManager:
    """Pool of sandbox instances for reuse."""

    def __init__(self, max_sandboxes: int = 8) -> None:
        self.max_sandboxes = max_sandboxes
        self._sandboxes: List[Sandbox] = []
        self._lock = asyncio.Lock()

    async def get_sandbox(
        self,
        limits: Optional[ResourceLimits] = None,
        storage: Optional[ArtifactStorage] = None,
    ) -> Sandbox:
        async with self._lock:
            for sandbox in self._sandboxes:
                if sandbox._state in {
                    SandboxState.COMPLETED,
                    SandboxState.FAILED,
                    SandboxState.KILLED,
                    SandboxState.TIMED_OUT,
                }:
                    await sandbox.cleanup_resources()
                    if limits is not None:
                        sandbox.limits = limits.copy()
                    if storage is not None:
                        sandbox.storage = storage
                    sandbox._state = SandboxState.PENDING
                    return sandbox

            if len(self._sandboxes) >= self.max_sandboxes:
                raise ResourceLimitExceeded("Maximum sandbox capacity reached")

            sandbox = Sandbox(limits=limits, storage=storage)
            self._sandboxes.append(sandbox)
            return sandbox

    async def cleanup(self) -> None:
        async with self._lock:
            for sandbox in list(self._sandboxes):
                await sandbox.cleanup_resources()
            self._sandboxes.clear()


_sandbox_manager: Optional[SandboxManager] = None


def get_sandbox_manager() -> SandboxManager:
    global _sandbox_manager
    if _sandbox_manager is None:
        _sandbox_manager = SandboxManager()
    return _sandbox_manager


async def create_sandbox(
    limits: Optional[ResourceLimits] = None,
    storage: Optional[ArtifactStorage] = None,
    workdir: Optional[Union[str, Path]] = None,
    cleanup: bool = False,
) -> Sandbox:
    sandbox = Sandbox(limits=limits, storage=storage, workdir=workdir, cleanup=cleanup)
    return sandbox


# Defer worker imports to avoid circular dependencies.
try:
    from .workers import (  # noqa: WPS433 - re-export pattern
        BaseWorker,
        CPUWorker,
        IOWorker,
        PythonWorker,
        ShellWorker,
        ToolWorker,
        WorkerConfig,
        WorkerResult,
        WorkerType,
        create_worker,
        worker_result_from_sandbox_result,
    )

    __all__.extend(
        [
            "BaseWorker",
            "CPUWorker",
            "IOWorker",
            "PythonWorker",
            "ShellWorker",
            "ToolWorker",
            "WorkerConfig",
            "WorkerResult",
            "WorkerType",
            "create_worker",
            "worker_result_from_sandbox_result",
        ]
    )
except Exception:  # pragma: no cover - the workers module imports us
    pass
