"""
Sandbox execution environment for LLM Orchestrator.

This module provides secure, isolated execution environments for running untrusted code
with configurable resource limits and security constraints.
"""
from __future__ import annotations

import asyncio
<<<<<<< HEAD
=======
import enum
>>>>>>> origin/main
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Awaitable

import psutil
from pydantic import BaseModel, Field, validator

from ..artifacts import Artifact, ArtifactStorage, create_json_artifact
from ..tools import ToolResult

<<<<<<< HEAD
# Import types first to avoid circular imports
from .types import (
    SandboxState, ResourceLimits, SandboxResult,
    WorkerType, WorkerConfig, WorkerResult
)

# Import workers after types are defined
# Note: We'll import these after the Sandbox class is defined to avoid circular imports
BaseWorker = None
CPUWorker = None
IOWorker = None
ShellWorker = None
PythonWorker = None
ToolWorker = None
create_worker = None

logger = logging.getLogger(__name__)

# Re-export types and components
__all__ = [
    # Core sandbox components
    'Sandbox', 'ResourceLimits', 'SandboxResult', 'SandboxState',
    'SandboxError', 'ResourceLimitExceeded', 'SandboxTimeoutError',
    
    # Worker components
    'BaseWorker', 'WorkerResult', 'WorkerConfig', 'WorkerType',
    'CPUWorker', 'IOWorker', 'ShellWorker', 'PythonWorker', 'ToolWorker',
    'create_worker',
    
    # Sandbox management
    'SandboxManager', 'get_sandbox_manager', 'create_sandbox'
]

# Lazy import workers after Sandbox is defined
def _import_workers():
    global BaseWorker, CPUWorker, IOWorker, ShellWorker, PythonWorker, ToolWorker, create_worker, worker_result_from_sandbox_result
    if BaseWorker is None:
        from .workers import (
            BaseWorker, CPUWorker, IOWorker, ShellWorker,
            PythonWorker, ToolWorker, create_worker, worker_result_from_sandbox_result
        )
        
        # Update the Sandbox class in the workers module
        from .workers import Sandbox as WorkersSandbox
        global Sandbox
        Sandbox = WorkersSandbox

=======
logger = logging.getLogger(__name__)

>>>>>>> origin/main
class SandboxError(Exception):
    """Base exception for sandbox-related errors."""
    pass

class ResourceLimitExceeded(SandboxError):
    """Raised when a resource limit is exceeded."""
    pass

class SandboxTimeoutError(SandboxError):
    """Raised when a sandbox operation times out."""
    pass

<<<<<<< HEAD
=======
class SandboxState(str, enum.Enum):
    """State of a sandboxed process."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    KILLED = "killed"

class ResourceLimits(BaseModel):
    """Resource limits for sandboxed execution."""
    # CPU limits (percentage of a single core)
    cpu_percent: float = 100.0
    
    # Memory limits (in bytes)
    memory_mb: int = 1024  # 1GB default
    
    # Timeout in seconds
    timeout_seconds: float = 300.0  # 5 minutes default
    
    # Maximum output size (in bytes)
    max_output_bytes: int = 10 * 1024 * 1024  # 10MB default
    
    # Maximum number of processes
    max_processes: int = 10
    
    # Maximum number of open files
    max_files: int = 100
    
    # Network access control
    allow_network: bool = False
    
    # Allowed network hosts (if allow_network is True)
    allowed_hosts: List[str] = Field(default_factory=list)
    
    # Environment variables to pass to the sandbox
    env_vars: Dict[str, str] = Field(default_factory=dict)
    
    # Working directory (if None, a temporary directory will be created)
    workdir: Optional[str] = None
    
    # Path to the Python interpreter to use (default: sys.executable)
    python_path: str = sys.executable
    
    class Config:
        json_encoders = {
            Path: str,
        }

@dataclass
class SandboxResult:
    """Result of a sandboxed execution."""
    # Execution status
    success: bool
    state: SandboxState
    
    # Execution details
    return_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    
    # Resource usage
    cpu_time: float = 0.0  # CPU time in seconds
    memory_used_mb: float = 0.0  # Peak memory usage in MB
    execution_time: float = 0.0  # Wall-clock time in seconds
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Captured artifacts
    artifacts: Dict[str, Artifact] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "success": self.success,
            "state": self.state.value,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "cpu_time": self.cpu_time,
            "memory_used_mb": self.memory_used_mb,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "artifacts": {k: v.to_dict() for k, v in self.artifacts.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SandboxResult':
        """Create a SandboxResult from a dictionary."""
        return cls(
            success=data["success"],
            state=SandboxState(data["state"]),
            return_code=data.get("return_code"),
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            cpu_time=data.get("cpu_time", 0.0),
            memory_used_mb=data.get("memory_used_mb", 0.0),
            execution_time=data.get("execution_time", 0.0),
            metadata=data.get("metadata", {}),
            artifacts={
                k: Artifact.from_dict(v) 
                for k, v in data.get("artifacts", {}).items()
            },
        )

>>>>>>> origin/main
class Sandbox:
    """
    A secure, isolated execution environment for running untrusted code.
    
    This class provides a sandboxed environment with configurable resource limits
    and security constraints. It supports running Python code, shell commands,
    and other executables.
    """
    
    def __init__(
        self,
        limits: Optional[ResourceLimits] = None,
        storage: Optional[ArtifactStorage] = None,
        workdir: Optional[Union[str, Path]] = None,
        cleanup: bool = True,
    ):
        """Initialize the sandbox.
        
        Args:
            limits: Resource limits for the sandbox
            storage: Artifact storage for saving results
            workdir: Working directory (if None, a temporary directory will be used)
            cleanup: Whether to clean up the working directory on exit
        """
<<<<<<< HEAD
        # Import workers now that Sandbox is defined
        _import_workers()
        
=======
>>>>>>> origin/main
        self.limits = limits or ResourceLimits()
        self.storage = storage
        self.cleanup = cleanup
        
        # Set up working directory
        if workdir is None:
            self._temp_workdir = tempfile.mkdtemp(prefix="sandbox_")
            self.workdir = Path(self._temp_workdir)
        else:
            self.workdir = Path(workdir).resolve()
            self.workdir.mkdir(parents=True, exist_ok=True)
            self._temp_workdir = None
        
        # Process tracking
        self._process: Optional[asyncio.subprocess.Process] = None
        self._start_time: float = 0.0
        self._state: SandboxState = SandboxState.PENDING
        
        # Resource monitoring
        self._monitor_task: Optional[asyncio.Task] = None
        self._peak_memory: float = 0.0
        self._cpu_time: float = 0.0
        
        logger.info(f"Initialized sandbox in {self.workdir}")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup_resources()
    
    async def cleanup_resources(self) -> None:
        """Clean up resources used by the sandbox."""
        # Stop any running processes
        if self._process and self._process.returncode is None:
            try:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            except ProcessLookupError:
                pass  # Process already terminated
        
        # Cancel monitoring task
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Clean up working directory if needed
        if self.cleanup and self._temp_workdir and os.path.exists(self._temp_workdir):
            try:
                shutil.rmtree(self._temp_workdir)
                logger.debug(f"Cleaned up sandbox directory: {self._temp_workdir}")
            except Exception as e:
                logger.warning(f"Failed to clean up sandbox directory: {e}")
    
    async def _monitor_resources(self) -> None:
        """Monitor resource usage of the sandboxed process."""
        if not self._process or not self._process.returncode is None:
            return
        
        process = psutil.Process(self._process.pid)
        
        try:
            while True:
                try:
                    # Get memory and CPU usage
                    mem_info = process.memory_info()
                    cpu_times = process.cpu_times()
                    
                    # Update peak memory
                    mem_mb = mem_info.rss / (1024 * 1024)  # Convert to MB
                    self._peak_memory = max(self._peak_memory, mem_mb)
                    
                    # Update CPU time
                    self._cpu_time = cpu_times.user + cpu_times.system
                    
                    # Check memory limit
                    if mem_mb > self.limits.memory_mb:
                        logger.warning(
                            f"Memory limit exceeded: {mem_mb:.2f}MB > "
                            f"{self.limits.memory_mb}MB"
                        )
                        self._process.terminate()
                        self._state = SandboxState.FAILED
                        break
                    
                    # Check timeout
                    elapsed = time.monotonic() - self._start_time
                    if elapsed > self.limits.timeout_seconds:
                        logger.warning(
                            f"Timeout exceeded: {elapsed:.2f}s > "
                            f"{self.limits.timeout_seconds}s"
                        )
                        self._process.terminate()
                        self._state = SandboxState.TIMED_OUT
                        break
                    
                    # Sleep before next check
                    await asyncio.sleep(0.1)
                    
                except (psutil.NoSuchProcess, ProcessLookupError):
                    # Process has terminated
                    break
                except Exception as e:
                    logger.error(f"Error monitoring process: {e}")
                    break
                    
        except asyncio.CancelledError:
            # Monitoring was cancelled
            pass
    
    async def _run_command(
        self,
        command: Union[str, List[str]],
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[Union[str, Path]] = None,
        input_data: Optional[bytes] = None,
    ) -> SandboxResult:
        """Run a command in the sandbox."""
        if isinstance(command, str):
            command = [command]
        
        # Set up environment
        env_vars = os.environ.copy()
        env_vars.update(self.limits.env_vars)
        if env:
            env_vars.update(env)
        
        # Set up working directory
        cwd = Path(cwd) if cwd else self.workdir
        cwd.mkdir(parents=True, exist_ok=True)
        
        # Set up process limits
        def preexec():
            # Set resource limits
            import resource
            
            # Set memory limit (RSS)
            if sys.platform == 'linux':
                try:
                    import resource
                    mem_bytes = self.limits.memory_mb * 1024 * 1024
                    resource.setrlimit(
                        resource.RLIMIT_AS, 
                        (mem_bytes, mem_bytes)
                    )
                except (ValueError, resource.error) as e:
                    logger.warning(f"Failed to set memory limit: {e}")
            
            # Set CPU limit
            try:
                import resource
                cpu_time = int(self.limits.timeout_seconds)
                resource.setrlimit(
                    resource.RLIMIT_CPU, 
                    (cpu_time, cpu_time + 60)  # Soft and hard limits
                )
            except (ValueError, resource.error) as e:
                logger.warning(f"Failed to set CPU limit: {e}")
            
            # Set process limits
            try:
                import resource
                resource.setrlimit(
                    resource.RLIMIT_NPROC,
                    (self.limits.max_processes, self.limits.max_processes)
                )
                resource.setrlimit(
                    resource.RLIMIT_NOFILE,
                    (self.limits.max_files, self.limits.max_files)
                )
            except (ValueError, resource.error) as e:
                logger.warning(f"Failed to set process limits: {e}")
            
            # Change working directory
            os.chdir(str(cwd))
            
            # Drop privileges if possible
            if hasattr(os, 'setgroups') and hasattr(os, 'setgid') and hasattr(os, 'setuid'):
                try:
                    os.setgroups([])
                    os.setgid(65534)  # nobody
                    os.setuid(65534)   # nobody
                except (OSError, PermissionError) as e:
                    logger.warning(f"Failed to drop privileges: {e}")
        
        # Start the process
        self._state = SandboxState.RUNNING
        self._start_time = time.monotonic()
        self._peak_memory = 0.0
        self._cpu_time = 0.0
        
        try:
            # Start the process
            self._process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE if input_data else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env_vars,
                cwd=str(cwd),
                preexec_fn=preexec if os.name == 'posix' else None,
                limit=self.limits.max_output_bytes,
            )
            
            # Start resource monitoring
            self._monitor_task = asyncio.create_task(self._monitor_resources())
            
            # Send input if provided
            if input_data and self._process.stdin:
                self._process.stdin.write(input_data)
                await self._process.stdin.drain()
                self._process.stdin.close()
            
            # Wait for the process to complete
            stdout_data, stderr_data = await self._process.communicate()
            
            # Get the return code
            return_code = self._process.returncode
            
            # Determine the final state
            if self._state == SandboxState.RUNNING:
                if return_code == 0:
                    self._state = SandboxState.COMPLETED
                else:
                    self._state = SandboxState.FAILED
            
            # Create the result
            result = SandboxResult(
                success=return_code == 0 and self._state == SandboxState.COMPLETED,
                state=self._state,
                return_code=return_code,
                stdout=stdout_data.decode('utf-8', errors='replace'),
                stderr=stderr_data.decode('utf-8', errors='replace'),
                cpu_time=self._cpu_time,
                memory_used_mb=self._peak_memory,
                execution_time=time.monotonic() - self._start_time,
            )
            
            return result
            
        except asyncio.TimeoutError:
            if self._process and self._process.returncode is None:
                self._process.kill()
                await self._process.wait()
            
            return SandboxResult(
                success=False,
                state=SandboxState.TIMED_OUT,
                execution_time=time.monotonic() - self._start_time,
                memory_used_mb=self._peak_memory,
                cpu_time=self._cpu_time,
                stderr=f"Execution timed out after {self.limits.timeout_seconds} seconds"
            )
            
        except Exception as e:
            return SandboxResult(
                success=False,
                state=SandboxState.FAILED,
                stderr=f"Failed to execute command: {str(e)}",
                execution_time=time.monotonic() - self._start_time if '_start_time' in locals() else 0.0,
                memory_used_mb=self._peak_memory,
                cpu_time=self._cpu_time,
            )
            
        finally:
            # Clean up
            if self._monitor_task and not self._monitor_task.done():
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
    
    async def run_python_code(
        self,
        code: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[Union[str, Path]] = None,
    ) -> SandboxResult:
        """Run Python code in the sandbox."""
        # Create a temporary file for the code
        script_path = self.workdir / "script.py"
        script_path.write_text(code, encoding='utf-8')
        
        # Build the command
        cmd = [self.limits.python_path, str(script_path)]
        if args:
            cmd.extend(args)
        
        # Run the command
        return await self._run_command(cmd, env=env, cwd=cwd)
    
    async def run_shell_command(
        self,
        command: str,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[Union[str, Path]] = None,
        input_data: Optional[bytes] = None,
    ) -> SandboxResult:
        """Run a shell command in the sandbox."""
        if sys.platform == 'win32':
            cmd = ["cmd", "/c", command]
        else:
            cmd = ["/bin/sh", "-c", command]
        
        return await self._run_command(cmd, env=env, cwd=cwd, input_data=input_data)
    
    async def execute_tool(
        self,
        tool_name: str,
        tool_version: str = "latest",
        args: Optional[Dict[str, Any]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ToolResult:
        """Execute a tool in the sandbox."""
        # TODO: Implement tool execution with proper tool registry integration
        raise NotImplementedError("Tool execution not yet implemented")

# Global sandbox manager
class SandboxManager:
    """Manages a pool of sandboxed environments."""
    
    def __init__(self, max_sandboxes: int = 10):
        self.max_sandboxes = max_sandboxes
        self.active_sandboxes: List[Sandbox] = []
        self._lock = asyncio.Lock()
    
    async def get_sandbox(
        self,
        limits: Optional[ResourceLimits] = None,
        storage: Optional[ArtifactStorage] = None,
    ) -> Sandbox:
        """Get a sandbox from the pool or create a new one."""
        async with self._lock:
            # Check for available sandboxes
            for sandbox in self.active_sandboxes:
                if sandbox._state in (SandboxState.COMPLETED, SandboxState.FAILED, SandboxState.TIMED_OUT):
                    await sandbox.cleanup_resources()
                    sandbox = Sandbox(limits=limits, storage=storage)
                    return sandbox
            
            # Create a new sandbox if under limit
            if len(self.active_sandboxes) < self.max_sandboxes:
                sandbox = Sandbox(limits=limits, storage=storage)
                self.active_sandboxes.append(sandbox)
                return sandbox
            
            # Wait for a sandbox to become available
            # (In a real implementation, we'd use a Semaphore or Queue)
            raise ResourceLimitExceeded(
                f"Maximum number of sandboxes ({self.max_sandboxes}) reached"
            )
    
    async def cleanup(self) -> None:
        """Clean up all sandboxes."""
        async with self._lock:
            for sandbox in self.active_sandboxes:
                await sandbox.cleanup_resources()
            self.active_sandboxes = []

# Global sandbox manager instance
_sandbox_manager: Optional[SandboxManager] = None

def get_sandbox_manager() -> SandboxManager:
    """Get the global sandbox manager instance."""
    global _sandbox_manager
    if _sandbox_manager is None:
        _sandbox_manager = SandboxManager()
    return _sandbox_manager

async def create_sandbox(
    limits: Optional[ResourceLimits] = None,
    storage: Optional[ArtifactStorage] = None,
    workdir: Optional[Union[str, Path]] = None,
    cleanup: bool = True,
) -> Sandbox:
    """Create a new sandbox."""
    return Sandbox(limits=limits, storage=storage, workdir=workdir, cleanup=cleanup)
