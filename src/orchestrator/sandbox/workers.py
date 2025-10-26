"""
Worker implementations for the sandbox environment.

This module provides different worker types for executing tasks in the sandbox:
<<<<<<< HEAD
- BaseWorker: Abstract base class for all workers
=======
>>>>>>> origin/main
- CPUWorker: For CPU-intensive tasks
- IOWorker: For I/O-bound tasks
- ShellWorker: For executing shell commands
- PythonWorker: For executing Python code
- ToolWorker: For executing registered tools
<<<<<<< HEAD

Workers are designed to be used with the Sandbox class to provide isolated execution
environments with configurable resource limits and security constraints.
=======
>>>>>>> origin/main
"""
from __future__ import annotations

import asyncio
import logging
import os
import shlex
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, TypeVar, Generic, Callable, Awaitable

<<<<<<< HEAD
from pydantic import BaseModel, validator

# Import types first to avoid circular imports
from .types import (
    WorkerType, WorkerConfig, WorkerResult, 
    ResourceLimits, SandboxState, SandboxResult
)
from ..artifacts import Artifact, ArtifactStorage, create_json_artifact
from ..tools import ToolVersion as Tool, ToolResult, ToolError, ToolVersionError

logger = logging.getLogger(__name__)

# Type variable for generic worker results
T = TypeVar('T')

# Import Sandbox here to avoid circular imports
class Sandbox:
    """Dummy Sandbox class that will be replaced with the real one."""
    pass

class SandboxError(Exception):
    """Base exception for sandbox-related errors."""
    pass

# Add from_sandbox_result as a standalone function
def worker_result_from_sandbox_result(
    sandbox_result: SandboxResult,
    output_parser: Optional[Callable[[str], Any]] = None
) -> WorkerResult:
    """Create a WorkerResult from a SandboxResult."""
    output = sandbox_result.stdout
    if output_parser:
        try:
            output = output_parser(output)
            return WorkerResult(
                success=sandbox_result.success,
                output=output,
                error=sandbox_result.stderr if not sandbox_result.success else None,
                metadata={
                    "return_code": sandbox_result.return_code,
                    "execution_time": sandbox_result.execution_time,
                    "memory_used_mb": sandbox_result.memory_used_mb,
                    **sandbox_result.metadata
                },
                artifacts=sandbox_result.artifacts
            )
        except Exception as e:
            return WorkerResult(
                success=False,
                output=output,
                error=f"Failed to parse output: {str(e)}",
                metadata={
                    "return_code": sandbox_result.return_code,
                    "stderr": sandbox_result.stderr,
                    "execution_time": sandbox_result.execution_time,
                    "memory_used_mb": sandbox_result.memory_used_mb,
                }
            )
    
    return WorkerResult(
        success=sandbox_result.success,
        output=output,
        error=sandbox_result.stderr if not sandbox_result.success else None,
        metadata={
            "return_code": sandbox_result.return_code,
            "execution_time": sandbox_result.execution_time,
            "memory_used_mb": sandbox_result.memory_used_mb,
            **sandbox_result.metadata
        },
        artifacts=sandbox_result.artifacts
    )

class BaseWorker(ABC):
=======
from pydantic import BaseModel, Field

from . import Sandbox, ResourceLimits, SandboxResult, SandboxError
from ..artifacts import Artifact, ArtifactStorage, create_json_artifact
from ..tools import ToolVersion as Tool, ToolResult, ToolError

logger = logging.getLogger(__name__)

T = TypeVar('T')

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
    limits: Optional[ResourceLimits] = Field(default=None, description="Resource limits for the worker")
    env_vars: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    workdir: Optional[str] = Field(default=None, description="Working directory")
    cleanup: bool = Field(default=True, description="Whether to clean up after execution")
    
    class Config:
        arbitrary_types_allowed = True

class WorkerResult(BaseModel):
    """Result of a worker execution."""
    success: bool = Field(..., description="Whether the execution was successful")
    output: Any = Field(default=None, description="Output of the execution")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    artifacts: Dict[str, Artifact] = Field(default_factory=dict, description="Generated artifacts")
    
    class Config:
        arbitrary_types_allowed = True
    
    @classmethod
    def from_sandbox_result(
        cls, 
        sandbox_result: SandboxResult,
        output_parser: Optional[Callable[[str], Any]] = None
    ) -> 'WorkerResult':
        """Create a WorkerResult from a SandboxResult."""
        output = sandbox_result.stdout
        if output_parser:
            try:
                output = output_parser(output)
            except Exception as e:
                return cls(
                    success=False,
                    error=f"Failed to parse output: {str(e)}",
                    metadata={"sandbox_result": sandbox_result.to_dict()}
                )
        
        return cls(
            success=sandbox_result.success,
            output=output,
            error=sandbox_result.stderr if not sandbox_result.success else None,
            metadata={
                "sandbox_result": sandbox_result.to_dict(),
                "execution_time": sandbox_result.execution_time,
                "cpu_time": sandbox_result.cpu_time,
                "memory_used_mb": sandbox_result.memory_used_mb,
            },
            artifacts=sandbox_result.artifacts,
        )

class BaseWorker(ABC, Generic[T]):
>>>>>>> origin/main
    """Base class for all workers."""
    
    def __init__(
        self,
        config: WorkerConfig,
        storage: Optional[ArtifactStorage] = None,
    ):
        """Initialize the worker.
        
        Args:
            config: Worker configuration
            storage: Artifact storage for saving results
        """
        self.config = config
        self.storage = storage
<<<<<<< HEAD
        self._sandbox = None  # Will be set after Sandbox class is defined
    
    async def __aenter__(self):
=======
        self._sandbox: Optional[Sandbox] = None
    
    async def __aenter__(self) -> 'BaseWorker[T]':
>>>>>>> origin/main
        """Context manager entry."""
        await self.initialize()
        return self
    
<<<<<<< HEAD
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize the worker."""
        if self._sandbox is None and Sandbox is not None:
            self._sandbox = Sandbox(
                limits=self.config.limits,
                workdir=self.config.workdir,
                cleanup=self.config.cleanup
            )
            await self._sandbox.__aenter__()
    
    async def cleanup(self):
        """Clean up resources used by the worker."""
        if self._sandbox is not None:
            await self._sandbox.__aexit__(None, None, None)
=======
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        await self.cleanup()
    
    async def initialize(self) -> None:
        """Initialize the worker."""
        self._sandbox = Sandbox(
            limits=self.config.limits or ResourceLimits(),
            storage=self.storage,
            workdir=self.config.workdir,
            cleanup=self.config.cleanup,
        )
        await self._sandbox.__aenter__()
    
    async def cleanup(self) -> None:
        """Clean up resources used by the worker."""
        if self._sandbox:
            await self._sandbox.cleanup_resources()
>>>>>>> origin/main
            self._sandbox = None
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> WorkerResult:
        """Execute the worker's task.
        
        Returns:
            WorkerResult containing the result of the execution
        """
        pass

class CPUWorker(BaseWorker):
    """Worker for CPU-intensive tasks."""
    
    def __init__(
        self,
        config: Optional[WorkerConfig] = None,
        storage: Optional[ArtifactStorage] = None,
    ):
        """Initialize the CPU worker."""
        if config is None:
            config = WorkerConfig(
                worker_type=WorkerType.CPU,
                limits=ResourceLimits(
                    cpu_percent=100.0,
                    memory_mb=2048,
                    timeout_seconds=300,
                    max_output_bytes=10 * 1024 * 1024,  # 10MB
                ),
            )
        super().__init__(config, storage)
    
    async def execute(
        self,
        function: Callable[..., T],
        *args,
        **kwargs
    ) -> WorkerResult:
        """Execute a CPU-intensive function.
        
        Args:
            function: The function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            WorkerResult containing the result of the execution
        """
        if not self._sandbox:
            raise RuntimeError("Worker not initialized. Call initialize() first or use as a context manager.")
        
        # Serialize the function and its arguments
        import cloudpickle
        import base64
        
        # Create a temporary file for the serialized function
        func_file = Path(self._sandbox.workdir) / "function.pkl"
        with open(func_file, "wb") as f:
            cloudpickle.dump((function, args, kwargs), f)
        
        # Create a runner script
        runner_script = """
import cloudpickle
import sys
import os

# Read the serialized function and arguments
with open('function.pkl', 'rb') as f:
    function, args, kwargs = cloudpickle.load(f)

# Execute the function
try:
    result = function(*args, **kwargs)
    print("SUCCESS")
    print(cloudpickle.dumps(result).decode('latin1'))
except Exception as e:
    print("ERROR")
    print(str(e))
    sys.exit(1)
        """
        
        # Write the runner script to a file
        runner_file = Path(self._sandbox.workdir) / "runner.py"
        runner_file.write_text(runner_script)
        
        # Install required packages
        await self._sandbox.run_shell_command(
            f"{sys.executable} -m pip install cloudpickle",
            env={"PYTHONPATH": "."}
        )
        
        # Run the function in the sandbox
        result = await self._sandbox.run_python_code(
            str(runner_file),
            env={"PYTHONPATH": "."}
        )
        
        # Parse the result
        if not result.success:
            return WorkerResult(
                success=False,
                error=result.stderr or "Unknown error",
                metadata={"sandbox_result": result.to_dict()}
            )
        
        # Parse the output
        output = result.stdout.strip()
        if output.startswith("SUCCESS\n"):
            try:
                result_data = cloudpickle.loads(
                    output[8:].encode('latin1')  # Skip "SUCCESS\n"
                )
                return WorkerResult(
                    success=True,
                    output=result_data,
                    metadata={"sandbox_result": result.to_dict()}
                )
            except Exception as e:
                return WorkerResult(
                    success=False,
                    error=f"Failed to deserialize result: {str(e)}",
                    metadata={"sandbox_result": result.to_dict()}
                )
        elif output.startswith("ERROR\n"):
            return WorkerResult(
                success=False,
                error=output[6:].strip(),  # Skip "ERROR\n"
                metadata={"sandbox_result": result.to_dict()}
            )
        else:
            return WorkerResult(
                success=False,
                error=f"Unexpected output: {output}",
                metadata={"sandbox_result": result.to_dict()}
            )

class IOWorker(BaseWorker):
    """Worker for I/O-bound tasks."""
    
    def __init__(
        self,
        config: Optional[WorkerConfig] = None,
        storage: Optional[ArtifactStorage] = None,
    ):
        """Initialize the IO worker."""
        if config is None:
            config = WorkerConfig(
                worker_type=WorkerType.IO,
                limits=ResourceLimits(
                    cpu_percent=50.0,  # Lower CPU priority
                    memory_mb=1024,
                    timeout_seconds=600,  # Longer timeout for I/O operations
                    max_output_bytes=100 * 1024 * 1024,  # 100MB for file operations
                ),
            )
        super().__init__(config, storage)
    
    async def execute(
        self,
        operation: str,
        *args,
        **kwargs
    ) -> WorkerResult:
        """Execute an I/O operation.
        
        Args:
            operation: The I/O operation to perform (read, write, copy, move, delete)
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            WorkerResult containing the result of the operation
        """
        if not self._sandbox:
            raise RuntimeError("Worker not initialized. Call initialize() first or use as a context manager.")
        
        operation = operation.lower()
        
        if operation == "read":
            return await self._read_file(*args, **kwargs)
        elif operation == "write":
            return await self._write_file(*args, **kwargs)
        elif operation == "copy":
            return await self._copy_file(*args, **kwargs)
        elif operation == "move":
            return await self._move_file(*args, **kwargs)
        elif operation == "delete":
            return await self._delete_file(*args, **kwargs)
        else:
            return WorkerResult(
                success=False,
                error=f"Unsupported I/O operation: {operation}"
            )
    
    async def _read_file(self, file_path: str, encoding: str = "utf-8") -> WorkerResult:
        """Read a file from the sandbox."""
        try:
            file_path = Path(file_path)
            if not file_path.is_absolute():
                file_path = self._sandbox.workdir / file_path
            
            content = file_path.read_text(encoding=encoding)
            return WorkerResult(
                success=True,
                output=content,
                metadata={
                    "file_path": str(file_path),
                    "size": len(content),
                    "encoding": encoding,
                }
            )
        except Exception as e:
            return WorkerResult(
                success=False,
                error=f"Failed to read file: {str(e)}",
                metadata={"file_path": str(file_path) if 'file_path' in locals() else None}
            )
    
    async def _write_file(
        self, 
        file_path: str, 
        content: str, 
        encoding: str = "utf-8"
    ) -> WorkerResult:
        """Write content to a file in the sandbox."""
        try:
            file_path = Path(file_path)
            if not file_path.is_absolute():
                file_path = self._sandbox.workdir / file_path
            
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_path.write_text(content, encoding=encoding)
            
            return WorkerResult(
                success=True,
                output=str(file_path),
                metadata={
                    "file_path": str(file_path),
                    "size": len(content),
                    "encoding": encoding,
                }
            )
        except Exception as e:
            return WorkerResult(
                success=False,
                error=f"Failed to write file: {str(e)}",
                metadata={"file_path": str(file_path) if 'file_path' in locals() else None}
            )
    
    async def _copy_file(self, src: str, dst: str) -> WorkerResult:
        """Copy a file within the sandbox."""
        try:
            src_path = Path(src) if Path(src).is_absolute() else self._sandbox.workdir / src
            dst_path = Path(dst) if Path(dst).is_absolute() else self._sandbox.workdir / dst
            
            # Create parent directories if they don't exist
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(src_path, dst_path)
            
            return WorkerResult(
                success=True,
                output=str(dst_path),
                metadata={
                    "source": str(src_path),
                    "destination": str(dst_path),
                }
            )
        except Exception as e:
            return WorkerResult(
                success=False,
                error=f"Failed to copy file: {str(e)}",
                metadata={
                    "source": str(src_path) if 'src_path' in locals() else src,
                    "destination": str(dst_path) if 'dst_path' in locals() else dst,
                }
            )
    
    async def _move_file(self, src: str, dst: str) -> WorkerResult:
        """Move a file within the sandbox."""
        try:
            src_path = Path(src) if Path(src).is_absolute() else self._sandbox.workdir / src
            dst_path = Path(dst) if Path(dst).is_absolute() else self._sandbox.workdir / dst
            
            # Create parent directories if they don't exist
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(src_path), str(dst_path))
            
            return WorkerResult(
                success=True,
                output=str(dst_path),
                metadata={
                    "source": str(src_path),
                    "destination": str(dst_path),
                }
            )
        except Exception as e:
            return WorkerResult(
                success=False,
                error=f"Failed to move file: {str(e)}",
                metadata={
                    "source": str(src_path) if 'src_path' in locals() else src,
                    "destination": str(dst_path) if 'dst_path' in locals() else dst,
                }
            )
    
    async def _delete_file(self, file_path: str) -> WorkerResult:
        """Delete a file from the sandbox."""
        try:
            file_path = Path(file_path)
            if not file_path.is_absolute():
                file_path = self._sandbox.workdir / file_path
            
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)
            else:
                return WorkerResult(
                    success=False,
                    error=f"Path does not exist: {file_path}",
                    metadata={"file_path": str(file_path)}
                )
            
            return WorkerResult(
                success=True,
                output=str(file_path),
                metadata={"file_path": str(file_path)}
            )
        except Exception as e:
            return WorkerResult(
                success=False,
                error=f"Failed to delete path: {str(e)}",
                metadata={"file_path": str(file_path) if 'file_path' in locals() else file_path}
            )

class ShellWorker(BaseWorker):
    """Worker for executing shell commands."""
    
    def __init__(
        self,
        config: Optional[WorkerConfig] = None,
        storage: Optional[ArtifactStorage] = None,
    ):
        """Initialize the shell worker."""
        if config is None:
            config = WorkerConfig(
                worker_type=WorkerType.SHELL,
                limits=ResourceLimits(
                    cpu_percent=100.0,
                    memory_mb=2048,
                    timeout_seconds=300,
                    max_output_bytes=10 * 1024 * 1024,  # 10MB
                ),
            )
        super().__init__(config, storage)
    
    async def execute(
        self,
        command: Union[str, List[str]],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        input_data: Optional[Union[str, bytes]] = None,
    ) -> WorkerResult:
        """Execute a shell command in the sandbox.
        
        Args:
            command: The shell command to execute
            cwd: Working directory (relative to sandbox root)
            env: Environment variables
            input_data: Input to send to the command's stdin
            
        Returns:
            WorkerResult containing the command output
        """
        if not self._sandbox:
            raise RuntimeError("Worker not initialized. Call initialize() first or use as a context manager.")
        
        # Prepare environment variables
        env_vars = {}
        if env:
            env_vars.update(env)
        
        # Set working directory
        workdir = None
        if cwd:
            workdir = Path(cwd)
            if not workdir.is_absolute():
                workdir = self._sandbox.workdir / workdir
        
        # Convert input to bytes if needed
        input_bytes = None
        if input_data is not None:
            if isinstance(input_data, str):
                input_bytes = input_data.encode('utf-8')
            else:
                input_bytes = input_data
        
        # Execute the command
        result = await self._sandbox.run_shell_command(
            command,
            env=env_vars,
            cwd=workdir,
            input_data=input_bytes,
        )
        
        # Create a worker result
        return WorkerResult.from_sandbox_result(result)

class PythonWorker(BaseWorker):
    """Worker for executing Python code."""
    
    def __init__(
        self,
        config: Optional[WorkerConfig] = None,
        storage: Optional[ArtifactStorage] = None,
    ):
        """Initialize the Python worker."""
        if config is None:
            config = WorkerConfig(
                worker_type=WorkerType.PYTHON,
                limits=ResourceLimits(
                    cpu_percent=100.0,
                    memory_mb=2048,
                    timeout_seconds=300,
                    max_output_bytes=10 * 1024 * 1024,  # 10MB
                ),
            )
        super().__init__(config, storage)
    
    async def execute(
        self,
        code: str,
        args: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> WorkerResult:
        """Execute Python code in the sandbox.
        
        Args:
            code: Python code to execute
            args: Command-line arguments
            cwd: Working directory (relative to sandbox root)
            env: Environment variables
            
        Returns:
            WorkerResult containing the code output
        """
        if not self._sandbox:
            raise RuntimeError("Worker not initialized. Call initialize() first or use as a context manager.")
        
        # Set working directory
        workdir = None
        if cwd:
            workdir = Path(cwd)
            if not workdir.is_absolute():
                workdir = self._sandbox.workdir / workdir
        
        # Execute the code
        result = await self._sandbox.run_python_code(
            code,
            args=args or [],
            env=env or {},
            cwd=workdir,
        )
        
        # Create a worker result
        return WorkerResult.from_sandbox_result(result)

class ToolWorker(BaseWorker):
<<<<<<< HEAD
    """Worker for executing registered tools in a sandboxed environment.
    
    This worker handles the execution of tools registered in the tool registry,
    managing their lifecycle, resource constraints, and result processing.
    
    Args:
        tool_name: Name of the tool to execute
        tool_version: Version of the tool (default: "latest")
        config: Worker configuration. If None, a default config will be used.
        storage: Optional storage for artifacts
        sandbox: Optional sandbox instance. If None, one will be created.
    """
=======
    """Worker for executing registered tools."""
>>>>>>> origin/main
    
    def __init__(
        self,
        tool_name: str,
        tool_version: str = "latest",
        config: Optional[WorkerConfig] = None,
        storage: Optional[ArtifactStorage] = None,
<<<<<<< HEAD
        sandbox: Optional[Sandbox] = None,
    ):
=======
    ):
        """Initialize the tool worker.
        
        Args:
            tool_name: Name of the tool to execute
            tool_version: Version of the tool (default: "latest")
            config: Worker configuration
            storage: Artifact storage
        """
>>>>>>> origin/main
        if config is None:
            config = WorkerConfig(
                worker_type=WorkerType.TOOL,
                limits=ResourceLimits(
                    cpu_percent=100.0,
                    memory_mb=2048,
                    timeout_seconds=300,
                    max_output_bytes=10 * 1024 * 1024,  # 10MB
                ),
            )
        
        self.tool_name = tool_name
        self.tool_version = tool_version
        self._tool = None
<<<<<<< HEAD
        self._sandbox = sandbox
        self._is_initialized = False
=======
>>>>>>> origin/main
        
        super().__init__(config, storage)
    
    async def initialize(self) -> None:
<<<<<<< HEAD
        """Initialize the worker and load the tool.
        
        Raises:
            ToolError: If the tool cannot be found or loaded
            ToolVersionError: If the specified version is not available
        """
        if self._is_initialized:
            return
            
        await super().initialize()
        
        try:
            # Import here to avoid circular imports
            from ..tools import tool_registry
            
            # Get the tool from the registry
            tool_cls = tool_registry.get_tool(self.tool_name, self.tool_version)
            if tool_cls is None:
                raise ToolError(f"Tool '{self.tool_name}' not found in registry")
                
            # Create a sandbox if one wasn't provided
            if self._sandbox is None:
                self._sandbox = Sandbox(limits=self.config.limits)
                await self._sandbox.initialize()
            
            # Initialize the tool
            self._tool = tool_cls(sandbox=self._sandbox, storage=self.storage)
            self._is_initialized = True
            
            logger.info(f"Initialized ToolWorker for {self.tool_name} v{self.tool_version}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ToolWorker: {e}")
            if isinstance(e, (ToolError, ToolVersionError)):
                raise
            raise ToolError(f"Failed to initialize tool: {e}") from e
=======
        """Initialize the worker and load the tool."""
        await super().initialize()
        
        # TODO: Load the tool from the tool registry
        # self._tool = await self._load_tool()
        raise NotImplementedError("Tool loading not implemented yet")
>>>>>>> origin/main
    
    async def execute(self, *args, **kwargs) -> WorkerResult:
        """Execute the tool with the given arguments.
        
        Args:
            *args: Positional arguments to pass to the tool
            **kwargs: Keyword arguments to pass to the tool
            
        Returns:
<<<<<<< HEAD
            WorkerResult containing the tool's output and metadata
            
        Raises:
            RuntimeError: If the worker is not initialized
            ToolError: If the tool execution fails
        """
        if not self._is_initialized or not self._tool:
            raise RuntimeError(
                "Tool not loaded. Call initialize() first or use as a context manager."
            )
        
        start_time = time.monotonic()
=======
            WorkerResult containing the tool's output
        """
        if not self._tool:
            raise RuntimeError("Tool not loaded. Call initialize() first or use as a context manager.")
>>>>>>> origin/main
        
        try:
            # Execute the tool
            result = await self._tool.execute(*args, **kwargs)
<<<<<<< HEAD
            execution_time = time.monotonic() - start_time
            
            # Create a result object
            worker_result = WorkerResult(
=======
            
            return WorkerResult(
>>>>>>> origin/main
                success=result.success,
                output=result.output,
                error=result.error,
                metadata={
                    "tool": self.tool_name,
                    "version": self.tool_version,
<<<<<<< HEAD
                    "execution_time": execution_time,
                    "tool_metadata": getattr(result, 'metadata', {})
                },
                artifacts={
                    name: create_json_artifact(
                        artifact.dict() if hasattr(artifact, 'dict') else artifact,
                        name=f"{self.tool_name}_{name}"
                    )
                    for name, artifact in getattr(result, 'artifacts', {}).items()
                }
            )
            
            # Log the result
            logger.info(
                f"Tool '{self.tool_name}' executed in {execution_time:.2f}s: "
                f"success={result.success}"
            )
            
            return worker_result
            
        except Exception as e:
            execution_time = time.monotonic() - start_time
            error_msg = f"Tool '{self.tool_name}' execution failed after {execution_time:.2f}s: {e}"
            logger.error(error_msg, exc_info=True)
            
            return WorkerResult(
                success=False,
                error=error_msg,
                metadata={
                    "tool": self.tool_name,
                    "version": self.tool_version,
                    "execution_time": execution_time,
                    "error_type": type(e).__name__,
                    "error_details": str(e)
=======
                    "execution_time": getattr(result, 'execution_time', None),
                },
                artifacts=getattr(result, 'artifacts', {})
            )
        except Exception as e:
            return WorkerResult(
                success=False,
                error=f"Tool execution failed: {str(e)}",
                metadata={
                    "tool": self.tool_name,
                    "version": self.tool_version,
>>>>>>> origin/main
                }
            )

# Factory function for creating workers
def create_worker(
    worker_type: Union[str, WorkerType],
    config: Optional[WorkerConfig] = None,
    storage: Optional[ArtifactStorage] = None,
    **kwargs
) -> BaseWorker:
    """Create a worker of the specified type.
    
    Args:
        worker_type: Type of worker to create (cpu, io, shell, python, tool)
        config: Worker configuration
        storage: Artifact storage
<<<<<<< HEAD
        **kwargs: Additional arguments for the worker constructor:
            - For ToolWorker: tool_name, tool_version, sandbox
            - For other workers: workdir, env_vars, etc.
            
    Returns:
        A worker instance
    """
    """Create a worker of the specified type.
    
    Args:
        worker_type: Type of worker to create (cpu, io, shell, python, tool)
        config: Worker configuration
        storage: Artifact storage
        **kwargs: Additional arguments for the worker constructor:
            - For ToolWorker: tool_name, tool_version, sandbox
            - For other workers: workdir, env_vars, etc.
            
    Returns:
        A worker instance
        
    Example:
        ```python
        # Create a shell worker
        shell_worker = create_worker('shell')
        
        # Create a tool worker
        tool_worker = create_worker(
            'tool',
            tool_name='calculator',
            tool_version='1.0.0',
            storage=my_storage
        )
        ```
    """
    if isinstance(worker_type, str):
        worker_type = WorkerType(worker_type.lower())
    
    # Special handling for ToolWorker
    if worker_type == WorkerType.TOOL:
        tool_name = kwargs.pop('tool_name', None)
        if tool_name is None:
            raise ValueError("tool_name is required for ToolWorker")
            
        tool_version = kwargs.pop('tool_version', 'latest')
        sandbox = kwargs.pop('sandbox', None)
        
        return ToolWorker(
            tool_name=tool_name,
            tool_version=tool_version,
            config=config,
            storage=storage,
            sandbox=sandbox,
            **kwargs
        )
    
    # Handle other worker types
    worker_class = {
        WorkerType.CPU: CPUWorker,
        WorkerType.IO: IOWorker,
        WorkerType.SHELL: ShellWorker,
        WorkerType.PYTHON: PythonWorker,
    }.get(worker_type)
    
    if worker_class is None:
        raise ValueError(f"Unknown worker type: {worker_type}")
    
    return worker_class(config=config, storage=storage, **kwargs)
=======
        **kwargs: Additional arguments for the worker constructor
        
    Returns:
        A worker instance
    """
    if isinstance(worker_type, str):
        worker_type = WorkerType(worker_type.lower())
    
    if worker_type == WorkerType.CPU:
        return CPUWorker(config=config, storage=storage, **kwargs)
    elif worker_type == WorkerType.IO:
        return IOWorker(config=config, storage=storage, **kwargs)
    elif worker_type == WorkerType.SHELL:
        return ShellWorker(config=config, storage=storage, **kwargs)
    elif worker_type == WorkerType.PYTHON:
        return PythonWorker(config=config, storage=storage, **kwargs)
    elif worker_type == WorkerType.TOOL:
        return ToolWorker(config=config, storage=storage, **kwargs)
    else:
        raise ValueError(f"Unknown worker type: {worker_type}")
>>>>>>> origin/main
