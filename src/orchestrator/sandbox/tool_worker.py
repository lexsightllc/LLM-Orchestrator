# SPDX-License-Identifier: MPL-2.0
"""
Tool Worker for executing registered tools in a sandboxed environment.

This module provides a worker implementation that can execute tools from the
tool registry in an isolated environment with proper resource limits.
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, get_type_hints

from dataclasses import dataclass

from pydantic import ValidationError

from ..tools import ToolVersion, ToolResult, tool_registry, ToolError, ToolVersionError
from .types import ResourceLimits
from .workers import BaseWorker, WorkerResult, WorkerConfig, WorkerType

logger = logging.getLogger(__name__)

@dataclass
class ToolWorkerConfig(WorkerConfig):
    """Configuration for the ToolWorker."""

    tool_name: str = ""
    tool_version: str = "latest"
    install_dependencies: bool = True
    dependency_timeout: int = 300

class ToolWorker(BaseWorker):
    """Worker for executing registered tools in a sandboxed environment."""
    
    def __init__(
        self,
        tool_name: str,
        tool_version: str = "latest",
        config: Optional[WorkerConfig] = None,
        storage: Optional[Any] = None,
        sandbox: Optional[Any] = None,
    ):
        """Initialize the ToolWorker.
        
        Args:
            tool_name: Name of the tool to execute
            tool_version: Version of the tool (default: "latest")
            config: Worker configuration. If None, a default config will be used.
            storage: Optional storage for artifacts
            sandbox: Optional sandbox instance to use (for testing)
        """
        if config is None:
            config = ToolWorkerConfig(
                worker_type=WorkerType.TOOL,
                tool_name=tool_name,
                tool_version=tool_version,
                limits=ResourceLimits(
                    cpu_percent=100.0,
                    memory_mb=2048,
                    timeout_seconds=300,
                    max_output_bytes=10 * 1024 * 1024,
                ),
            )
        
        self.tool_name = tool_name
        self.tool_version = tool_version
        self._tool: Optional[ToolVersion] = None
        self.__sandbox = sandbox  # Store sandbox instance if provided

        super().__init__(config, storage)
        if sandbox is not None:
            self._sandbox = sandbox
            self._is_initialized = True
    
    async def initialize(self) -> None:
        """Initialize the worker and load the tool."""
        if self.__sandbox is None:
            await super().initialize()
        else:
            self._sandbox = self.__sandbox

        # Load the tool from the registry
        try:
            self._tool = tool_registry.get_tool(self.tool_name, self.tool_version)
            logger.info(f"Loaded tool: {self.tool_name}@{self._tool.version}")
            
            # Install dependencies if needed
            if self.config.install_dependencies and hasattr(self._tool.implementation, '__requires__'):
                await self._install_dependencies()
                
        except (ToolError, ToolVersionError) as e:
            logger.error(f"Failed to load tool {self.tool_name}@{self.tool_version}: {str(e)}")
            raise
    
    async def _install_dependencies(self) -> None:
        """Install the tool's dependencies in the sandbox."""
        if not self._tool or not hasattr(self._tool.implementation, '__requires__'):
            return
        
        requirements = self._tool.implementation.__requires__
        if not isinstance(requirements, (list, tuple)):
            logger.warning(f"Invalid requirements format for {self.tool_name}")
            return
        
        logger.info(f"Installing dependencies for {self.tool_name}: {', '.join(requirements)}")
        
        # Create a requirements file
        req_file = self._sandbox.workdir / "requirements.txt"
        req_file.write_text("\n".join(requirements))
        
        # Install the dependencies
        cmd = f"{sys.executable} -m pip install -r {req_file.name}"
        
        try:
            result = await self._sandbox.run_shell_command(
                cmd,
                timeout=self.config.dependency_timeout
            )
            
            if not result.success:
                logger.error(f"Failed to install dependencies: {result.stderr}")
                raise ToolError(f"Failed to install dependencies: {result.stderr}")
                
            logger.info(f"Successfully installed dependencies for {self.tool_name}")
            
        except Exception as e:
            logger.error(f"Error installing dependencies: {str(e)}")
            raise ToolError(f"Dependency installation failed: {str(e)}") from e
    
    async def execute(self, *args, **kwargs) -> WorkerResult:
        """Execute the tool with the given arguments.
        
        Args:
            *args: Positional arguments to pass to the tool
            **kwargs: Keyword arguments to pass to the tool
            
        Returns:
            WorkerResult containing the tool's output
        """
        if not self._tool:
            raise RuntimeError("Tool not loaded. Call initialize() first or use as a context manager.")
        
        try:
            # Validate input against the tool's schema if available
            if self._tool.input_schema:
                self._validate_input(kwargs, self._tool.input_schema)
            
            # Execute the tool
            start_time = asyncio.get_event_loop().time()
            
            async def _invoke(payload: Dict[str, Any]) -> ToolResult:
                impl = self._tool.implementation
                if inspect.iscoroutinefunction(impl):
                    try:
                        return await impl(**payload)
                    except TypeError:
                        return await impl(payload)

                loop = asyncio.get_event_loop()

                def _call() -> Any:
                    try:
                        return impl(**payload)
                    except TypeError:
                        return impl(payload)

                return await loop.run_in_executor(None, _call)

            payload = dict(kwargs)
            tool_result = await _invoke(payload)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Convert to ToolResult if needed
            if not isinstance(tool_result, ToolResult):
                tool_result = ToolResult.from_success(tool_result)
            
            # Add execution time
            tool_result.metadata["execution_time"] = execution_time
            
            # Validate output against the tool's schema if available
            if self._tool.output_schema and tool_result.success and tool_result.data is not None:
                self._validate_output(tool_result.data, self._tool.output_schema)
            
            # Convert to WorkerResult
            return WorkerResult(
                success=tool_result.success,
                output=tool_result.data,
                error=tool_result.error,
                metadata={
                    "tool": self.tool_name,
                    "version": self._tool.version,
                    **tool_result.metadata,
                },
                artifacts=getattr(tool_result, 'artifacts', {})
            )
            
        except ValidationError as e:
            error_msg = f"Input validation error: {str(e)}"
            logger.error(error_msg)
            return WorkerResult(
                success=False,
                error=error_msg,
                metadata={
                    "tool": self.tool_name,
                    "version": self._tool.version if self._tool else "unknown",
                    "error_type": "ValidationError",
                }
            )
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return WorkerResult(
                success=False,
                error=error_msg,
                metadata={
                    "tool": self.tool_name,
                    "version": self._tool.version if self._tool else "unknown",
                    "error_type": e.__class__.__name__,
                }
            )
    
    def _validate_input(self, data: Any, schema: dict) -> None:
        """Validate input data against a JSON Schema.
        
        Args:
            data: The data to validate
            schema: The JSON Schema to validate against
            
        Raises:
            ValidationError: If the data is invalid
        """
        try:
            from jsonschema import validate
            validate(instance=data, schema=schema)
        except ImportError:
            logger.warning("jsonschema not installed, input validation skipped")
        except Exception as e:
            raise ValidationError(f"Input validation failed: {str(e)}") from e
    
    def _validate_output(self, data: Any, schema: dict) -> None:
        """Validate output data against a JSON Schema.
        
        Args:
            data: The data to validate
            schema: The JSON Schema to validate against
            
        Raises:
            ValidationError: If the data is invalid
        """
        try:
            from jsonschema import validate
            validate(instance=data, schema=schema)
        except ImportError:
            logger.warning("jsonschema not installed, output validation skipped")
        except Exception as e:
            raise ValidationError(f"Output validation failed: {str(e)}") from e

# Update the worker factory to include ToolWorker
def create_tool_worker(
    tool_name: str,
    tool_version: str = "latest",
    config: Optional[WorkerConfig] = None,
    storage: Optional[Any] = None,
) -> ToolWorker:
    """Create a new ToolWorker instance.
    
    Args:
        tool_name: Name of the tool to execute
        tool_version: Version of the tool (default: "latest")
        config: Worker configuration. If None, a default config will be used.
        storage: Optional storage for artifacts
        
    Returns:
        A new ToolWorker instance
    """
    return ToolWorker(
        tool_name=tool_name,
        tool_version=tool_version,
        config=config,
        storage=storage,
    )
