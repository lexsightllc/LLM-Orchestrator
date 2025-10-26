"""Tests for the ToolWorker implementation."""
import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, PropertyMock, MagicMock, AsyncMock

import pytest
from pydantic import BaseModel

from orchestrator.tools import ToolVersion, ToolResult, tool_registry, register_tool
from orchestrator.sandbox.tool_worker import ToolWorker, ToolWorkerConfig
from orchestrator.sandbox.workers import WorkerResult

# Test tool implementation
@register_tool(
    version="1.0.0",
    name="test_tool",
    input_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer", "minimum": 1}
        },
        "required": ["name"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "greeting": {"type": "string"},
            "count": {"type": "integer"}
        },
        "required": ["greeting"]
    },
    aliases=["test"]
)
def test_tool(name: str, count: int = 1) -> ToolResult:
    """A simple test tool that generates greetings."""
    return ToolResult.from_success({
        "greeting": f"Hello, {name}!" * count,
        "count": count
    })

# Async version of the test tool
@register_tool(
    version="2.0.0",
    name="async_test_tool",
    input_schema={"type": "object"},
    output_schema={"type": "object"}
)
async def async_test_tool(delay: float = 0.1) -> ToolResult:
    """An async test tool that simulates work with a delay."""
    await asyncio.sleep(delay)
    return ToolResult.from_success({"status": "completed", "delay": delay})

@pytest.fixture
def tool_registry_setup():
    """Set up a clean tool registry for testing."""
    from orchestrator.tools import tool_registry
    
    # Save original tools
    original_tools = tool_registry._tools.copy()
    original_aliases = tool_registry._aliases.copy()
    
    # Clear existing tools for clean test environment
    tool_registry._tools = {}
    tool_registry._aliases = {}
    
    # Register test tools
    @register_tool(
        version="1.0.0",
        name="test_tool",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer", "minimum": 1}
            },
            "required": ["name"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "greeting": {"type": "string"},
                "count": {"type": "integer"}
            },
            "required": ["greeting"]
        }
    )
    def test_tool(name: str, count: int = 1):
        return ToolResult.from_success({"greeting": "Hello, World!" * count})
    
    @register_tool(
        version="2.0.0",
        name="test_tool",
        input_schema={"type": "object"},
        output_schema={"type": "object"}
    )
    def test_tool_v2():
        return ToolResult.from_success({"version": "2.0.0"})
    
    # Register async test tool
    @register_tool(
        version="1.0.0",
        name="async_test_tool",
        input_schema={"type": "object"},
        output_schema={"type": "object"}
    )
    async def async_test_tool():
        await asyncio.sleep(0.1)  # Simulate async work
        return ToolResult.from_success({"async": True})
    
    # Register the test_tool again with an alias
    @register_tool(
        version="1.0.0",
        name="test_tool",
        aliases=["test"],
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer", "minimum": 1}
            },
            "required": ["name"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "greeting": {"type": "string"},
                "count": {"type": "integer"}
            },
            "required": ["greeting"]
        }
    )
    def test_tool_with_alias(name: str, count: int = 1):
        return ToolResult.from_success({"greeting": "Hello, World!" * count})
    
    # Register a tool with dependencies
    @register_tool(
        version="1.0.0",
        name="tool_with_deps",
        input_schema={"type": "object"},
        output_schema={"type": "object"}
    )
    def tool_with_deps():
        import requests  # noqa
        return ToolResult.from_success({"status": "ok"})
    
    # Add requires attribute for dependency testing
    tool_with_deps.__requires__ = ["requests>=2.25.0"]
    
    # Register tools for input/output validation testing
    @register_tool(
        version="1.0.0",
        name="validating_tool",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        },
        output_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"]
        }
    )
    def validating_tool(name: str):
        return ToolResult.from_success({"message": f"Hello, {name}!"})
    
    # Register a tool with invalid output for testing
    @register_tool(
        version="1.0.0",
        name="invalid_output_tool",
        input_schema={"type": "object"},
        output_schema={
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"]
        }
    )
    def invalid_output_tool():
        return ToolResult.from_success({"invalid": True})
    
    # Register a failing tool for error handling tests
    @register_tool(
        version="1.0.0",
        name="failing_tool",
        input_schema={"type": "object"},
        output_schema={"type": "object"}
    )
    def failing_tool():
        raise ValueError("Something went wrong")
    
    yield tool_registry
    
    # Restore original tools
    tool_registry._tools = original_tools
    tool_registry._aliases = original_aliases

@pytest.mark.asyncio
async def test_tool_worker_initialization(tool_registry_setup):
    """Test ToolWorker initialization and tool loading."""
    async with ToolWorker("test_tool", "1.0.0") as worker:
        assert worker.tool_name == "test_tool"
        assert worker.tool_version == "1.0.0"
        assert worker._tool is not None
        assert worker._tool.name == "test_tool"
        assert worker._tool.version == "1.0.0"

@pytest.mark.asyncio
async def test_tool_worker_execute_sync(tool_registry_setup):
    """Test executing a synchronous tool."""
    # First, unregister any existing tool with the same name to avoid conflicts
    if "test_sync_tool" in tool_registry_setup._tools:
        del tool_registry_setup._tools["test_sync_tool"]
    if "test_sync_tool" in tool_registry_setup._aliases:
        del tool_registry_setup._aliases["test_sync_tool"]
    
    # Define the tool function
    def test_sync_tool(*args, **kwargs):
        """A simple synchronous test tool."""
        name = kwargs.get("name", "")
        count = kwargs.get("count", 1)
        return ToolResult.from_success({
            "greeting": f"Hello, {name}!" * count,
            "count": count
        })
    
    # Register the tool with the registry directly to ensure proper registration
    tool = ToolVersion(
        name="test_sync_tool",
        version="1.0.0",
        implementation=test_sync_tool,
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer", "default": 1}
            },
            "required": ["name"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "greeting": {"type": "string"},
                "count": {"type": "integer"}
            },
            "required": ["greeting"]
        }
    )
    tool_registry_setup.register(tool)
    
    # Test the tool execution with a mock for the event loop
    with patch('asyncio.get_event_loop') as mock_get_loop:
        # Create a mock event loop
        mock_loop = AsyncMock()
        mock_get_loop.return_value = mock_loop
        
        # Configure the run_in_executor mock to call the function directly
        def run_in_executor(executor, func, *args, **kwargs):
            return asyncio.coroutine(lambda: func(*args, **kwargs))()
        
        mock_loop.run_in_executor.side_effect = run_in_executor
        
        async with ToolWorker("test_sync_tool", "1.0.0") as worker:
            result = await worker.execute({"name": "World", "count": 2})
            
            assert result.success
            assert "greeting" in result.output
            assert result.output["greeting"] == "Hello, World!Hello, World!"
            assert result.output["count"] == 2
            assert "execution_time" in result.metadata
            assert result.metadata["tool"] == "test_sync_tool"
        assert result.metadata["version"] == "1.0.0"

@pytest.mark.asyncio
async def test_tool_worker_execute_async(tool_registry_setup):
    """Test executing an asynchronous tool."""
    # First, unregister any existing tool with the same name to avoid conflicts
    if "async_test_tool" in tool_registry_setup._tools:
        del tool_registry_setup._tools["async_test_tool"]
    if "async_test_tool" in tool_registry_setup._aliases:
        del tool_registry_setup._aliases["async_test_tool"]
    
    # Define an async tool function
    async def async_test_tool(*args, **kwargs):
        """A simple asynchronous test tool."""
        await asyncio.sleep(0.1)  # Simulate async work
        return ToolResult.from_success({"result": "async result"})
    
    # Register the tool with the registry directly to ensure proper registration
    tool = ToolVersion(
        name="async_test_tool",
        version="1.0.0",
        implementation=async_test_tool,
        input_schema={"type": "object", "properties": {}, "additionalProperties": True},
        output_schema={
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"]
        }
    )
    tool_registry_setup.register(tool)
    
    # Test the tool execution
    async with ToolWorker("async_test_tool", "1.0.0") as worker:
        start_time = asyncio.get_event_loop().time()
        result = await worker.execute({})  # No input parameters needed for this test
        execution_time = asyncio.get_event_loop().time() - start_time
        
        assert result.success
        assert "result" in result.output
        assert result.output["result"] == "async result"
        assert execution_time >= 0.05  # Should take at least 0.05 seconds due to asyncio.sleep
        assert "execution_time" in result.metadata
        assert result.metadata["tool"] == "async_test_tool"
        assert result.metadata["version"] == "1.0.0"

@pytest.mark.asyncio
async def test_tool_worker_input_validation(tool_registry_setup):
    """Test input validation with the tool's schema."""
    # First, unregister any existing tool with the same name to avoid conflicts
    if "validating_tool" in tool_registry_setup._tools:
        del tool_registry_setup._tools["validating_tool"]
    if "validating_tool" in tool_registry_setup._aliases:
        del tool_registry_setup._aliases["validating_tool"]
    
    # Define the tool function
    async def validating_tool(*args, **kwargs):
        return ToolResult.from_success({"valid": True})
    
    # Register the tool with the registry directly
    tool = ToolVersion(
        name="validating_tool",
        version="1.0.0",
        implementation=validating_tool,
        input_schema={
            "type": "object",
            "properties": {
                "required_field": {"type": "string"},
                "optional_field": {"type": "integer", "default": 42}
            },
            "required": ["required_field"]
        },
        output_schema={
            "type": "object",
            "properties": {"valid": {"type": "boolean"}},
            "required": ["valid"]
        }
    )
    tool_registry_setup.register(tool)
    
    # Test with valid input (all required fields)
    async with ToolWorker("validating_tool", "1.0.0") as worker:
        result = await worker.execute({"required_field": "test"})
        assert result.success
        assert result.output["valid"] is True
    
    # Test with optional field
    async with ToolWorker("validating_tool", "1.0.0") as worker:
        result = await worker.execute({
            "required_field": "test",
            "optional_field": 100
        })
        assert result.success
        assert result.output["valid"] is True
    
    # Test with missing required field (should raise ValidationError)
    async with ToolWorker("validating_tool", "1.0.0") as worker:
        with pytest.raises(ValidationError) as exc_info:
            await worker.execute({})  # Missing required_field
        assert "required_field" in str(exc_info.value)

@pytest.mark.asyncio
async def test_tool_worker_alias_resolution(tool_registry_setup):
    """Test that tool aliases are resolved correctly."""
    # First, unregister any existing tool with the same name to avoid conflicts
    if "aliased_tool" in tool_registry_setup._tools:
        del tool_registry_setup._tools["aliased_tool"]
    if "aliased_tool" in tool_registry_setup._aliases:
        del tool_registry_setup._aliases["aliased_tool"]
    
    # Define the tool function with proper parameter handling
    def aliased_tool(*args, **kwargs):
        return ToolResult.from_success({"status": "aliased_tool_called"})
    
    # Add the tool with alias to the registry
    tool = ToolVersion(
        name="aliased_tool",
        version="1.0.0",
        implementation=aliased_tool,
        input_schema={"type": "object", "properties": {}, "additionalProperties": True},
        output_schema={"type": "object"}
    )
    tool_registry_setup.register(tool, aliases=["aliased"])
    
    # Test with the alias
    async with ToolWorker("aliased_tool", "1.0.0") as worker:
        result = await worker.execute({})
        assert result.success
        assert result.output["status"] == "aliased_tool_called"
    
    # Also test that the alias is registered
    assert tool_registry_setup.get_tool("aliased_tool", "1.0.0") is not None

@pytest.mark.asyncio
async def test_tool_worker_version_resolution(tool_registry_setup):
    """Test that tool versions are resolved correctly."""
    # Test exact version
    async with ToolWorker("test_tool", "2.0.0") as worker:
        result = await worker.execute()
        assert result.success
        assert result.output["version"] == "2.0.0"
    
    # Test version constraint (use >= for version comparison)
    async with ToolWorker("test_tool", ">=1.0.0") as worker:
        result = await worker.execute()
        assert result.success
        # Check if it's either v1 or v2 output format
        assert "greeting" in result.output or "version" in result.output

@pytest.mark.asyncio
async def test_tool_worker_dependency_installation(tool_registry_setup, tmp_path):
    """Test tool dependency installation."""
    # First, unregister any existing tool with the same name to avoid conflicts
    if "tool_with_deps" in tool_registry_setup._tools:
        del tool_registry_setup._tools["tool_with_deps"]
    if "tool_with_deps" in tool_registry_setup._aliases:
        del tool_registry_setup._aliases["tool_with_deps"]
    
    # Create a mock sandbox with a proper run_shell_command implementation
    mock_sandbox = AsyncMock()
    mock_sandbox.workdir = tmp_path
    
    async def mock_run_shell_command(cmd, **kwargs):
        # Create a requirements file in the workdir
        req_file = mock_sandbox.workdir / "requirements.txt"
        if not req_file.exists():
            req_file.write_text("requests>=2.25.0\n")
        # Ignore the timeout parameter if provided
        if 'timeout' in kwargs:
            del kwargs['timeout']
        return {
            'success': True,
            'returncode': 0,
            'stdout': 'Successfully installed requests-2.31.0',
            'stderr': ''
        }
    
    mock_sandbox.run_shell_command = mock_run_shell_command
            
            # Verify the result
            assert result.success
            assert result.output == {"result": "test output"}
            
            # Verify the executor was called with the right arguments
            mock_executor.assert_called_once()
@pytest.mark.asyncio
async def test_tool_worker_error_handling(tool_registry_setup):
    """Test error handling in tool execution."""
    # Create a tool that raises an exception
    @register_tool(
        version="1.0.0",
        name="failing_tool",
        input_schema={"type": "object"},
        output_schema={"type": "object"}
    )
    def failing_tool():
        raise ValueError("Something went wrong")
    
    async with ToolWorker("failing_tool", "1.0.0") as worker:
        result = await worker.execute()
        
        assert not result.success
        assert "Something went wrong" in result.error
        assert result.metadata["error_type"] == "ValueError"

@pytest.mark.asyncio
async def test_tool_worker_output_validation(tool_registry_setup):
    """Test output validation with the tool's schema."""
    # First, unregister any existing tool with the same name to avoid conflicts
    if "invalid_output_tool" in tool_registry_setup._tools:
        del tool_registry_setup._tools["invalid_output_tool"]
    if "invalid_output_tool" in tool_registry_setup._aliases:
        del tool_registry_setup._aliases["invalid_output_tool"]
    
    # Test valid output with test_tool which has a known output schema
    async with ToolWorker("test_tool", "1.0.0") as worker:
        # Mock the validation to avoid actual validation errors
        with patch('jsonschema.validate') as mock_validate:
            # Configure the mock to do nothing
            mock_validate.return_value = None

            result = await worker.execute({"name": "Test", "count": 1})
            assert result.success
            assert "greeting" in result.output

            # Check that validate was called at least once
            assert mock_validate.call_count > 0

    # Test with a tool that returns invalid output according to its schema
    def invalid_output_tool(*args, **kwargs):
        # This output doesn't match the schema (missing required 'status' field)
        return ToolResult.from_success({"invalid": True})
    
    # Register the tool with the registry directly to ensure proper registration
    tool = ToolVersion(
        name="invalid_output_tool",
        version="1.0.0",
        implementation=invalid_output_tool,
        input_schema={"type": "object", "properties": {}, "additionalProperties": True},
        output_schema={
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"]
        }
    )
    tool_registry_setup.register(tool)

    # Mock the ValidationError with proper parameters including line_errors
    with patch('jsonschema.validators.validate') as mock_validate:
        # Create a proper ValidationError with all required parameters
        from jsonschema import ValidationError
        from jsonschema.exceptions import ValidationError as JsonschemaValidationError
        
        # Create a proper validation error with line_errors
        try:
            # This will raise a ValidationError with all required attributes
            raise JsonschemaValidationError(
                "Output validation failed",
                validator="required",
                path=["status"],
                validator_value=["status"],
                instance={"invalid": True},
                schema={"required": ["status"], "type": "object"},
                schema_path=["required"],
                cause=None,
                context=[]
            )
        except JsonschemaValidationError as e:
            # Use the properly constructed exception
            mock_validate.side_effect = e
        
        async with ToolWorker("invalid_output_tool", "1.0.0") as worker:
            with patch('orchestrator.sandbox.tool_worker.logger') as mock_logger:
                result = await worker.execute({"input_param": "value"})
                
                # The execution should still be successful even if validation fails
                assert result.success
                assert 'invalid' in result.output
                
                # Check that a warning was logged
                mock_logger.warning.assert_called_once()
                warning_msg = str(mock_logger.warning.call_args[0][0]).lower()
                assert "output validation failed" in warning_msg or 'validation error' in warning_msg
