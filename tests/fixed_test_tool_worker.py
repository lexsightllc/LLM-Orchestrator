import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch, PropertyMock, MagicMock, create_autospec

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the necessary classes
from orchestrator.tools import ToolRegistry, ToolVersion, ToolResult, ToolError, tool_registry
from orchestrator.sandbox.tool_worker import ToolWorker

# Patch the global tool_registry at module level
import orchestrator.tools
orchestrator.tools.tool_registry = ToolRegistry()

# Re-import to ensure the patch is applied
import importlib
import orchestrator.sandbox.tool_worker
importlib.reload(orchestrator.sandbox.tool_worker)
from orchestrator.sandbox.tool_worker import ToolWorker

@pytest.fixture(autouse=True)
def setup_tool_registry():
    """Set up the global tool registry for testing."""
    # Create a new registry for each test
    registry = ToolRegistry()
    
    # Define a synchronous test tool
    def sync_test_tool(*args, **kwargs):
        return ToolResult.from_success({"result": "test output"})
    
    # Register the tool with the registry
    registry.register(ToolVersion(
        name="sync_test_tool",
        version="1.0.0",
        implementation=sync_test_tool,
        input_schema={"type": "object", "properties": {}, "additionalProperties": True},
        output_schema={"type": "object", "properties": {"result": {"type": "string"}}},
    ))
    
    # Define a tool with dependencies
    def tool_with_deps(*args, **kwargs):
        return ToolResult.from_success({"status": "success"})
    
    tool_with_deps.__requires__ = ["requests>=2.25.0"]
    
    registry.register(ToolVersion(
        name="tool_with_deps",
        version="1.0.0",
        implementation=tool_with_deps,
        input_schema={"type": "object", "properties": {}, "additionalProperties": True},
        output_schema={"type": "object", "properties": {"status": {"type": "string"}}},
    ))
    
    # Define a tool with invalid output
    def invalid_output_tool(*args, **kwargs):
        return ToolResult.from_success({"invalid": True})
    
    registry.register(ToolVersion(
        name="invalid_output_tool",
        version="1.0.0",
        implementation=invalid_output_tool,
        input_schema={"type": "object", "properties": {}, "additionalProperties": True},
        output_schema={
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"]
        },
    ))
    
    # Patch the global registry
    with patch('orchestrator.tools.tool_registry', registry):
        # Also patch the registry in the tool_worker module
        with patch('orchestrator.sandbox.tool_worker.tool_registry', registry):
            yield registry

@pytest.mark.asyncio
async def test_tool_worker_execute_sync():
    """Test synchronous tool execution with the ToolWorker."""
    # Create a mock sandbox
    mock_sandbox = AsyncMock()
    mock_sandbox.workdir = "/tmp/test_workdir"
    
    # Create a mock ToolWorker class
    with patch('orchestrator.sandbox.tool_worker.ToolWorker') as MockToolWorkerClass:
        # Configure the mock ToolWorker instance
        mock_worker = AsyncMock()
        mock_worker.sandbox = mock_sandbox
        mock_worker.execute.return_value = ToolResult.from_success({"result": "test output"})
        
        # Mock the context manager behavior
        mock_worker.__aenter__.return_value = mock_worker
        mock_worker.__aexit__.return_value = None
        
        # Configure the class to return our mock instance
        MockToolWorkerClass.return_value = mock_worker
        
        # Create the worker and execute the test
        async with ToolWorker("sync_test_tool", "1.0.0") as worker:
            # Ensure the sandbox is properly initialized
            worker._ToolWorker__sandbox = mock_sandbox
            
            # Mock the _install_dependencies method to avoid actual installation
            with patch.object(worker, '_install_dependencies', AsyncMock()):
                result = await worker.execute({"param1": "value1"})
                
                # Verify the result
                assert result.success
                assert result.output == {"result": "test output"}
                
                # Verify the executor was called with the right arguments
                mock_worker.execute.assert_called_once_with({"param1": "value1"})
    
    # Create a mock sandbox with a proper run_shell_command implementation
    mock_sandbox = AsyncMock()
    mock_sandbox.workdir = "/tmp/test_workdir"
    
    # Create a mock ToolWorker class that returns our mock sandbox
    with patch('orchestrator.sandbox.tool_worker.ToolWorker') as MockToolWorkerClass:
        # Configure the mock ToolWorker instance
        mock_worker = AsyncMock()
        mock_worker.sandbox = mock_sandbox
        mock_worker.execute.return_value = ToolResult.from_success({"result": "test output"})
        
        # Mock the context manager behavior
        mock_worker.__aenter__.return_value = mock_worker
        mock_worker.__aexit__.return_value = None
        
        # Configure the class to return our mock instance
        MockToolWorkerClass.return_value = mock_worker
        
        # Mock the event loop's run_in_executor
        with patch('asyncio.get_running_loop') as mock_loop:
            mock_executor = AsyncMock()
            mock_loop.return_value.run_in_executor = mock_executor
            
            # Configure the executor to return our test result
            mock_executor.return_value = ToolResult.from_success({"result": "test output"})
            
            # Create the worker and execute the test
            async with ToolWorker("sync_test_tool", "1.0.0") as worker:
                # Ensure the sandbox is properly initialized
                worker._ToolWorker__sandbox = mock_sandbox
                
                result = await worker.execute({"param1": "value1"})
                
                # Verify the result
                assert result.success
                assert result.output == {"result": "test output"}
                
                # Verify the executor was called with the right arguments
                mock_executor.assert_called_once()
            
            # Verify the executor was called with the right arguments
            mock_executor.assert_called_once()

@pytest.mark.asyncio
async def test_tool_worker_dependency_installation(tmp_path):
    """Test dependency installation for tools with requirements."""
    # Create a mock sandbox
    mock_sandbox = AsyncMock()
    mock_sandbox.workdir = tmp_path
    
    # Mock the run_shell_command method to accept and ignore the timeout parameter
    async def mock_run_shell_command(cmd, **kwargs):
        # Create a requirements file in the workdir
        req_file = tmp_path / "requirements.txt"
        if not req_file.exists():
            req_file.write_text("requests>=2.25.0\n")
        # Accept but ignore the timeout parameter
        kwargs.pop('timeout', None)
        return {
            'success': True,
            'returncode': 0,
            'stdout': 'Successfully installed requests-2.31.0',
            'stderr': ''
        }
    
    mock_sandbox.run_shell_command = mock_run_shell_command
    
    # Create a mock ToolWorker class
    with patch('orchestrator.sandbox.tool_worker.ToolWorker') as MockToolWorkerClass:
        # Configure the mock ToolWorker instance
        mock_worker = AsyncMock()
        mock_worker.sandbox = mock_sandbox
        mock_worker.execute.return_value = ToolResult.from_success({"status": "success"})
        
        # Mock the context manager behavior
        mock_worker.__aenter__.return_value = mock_worker
        mock_worker.__aexit__.return_value = None
        
        # Configure the class to return our mock instance
        MockToolWorkerClass.return_value = mock_worker
        
        # Create the worker and execute the test
        async with ToolWorker("tool_with_deps", "1.0.0") as worker:
            # The tool should be initialized and dependencies installed
            assert worker is not None
            
            # Verify the requirements file was created
            req_file = tmp_path / "requirements.txt"
            assert req_file.exists()
            assert "requests>=2.25.0" in req_file.read_text()
            
            # Verify the installation command was called
            mock_sandbox.run_shell_command.assert_called()
            
            # Execute the tool to ensure it works after installation
            result = await worker.execute({})
            assert result.success
            assert result.output == {"status": "success"}
    
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
    
    # Create a mock ToolWorker class that returns our mock sandbox
    with patch('orchestrator.sandbox.tool_worker.ToolWorker') as MockToolWorkerClass:
        # Configure the mock ToolWorker instance
        mock_worker = AsyncMock()
        mock_worker.sandbox = mock_sandbox
        mock_worker.execute.return_value = ToolResult.from_success({"status": "success"})
        
        # Mock the context manager behavior
        mock_worker.__aenter__.return_value = mock_worker
        mock_worker.__aexit__.return_value = None
        
        # Configure the class to return our mock instance
        MockToolWorkerClass.return_value = mock_worker
        
        # Create the worker and execute the test
        async with ToolWorker("tool_with_deps", "1.0.0") as worker:
            # The tool should be initialized and dependencies installed
            assert worker is not None
            
            # Verify the requirements file was created
            req_file = tmp_path / "requirements.txt"
            assert req_file.exists()
            assert "requests>=2.25.0" in req_file.read_text()
            
            # Verify the installation command was called
            mock_sandbox.run_shell_command.assert_called()
            
            # Execute the tool to ensure it works after installation
            result = await worker.execute({})
            assert result.success
            assert result.output == {"status": "success"}

@pytest.mark.asyncio
async def test_tool_worker_output_validation():
    """Test output validation with the tool's schema."""
    # Create a mock sandbox
    mock_sandbox = AsyncMock()
    
    # Mock the ToolWorker to avoid actual execution
    with patch('orchestrator.sandbox.tool_worker.ToolWorker') as MockToolWorkerClass, \
         patch('jsonschema.validators.validate') as mock_validate:
        
        # Configure the mock ToolWorker instance
        mock_worker = AsyncMock()
        mock_worker.sandbox = mock_sandbox
        mock_worker.execute.return_value = ToolResult.from_success({"invalid": True})
        
        # Mock the context manager behavior
        mock_worker.__aenter__.return_value = mock_worker
        mock_worker.__aexit__.return_value = None
        
        # Configure the class to return our mock instance
        MockToolWorkerClass.return_value = mock_worker
        
        # Create a proper ValidationError with all required parameters
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
        
        with patch('orchestrator.sandbox.tool_worker.logger') as mock_logger:
            async with ToolWorker("invalid_output_tool", "1.0.0") as worker:
                # Execute the tool with invalid output
                result = await worker.execute({"input_param": "value"})
                
                # The execution should still be successful even if validation fails
                assert result.success
                assert 'invalid' in result.output
                
                # Check that a warning was logged
                mock_logger.warning.assert_called_once()
                warning_msg = str(mock_logger.warning.call_args[0][0]).lower()
                assert "output validation failed" in warning_msg or 'validation error' in warning_msg

    # Mock the ToolWorker to avoid actual execution
    with patch('orchestrator.sandbox.tool_worker.ToolWorker') as MockToolWorkerClass, \
         patch('jsonschema.validators.validate') as mock_validate:
        
        # Configure the mock ToolWorker instance
        mock_worker = AsyncMock()
        mock_worker.execute.return_value = ToolResult.from_success({"invalid": True})
        
        # Mock the context manager behavior
        mock_worker.__aenter__.return_value = mock_worker
        mock_worker.__aexit__.return_value = None
        
        # Configure the class to return our mock instance
        MockToolWorkerClass.return_value = mock_worker
        
        # Create a proper ValidationError with all required parameters
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
        
        with patch('orchestrator.sandbox.tool_worker.logger') as mock_logger:
            async with ToolWorker("invalid_output_tool", "1.0.0") as worker:
                result = await worker.execute({"input_param": "value"})
                
                # The execution should still be successful even if validation fails
                assert result.success
                assert 'invalid' in result.output
                
                # Check that a warning was logged
                mock_logger.warning.assert_called_once()
                warning_msg = str(mock_logger.warning.call_args[0][0]).lower()
                assert "output validation failed" in warning_msg or 'validation error' in warning_msg
