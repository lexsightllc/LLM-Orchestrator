""Tests for the sandbox workers."""
import asyncio
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import cloudpickle
import pytest

from orchestrator.sandbox.workers import (
    WorkerType,
    WorkerConfig,
    WorkerResult,
    CPUWorker,
    IOWorker,
    ShellWorker,
    PythonWorker,
    create_worker,
)

# Skip resource-intensive tests on Windows
SKIP_RESOURCE_TESTS = sys.platform == 'win32'

@pytest.fixture
def temp_dir():
    ""Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def sample_worker_config():
    ""Create a sample worker config for testing."""
    return WorkerConfig(
        worker_type=WorkerType.CPU,
        limits={
            "cpu_percent": 100.0,
            "memory_mb": 512,
            "timeout_seconds": 30,
            "max_output_bytes": 1024 * 1024,
        },
        env_vars={"TEST_ENV": "test_value"},
    )

@pytest.mark.asyncio
async def test_cpu_worker_simple_function(temp_dir):
    ""Test CPU worker with a simple function."""
    # Define a simple function to execute
    def add(a, b):
        return a + b
    
    # Create and initialize the worker
    async with CPUWorker() as worker:
        # Execute the function
        result = await worker.execute(add, 2, 3)
        
        # Check the result
        assert result.success
        assert result.output == 5
        assert result.error is None
        assert "sandbox_result" in result.metadata

@pytest.mark.asyncio
async def test_cpu_worker_complex_function(temp_dir):
    ""Test CPU worker with a more complex function."""
    # Define a function that uses external libraries
    def process_data(data):
        import numpy as np
        arr = np.array(data)
        return {
            "sum": float(np.sum(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }
    
    # Create and initialize the worker
    async with CPUWorker() as worker:
        # Install numpy in the sandbox
        await worker._sandbox.run_shell_command(
            f"{sys.executable} -m pip install numpy"
        )
        
        # Execute the function
        test_data = [1, 2, 3, 4, 5]
        result = await worker.execute(process_data, test_data)
        
        # Check the result
        assert result.success
        assert result.output["sum"] == 15.0
        assert result.output["mean"] == 3.0
        assert result.output["std"] == pytest.approx(1.4142, abs=1e-4)

@pytest.mark.asyncio
async def test_cpu_worker_error_handling(temp_dir):
    ""Test CPU worker error handling."""
    # Define a function that raises an exception
    def raise_error():
        raise ValueError("Test error")
    
    # Create and initialize the worker
    async with CPUWorker() as worker:
        # Execute the function
        result = await worker.execute(raise_error)
        
        # Check the result
        assert not result.success
        assert "Test error" in result.error
        assert "sandbox_result" in result.metadata

@pytest.mark.asyncio
async def test_io_worker_file_operations(temp_dir):
    ""Test IO worker file operations."""
    # Create and initialize the worker
    async with IOWorker() as worker:
        # Test writing a file
        write_result = await worker.execute(
            "write",
            "test.txt",
            "Hello, World!"
        )
        assert write_result.success
        
        # Test reading the file
        read_result = await worker.execute("read", "test.txt")
        assert read_result.success
        assert read_result.output == "Hello, World!"
        
        # Test copying the file
        copy_result = await worker.execute("copy", "test.txt", "test_copy.txt")
        assert copy_result.success
        
        # Test reading the copied file
        read_copy_result = await worker.execute("read", "test_copy.txt")
        assert read_copy_result.success
        assert read_copy_result.output == "Hello, World!"
        
        # Test deleting the file
        delete_result = await worker.execute("delete", "test.txt")
        assert delete_result.success
        
        # Verify the file was deleted
        read_deleted_result = await worker.execute("read", "test.txt")
        assert not read_deleted_result.success
        assert "No such file or directory" in read_deleted_result.error

@pytest.mark.asyncio
async def test_shell_worker_basic_commands(temp_dir):
    ""Test ShellWorker with basic commands."""
    async with ShellWorker() as worker:
        # Test echo command
        result = await worker.execute("echo Hello, World!")
        assert result.success
        assert result.output.strip() == "Hello, World!"
        
        # Test command with arguments
        result = await worker.execute("echo", ["Hello", "World"])
        assert result.success
        assert result.output.strip() == "Hello World"
        
        # Test command with environment variables
        result = await worker.execute(
            "echo $TEST_VAR",
            env={"TEST_VAR": "test_value"}
        )
        assert result.success
        assert result.output.strip() == "test_value"

@pytest.mark.asyncio
async def test_shell_worker_working_directory(temp_dir):
    ""Test ShellWorker with custom working directory."""
    async with ShellWorker() as worker:
        # Create a subdirectory
        subdir = worker._sandbox.workdir / "subdir"
        subdir.mkdir()
        
        # Test command with custom working directory
        result = await worker.execute(
            "pwd",
            cwd="subdir"
        )
        assert result.success
        if sys.platform != 'win32':
            assert str(subdir) in result.output.strip()

@pytest.mark.asyncio
async def test_python_worker_basic_code(temp_dir):
    ""Test PythonWorker with basic code execution."""
    async with PythonWorker() as worker:
        # Test simple expression
        result = await worker.execute("print(2 + 2)")
        assert result.success
        assert result.output.strip() == "4"
        
        # Test multi-line code
        code = """
import sys
print("Python version:", sys.version_info[0])
print("Arguments:", sys.argv[1:])
        """
        result = await worker.execute(code, args=["arg1", "arg2"])
        assert result.success
        assert "Python version:" in result.output
        assert "Arguments: ['arg1', 'arg2']" in result.output

@pytest.mark.asyncio
async def test_python_worker_external_packages(temp_dir):
    ""Test PythonWorker with external packages."""
    async with PythonWorker() as worker:
        # Install a package in the sandbox
        await worker._sandbox.run_shell_command(
            f"{sys.executable} -m pip install numpy"
        )
        
        # Test using the package
        code = """
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f"Sum: {np.sum(arr)}")
print(f"Mean: {np.mean(arr)}")
        """
        
        result = await worker.execute(code)
        assert result.success
        assert "Sum: 15" in result.output
        assert "Mean: 3.0" in result.output

@pytest.mark.asyncio
async def test_worker_factory(temp_dir):
    ""Test the worker factory function."""
    # Test creating a CPU worker
    cpu_worker = create_worker("cpu")
    assert isinstance(cpu_worker, CPUWorker)
    
    # Test creating an IO worker
    io_worker = create_worker("io")
    assert isinstance(io_worker, IOWorker)
    
    # Test creating a shell worker
    shell_worker = create_worker("shell")
    assert isinstance(shell_worker, ShellWorker)
    
    # Test creating a Python worker
    python_worker = create_worker("python")
    assert isinstance(python_worker, PythonWorker)
    
    # Test with custom config
    config = WorkerConfig(
        worker_type=WorkerType.CPU,
        limits={"memory_mb": 1024},
    )
    custom_worker = create_worker("cpu", config=config)
    assert custom_worker.config.limits.memory_mb == 1024

@pytest.mark.asyncio
async def test_worker_result_serialization():
    ""Test serialization and deserialization of WorkerResult."""
    # Create a sample result
    result = WorkerResult(
        success=True,
        output={"key": "value"},
        error=None,
        metadata={"execution_time": 1.23},
    )
    
    # Convert to dict and back
    result_dict = result.dict()
    new_result = WorkerResult(**result_dict)
    
    # Check that the data was preserved
    assert new_result.success == result.success
    assert new_result.output == result.output
    assert new_result.error == result.error
    assert new_result.metadata == result.metadata

@pytest.mark.skipif(SKIP_RESOURCE_TESTS, reason="Resource tests not supported on this platform")
@pytest.mark.asyncio
async def test_cpu_worker_resource_limits():
    ""Test CPU worker with resource limits."""
    # Define a CPU-intensive function
    def cpu_intensive():
        total = 0
        for i in range(10**7):
            total += i
        return total
    
    # Create a worker with strict limits
    config = WorkerConfig(
        worker_type=WorkerType.CPU,
        limits={
            "cpu_percent": 50.0,
            "memory_mb": 100,  # Very low memory limit
            "timeout_seconds": 5,
        },
    )
    
    async with CPUWorker(config=config) as worker:
        # This should fail due to memory limits
        result = await worker.execute(cpu_intensive)
        
        # The function should be killed due to memory limits
        assert not result.success
        assert "MemoryError" in result.error or "Killed" in result.error
