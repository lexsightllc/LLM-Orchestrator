"""Tests for the sandbox execution environment."""
import asyncio
import json
import os
import platform
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import psutil

from orchestrator.sandbox import (
    Sandbox,
    SandboxState,
    ResourceLimits,
    SandboxError,
    ResourceLimitExceeded,
    SandboxTimeoutError,
    SandboxResult,
    get_sandbox_manager
)

# Skip resource limit tests on Windows
SKIP_RESOURCE_TESTS = sys.platform == 'win32'

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def sandbox(temp_dir):
    """Create a sandbox for testing."""
    # Set up resource limits for testing
    limits = ResourceLimits(
        memory_mb=100,  # 100MB memory limit
        timeout_seconds=30,  # 30 second timeout
        max_output_bytes=1024 * 1024,  # 1MB output limit
        allow_network=False,
    )
    
    # Create and return the sandbox
    sandbox = Sandbox(limits=limits, workdir=temp_dir / "sandbox")
    return sandbox

@pytest.mark.asyncio
async def test_sandbox_initialization(sandbox):
    """Test sandbox initialization."""
    assert sandbox is not None
    assert sandbox.workdir.exists()
    assert sandbox.workdir.is_dir()
    assert sandbox._state == SandboxState.PENDING

@pytest.mark.asyncio
async def test_run_simple_command(sandbox):
    """Test running a simple command in the sandbox."""
    if sys.platform == 'win32':
        cmd = "echo Hello, World!"
    else:
        cmd = "echo -n 'Hello, World!'"
    
    result = await sandbox.run_shell_command(cmd)
    
    assert result.success
    assert result.state == SandboxState.COMPLETED
    assert result.return_code == 0
    assert "Hello, World!" in result.stdout

@pytest.mark.asyncio
async def test_run_python_code(sandbox):
    """Test running Python code in the sandbox."""
    code = """
import sys
print("Hello from Python", sys.version_info[0])
sys.stderr.write("Error output\n")
print("Arguments:", ' '.join(sys.argv[1:]))
    """
    
    result = await sandbox.run_python_code(code, args=["arg1", "arg2"])
    
    assert result.success
    assert f"Hello from Python {sys.version_info[0]}" in result.stdout
    assert "Error output" in result.stderr
    assert "Arguments: arg1 arg2" in result.stdout

@pytest.mark.asyncio
async def test_memory_limit(sandbox):
    """Test memory limit enforcement."""
    if SKIP_RESOURCE_TESTS:
        pytest.skip("Resource limit tests not supported on Windows")
    
    # Try to allocate more memory than allowed
    code = """
import numpy as np
# Try to allocate 200MB (twice the limit)
data = np.zeros((200, 1024, 1024), dtype=np.uint8)
print("Allocated memory:", data.nbytes / (1024 * 1024), "MB")
    """
    
    result = await sandbox.run_python_code(code)
    
    # Should fail due to memory limit
    assert not result.success
    assert result.state == SandboxState.FAILED

@pytest.mark.asyncio
async def test_timeout(sandbox):
    """Test timeout enforcement."""
    if SKIP_RESOURCE_TESTS:
        pytest.skip("Resource limit tests not supported on Windows")
    
    # Try to sleep longer than the timeout
    code = """
import time
time.sleep(60)  # Longer than the 30s timeout
print("This should not be reached")
    """
    
    with pytest.raises(SandboxTimeoutError):
        await sandbox.run_python_code(code, timeout=1)  # 1 second timeout for test

@pytest.mark.asyncio
async def test_output_limit(sandbox):
    """Test output size limit enforcement."""
    if SKIP_RESOURCE_TESTS:
        pytest.skip("Resource limit tests not supported on Windows")
    
    # Generate more output than allowed
    large_output = "x" * (2 * 1024 * 1024)  # 2MB, more than the 1MB limit
    code = f"print('{large_output}')"
    
    result = await sandbox.run_python_code(code)
    
    # Should fail due to output limit
    assert not result.success
    assert result.state == SandboxState.FAILED
    assert len(result.stdout) <= sandbox.limits.max_output_bytes * 2  # Some buffer

@pytest.mark.asyncio
async def test_network_access_denied(sandbox):
    """Test that network access is denied by default."""
    if sys.platform == 'win32':
        pytest.skip("Network access tests not supported on Windows")
    
    # Try to make a network request
    code = """
import urllib.request
import json

try:
    response = urllib.request.urlopen('http://example.com')
    print(response.read().decode('utf-8'))
except Exception as e:
    print(f"Error: {str(e)}")
    raise
    """
    
    result = await sandbox.run_python_code(code)
    
    # The request should fail due to network access being denied
    assert not result.success
    assert "Error" in result.stdout or "Error" in result.stderr

@pytest.mark.asyncio
async def test_environment_variables(sandbox):
    """Test that environment variables are passed correctly."""
    env_vars = {
        "TEST_VAR": "test_value",
        "ANOTHER_VAR": "123"
    }
    
    if sys.platform == 'win32':
        cmd = "echo %TEST_VAR% %ANOTHER_VAR%"
    else:
        cmd = "echo $TEST_VAR $ANOTHER_VAR"
    
    result = await sandbox.run_shell_command(cmd, env=env_vars)
    
    assert result.success
    assert "test_value 123" in result.stdout.strip()

@pytest.mark.asyncio
async def test_working_directory(sandbox, temp_dir):
    """Test that the working directory is set correctly."""
    # Create a test file in the working directory
    test_file = sandbox.workdir / "test.txt"
    test_file.write_text("test content")
    
    # Create a subdirectory
    subdir = sandbox.workdir / "subdir"
    subdir.mkdir()
    
    # Run a command that lists the working directory
    if sys.platform == 'win32':
        cmd = f"dir /b"
    else:
        cmd = f"ls -la"
    
    result = await sandbox.run_shell_command(cmd)
    
    assert result.success
    
    # Check that the working directory is correct
    if sys.platform != 'win32':
        assert str(subdir) in result.stdout
    
    # Check that the test file is listed
    assert "test.txt" in result.stdout

@pytest.mark.asyncio
async def test_sandbox_manager():
    """Test the sandbox manager."""
    manager = get_sandbox_manager()
    
    # Get a sandbox
    sandbox1 = await manager.get_sandbox()
    assert sandbox1 is not None
    
    # Get another sandbox
    sandbox2 = await manager.get_sandbox()
    assert sandbox2 is not None
    assert sandbox1 is not sandbox2
    
    # Clean up
    await manager.cleanup()

@pytest.mark.asyncio
async def test_sandbox_cleanup(sandbox):
    """Test that sandbox cleanup works."""
    # Create a file in the sandbox
    test_file = sandbox.workdir / "test.txt"
    test_file.write_text("test")
    
    # Run a command
    result = await sandbox.run_shell_command("echo Hello")
    assert result.success
    
    # Clean up
    await sandbox.cleanup_resources()
    
    # The sandbox directory should still exist (we didn't set cleanup=True)
    assert sandbox.workdir.exists()
    
    # But the process should be terminated
    if hasattr(sandbox, '_process') and sandbox._process:
        assert sandbox._process.returncode is not None

@pytest.mark.asyncio
async def test_sandbox_with_artifacts(sandbox, temp_dir):
    """Test sandbox with artifact storage."""
    from orchestrator.artifacts import ArtifactStorage
    
    # Set up artifact storage
    storage = ArtifactStorage(temp_dir / "artifacts")
    sandbox.storage = storage
    
    # Run a command that creates an artifact
    code = """
import json
import os

# Create some data to store as an artifact
data = {"key": "value", "number": 42}

# Save as an artifact
artifact_path = os.path.join(os.getcwd(), "output.json")
with open(artifact_path, 'w') as f:
    json.dump(data, f)

# Register the artifact
if hasattr(sandbox, 'storage'):
    sandbox.storage.store_artifact("test_artifact", artifact_path, "json")

print(f"Artifact created at: {artifact_path}")
    """
    
    result = await sandbox.run_python_code(code)
    assert result.success
    
    # The output file should exist in the sandbox
    output_file = sandbox.workdir / "output.json"
    assert output_file.exists()
    
    # The artifact should be in the storage
    artifacts = list((temp_dir / "artifacts" / "refs").glob("*.json"))
    assert len(artifacts) > 0

@pytest.mark.asyncio
async def test_sandbox_concurrent_execution():
    """Test that multiple sandboxes can run concurrently."""
    async def run_in_sandbox(i):
        async with Sandbox() as sandbox:
            result = await sandbox.run_shell_command(f"echo {i} && sleep 0.1")
            return result.stdout.strip()
    
    # Run multiple sandboxes concurrently
    tasks = [run_in_sandbox(i) for i in range(5)]
    results = await asyncio.gather(*tasks)
    
    # Check that all sandboxes completed successfully
    assert len(results) == 5
    assert set(results) == {"0", "1", "2", "3", "4"}
