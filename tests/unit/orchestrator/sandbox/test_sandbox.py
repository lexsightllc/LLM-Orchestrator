# SPDX-License-Identifier: MPL-2.0
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
    assert result.stdout.strip() == "Hello, World!"
    assert result.stderr == ""
    assert result.execution_time > 0

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
    assert result.state == SandboxState.COMPLETED
    assert result.return_code == 0
    assert "Hello from Python" in result.stdout
    assert "Arguments: arg1 arg2" in result.stdout
    assert "Error output" in result.stderr

@pytest.mark.asyncio
async def test_memory_limit(sandbox):
    """Test memory limit enforcement."""
    if SKIP_RESOURCE_TESTS:
        pytest.skip("Memory limit tests not supported on this platform")
    
    # This Python code will try to allocate a large amount of memory
    code = """
import sys
if sys.platform == 'win32':
    # Windows doesn't have resource module
    print("Skipping memory test on Windows")
    sys.exit(0)

try:
    # Try to allocate 200MB of memory (more than our 100MB limit)
    data = 'x' * (200 * 1024 * 1024)
    print(f"Allocated {len(data) / (1024 * 1024):.1f}MB")
except MemoryError:
    print("MemoryError caught as expected")
    sys.exit(0)
else:
    print("MemoryError not raised!")
    sys.exit(1)
    """
    
    result = await sandbox.run_python_code(code)
    
    # The process should be killed due to memory limit
    assert not result.success
    assert result.state in (SandboxState.FAILED, SandboxState.KILLED)

@pytest.mark.asyncio
async def test_timeout(sandbox):
    """Test timeout enforcement."""
    # Set a short timeout
    sandbox.limits.timeout_seconds = 1.0
    
    # This command will sleep for 5 seconds (longer than our 1s timeout)
    if sys.platform == 'win32':
        cmd = "timeout 5 /nobreak"
    else:
        cmd = "sleep 5"
    
    start_time = time.monotonic()
    result = await sandbox.run_shell_command(cmd)
    duration = time.monotonic() - start_time
    
    # Should be killed after ~1 second (with some buffer)
    assert 0.9 <= duration <= 3.0
    assert not result.success
    assert result.state == SandboxState.TIMED_OUT

@pytest.mark.asyncio
async def test_output_limit(sandbox):
    """Test output size limit enforcement."""
    # Set a small output limit
    sandbox.limits.max_output_bytes = 100  # 100 bytes
    
    # Generate output that exceeds the limit
    if sys.platform == 'win32':
        cmd = "for /L %i in (1,1,1000) do @echo 1234567890"
    else:
        cmd = "for i in {1..1000}; do echo 1234567890; done"
    
    result = await sandbox.run_shell_command(cmd)
    
    # The process should be killed due to output limit
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
import sys
try:
    import urllib.request
    response = urllib.request.urlopen('http://example.com')
    print(f"Status: {response.status}")
    sys.exit(0)
except Exception as e:
    print(f"Error: {e}")
    # Exit with non-zero to indicate error
    sys.exit(1)
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
    
    # Run a command that checks the working directory and lists files
    if sys.platform == 'win32':
        cmd = "echo %CD% && dir /b"
    else:
        cmd = "pwd && ls -1"
    
    result = await sandbox.run_shell_command(cmd, cwd=subdir)
    
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

# Create a test file
with open("output.json", "w") as f:
    json.dump({"result": 42}, f)

print("DONE")
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
