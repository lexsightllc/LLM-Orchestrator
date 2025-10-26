"""Integration tests for sandbox tool execution."""
import pytest

from orchestrator.sandbox import Sandbox
from orchestrator.tools import ToolResult, register_tool, tool_registry


@pytest.fixture
def isolated_registry():
    """Provide an isolated tool registry for the test run."""
    original_tools = tool_registry._tools.copy()
    original_aliases = tool_registry._aliases.copy()
    try:
        tool_registry._tools = {}
        tool_registry._aliases = {}
        yield tool_registry
    finally:
        tool_registry._tools = original_tools
        tool_registry._aliases = original_aliases


@pytest.mark.asyncio
async def test_execute_registered_tool(tmp_path, isolated_registry):
    """Tools registered in the global registry can be executed inside the sandbox."""

    @register_tool(version="1.0.0", name="adder")
    def adder(a: int, b: int) -> ToolResult:
        return ToolResult.from_success({"sum": a + b})

    sandbox = Sandbox(workdir=tmp_path / "sbx")
    result = await sandbox.execute_tool("adder", args={"a": 2, "b": 3})

    assert result.success
    assert result.data["sum"] == 5
    assert result.metadata["execution_time"] >= 0.0
