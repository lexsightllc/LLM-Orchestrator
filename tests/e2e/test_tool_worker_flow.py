"""End-to-end scenario covering sandbox tool execution."""
import pytest

from orchestrator.sandbox import Sandbox
from orchestrator.sandbox.tool_worker import ToolWorker
from orchestrator.tools import ToolResult, register_tool, tool_registry


@pytest.fixture
def isolated_registry():
    """Isolate the tool registry to keep tests deterministic."""
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
async def test_tool_worker_end_to_end(tmp_path, isolated_registry):
    """ToolWorker executes registered tools inside a managed sandbox."""

    # Given a deterministic tool registered in the global registry
    @register_tool(version="1.0.0", name="doubler")
    def doubler(value: int) -> ToolResult:
        return ToolResult.from_success({"value": value * 2})

    sandbox = Sandbox(workdir=tmp_path / "e2e-sandbox")

    # When the ToolWorker runs the tool with structured arguments
    async with ToolWorker(tool_name="doubler", sandbox=sandbox) as worker:
        result = await worker.execute(value=21)

    # Then the workflow succeeds and returns the expected payload
    assert result.success
    assert result.output["value"] == 42
    assert "execution_time" in result.metadata
