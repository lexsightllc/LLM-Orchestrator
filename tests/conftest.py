import pytest
from orchestrator.tools import ToolRegistry, ToolVersion, ToolResult

@pytest.fixture
def tool_registry_setup():
    """Fixture that provides a clean ToolRegistry instance for each test with test tools."""
    # Create a new registry
    registry = ToolRegistry()
    
    # 1. Test tool for basic sync execution
    def sync_test_tool(*args, **kwargs):
        return ToolResult.from_success({"result": "test output"})
    
    registry.register(ToolVersion(
        name="sync_test_tool",
        version="1.0.0",
        implementation=sync_test_tool,
        input_schema={"type": "object", "properties": {}, "additionalProperties": True},
        output_schema={"type": "object", "properties": {"result": {"type": "string"}}},
    ))
    
    # 2. Test tool with dependencies
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
    
    # 3. Test tool with invalid output
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
    
    return registry
