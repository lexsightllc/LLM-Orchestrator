# SPDX-License-Identifier: MPL-2.0
import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from orchestrator.tools import (
    ToolRegistry, ToolVersion, ToolError, ToolVersionError, 
    tool_registry, register_tool, ToolResult
)

@pytest.fixture
def registry():
    """Create a fresh registry for each test."""
    return ToolRegistry()

def test_register_and_retrieve_tool(registry):
    """Test registering and retrieving a tool."""
    # Create a mock tool
    def mock_impl():
        return "test"
    
    tool = ToolVersion(
        name="test_tool",
        version="1.0.0",
        input_schema={"type": "object"},
        output_schema={"type": "string"},
        implementation=mock_impl
    )
    
    # Register the tool
    registry.register(tool, aliases=["latest"])
    
    # Retrieve by exact version
    retrieved = registry.get_tool("test_tool", "1.0.0")
    assert retrieved.name == "test_tool"
    assert retrieved.version == "1.0.0"
    assert retrieved.implementation() == "test"
    
    # Retrieve by alias
    aliased = registry.get_tool("test_tool", "latest")
    assert aliased.version == "1.0.0"

def test_version_constraints(registry):
    """Test version constraint resolution."""
    # Register multiple versions
    for version in ["1.0.0", "1.1.0", "2.0.0", "2.1.0"]:
        tool = ToolVersion(
            name="versioned_tool",
            version=version,
            input_schema={},
            output_schema={},
            implementation=lambda: version
        )
        registry.register(tool)
    
    # Test version constraints
    assert registry.get_tool("versioned_tool", "^1.0.0").version == "1.1.0"  # Latest 1.x
    assert registry.get_tool("versioned_tool", "2.0.0").version == "2.0.0"    # Exact version
    assert registry.get_tool("versioned_tool", ">=1.0.0 <2.0.0").version == "1.1.0"  # Range

def test_missing_tool(registry):
    """Test error handling for missing tools."""
    with pytest.raises(ToolError):
        registry.get_tool("nonexistent", "1.0.0")
    
    # Register a tool but request non-existent version
    tool = ToolVersion(
        name="existing_tool",
        version="1.0.0",
        input_schema={},
        output_schema={},
        implementation=lambda: None
    )
    registry.register(tool)
    
    with pytest.raises(ToolVersionError):
        registry.get_tool("existing_tool", "2.0.0")

def test_tool_result():
    """Test ToolResult helper methods."""
    # Test success result
    success = ToolResult.from_success("data", extra="info")
    assert success.success is True
    assert success.data == "data"
    assert success.metadata["extra"] == "info"
    
    # Test error result
    error = ToolResult.from_error("Something went wrong", code=500)
    assert error.success is False
    assert "Something went wrong" in error.error
    assert error.metadata["code"] == 500

@patch('importlib.util.spec_from_file_location')
@patch('importlib.util.module_from_spec')
def test_load_from_directory(mock_module, mock_spec, registry, tmp_path):
    """Test loading tools from a directory."""
    # Create a temporary tool directory
    tool_dir = tmp_path / "test_tool"
    tool_dir.mkdir()
    
    # Create a tool manifest
    manifest = {
        "name": "test_tool",
        "version": "1.0.0",
        "aliases": ["latest"],
        "input_schema": {"type": "object"},
        "output_schema": {"type": "string"}
    }
    
    with open(tool_dir / "tool.json", 'w') as f:
        json.dump(manifest, f)
    
    # Create a simple implementation
    with open(tool_dir / "impl.py", 'w') as f:
        f.write("def execute():")
        f.write("    return 'test'")
    
    # Mock the module import
    mock_module.return_value.spec = MagicMock()
    mock_module.return_value.spec.loader = MagicMock()
    
    # Load the tool
    registry.load_from_directory(str(tmp_path))
    
    # Verify the tool was loaded
    tool = registry.get_tool("test_tool", "latest")
    assert tool.version == "1.0.0"

def test_register_decorator():
    """Test the @register_tool decorator."""
    # Reset the global registry for this test
    global tool_registry
    tool_registry = ToolRegistry()
    
    # Register a tool using the decorator
    @register_tool(
        version="1.0.0",
        name="decorated_tool",
        aliases=["latest"],
        input_schema={"type": "object"},
        output_schema={"type": "string"}
    )
    def my_tool():
        return "decorated"
    
    # Verify registration
    tool = tool_registry.get_tool("decorated_tool", "latest")
    assert tool.version == "1.0.0"
    assert tool.implementation() == "decorated"
    
    # Clean up
    tool_registry = ToolRegistry()  # Reset to default

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
