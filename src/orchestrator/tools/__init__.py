"""Tool versioning and registry for LLM Orchestrator."""

from typing import Dict, List, Optional, Protocol, TypeVar, Any
import functools
import inspect
from dataclasses import dataclass, field
from pathlib import Path
import importlib.util
import json
import logging
from pydantic import BaseModel, Field

# The original implementation depended on the external `semver` package for
# version parsing and constraint matching. To keep the test environment
# lightweight and avoid an additional dependency, we provide a minimal
# compatible implementation in ``orchestrator._compat.semver``.
from orchestrator._compat.semver import VersionInfo
import hashlib
import os

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ToolError(Exception):
    """Base exception for tool-related errors."""
    pass

class ToolVersionError(ToolError):
    """Raised when there's an issue with tool versions."""
    pass

class ToolMigrationError(ToolError):
    """Raised when a tool migration fails."""
    pass

@dataclass
class ToolVersion:
    """Represents a specific version of a tool."""
    name: str
    version: str
    input_schema: dict
    output_schema: dict
    implementation: callable
    migrations: Dict[str, str] = field(default_factory=dict)
    
    def get_semver(self) -> VersionInfo:
        """Get the semantic version object."""
        return VersionInfo.parse(self.version)
    
    def is_compatible(self, version_constraint: str) -> bool:
        """Check if this version satisfies the given constraint."""
        return self.get_semver().match(version_constraint)

class ToolRegistry:
    """Manages tool versions and provides version resolution."""
    
    def __init__(self):
        self._tools: Dict[str, Dict[str, ToolVersion]] = {}
        self._aliases: Dict[str, Dict[str, str]] = {}
    
    def register(self, tool: ToolVersion, aliases: Optional[List[str]] = None) -> None:
        """Register a new tool version."""
        if tool.name not in self._tools:
            self._tools[tool.name] = {}
            self._aliases[tool.name] = {}
        
        self._tools[tool.name][tool.version] = tool
        
        # Register aliases if provided
        if aliases:
            for alias in aliases:
                self._aliases[tool.name][alias] = tool.version
    
    def get_tool(self, name: str, version_spec: str = "latest") -> ToolVersion:
        """Get a tool by name and version specification."""
        if name not in self._tools:
            raise ToolError(f"No tool named '{name}' found")
        
        # Check for alias first
        if version_spec in self._aliases.get(name, {}):
            version_spec = self._aliases[name][version_spec]
        
        # Try to find exact match first
        if version_spec in self._tools[name]:
            return self._tools[name][version_spec]
        
        # Try to find a compatible version
        try:
            # If it's a version constraint (e.g., ^1.0.0, >=2.0.0, etc.)
            for version, tool in sorted(
                self._tools[name].items(),
                key=lambda x: VersionInfo.parse(x[0]),
                reverse=True
            ):
                if tool.is_compatible(version_spec):
                    return tool
        except ValueError:
            # Not a valid version constraint, try as a version string
            pass
        
        raise ToolVersionError(
            f"No version of '{name}' matching '{version_spec}' found. "
            f"Available versions: {', '.join(self._tools[name].keys())}"
        )
    
    def load_from_directory(self, directory: str) -> None:
        """Load tool definitions from a directory."""
        tool_dir = Path(directory)
        if not tool_dir.exists() or not tool_dir.is_dir():
            raise ToolError(f"Tool directory not found: {directory}")
        
        for tool_name in os.listdir(tool_dir):
            tool_path = tool_dir / tool_name
            if tool_path.is_dir() and (tool_path / "tool.json").exists():
                self._load_tool_definition(tool_path)
    
    def _load_tool_definition(self, tool_dir: Path) -> None:
        """Load a single tool definition from a directory."""
        try:
            # Load the tool manifest
            with open(tool_dir / "tool.json") as f:
                manifest = json.load(f)
            
            # Load the implementation
            impl_path = tool_dir / "impl.py"
            if not impl_path.exists():
                raise ToolError(f"Implementation not found for {tool_dir.name}")
            
            # Import the module
            spec = importlib.util.spec_from_file_location(
                f"orchestrator.tools.{tool_dir.name}",
                impl_path
            )
            if spec is None or spec.loader is None:
                raise ToolError(f"Failed to load module: {impl_path}")
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the tool implementation function
            if not hasattr(module, "execute"):
                raise ToolError(f"Tool {tool_dir.name} is missing 'execute' function")
            
            # Create and register the tool version
            tool = ToolVersion(
                name=manifest["name"],
                version=manifest["version"],
                input_schema=manifest.get("input_schema", {}),
                output_schema=manifest.get("output_schema", {}),
                implementation=module.execute,
                migrations=manifest.get("migrations", {})
            )
            
            self.register(
                tool,
                aliases=manifest.get("aliases", [])
            )
            
            logger.info(f"Loaded tool: {tool.name}@{tool.version}")
            
        except Exception as e:
            logger.error(f"Failed to load tool from {tool_dir}: {str(e)}")
            raise ToolError(f"Failed to load tool: {str(e)}") from e

class ToolResult(BaseModel):
    """Structured result from a tool execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: dict = Field(default_factory=dict)
    
    @classmethod
    def from_success(cls, data: Any, **metadata) -> 'ToolResult':
        """Create a successful result."""
        return cls(success=True, data=data, metadata=metadata)
    
    @classmethod
    def from_error(cls, error: str, **metadata) -> 'ToolResult':
        """Create an error result."""
        return cls(success=False, error=error, metadata=metadata)

# Global registry instance
tool_registry = ToolRegistry()

from .normalization import patch_registry  # noqa: E402

try:
    patch_registry(tool_registry)
except Exception as _e:  # pragma: no cover - defensive
    pass

def register_tool(version: str, **kwargs):
    """Decorator to register a tool function."""
    def decorator(func):
        # Extract name from function if not provided
        name = kwargs.get('name', func.__name__)
        
        implementation = _adapt_tool_callable(func)

        # Create and register the tool
        tool = ToolVersion(
            name=name,
            version=version,
            input_schema=kwargs.get('input_schema', {}),
            output_schema=kwargs.get('output_schema', {}),
            implementation=implementation,
            migrations=kwargs.get('migrations', {})
        )
        
        aliases = list(kwargs.get('aliases', []))
        if "latest" not in aliases:
            aliases.append("latest")
        tool_registry.register(tool, aliases=aliases)
        return func
    return decorator


def _adapt_tool_callable(func):
    """Adapt tool callables so they accept a payload dictionary."""

    signature = inspect.signature(func)
    params = list(signature.parameters.values())

    expects_payload = False
    if len(params) == 1 and params[0].kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        param = params[0]
        annotation = param.annotation
        expects_payload = (
            param.name == "payload"
            or annotation in {dict, Dict[str, Any]}
        )

    if expects_payload:
        return func

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(payload: Dict[str, Any]):
            return await func(**payload)

        return async_wrapper

    @functools.wraps(func)
    def wrapper(payload: Dict[str, Any]):
        return func(**payload)

    return wrapper
