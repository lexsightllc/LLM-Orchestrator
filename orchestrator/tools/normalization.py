"""
Recursion-safe tool normalization via Proxy.

This module wraps tools at retrieval time so every call to `.implementation(payload)`
returns a proper ToolResult; raw scalars are boxed into a dict for subscript safety.
No monkey-patching of ToolVersion or individual tools is required.
"""

from __future__ import annotations
from typing import Any, Optional, Callable
import inspect

try:
    # Import from sibling package; orchestrator/tools/__init__.py exports ToolResult
    from . import ToolResult  # type: ignore
except Exception as e:
    # Fallback shim if importing standalone in examples/tests
    class ToolResult:  # type: ignore
        def __init__(self, success: bool, data: Any = None, metadata: Optional[dict] = None):
            self.success = bool(success)
            self.data = data
            self.metadata = metadata or {}
        def __repr__(self) -> str:
            return f"ToolResult(success={self.success}, data={self.data!r}, metadata={self.metadata!r})"


def _box_scalar(v: Any) -> Any:
    """If v is a number/bool, box into a dict so later subscripting is safe."""
    if isinstance(v, (int, float, bool)):
        return {"data": v, "value": v}
    return v


def _to_toolresult(x: Any) -> ToolResult:
    """Normalize anything to a ToolResult; keep success/metadata if present; box scalars."""
    if isinstance(x, ToolResult):
        return ToolResult(
            success=getattr(x, "success", True),
            data=_box_scalar(getattr(x, "data", None)),
            metadata=getattr(x, "metadata", {}) or {},
        )
    if isinstance(x, dict) and "data" in x:
        return ToolResult(
            success=bool(x.get("success", True)),
            data=_box_scalar(x.get("data")),
            metadata=x.get("metadata", {}) or {},
        )
    return ToolResult(success=True, data=_box_scalar(x), metadata={"wrapped": True})


class ToolProxy:
    """
    Thin, recursion-safe wrapper around a tool instance.
    Only intercepts implementation(payload) to normalize outputs; everything else delegates.
    """

    __slots__ = ("_inner", "_proxied")

    def __init__(self, inner: Any):
        self._inner = inner
        self._proxied = True  # marker for idempotence

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def name(self) -> str:
        return getattr(self._inner, "name", self._inner.__class__.__name__)

    @property
    def version(self) -> str:
        return getattr(self._inner, "version", "*")

    def implementation(self, payload: dict) -> ToolResult:
        out = self._inner.implementation(payload)
        return _to_toolresult(out)

    # Many tool classes also expose execute(); keep it transparent.
    def execute(self, payload: dict) -> Any:
        return getattr(self._inner, "execute")(payload)


def _ensure_instance(obj: Any) -> Any:
    """Instantiate classes; leave instances alone."""
    return obj() if inspect.isclass(obj) else obj


def _ensure_proxied(obj: Any) -> Any:
    """Wrap instances in ToolProxy exactly once."""
    if getattr(obj, "_proxied", False):
        return obj
    return ToolProxy(obj)


def _normalize_map(internal: Any) -> None:
    """Convert classes→instances and wrap in proxies in the registry's internal map."""
    if not isinstance(internal, dict):
        return
    for name, versions in list(internal.items()):
        if not isinstance(versions, dict):
            continue
        for ver, obj in list(versions.items()):
            obj = _ensure_instance(obj)
            obj = _ensure_proxied(obj)
            versions[ver] = obj


def patch_registry(registry: Any) -> None:
    """
    Apply normalization to a ToolRegistry-like object.
    Patches get_tool to return proxied instances and normalizes existing/loaded/registered tools.
    Idempotent: only patches once per registry.
    """
    if getattr(registry, "_normalization_patched", False):
        return

    # 1) Patch get_tool via the *class* method to avoid recursion
    orig_cls_get_tool: Callable = registry.__class__.get_tool  # type: ignore[attr-defined]
    registry.get_tool = orig_cls_get_tool.__get__(registry, registry.__class__)  # rebind original

    def _safe_get_tool(self, name: str, version_spec: str) -> Any:
        obj = orig_cls_get_tool(self, name, version_spec)
        obj = _ensure_instance(obj)
        obj = _ensure_proxied(obj)
        internal = getattr(self, "_tools", None)
        if isinstance(internal, dict):
            internal.setdefault(name, {})[version_spec] = obj
        return obj

    registry.get_tool = _safe_get_tool.__get__(registry, registry.__class__)  # type: ignore

    # 2) Patch register to normalize future registrations without touching originals
    if hasattr(registry, "register"):
        _orig_reg = registry.register

        def _safe_register(tool_obj: Any) -> Any:
            tool_obj = _ensure_instance(tool_obj)
            tool_obj = _ensure_proxied(tool_obj)
            return _orig_reg(tool_obj)

        registry.register = _safe_register  # type: ignore[assignment]

    # 3) Patch load_from_directory to normalize after loading from disk
    if hasattr(registry, "load_from_directory"):
        _orig_loader = registry.load_from_directory

        def _safe_loader(path: str) -> None:
            _orig_loader(path)
            internal = getattr(registry, "_tools", {})
            _normalize_map(internal)

        registry.load_from_directory = _safe_loader  # type: ignore[assignment]

    # 4) Normalize current registry contents
    internal = getattr(registry, "_tools", {})
    _normalize_map(internal)

    # Done
    registry._normalization_patched = True
