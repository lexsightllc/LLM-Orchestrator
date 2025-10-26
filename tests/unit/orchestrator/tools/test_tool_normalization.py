# SPDX-License-Identifier: MPL-2.0
import pytest
from orchestrator import tools
from orchestrator.tools import ToolResult
from orchestrator.tools.normalization import patch_registry


class DummyCalc:
    name = "calculator"
    version = "1.0.0"

    def implementation(self, payload: dict):
        op = payload.get("operation")
        a, b = payload.get("a", 0), payload.get("b", 0)
        if op == "add":
            return a + b
        if op == "sub":
            return a - b
        return {"success": True, "data": 42, "metadata": {"op": op}}


def test_proxy_patch_is_idempotent():
    reg = getattr(tools, "tool_registry")
    patch_registry(reg)
    patch_registry(reg)  # no-op
    assert getattr(reg, "_normalization_patched", False) is True


def test_scalar_results_are_boxed_and_toolresult():
    reg = tools.tool_registry
    internal = getattr(reg, "_tools", {})
    internal.setdefault("calculator", {})["1.0.0"] = DummyCalc()
    patch_registry(reg)

    calc = reg.get_tool("calculator", "1.0.0")
    out = calc.implementation({"operation": "add", "a": 3, "b": 9})
    assert isinstance(out, ToolResult)
    assert out.success is True
    assert isinstance(out.data, dict)
    assert out.data["data"] == 12
    assert out.data["value"] == 12


def test_dict_shaped_results_pass_through_and_box_inner():
    reg = tools.tool_registry
    internal = getattr(reg, "_tools", {})
    internal.setdefault("calculator", {})["1.0.0"] = DummyCalc()
    patch_registry(reg)

    calc = reg.get_tool("calculator", "1.0.0")
    out = calc.implementation({"operation": "noop"})
    assert isinstance(out, ToolResult)
    assert out.success is True
    assert out.metadata.get("op") == "noop"
    assert isinstance(out.data, dict)
    assert out.data["data"] == 42
    assert out.data["value"] == 42


def test_register_and_load_are_safe_to_call(monkeypatch, tmp_path):
    reg = tools.tool_registry
    patch_registry(reg)

    tool = DummyCalc()
    if hasattr(reg, "register"):
        reg.register(tool)
    got = reg.get_tool("calculator", "1.0.0")
    assert hasattr(got, "_proxied") and got._proxied is True

    if hasattr(reg, "load_from_directory"):
        reg.load_from_directory(str(tmp_path))
