# SPDX-License-Identifier: MPL-2.0
"""
Demonstration of recursion-safe tool normalization via ToolProxy.
Run this to confirm ToolResult shape and scalar boxing.
"""

from orchestrator import tools
from orchestrator.tools.normalization import patch_registry


def main():
    patch_registry(tools.tool_registry)

    class Calculator:
        name = "calculator"
        version = "1.0.0"

        def implementation(self, payload: dict):
            op = payload.get("operation")
            a, b = payload.get("a", 0), payload.get("b", 0)
            if op == "add":
                return a + b
            if op == "mul":
                return a * b
            return {"success": True, "data": 7, "metadata": {"note": "default"}}

    internal = getattr(tools.tool_registry, "_tools", {})
    internal.setdefault("calculator", {})["1.0.0"] = Calculator()

    calc = tools.tool_registry.get_tool("calculator", "1.0.0")
    r1 = calc.implementation({"operation": "add", "a": 5, "b": 7})
    r2 = calc.implementation({"operation": "mul", "a": 3, "b": 9})
    r3 = calc.implementation({"operation": "noop"})

    print("add ->", r1)
    print("mul ->", r2)
    print("noop->", r3)
    print("boxed access ok:", r1.data["data"], r2.data["value"])


if __name__ == "__main__":
    main()
