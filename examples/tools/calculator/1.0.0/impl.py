"""Calculator tool implementation."""
from typing import Dict, Any
from orchestrator.tools import ToolResult


def execute(params: Dict[str, Any]) -> ToolResult:
    """
    Execute a calculation.

    Args:
        params: Dictionary containing:
            - operation: One of 'add', 'subtract', 'multiply', 'divide'
            - a: First number
            - b: Second number

    Returns:
        ToolResult with the calculation result
    """
    operation = params.get('operation')
    a = params.get('a')
    b = params.get('b')
    
    if operation == 'add':
        result = a + b
    elif operation == 'subtract':
        result = a - b
    elif operation == 'multiply':
        result = a * b
    elif operation == 'divide':
        if b == 0:
            return ToolResult.from_error("Cannot divide by zero")
        result = a / b
    else:
        return ToolResult.from_error(f"Unknown operation: {operation}")
    
    return ToolResult.from_success({
        'result': result,
        'operation': operation
    })
