"""Lightweight jsonschema validator used when the external dependency is unavailable."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping


class ValidationError(Exception):
    """Raised when validation fails."""


_TYPE_MAP = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "object": Mapping,
    "array": Iterable,
    "boolean": bool,
}


def validate(instance: Any, schema: Dict[str, Any]) -> None:
    """Validate *instance* against a tiny subset of JSON Schema."""

    expected_type = schema.get("type")
    if expected_type:
        python_type = _TYPE_MAP.get(expected_type)
        if python_type is None:
            raise ValidationError(f"Unsupported type in schema: {expected_type}")
        if not isinstance(instance, python_type):
            raise ValidationError(f"Expected type {expected_type}")

    if expected_type == "object" and isinstance(instance, Mapping):
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for field in required:
            if field not in instance:
                raise ValidationError(f"Missing required field: {field}")

        for field, subschema in properties.items():
            if field in instance:
                validate(instance[field], subschema)

    if expected_type == "array" and isinstance(instance, Iterable):
        item_schema = schema.get("items")
        if item_schema:
            for item in instance:
                validate(item, item_schema)


__all__ = ["validate", "ValidationError"]
