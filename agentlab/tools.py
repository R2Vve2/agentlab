"""Tool registry.

A tool is a plain Python function registered with ``@registry.tool(...)``. The registry
derives the JSON Schema the model sees from the function's type hints, validates every
model-supplied input against that schema with pydantic before calling the function, and
turns exceptions into error results the model can recover from (instead of crashing the
agent loop).
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

from pydantic import ValidationError, create_model

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass
class ToolSpec:
    name: str
    description: str
    func: Callable[..., Any]
    input_schema: dict[str, Any]
    validator: Any  # pydantic model class

    def to_api(self) -> dict[str, Any]:
        """Shape expected by the Claude Messages API ``tools`` parameter."""
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema, "strict": True}


@dataclass
class ToolResult:
    name: str
    input: dict[str, Any]
    output: str
    is_error: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


def _schema_for(func: Callable[..., Any], param_docs: dict[str, str]) -> tuple[dict[str, Any], Any]:
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    fields: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        hint = hints.get(name, str)
        if hint not in _JSON_TYPES:
            raise TypeError(f"tool {func.__name__}: parameter {name!r} must be str/int/float/bool, got {hint}")
        prop: dict[str, Any] = {"type": _JSON_TYPES[hint]}
        if name in param_docs:
            prop["description"] = param_docs[name]
        if param.default is inspect.Parameter.empty:
            required.append(name)
            fields[name] = (hint, ...)
        else:
            prop["default"] = param.default
            fields[name] = (hint, param.default)
        properties[name] = prop
    schema = {"type": "object", "properties": properties, "required": required, "additionalProperties": False}
    return schema, create_model(f"{func.__name__}_Input", **fields)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def tool(self, description: str, params: dict[str, str] | None = None, name: str | None = None):
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or func.__name__
            if tool_name in self._tools:
                raise ValueError(f"tool {tool_name!r} already registered")
            schema, validator = _schema_for(func, params or {})
            self._tools[tool_name] = ToolSpec(tool_name, description.strip(), func, schema, validator)
            return func

        return decorator

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> ToolSpec:
        return self._tools[name]

    def to_api(self) -> list[dict[str, Any]]:
        return [t.to_api() for t in self._tools.values()]

    def call(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
        """Validate and run a tool. Never raises: failures become ``is_error`` results."""
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(name, raw_input, f"Unknown tool {name!r}. Available: {', '.join(self._tools)}", is_error=True)
        try:
            validated = spec.validator(**raw_input)
        except ValidationError as exc:
            return ToolResult(name, raw_input, f"Invalid input for {name}: {exc.errors()[0]['msg']} ({exc.errors()[0]['loc']})", is_error=True)
        try:
            output = spec.func(**validated.model_dump())
        except Exception as exc:  # noqa: BLE001 - the model should see tool failures, not crash the loop
            return ToolResult(name, raw_input, f"{type(exc).__name__}: {exc}", is_error=True)
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False, default=str)
        return ToolResult(name, raw_input, output)
