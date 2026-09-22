"""Tool registry: JSON-schema tool definitions, argument validation and safe dispatch.

Every failure mode (unknown tool, bad JSON, missing/unknown/mistyped args, exceptions
inside the tool) becomes a ToolResult the model can read and recover from, never a crash.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

_PY_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool}


@dataclass
class ToolResult:
    text: str
    done: bool = False      # ends the episode (e.g. accepted submission)
    error: bool = False


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, dict]                 # JSON-schema "properties"
    fn: Callable[..., Any]
    required: list[str] = field(default_factory=list)

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": self.params,
                           "required": self.required, "additionalProperties": False}}}


def _coerce(value: Any, typ: str) -> Any:
    """Accept the common near-misses models produce ("30" for 30, "true" for True)."""
    if typ == "integer" and isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    if typ == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    if typ == "number" and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    if typ == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool {tool.name}")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def validate(self, tool: Tool, args: dict) -> tuple[dict, list[str]]:
        errors, clean = [], {}
        for k in tool.required:
            if k not in args:
                errors.append(f"missing required argument '{k}'")
        for k, v in args.items():
            spec = tool.params.get(k)
            if spec is None:
                errors.append(f"unknown argument '{k}' (allowed: {', '.join(tool.params)})")
                continue
            typ = spec.get("type")
            v = _coerce(v, typ)
            py = _PY_TYPES.get(typ)
            # bool is a subclass of int; don't let True pass as an integer
            if py and (not isinstance(v, py) or (typ in ("integer", "number") and isinstance(v, bool))):
                errors.append(f"argument '{k}' must be {typ}, got {type(v).__name__}")
                continue
            clean[k] = v
        return clean, errors

    def dispatch(self, name: str, raw_args: str | dict | None) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(f"error: unknown tool '{name}'. Available: {', '.join(self._tools)}", error=True)
        if isinstance(raw_args, dict):
            args = raw_args
        else:
            try:
                args = json.loads(raw_args or "{}")
            except json.JSONDecodeError as e:
                return ToolResult(f"error: arguments are not valid JSON ({e}). Re-issue the call.", error=True)
        if not isinstance(args, dict):
            return ToolResult("error: arguments must be a JSON object", error=True)
        clean, errors = self.validate(tool, args)
        if errors:
            return ToolResult("error: " + "; ".join(errors), error=True)
        try:
            out = tool.fn(**clean)
        except Exception as e:  # tool bugs must not kill the episode
            return ToolResult(f"tool error: {type(e).__name__}: {e}", error=True)
        return out if isinstance(out, ToolResult) else ToolResult(str(out))
