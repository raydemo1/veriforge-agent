"""Validation boundary for model-produced tool calls.

Tool providers are allowed to differ in how strictly they enforce function
schemas.  This module keeps that variability outside the execution path:
arguments are decoded, structurally validated, and semantically preflighted
before permission or resource planning can see them.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from .tool_failures import ToolFailure
from .tool_result import ToolResult

if TYPE_CHECKING:
    from .tool_context import ToolContext
    from .tool_registry import ToolRegistry


@dataclass(frozen=True)
class ToolCall:
    """Provider-neutral representation of one function call."""

    index: int
    tool_call_id: str
    name: str
    raw_arguments: Any
    raw: Any = None

    @classmethod
    def from_raw(cls, raw: Any, index: int) -> ToolCall:
        function = _read_field(raw, "function", {})
        if function is None:
            function = {}
        name_value = _read_field(function, "name", "")
        name = name_value.strip() if isinstance(name_value, str) else ""

        call_id_value = _read_field(raw, "id", None)
        call_id = str(call_id_value).strip() if call_id_value is not None else ""
        if not call_id:
            call_id = f"call_{index}"

        return cls(
            index=index,
            tool_call_id=call_id,
            name=name,
            raw_arguments=_read_field(function, "arguments", None),
            raw=raw,
        )


@dataclass(frozen=True)
class ToolError:
    """Structured failure returned when a tool call cannot enter execution."""

    kind: str
    message: str
    retryable: bool
    phase: str

    def to_result(self, tool: str, *, extra_metadata: Mapping[str, Any] | None = None) -> ToolResult:
        error = f"{self.kind}: {self.message}"
        retryable = "true" if self.retryable else "false"
        metadata = dict(extra_metadata or {})
        metadata.update(
            {
                "status_source": "validation",
                "error_kind": self.kind,
                "retryable": self.retryable,
                "validation_phase": self.phase,
            }
        )
        result = ToolResult(
            tool=tool,
            status="failed",
            output=(
                f"[error] {self.message}\n"
                f"[tool_error] kind={self.kind} retryable={retryable} phase={self.phase}"
            ),
            error=error,
            metadata=metadata,
        )
        # Add the canonical failure_* keys while keeping the legacy keys above.
        return ToolFailure.from_tool_error(
            tool_call_id="",
            tool_name=tool,
            error=self,
        ).stamp_metadata(result)


@dataclass(frozen=True)
class ToolCallValidationResult:
    arguments: dict[str, Any] = field(default_factory=dict)
    error: ToolError | None = None

    @property
    def valid(self) -> bool:
        return self.error is None


def strict_tool_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of a tool schema with object maps closed by default.

    Explicit ``additionalProperties`` values are preserved.  This lets a
    tool author intentionally declare an open map while making ordinary tool
    argument objects reject undeclared fields.
    """

    strict = copy.deepcopy(dict(schema))
    _close_object_maps(strict)
    return strict


def validate_tool_call(
    call: ToolCall,
    registry: ToolRegistry,
    tool_context: ToolContext | None = None,
) -> ToolCallValidationResult:
    """Validate one normalized call through syntax, schema, and semantics."""

    arguments, decode_error = _decode_arguments(call.raw_arguments)
    if decode_error is not None:
        return ToolCallValidationResult(arguments=arguments, error=decode_error)

    if not call.name:
        return ToolCallValidationResult(
            arguments=arguments,
            error=ToolError(
                kind="invalid_tool_call",
                message="tool name is required",
                retryable=True,
                phase="structural",
            ),
        )

    schema = registry.schema_for(call.name)
    if schema is None:
        return ToolCallValidationResult(
            arguments=arguments,
            error=ToolError(
                kind="unknown_tool",
                message=f"Tool '{call.name}' is not registered",
                retryable=False,
                phase="structural",
            ),
        )

    parameters = _parameters_schema(schema)
    if parameters is None:
        return ToolCallValidationResult(
            arguments=arguments,
            error=ToolError(
                kind="tool_schema_error",
                message=f"Tool '{call.name}' has no valid parameters schema",
                retryable=False,
                phase="structural",
            ),
        )

    try:
        Draft202012Validator.check_schema(parameters)
        schema_errors = list(Draft202012Validator(parameters).iter_errors(arguments))
    except SchemaError as exc:
        return ToolCallValidationResult(
            arguments=arguments,
            error=ToolError(
                kind="tool_schema_error",
                message=f"Tool '{call.name}' schema is invalid: {exc.message}",
                retryable=False,
                phase="structural",
            ),
        )

    if schema_errors:
        return ToolCallValidationResult(
            arguments=arguments,
            error=ToolError(
                kind="invalid_arguments",
                message=_format_schema_error(schema_errors[0]),
                retryable=True,
                phase="structural",
            ),
        )

    semantic_error = _validate_semantics(call.name, arguments, tool_context)
    if semantic_error is not None:
        return ToolCallValidationResult(arguments=arguments, error=semantic_error)
    return ToolCallValidationResult(arguments=arguments)


def validate_tool_arguments(
    name: str,
    arguments: Any,
    registry: ToolRegistry,
    tool_context: ToolContext | None = None,
) -> ToolCallValidationResult:
    """Validate already-decoded arguments used by direct runtime callers."""

    return validate_tool_call(
        ToolCall(index=-1, tool_call_id="direct", name=name, raw_arguments=arguments),
        registry,
        tool_context,
    )


def _read_field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _decode_arguments(raw_arguments: Any) -> tuple[dict[str, Any], ToolError | None]:
    if raw_arguments is None:
        return {}, None
    if isinstance(raw_arguments, str):
        text = raw_arguments.strip()
        if not text:
            return {}, None
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            return {}, ToolError(
                kind="invalid_json",
                message=f"arguments are not valid JSON ({exc.msg} at character {exc.pos})",
                retryable=True,
                phase="syntactic",
            )
    elif isinstance(raw_arguments, Mapping):
        decoded = dict(raw_arguments)
    else:
        return {}, ToolError(
            kind="invalid_arguments",
            message="arguments must be a JSON string or object",
            retryable=True,
            phase="syntactic",
        )

    if not isinstance(decoded, dict):
        root_type = _json_type_name(decoded)
        return {}, ToolError(
            kind="invalid_arguments",
            message=f"arguments root must be an object, got {root_type}",
            retryable=True,
            phase="structural",
        )
    return decoded, None


def _parameters_schema(schema: Mapping[str, Any]) -> dict[str, Any] | None:
    function = schema.get("function")
    if isinstance(function, Mapping):
        parameters = function.get("parameters")
    else:
        parameters = schema
    return dict(parameters) if isinstance(parameters, Mapping) else None


def _close_object_maps(node: Any) -> None:
    if isinstance(node, dict):
        if "properties" in node and "additionalProperties" not in node:
            node["additionalProperties"] = False
        for value in node.values():
            _close_object_maps(value)
    elif isinstance(node, list):
        for value in node:
            _close_object_maps(value)


def _format_schema_error(error: ValidationError) -> str:
    for child in _leaf_errors(error):
        return _format_leaf_schema_error(child)
    return "arguments do not match the tool schema"


def _leaf_errors(error: ValidationError):
    if error.context:
        for child in error.context:
            yield from _leaf_errors(child)
        return
    yield error


def _format_leaf_schema_error(error: ValidationError) -> str:
    location = _error_location(error)
    params = getattr(error, "params", {}) or {}
    validator = error.validator

    if validator == "required":
        required = params.get("property")
        if required is None:
            required = _extract_quoted(error.message)
        return f"{required or location} is required"
    if validator == "additionalProperties":
        unexpected = params.get("additionalProperties")
        if isinstance(unexpected, (set, list, tuple)):
            names = ", ".join(sorted(str(item) for item in unexpected))
        else:
            names = str(unexpected or "an undeclared field")
        return f"additional properties are not allowed: {names}"
    if validator == "type":
        return f"{location} must be {_human_type(error.validator_value)}"
    if validator == "enum":
        choices = ", ".join(
            item if isinstance(item, str) else repr(item)
            for item in error.validator_value
        )
        return f"{location} must be one of: {choices}"
    if validator in {"minimum", "exclusiveMinimum"}:
        bound = error.validator_value
        operator = ">" if validator == "exclusiveMinimum" else ">="
        return _format_numeric_constraint(location, "must be", operator, bound, error)
    if validator in {"maximum", "exclusiveMaximum"}:
        bound = error.validator_value
        operator = "<" if validator == "exclusiveMaximum" else "<="
        return _format_numeric_constraint(location, "must be", operator, bound, error)
    if validator == "minLength":
        return f"{location} must contain at least {error.validator_value} characters"
    if validator == "maxLength":
        return f"{location} must contain at most {error.validator_value} characters"
    if validator == "minItems":
        return f"{location} must contain at least {error.validator_value} items"
    if validator == "maxItems":
        return f"{location} must contain at most {error.validator_value} items"
    if validator == "pattern":
        return f"{location} must match pattern {error.validator_value!r}"
    return f"{location} is invalid: {error.message}"


def _format_numeric_constraint(
    location: str,
    _verb: str,
    operator: str,
    bound: Any,
    error: ValidationError,
) -> str:
    message = f"{location} must be {operator} {bound}"
    schema_type = error.schema.get("type") if isinstance(error.schema, Mapping) else None
    if schema_type == "integer":
        message += f"; {location} must be an integer"
    elif schema_type == "number":
        message += f"; {location} must be a number"
    return message


def _error_location(error: ValidationError) -> str:
    if not error.path:
        return "arguments"
    parts: list[str] = []
    for item in error.path:
        if isinstance(item, int):
            parts.append(f"[{item}]")
        elif parts:
            parts.append(f".{item}")
        else:
            parts.append(str(item))
    return "".join(parts)


def _extract_quoted(message: str) -> str | None:
    if not message.startswith("'"):
        return None
    end = message.find("'", 1)
    return message[1:end] if end > 1 else None


def _human_type(value: Any) -> str:
    values = value if isinstance(value, list) else [value]
    labels = {
        "boolean": "a boolean",
        "integer": "an integer",
        "number": "a number",
        "object": "an object",
        "array": "an array",
        "string": "a string",
        "null": "null",
    }
    return " or ".join(labels.get(str(item), f"{item}") for item in values)


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


_PATH_ARGUMENTS = {
    "read_file": "path",
    "write_file": "path",
    "apply_patch": "path",
    "repo_search": "path",
    "list_files": "directory",
}


def _validate_semantics(
    name: str,
    arguments: dict[str, Any],
    tool_context: ToolContext | None,
) -> ToolError | None:
    path_argument = _PATH_ARGUMENTS.get(name)
    if path_argument is not None:
        path = arguments.get(path_argument, ".")
        if name in {"read_file", "write_file", "apply_patch"} and not str(path).strip():
            message = "Empty file path. You must specify a path."
            if name == "apply_patch":
                message = "Empty file path"
            return ToolError(
                kind="invalid_arguments",
                message=message,
                retryable=True,
                phase="semantic",
            )
        if tool_context is not None:
            try:
                tool_context.workspace.resolve(path or ".")
            except (OSError, TypeError, ValueError):
                return ToolError(
                    kind="workspace_escape",
                    message=f"path `{path}` is outside the workspace",
                    retryable=True,
                    phase="semantic",
                )

    if name == "apply_patch" and not str(arguments.get("search", "")).strip():
        return ToolError(
            kind="invalid_arguments",
            message="Patch search text must not be empty",
            retryable=True,
            phase="semantic",
        )

    if name == "run_bash":
        command = str(arguments.get("command", ""))
        if not command.strip():
            return ToolError(
                kind="invalid_arguments",
                message="Empty command. You must specify a command to run.",
                retryable=True,
                phase="semantic",
            )
        first_word = command.strip().split(maxsplit=1)[0].lower()
        if first_word in {"vim", "nano", "vi", "less", "more", "top", "htop"}:
            viewer = "type/more" if os.name == "nt" else "cat/head/tail"
            return ToolError(
                kind="blocked_command",
                message=(
                    f"'{first_word}' is an interactive command that will hang. "
                    f"Use non-interactive alternatives: for editing use write_file, "
                    f"for viewing use {viewer}."
                ),
                retryable=False,
                phase="semantic",
            )
    return None
