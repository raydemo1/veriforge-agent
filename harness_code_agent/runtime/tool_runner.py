"""Tool execution and event finalization."""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from ..agent.cancellation import CancelledError
from ..sessions.events import (
    FailureEvent,
    FileChangeEvent,
    ToolCallEvent,
    ToolResultEvent,
    classify_tool_failure,
)
from .tool_call_validation import validate_tool_arguments
from .tool_context import ToolContext
from .tool_registry import ToolRegistry
from .tool_result import ToolResult, unstructured_tool_result_from_text


TOOL_EVENT_OUTPUT_LIMIT = 2_000


def execute_tool(
    name: str,
    arguments: dict,
    runtime_state=None,
    agent_name: str | None = None,
    tool_context: ToolContext | None = None,
    cancellation_token=None,
) -> str:
    return execute_tool_result(
        name,
        arguments,
        runtime_state=runtime_state,
        agent_name=agent_name,
        tool_context=tool_context,
        cancellation_token=cancellation_token,
    ).to_text()


def execute_tool_result(
    name: str,
    arguments: Any,
    runtime_state=None,
    agent_name: str | None = None,
    tool_context: ToolContext | None = None,
    emit_events: bool = True,
    cancellation_token=None,
) -> ToolResult:
    """Execute a registered tool only after the runtime validation boundary."""
    registry = _registry_for_context(tool_context)

    if emit_events and tool_context is not None:
        emit_tool_call_started(
            name=name,
            arguments=arguments if isinstance(arguments, dict) else {},
            tool_context=tool_context,
            agent_name=agent_name,
        )

    fn = registry.get(name)
    if fn is None:
        return _finalize_tool_result_object(
            ToolResult(
                tool=name,
                status="failed",
                output=f"[error] Unknown tool: {name}",
                error=f"Unknown tool: {name}",
                metadata={"status_source": "registry"},
            ),
            tool_context=tool_context,
            agent_name=agent_name,
            emit_events=emit_events,
        )

    if registry.schema_for(name) is not None:
        validation = validate_tool_arguments(name, arguments, registry, tool_context)
        if validation.error is not None:
            return _finalize_tool_result_object(
                validation.error.to_result(name),
                tool_context=tool_context,
                agent_name=agent_name,
                emit_events=emit_events,
            )
        arguments = validation.arguments
    else:
        # Direct-call path for handlers injected by integrations that do not
        # register a schema. Model tool calls never use it: ToolExecutor
        # requires a registered schema first.
        arguments = dict(arguments or {})

    try:
        result = _invoke_registered_tool(
            fn,
            arguments,
            runtime_state=runtime_state,
            agent_name=agent_name,
            tool_context=tool_context,
            cancellation_token=cancellation_token,
        )
    except CancelledError:
        raise
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        tool_result = ToolResult(
            tool=name,
            status="failed",
            output=f"[error] {error}",
            error=error,
            metadata={"status_source": "exception"},
        )
    else:
        tool_result = _coerce_tool_result(name, result)

    return _finalize_tool_result_object(
        tool_result,
        tool_context=tool_context,
        agent_name=agent_name,
        emit_events=emit_events,
    )


def finalize_executed_tool_result(
    tool_result: ToolResult,
    *,
    arguments: dict | None = None,
    tool_context: ToolContext | None,
    agent_name: str | None,
    emit_call: bool = True,
) -> ToolResult:
    """Record a tool call and its already-computed result on the main thread."""
    if emit_call:
        emit_tool_call_started(
            name=tool_result.tool,
            arguments=arguments or {},
            tool_context=tool_context,
            agent_name=agent_name,
        )
    return _finalize_tool_result_object(
        tool_result,
        tool_context=tool_context,
        agent_name=agent_name,
    )


def finalize_intercepted_tool_result(
    tool_result: ToolResult,
    *,
    arguments: dict | None = None,
    tool_context: ToolContext | None,
    agent_name: str | None,
) -> ToolResult:
    """Record a tool call that was intercepted before native execution."""
    if tool_context is not None:
        tool_context.event_bus.emit_event(
            ToolCallEvent(
                tool=tool_result.tool,
                args=_redact_tool_args(arguments or {}),
                agent=agent_name,
            ).to_event()
        )
    return _finalize_tool_result_object(
        tool_result,
        tool_context=tool_context,
        agent_name=agent_name,
    )


def emit_tool_call_started(
    *,
    name: str,
    arguments: dict,
    tool_context: ToolContext | None,
    agent_name: str | None,
) -> None:
    if tool_context is None:
        return
    tool_context.event_bus.emit_event(
        ToolCallEvent(
            tool=name,
            args=_redact_tool_args(arguments),
            agent=agent_name,
        ).to_event()
    )


def _registry_for_context(tool_context: ToolContext | None) -> ToolRegistry:
    if tool_context is not None and tool_context.tool_registry is not None:
        return tool_context.tool_registry
    from .builtins.registry import BUILTIN_TOOL_REGISTRY

    return BUILTIN_TOOL_REGISTRY


def _invoke_registered_tool(
    fn: Callable,
    arguments: dict,
    *,
    runtime_state,
    agent_name: str | None,
    tool_context: ToolContext | None,
    cancellation_token=None,
):
    kwargs = dict(arguments)
    parameters = inspect.signature(fn).parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    extras = {
        "runtime_state": runtime_state,
        "agent_name": agent_name,
        "tool_context": tool_context,
    }
    if cancellation_token is not None:
        extras["cancellation_token"] = cancellation_token
    for key, value in extras.items():
        if key not in kwargs and (key in parameters or accepts_kwargs):
            kwargs[key] = value
    return fn(**kwargs)


def _finalize_tool_result_object(
    tool_result: ToolResult,
    *,
    tool_context: ToolContext | None,
    agent_name: str | None,
    emit_events: bool = True,
) -> ToolResult:
    if emit_events and tool_context is not None:
        _emit_structured_tool_result(tool_result, tool_context=tool_context, agent_name=agent_name)
        _emit_file_change_events(tool_result, tool_context=tool_context, agent_name=agent_name)
    return tool_result


def _coerce_tool_result(name: str, result) -> ToolResult:
    if isinstance(result, ToolResult):
        return result
    return unstructured_tool_result_from_text(tool=name, text=str(result))


def _event_safe_tool_output(tool_result: ToolResult) -> tuple[str, dict]:
    metadata = dict(tool_result.metadata)
    output = tool_result.output or ""
    metadata["output_length"] = len(output)
    if tool_result.tool in {"read_file", "read_agent_changes", "read_agent_conflicts"} and output:
        metadata["output_redacted"] = True
        return f"[redacted {tool_result.tool} output: {len(output)} chars]", metadata
    if len(output) > TOOL_EVENT_OUTPUT_LIMIT:
        metadata["output_truncated"] = True
        metadata["output_preview_chars"] = TOOL_EVENT_OUTPUT_LIMIT
        return (
            output[:TOOL_EVENT_OUTPUT_LIMIT]
            + f"\n\n[TRUNCATED in session event: {len(output) - TOOL_EVENT_OUTPUT_LIMIT} chars omitted]",
            metadata,
        )
    return output, metadata


def _emit_structured_tool_result(
    tool_result: ToolResult,
    *,
    tool_context: ToolContext,
    agent_name: str | None,
) -> None:
    event_output, event_metadata = _event_safe_tool_output(tool_result)
    tool_context.event_bus.emit_event(
        ToolResultEvent(
            tool=tool_result.tool,
            status=tool_result.status,
            output=event_output,
            error=tool_result.error,
            return_code=tool_result.return_code,
            metadata=event_metadata,
            agent=agent_name,
        ).to_event()
    )
    if tool_result.status == "failed":
        source = tool_result.metadata.get("status_source")
        message = tool_result.error or event_output
        failure_metadata = {
            key: event_metadata[key]
            for key in (
                "failure_category",
                "failure_kind",
                "failure_phase",
                "failure_visibility",
                "failure_retryable",
                "failure_user_action",
                "failure_attempt",
                "failure_intercepted",
            )
            if key in event_metadata
        }
        tool_context.event_bus.emit_event(
            FailureEvent(
                category=classify_tool_failure(tool_result),
                message=message,
                tool=tool_result.tool,
                source=str(source) if source else None,
                metadata=failure_metadata or None,
                agent=agent_name,
            ).to_event()
        )


def _emit_file_change_events(
    tool_result: ToolResult,
    *,
    tool_context: ToolContext,
    agent_name: str | None,
) -> None:
    file_changes = tool_result.metadata.get("file_changes")
    if not isinstance(file_changes, list):
        return
    for change in file_changes:
        if not isinstance(change, dict):
            continue
        path = change.get("path")
        if not path:
            continue
        payload = {
            "path": str(path),
            "snapshot_path": change.get("snapshot_path"),
        }
        if change.get("operation"):
            payload["operation"] = change["operation"]
        if change.get("additions") is not None:
            payload["additions"] = change["additions"]
        if change.get("deletions") is not None:
            payload["deletions"] = change["deletions"]
        tool_context.event_bus.emit_event(
            FileChangeEvent(
                path=str(path),
                operation=change.get("operation"),
                snapshot_path=change.get("snapshot_path"),
                diff=change.get("diff"),
                additions=change.get("additions"),
                deletions=change.get("deletions"),
                agent=agent_name,
            ).to_event()
        )


def _redact_tool_args(arguments: dict) -> dict:
    value = _redact_tool_arg_value(arguments or {})
    return value if isinstance(value, dict) else {}


def _redact_tool_arg_value(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if key == "content":
                redacted[key] = f"[{len(str(item))} chars]"
            else:
                redacted[key] = _redact_tool_arg_value(item)
        return redacted
    if isinstance(value, list):
        return [_redact_tool_arg_value(item) for item in value]
    return value
