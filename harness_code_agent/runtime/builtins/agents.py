"""Tool adapters for the session-scoped agent coordinator."""
from __future__ import annotations

import json
import time
from typing import Any

from ..errors import SubagentCapacityError
from ..tool_context import ToolContext
from ..tool_result import ToolResult


def spawn_agent(
    name: str,
    role: str,
    task: str,
    expected_output: str = "",
    allowed_paths: list[str] | None = None,
    fork_turns: str | int = "none",
    model_intensity: str | None = None,
    tool_context: ToolContext | None = None,
    cancellation_token=None,
) -> ToolResult:
    coordinator = _coordinator(tool_context)
    try:
        result = coordinator.spawn(
            name=name,
            role=role,
            task=task,
            expected_output=expected_output,
            allowed_paths=allowed_paths,
            fork_turns=fork_turns,
            model_intensity=model_intensity,
            parent_cancellation_token=cancellation_token,
        )
    except SubagentCapacityError as exc:
        return _capacity_error("spawn_agent", str(exc))
    except ValueError as exc:
        return _policy_error("spawn_agent", str(exc))
    return _result("spawn_agent", result)


def _policy_error(tool: str, reason: str) -> ToolResult:
    return ToolResult(
        tool=tool,
        status="failed",
        output=f"[blocked] {reason}",
        error=reason,
        metadata={"status_source": "agent_policy"},
    )


def _capacity_error(tool: str, reason: str) -> ToolResult:
    return ToolResult(
        tool=tool,
        status="failed",
        output=f"[resource_unavailable] {reason}",
        error=reason,
        metadata={"status_source": "resource", "resource_kind": "subagent_capacity"},
    )


def send_agent_message(agent_id: str, message: str, tool_context: ToolContext | None = None) -> ToolResult:
    return _result("send_agent_message", _coordinator(tool_context).send(agent_id, message))


def send_parent_message(message: str, tool_context: ToolContext | None = None) -> ToolResult:
    # Child -> Main only. The tool is absent from the main registry, and the
    # context carries an id only for child-agent contexts.
    agent_id = None if tool_context is None else tool_context.agent_id
    if not agent_id:
        return _policy_error(
            "send_parent_message",
            "send_parent_message is only available to spawned agents",
        )
    try:
        payload = _coordinator(tool_context).send_to_parent(agent_id, message)
    except ValueError as exc:
        return _policy_error("send_parent_message", str(exc))
    return _result("send_parent_message", payload)


def followup_agent(agent_id: str, task: str, tool_context: ToolContext | None = None) -> ToolResult:
    return _result("followup_agent", _coordinator(tool_context).followup(agent_id, task))


def wait_agents(
    agent_ids: list[str] | None = None,
    timeout_seconds: float = 30,
    tool_context: ToolContext | None = None,
    cancellation_token=None,
) -> ToolResult:
    coordinator = _coordinator(tool_context)
    deadline = time.monotonic() + max(0.0, min(300.0, float(timeout_seconds)))
    payload = coordinator.wait(agent_ids, timeout_seconds=0)
    while payload.get("timed_out") and time.monotonic() < deadline:
        if cancellation_token is not None:
            cancellation_token.check()
        payload = coordinator.wait(agent_ids, timeout_seconds=min(0.25, deadline - time.monotonic()))
    return _result("wait_agents", payload)


def interrupt_agent(agent_id: str, tool_context: ToolContext | None = None) -> ToolResult:
    return _result("interrupt_agent", _coordinator(tool_context).interrupt(agent_id))


def read_agent_changes(
    proposal_id: str = "",
    conflict_id: str = "",
    path: str = "",
    offset: int = 0,
    limit: int = 12_000,
    tool_context: ToolContext | None = None,
) -> ToolResult:
    if not proposal_id and not conflict_id:
        raise ValueError("read_agent_changes requires proposal_id or conflict_id")
    changes = _coordinator(tool_context).changes
    if conflict_id:
        payload = changes.read_conflicts(
            conflict_id, proposal_id=proposal_id, path=path, offset=offset, limit=limit
        )
    else:
        payload = changes.read_changes(proposal_id, path=path, offset=offset, limit=limit)
    return _result("read_agent_changes", payload)


def apply_agent_changes(
    proposal_id: str = "",
    conflict_id: str = "",
    resolutions: dict[str, str] | None = None,
    tool_context: ToolContext | None = None,
) -> ToolResult:
    changes = _coordinator(tool_context).changes
    if resolutions is not None:
        if not conflict_id:
            raise ValueError("resolutions requires the conflict_id returned by the previous apply")
        payload = changes.resolve(
            conflict_id, resolutions, tool_context.workspace, proposal_id=proposal_id
        )
    elif conflict_id:
        raise ValueError(
            "conflict_id requires resolutions with the final content for every conflicted path"
        )
    elif proposal_id:
        payload = changes.apply(proposal_id, tool_context.workspace)
    else:
        raise ValueError("apply_agent_changes requires proposal_id or conflict_id")
    return _result("apply_agent_changes", payload, metadata={"file_changes": payload.get("file_changes", [])})


def close_agent(
    agent_id: str,
    discard_changes: bool = False,
    tool_context: ToolContext | None = None,
) -> ToolResult:
    return _result("close_agent", _coordinator(tool_context).close_agent(agent_id, discard_changes=discard_changes))


def _coordinator(tool_context: ToolContext | None):
    if tool_context is None or tool_context.agent_coordinator is None:
        raise RuntimeError("agent coordinator is unavailable")
    return tool_context.agent_coordinator


def _result(tool: str, payload: dict[str, Any], *, metadata: dict | None = None) -> ToolResult:
    merged_metadata = {"status_source": "native"}
    merged_metadata.update(metadata or {})
    return ToolResult(
        tool=tool,
        status="success",
        output=json.dumps(payload, ensure_ascii=False, sort_keys=True),
        metadata=merged_metadata,
    )
