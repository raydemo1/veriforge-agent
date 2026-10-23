"""Read-only projections of the authoritative execution state."""

from __future__ import annotations

import hashlib
import json
import time

from ... import config
from ..tool_result import ToolResult


def get_runtime_status(
    section: str, *, runtime_state=None, tool_context=None, agent_name=None
):
    if runtime_state is None:
        return ToolResult(
            "get_runtime_status", "failed", error="Runtime state is unavailable"
        )
    state = runtime_state
    result = {
        "section": section,
        "turn_id": state.turn_id,
        "epoch_id": state.epoch_id,
        "observation_sequence": state.observation_sequence,
    }
    if section == "task":
        todo = state.todo
        result.update(
            task=state.current_user_task,
            todo=todo.snapshot() if todo else None,
            counts={
                status: sum(item.status == status for item in todo.items)
                for status in ("pending", "in_progress", "completed", "cancelled")
            }
            if todo
            else {},
        )
    elif section == "budget":

        def budget(used, limit):
            return {
                "used": used,
                "limit": limit if limit and limit > 0 else None,
                "remaining": max(0, limit - used) if limit and limit > 0 else None,
                "unlimited": not limit or limit <= 0,
            }

        elapsed = (
            time.monotonic() - state.run_started
            if state.run_started is not None
            else None
        )
        result.update(
            scope="current_turn",
            tool_calls=budget(
                state.fallback.tool_call_count, config.MAX_AGENT_TOOL_CALLS
            ),
            total_tokens=budget(
                state.fallback.total_tokens, config.MAX_AGENT_TOTAL_TOKENS
            ),
            seconds=budget(elapsed, state.time_budget)
            if elapsed is not None
            else {"used": None, "limit": state.time_budget},
        )
    elif section == "verification":
        record = state.verification
        stale = state.verification_stale
        if record and tool_context:
            for path, expected in record["fingerprints"].items():
                try:
                    actual = hashlib.sha256(
                        (tool_context.workspace.root / path).read_bytes()
                    ).hexdigest()
                except OSError:
                    actual = None
                if actual != expected:
                    stale = True
                    break
        result.update(
            status="recorded" if record else "not_recorded",
            stale=stale if record else None,
            checks=record["checks"] if record else [],
            evidence_sequence=record.get("sequence") if record else None,
            files=sorted(record["fingerprints"]) if record else [],
        )
    elif section == "environment":
        agent_config = state.agent_config
        profile = config.resolve_model_profile(
            getattr(agent_config, "model_intensity", None) or config.MODEL_INTENSITY
        )
        result.update(
            profile=getattr(agent_config, "name", None),
            model=profile.model,
            provider=profile.provider,
            thinking=profile.thinking,
            reasoning_effort=profile.reasoning_effort,
            input_mode=profile.input_mode,
            cwd=str(tool_context.workspace.root) if tool_context else None,
            agent=agent_name,
            permission_mode=state.permission_mode or None,
        )
    elif section == "agents":
        coordinator = getattr(tool_context, "agent_coordinator", None)
        if coordinator is None or getattr(tool_context, "agent_id", None):
            result["agents"] = []
        else:
            result["agents"] = [
                {key: agent.get(key) for key in ("agent_id", "name", "role", "status")}
                for agent in coordinator.list().get("agents", [])
            ]
    else:
        return ToolResult(
            "get_runtime_status", "failed", error="Unknown runtime status section"
        )
    return ToolResult(
        "get_runtime_status",
        "success",
        json.dumps(result, ensure_ascii=False),
        metadata={"status_source": "runtime"},
    )
