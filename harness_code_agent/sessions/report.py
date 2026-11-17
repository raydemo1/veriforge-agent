from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ._event_helpers import (
    changed_files as _changed_files,
)
from ._event_helpers import (
    count_events as _count_events,
)
from ._event_helpers import (
    event_type as _event_type,
)
from ._event_helpers import (
    failure_categories as _failure_categories,
)
from ._event_helpers import (
    payload as _payload,
)
from ._event_helpers import (
    tool_counts as _tool_counts,
)


def build_final_report(
    metadata: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    status: str,
    reason: str,
    summary: str = "",
) -> dict[str, Any]:
    """Build the phase-two final report payload from persisted session data."""
    metadata = metadata or {}
    events = events or []
    return {
        "session_id": str(metadata.get("id", "")),
        "status": status,
        "reason": reason,
        "summary": _report_summary(summary, events),
        "statistics": {
            "events": len(events),
            "user_inputs": _count_events(events, "user_input"),
            "assistant_messages": _count_events(events, "assistant_message"),
            "tool_calls": _tool_result_count(events),
            "failures": _count_events(events, "failure"),
            "fallbacks": _count_events(events, "agent_fallback"),
            "latest_fallback": _latest_fallback_reason(events),
            "file_changes": len(_changed_files(events)),
        },
        "failure_categories": dict(_failure_categories(events)),
        "tool_counts": dict(_tool_counts(events)),
        "changed_files": _changed_files(events),
        "started_at": str(metadata.get("created_at", "")),
        "ended_at": datetime.now(UTC).isoformat(),
    }


def _tool_result_count(events: list[dict[str, Any]]) -> int:
    return _count_events(events, "tool_result")


def _report_summary(summary: str, events: list[dict[str, Any]]) -> str:
    text = (summary or "").strip()
    if not text:
        text = _latest_assistant_message(events)
    if not text:
        return "No assistant response recorded."
    text = " ".join(text.split())
    if len(text) > 500:
        return text[:497] + "..."
    return text


def _latest_assistant_message(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if _event_type(event) == "assistant_message":
            text = _payload(event).get("text")
            if text:
                return str(text)
    return ""


def _latest_fallback_reason(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if _event_type(event) == "agent_fallback":
            reason = _payload(event).get("reason")
            return str(reason or "unknown")
    return ""
