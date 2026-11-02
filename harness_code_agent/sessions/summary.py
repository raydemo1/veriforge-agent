from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

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
    failure_categories as _event_failure_categories,
)
from ._event_helpers import (
    payload as _payload,
)
from ._event_helpers import (
    tool_counts as _tool_counts,
)

if TYPE_CHECKING:
    from .store import SessionStore


def load_session_summary(store: SessionStore, session_id: str) -> str:
    metadata = store.read_metadata(session_id)
    events = store.read_events(session_id)
    return format_session_summary(metadata, events, session_id=session_id)


def format_session_summary(
    metadata: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    session_id: str | None = None,
    recent_limit: int = 5,
) -> str:
    """Render a compact human-readable summary from durable session data."""
    metadata = metadata or {}
    events = events or []
    resolved_id = metadata.get("id") or session_id or ""
    status = _derived_status(metadata, events)
    user_inputs = _count_events(events, "user_input")
    assistant_messages = _count_events(events, "assistant_message")
    turn_started = _count_events(events, "turn_started")
    turn_finished = _count_events(events, "turn_finished")
    tool_counts = _tool_counts(events)
    changed_files = _changed_files(events)
    approvals = _approval_counts(events)
    switches = _profile_switches(events)
    failures = _count_events(events, "failure")
    failure_categories = _failure_categories(events)
    fallbacks = _count_events(events, "agent_fallback")
    latest_fallback = _latest_fallback_reason(events)
    final_report = _latest_final_report(events)
    plans_ready = _count_events(events, "plan_ready")
    task_outcome = _latest_task_outcome(events)

    lines = [
        "Session summary",
        f"id: {resolved_id}",
        f"profile: {metadata.get('profile', '')}",
        f"model: {metadata.get('model', '')}",
        f"permission_mode: {metadata.get('permission_mode', '')}",
        f"status: {status}",
        f"cwd: {metadata.get('cwd', '')}",
        f"created_at: {metadata.get('created_at', '')}",
    ]
    if metadata.get("forked_from"):
        lines.append(f"forked_from: {metadata.get('forked_from')}")

    lines.extend([
        f"events: {len(events)}",
        f"user_inputs: {user_inputs}",
        f"assistant_messages: {assistant_messages}",
        f"turns: {turn_started} started, {turn_finished} finished",
        f"tools: {_format_tool_counts(tool_counts)}",
        f"changed_files: {_format_list(changed_files)} ({len(changed_files)})",
        f"failures: {failures}",
        f"failure_categories: {_format_counts(failure_categories)}",
        f"fallbacks: {_format_fallbacks(fallbacks, latest_fallback)}",
        (
            "approvals: "
            f"{approvals['requested']} requested, "
            f"{approvals['approved']} approved, "
            f"{approvals['denied']} denied"
        ),
        f"profile_switches: {_format_list(switches)}",
        f"plans_ready: {plans_ready}",
    ])
    if final_report:
        lines.append(f"final_report: {_format_final_report(final_report)}")
    lines.extend([
        f"task_outcome: {task_outcome}",
        "recent_events:",
    ])

    recent = events[-recent_limit:] if recent_limit > 0 else []
    if recent:
        lines.extend(f"- {_event_summary(event)}" for event in recent)
    else:
        lines.append("- unknown")
    return "\n".join(lines)


def _derived_status(metadata: dict[str, Any], events: list[dict[str, Any]]) -> str:
    latest_finished = _latest_session_finished_payload(events)
    if latest_finished.get("status"):
        return str(latest_finished["status"])
    latest_report = _latest_final_report(events)
    if latest_report.get("status"):
        return str(latest_report["status"])
    return str(metadata.get("status", ""))


def _approval_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"requested": 0, "approved": 0, "denied": 0}
    for event in events:
        event_type = _event_type(event)
        if event_type == "approval_requested":
            counts["requested"] += 1
        elif event_type == "approval_decided":
            if _payload(event).get("approved") is True:
                counts["approved"] += 1
            else:
                counts["denied"] += 1
    return counts


def _failure_categories(events: list[dict[str, Any]]) -> Counter[str]:
    latest_report = _latest_final_report(events)
    report_categories = latest_report.get("failure_categories")
    if isinstance(report_categories, dict) and report_categories:
        return Counter({str(key): int(value) for key, value in report_categories.items()})

    return _event_failure_categories(events)


def _profile_switches(events: list[dict[str, Any]]) -> list[str]:
    switches: list[str] = []
    for event in events:
        if _event_type(event) != "profile_switched":
            continue
        payload = _payload(event)
        previous = payload.get("previous_profile", "")
        current = payload.get("profile", "")
        reason = payload.get("reason", "")
        text = f"{previous} -> {current}".strip()
        if reason:
            text = f"{text} ({reason})"
        switches.append(text)
    return switches


def _latest_task_outcome(events: list[dict[str, Any]]) -> str:
    payload = _latest_task_outcome_payload(events)
    status = str(payload.get("status", "")).strip()
    summary = str(payload.get("summary", "")).strip()
    if status and summary:
        return f"{status} - {summary}"
    if status:
        return status
    return "unknown"


def _latest_task_outcome_payload(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        if _event_type(event) == "task_outcome":
            return _payload(event)
    return {}


def _latest_final_report(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        if _event_type(event) == "final_report":
            return _payload(event)
    return {}


def _latest_fallback_reason(events: list[dict[str, Any]]) -> str:
    latest_report = _latest_final_report(events)
    statistics = latest_report.get("statistics")
    if isinstance(statistics, dict) and statistics.get("latest_fallback"):
        return str(statistics["latest_fallback"])
    for event in reversed(events):
        if _event_type(event) == "agent_fallback":
            reason = _payload(event).get("reason")
            return str(reason or "unknown")
    return ""


def _latest_session_finished_payload(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        if _event_type(event) == "session_finished":
            return _payload(event)
    return {}


def _format_tool_counts(counts: Counter[str]) -> str:
    total = sum(counts.values())
    if not counts:
        return "0 call(s)"
    details = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    return f"{total} call(s): {details}"


def _format_counts(counts: Counter[str]) -> str:
    if not counts:
        return "unknown"
    return ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))


def _format_fallbacks(count: int, latest_reason: str) -> str:
    if count <= 0:
        return "0"
    if latest_reason:
        return f"{count} (latest: {latest_reason})"
    return str(count)


def _format_final_report(payload: dict[str, Any]) -> str:
    status = str(payload.get("status", "")).strip()
    summary = str(payload.get("summary", "")).strip()
    if status and summary:
        return f"{status} - {summary}"
    if status:
        return status
    if summary:
        return summary
    return "unknown"


def _format_list(items: list[str]) -> str:
    return ", ".join(items) if items else "unknown"


def _event_summary(event: dict[str, Any]) -> str:
    payload = _payload(event)
    payload_bits = []
    for key in sorted(payload)[:4]:
        value = payload[key]
        text = str(value).replace("\n", " ")
        if len(text) > 80:
            text = text[:77] + "..."
        payload_bits.append(f"{key}={text}")
    suffix = f" ({', '.join(payload_bits)})" if payload_bits else ""
    return f"#{event.get('sequence')} {_event_type(event)} agent={event.get('agent')}{suffix}"
