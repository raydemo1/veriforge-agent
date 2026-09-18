from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .. import config
from ..agent.providers import ProviderAdapter, client_scope
from ._event_helpers import changed_files as _changed_files
from ._event_helpers import event_type as _event_type
from ._event_helpers import payload as _payload
from ._event_helpers import tool_counts as _tool_counts

log = logging.getLogger("harness")

LONG_TURN_TOOL_RESULT_THRESHOLD = 3
LONG_TURN_SECONDS_THRESHOLD = 45.0
LONG_TURN_TOOLS = {"run_bash", "browser_test"}


@dataclass(frozen=True)
class TurnSummaryResult:
    summary: str
    tool_counts: dict[str, int]
    changed_files: list[str]
    generated_by: dict[str, Any]


def should_summarize_turn(
    events: list[dict[str, Any]],
    *,
    profile_name: str,
    duration_seconds: float,
) -> bool:
    """Return whether a turn is long enough to summarize for TUI display."""
    if profile_name == "plan":
        return False

    counts = _tool_counts(events)
    if sum(counts.values()) >= LONG_TURN_TOOL_RESULT_THRESHOLD:
        return True
    if _changed_files(events):
        return True
    if any(tool in LONG_TURN_TOOLS for tool in counts):
        return True
    if any(_event_type(event) == "agent_fallback" for event in events):
        return True
    return duration_seconds >= LONG_TURN_SECONDS_THRESHOLD


def generate_turn_summary(
    events: list[dict[str, Any]],
    *,
    user_prompt: str,
    assistant_text: str,
    checkpoint: str,
    llm_create: Callable[..., Any] | None = None,
) -> TurnSummaryResult:
    """Generate a concise summary using the configured fast model profile."""
    profile = config.resolve_model_profile("fast")
    generated_by = {
        "intensity": "fast",
        "provider": profile.provider,
        "model": profile.model,
        "thinking": profile.thinking,
        "reasoning_effort": profile.reasoning_effort,
    }
    counts = dict(_tool_counts(events))
    files = _changed_files(events)
    fallback_summary = _fallback_summary(
        user_prompt=user_prompt,
        assistant_text=assistant_text,
        tool_counts=counts,
        changed_files=files,
        checkpoint=checkpoint,
    )

    try:
        messages = _summary_messages(
            events,
            user_prompt=user_prompt,
            assistant_text=assistant_text,
            checkpoint=checkpoint,
        )
        adapter = ProviderAdapter(profile.provider)
        kwargs = adapter.chat_kwargs(
            profile=profile,
            messages=messages,
            max_tokens=800,
        )
        if llm_create is not None:
            response = llm_create(**kwargs)
        else:
            with client_scope() as client:
                response = client.chat.completions.create(**kwargs)
        summary = str(response.choices[0].message.content or "").strip()
        if not summary:
            summary = fallback_summary
    except Exception as exc:
        log.warning("Failed to generate turn summary: %s", exc)
        summary = fallback_summary

    return TurnSummaryResult(
        summary=summary,
        tool_counts=counts,
        changed_files=files,
        generated_by=generated_by,
    )


def _summary_messages(
    events: list[dict[str, Any]],
    *,
    user_prompt: str,
    assistant_text: str,
    checkpoint: str,
) -> list[dict[str, str]]:
    facts = {
        "user_prompt": _truncate(user_prompt, 2000),
        "assistant_text": _truncate(assistant_text, 3000),
        "tool_counts": dict(_tool_counts(events)),
        "changed_files": _changed_files(events),
        "failures": _failure_summaries(events),
        "checkpoint": checkpoint,
    }
    return [
        {
            "role": "system",
            "content": (
                "Summarize the completed coding turn for the user. Use the same language "
                "as the user prompt. Return 4-6 short bullet points. Summarize only "
                "observable facts: work done, files changed, validation, remaining issues. "
                "Do not mention hidden reasoning or chain of thought."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(facts, ensure_ascii=False, indent=2),
        },
    ]


def _fallback_summary(
    *,
    user_prompt: str,
    assistant_text: str,
    tool_counts: dict[str, int],
    changed_files: list[str],
    checkpoint: str,
) -> str:
    lines = []
    if user_prompt:
        lines.append(f"- Task: {_one_line(user_prompt, 140)}")
    if assistant_text:
        lines.append(f"- Result: {_one_line(assistant_text, 180)}")
    if changed_files:
        lines.append("- Changed files: " + ", ".join(changed_files[:8]))
    if tool_counts:
        details = ", ".join(f"{name}={count}" for name, count in sorted(tool_counts.items()))
        lines.append(f"- Tools used: {details}")
    if checkpoint:
        lines.append(f"- Checkpoint: {checkpoint}")
    return "\n".join(lines) if lines else "- Turn completed."


def _failure_summaries(events: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for event in events:
        if _event_type(event) != "failure":
            continue
        payload = _payload(event)
        category = payload.get("category") or "unknown"
        message = _one_line(str(payload.get("message") or ""), 180)
        failures.append(f"{category}: {message}" if message else str(category))
    return failures


def _one_line(text: str, limit: int) -> str:
    return _truncate(" ".join(str(text).split()), limit)


def _truncate(text: str, limit: int) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."
