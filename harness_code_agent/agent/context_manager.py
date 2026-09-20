from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from . import context

SUMMARY_MARKER = "[COMPACTED CONTEXT — structured working state]"
SUMMARY_HEADINGS = (
    "Task goal",
    "User constraints",
    "Completed work",
    "Changed files",
    "Validation",
    "Unresolved issues",
    "Next action",
    "Evidence references",
)


@dataclass(frozen=True)
class CompactionResult:
    messages: list[dict]
    summary: str
    first_kept_index: int


class ContextManager:
    """Single compaction boundary for manual and automatic context management."""

    def compact(
        self,
        messages: list[dict],
        llm_call: Callable,
        *,
        current_turn_start_index: int,
        state: dict,
        force: bool = False,
    ) -> CompactionResult | None:
        if not messages:
            return None
        system = [messages[0]] if messages[0].get("role") == "system" else []
        system_len = len(system)
        split = max(system_len, min(current_turn_start_index, len(messages)))
        older, current = messages[system_len:split], messages[split:]
        if not older or (not force and not context._compaction_economics(older)):
            return None
        prompt = (
            "Summarize this coding-agent history as durable working state. Return exactly these Markdown headings:\n"
            "## Task goal\n## User constraints\n## Completed work\n## Changed files\n"
            "## Validation\n## Unresolved issues\n## Next action\n## Evidence references\n"
            "Preserve concrete paths, commands, failures, decisions, tool-result relationships, and current status. "
            "Exclude hidden reasoning, repeated logs, and resolved branches. Do not claim unverified success.\n\n"
            "Runtime state:\n" + _state_text(state) + "\n\nHistory:\n" + context._messages_to_text(older)
        )
        try:
            summary = llm_call([
                {"role": "system", "content": "You create precise recovery state for a coding agent."},
                {"role": "user", "content": prompt},
            ])
        except Exception:  # noqa: BLE001 - compaction failure must preserve the original context
            return None
        if not isinstance(summary, str) or not summary.strip():
            return None
        summary = _normalize_summary(summary)
        return CompactionResult(
            messages=system + [{"role": "user", "content": f"{SUMMARY_MARKER}\n{summary}"}] + current,
            summary=summary,
            first_kept_index=split,
        )

    @staticmethod
    def breakdown(messages: list[dict], tool_schemas: list[dict] | None = None) -> dict[str, int]:
        result = {"system": 0, "summary": 0, "recent": 0, "memory": 0, "tools": 0}
        for index, message in enumerate(messages):
            tokens = context.count_tokens([message])
            content = str(message.get("content") or "")
            if index == 0 and message.get("role") == "system":
                if "[HARNESS_MEMORY_INDEX]" in content:
                    stable, memory = content.split("[HARNESS_MEMORY_INDEX]", 1)
                    result["system"] += context.count_text_tokens(stable)
                    result["memory"] += context.count_text_tokens("[HARNESS_MEMORY_INDEX]" + memory)
                else:
                    result["system"] += tokens
            elif SUMMARY_MARKER in content:
                result["summary"] += tokens
            elif "Relevant long-term memory" in content:
                result["memory"] += tokens
            else:
                result["recent"] += tokens
        if tool_schemas:
            result["tools"] = context.count_request_tokens([], tool_schemas=tool_schemas)
        return result


def _state_text(state: dict) -> str:
    return "\n".join(
        f"- {key}: {value}" for key, value in state.items() if value not in (None, "", [], {})
    )


def _normalize_summary(summary: str) -> str:
    """Keep recovery summaries structurally stable even when a model omits a section."""
    sections: dict[str, list[str]] = {heading: [] for heading in SUMMARY_HEADINGS}
    current: str | None = None
    preface: list[str] = []
    for line in summary.strip().splitlines():
        if line.startswith("## ") and line[3:].strip() in sections:
            current = line[3:].strip()
        elif current is None:
            preface.append(line)
        else:
            sections[current].append(line)
    if preface and not sections["Task goal"]:
        sections["Task goal"] = preface
    lines: list[str] = []
    for heading in SUMMARY_HEADINGS:
        body = "\n".join(sections[heading]).strip() or "- None recorded."
        lines.extend((f"## {heading}", body))
    return "\n".join(lines)
