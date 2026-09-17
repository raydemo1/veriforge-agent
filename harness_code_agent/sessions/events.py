from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("harness")



FAILURE_CATEGORIES = {
    "tool_error",
    "runtime_error",
    "user_cancelled",
    "validation_error",
    "unknown",
}


@dataclass
class SessionEvent:
    sequence: int
    timestamp: float
    type: str
    agent: str | None
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "type": self.type,
            "agent": self.agent,
            "payload": self.payload,
        }


@dataclass(frozen=True)
class StructuredEvent:
    type: str
    payload: dict[str, Any]
    agent: str | None = None

    def to_event(self) -> SessionEvent:
        return SessionEvent(
            sequence=0,
            timestamp=0.0,
            type=self.type,
            agent=self.agent,
            payload=self.payload,
        )


@dataclass(frozen=True)
class UserInputEvent:
    text: str
    turn: int | None = None
    mentions: list[str] | None = None
    attachments: list[dict[str, Any]] | None = None
    agent: str | None = "main_agent"

    def to_event(self) -> StructuredEvent:
        payload: dict[str, Any] = {"text": self.text}
        if self.turn is not None:
            payload["turn"] = self.turn
        if self.mentions is not None:
            payload["mentions"] = self.mentions
        if self.attachments is not None:
            payload["attachments"] = self.attachments
        return StructuredEvent("user_input", payload, self.agent)


@dataclass(frozen=True)
class AssistantMessageEvent:
    text: str
    turn: int | None = None
    streamed: bool = False
    agent: str | None = "main_agent"

    def to_event(self) -> StructuredEvent:
        payload: dict[str, Any] = {"text": self.text}
        if self.turn is not None:
            payload["turn"] = self.turn
        if self.streamed:
            payload["streamed"] = self.streamed
        return StructuredEvent("assistant_message", payload, self.agent)


@dataclass(frozen=True)
class ToolCallEvent:
    tool: str
    args: dict[str, Any]
    agent: str | None = "main_agent"

    def to_event(self) -> StructuredEvent:
        return StructuredEvent("tool_call", {"tool": self.tool, "args": self.args}, self.agent)


@dataclass(frozen=True)
class ToolResultEvent:
    tool: str
    status: str | None = None
    output: str = ""
    error: str | None = None
    return_code: int | None = None
    metadata: dict[str, Any] | None = None
    ok: bool | None = None
    agent: str | None = "main_agent"

    def to_event(self) -> StructuredEvent:
        status = self.status or _tool_status_from_ok(self.ok)
        payload: dict[str, Any] = {
            "tool": self.tool,
            "status": status,
            "ok": _ok_from_tool_status(status),
            "output": self.output,
        }
        if self.error is not None:
            payload["error"] = self.error
        if self.return_code is not None:
            payload["return_code"] = self.return_code
        if self.metadata:
            payload["metadata"] = self.metadata
        return StructuredEvent("tool_result", payload, self.agent)


@dataclass(frozen=True)
class FileChangeEvent:
    path: str
    operation: str | None = None
    snapshot_path: str | None = None
    diff: str | None = None
    additions: int | None = None
    deletions: int | None = None
    agent: str | None = "main_agent"

    def to_event(self) -> StructuredEvent:
        payload: dict[str, Any] = {"path": self.path}
        if self.operation:
            payload["operation"] = self.operation
        if self.snapshot_path:
            payload["snapshot_path"] = self.snapshot_path
        if self.diff:
            payload["diff"] = self.diff
        if self.additions is not None:
            payload["additions"] = self.additions
        if self.deletions is not None:
            payload["deletions"] = self.deletions
        return StructuredEvent("file_change", payload, self.agent)


@dataclass(frozen=True)
class FailureEvent:
    category: str
    message: str
    tool: str | None = None
    source: str | None = None
    metadata: dict[str, Any] | None = None
    agent: str | None = "main_agent"

    def to_event(self) -> StructuredEvent:
        category = self.category if self.category in FAILURE_CATEGORIES else "unknown"
        payload: dict[str, Any] = {"category": category, "message": self.message}
        if self.tool:
            payload["tool"] = self.tool
        if self.source:
            payload["source"] = self.source
        if self.metadata:
            payload["metadata"] = self.metadata
        return StructuredEvent("failure", payload, self.agent)


@dataclass(frozen=True)
class FinalReportEvent:
    status: str
    reason: str
    summary: str
    session_id: str | None = None
    statistics: dict[str, Any] | None = None
    failure_categories: dict[str, int] | None = None
    tool_counts: dict[str, int] | None = None
    changed_files: list[str] | None = None
    started_at: str | None = None
    ended_at: str | None = None
    agent: str | None = "main_agent"

    def to_event(self) -> StructuredEvent:
        payload: dict[str, Any] = {
            "status": self.status,
            "reason": self.reason,
            "summary": self.summary,
            "statistics": dict(self.statistics or {}),
            "failure_categories": dict(self.failure_categories or {}),
            "tool_counts": dict(self.tool_counts or {}),
            "changed_files": list(self.changed_files or []),
        }
        if self.session_id:
            payload["session_id"] = self.session_id
        if self.started_at:
            payload["started_at"] = self.started_at
        if self.ended_at:
            payload["ended_at"] = self.ended_at
        return StructuredEvent("final_report", payload, self.agent)


@dataclass(frozen=True)
class SessionFinishedEvent:
    reason: str
    status: str
    agent: str | None = "main_agent"

    def to_event(self) -> StructuredEvent:
        return StructuredEvent(
            "session_finished",
            {"reason": self.reason, "status": self.status},
            self.agent,
        )


@dataclass(frozen=True)
class TaskOutcomeEvent:
    status: str
    evidence: list[str]
    summary: str
    agent: str | None = "main_agent"

    def to_event(self) -> StructuredEvent:
        return StructuredEvent(
            "task_outcome",
            {
                "status": self.status,
                "evidence": self.evidence,
                "summary": self.summary,
            },
            self.agent,
        )


@dataclass(frozen=True)
class TurnSummaryEvent:
    turn: int
    summary: str
    duration_seconds: float
    tool_counts: dict[str, int] | None = None
    changed_files: list[str] | None = None
    checkpoint: str = ""
    generated_by: dict[str, Any] | None = None
    long_task: bool = True
    fold_details: bool = True
    agent: str | None = "main_agent"

    def to_event(self) -> StructuredEvent:
        return StructuredEvent(
            "turn_summary",
            {
                "turn": self.turn,
                "summary": self.summary,
                "long_task": self.long_task,
                "fold_details": self.fold_details,
                "duration_seconds": self.duration_seconds,
                "tool_counts": dict(self.tool_counts or {}),
                "changed_files": list(self.changed_files or []),
                "checkpoint": self.checkpoint,
                "generated_by": dict(self.generated_by or {}),
            },
            self.agent,
        )


@dataclass(frozen=True)
class AgentBudgetWarningEvent:
    limit_type: str
    used: int
    limit: int
    fraction: float | None = None
    agent: str | None = "main_agent"

    def to_event(self) -> StructuredEvent:
        payload: dict[str, Any] = {
            "limit_type": self.limit_type,
            "used": self.used,
            "limit": self.limit,
        }
        if self.fraction is not None:
            payload["fraction"] = self.fraction
        return StructuredEvent("agent_budget_warning", payload, self.agent)


@dataclass(frozen=True)
class AgentFallbackEvent:
    reason: str
    limit_type: str | None = None
    used: int | None = None
    limit: int | None = None
    last_tool: str | None = None
    fingerprint_hash: str | None = None
    recent_action_summary: list[str] | None = None
    agent: str | None = "main_agent"

    def to_event(self) -> StructuredEvent:
        payload: dict[str, Any] = {"reason": self.reason}
        if self.limit_type:
            payload["limit_type"] = self.limit_type
        if self.used is not None:
            payload["used"] = self.used
        if self.limit is not None:
            payload["limit"] = self.limit
        if self.last_tool:
            payload["last_tool"] = self.last_tool
        if self.fingerprint_hash:
            payload["fingerprint_hash"] = self.fingerprint_hash
        if self.recent_action_summary:
            payload["recent_action_summary"] = list(self.recent_action_summary)
        return StructuredEvent("agent_fallback", payload, self.agent)


@dataclass(frozen=True)
class LlmUsageEvent:
    provider: str
    model: str
    prompt_tokens: int | None = None
    cached_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_cache_key_hash: str | None = None
    cache_diagnostics: dict[str, Any] | None = None
    agent: str | None = "main_agent"

    def to_event(self) -> StructuredEvent:
        cache_hit_tokens = self.cache_hit_tokens or self.cached_tokens
        ratio = 0.0
        if self.prompt_tokens is not None and self.prompt_tokens > 0:
            ratio = cache_hit_tokens / self.prompt_tokens
        payload: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": cache_hit_tokens,
            "cache_hit_tokens": cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cache_hit_ratio": ratio,
        }
        if self.prompt_cache_key_hash:
            payload["prompt_cache_key_hash"] = self.prompt_cache_key_hash
        if self.cache_diagnostics is not None:
            payload["cache_diagnostics"] = dict(self.cache_diagnostics)
        return StructuredEvent("llm_usage", payload, self.agent)


@dataclass(frozen=True)
class LlmRequestStartedEvent:
    call_id: str
    provider: str
    model: str
    streamed: bool = False
    agent: str | None = "main_agent"

    def to_event(self) -> StructuredEvent:
        return StructuredEvent(
            "llm_request_started",
            {
                "call_id": self.call_id,
                "provider": self.provider,
                "model": self.model,
                "streamed": self.streamed,
            },
            self.agent,
        )


@dataclass(frozen=True)
class LlmFirstTokenEvent:
    call_id: str
    elapsed_ms: int
    provider: str
    model: str
    agent: str | None = "main_agent"

    def to_event(self) -> StructuredEvent:
        return StructuredEvent(
            "llm_first_token",
            {
                "call_id": self.call_id,
                "elapsed_ms": self.elapsed_ms,
                "provider": self.provider,
                "model": self.model,
            },
            self.agent,
        )


@dataclass(frozen=True)
class LlmResponseFinishedEvent:
    call_id: str
    duration_ms: int
    provider: str
    model: str
    streamed: bool = False
    finish_reason: str | None = None
    first_token_ms: int | None = None
    agent: str | None = "main_agent"

    def to_event(self) -> StructuredEvent:
        payload: dict[str, Any] = {
            "call_id": self.call_id,
            "duration_ms": self.duration_ms,
            "provider": self.provider,
            "model": self.model,
            "streamed": self.streamed,
        }
        if self.finish_reason is not None:
            payload["finish_reason"] = self.finish_reason
        if self.first_token_ms is not None:
            payload["first_token_ms"] = self.first_token_ms
        return StructuredEvent("llm_response_finished", payload, self.agent)


@dataclass(frozen=True)
class ContextCompactionStartedEvent:
    token_count: int
    threshold: int
    forced: bool = False
    phase: str = ""
    agent: str | None = "main_agent"

    def to_event(self) -> StructuredEvent:
        return StructuredEvent(
            "context_compaction_started",
            {
                "token_count": self.token_count,
                "threshold": self.threshold,
                "forced": self.forced,
                "phase": self.phase,
            },
            self.agent,
        )


@dataclass(frozen=True)
class ContextCompactionCommittedEvent:
    summary_chars: int
    messages_before: int
    messages_after: int
    tokens_saved: int = 0
    agent: str | None = "main_agent"

    def to_event(self) -> StructuredEvent:
        return StructuredEvent(
            "context_compaction_committed",
            {
                "summary_chars": self.summary_chars,
                "messages_before": self.messages_before,
                "messages_after": self.messages_after,
                "tokens_saved": self.tokens_saved,
            },
            self.agent,
        )


@dataclass(frozen=True)
class ContextAnxietyObservedEvent:
    token_count: int
    threshold: int
    score: int
    reasons: list[str]
    source: str = "assistant_recent_messages"
    agent: str | None = "main_agent"

    def to_event(self) -> StructuredEvent:
        return StructuredEvent(
            "context_anxiety_observed",
            {
                "token_count": self.token_count,
                "threshold": self.threshold,
                "score": self.score,
                "reasons": list(self.reasons),
                "source": self.source,
            },
            self.agent,
        )


@dataclass(frozen=True)
class ThoughtStartedEvent:
    agent: str | None = "main_agent"

    def to_event(self) -> StructuredEvent:
        return StructuredEvent("thought_started", {}, self.agent)


@dataclass(frozen=True)
class ThoughtFinishedEvent:
    duration_seconds: float
    truncated: bool = False
    source: str = ""
    agent: str | None = "main_agent"

    def to_event(self) -> StructuredEvent:
        return StructuredEvent(
            "thought_finished",
            {
                "duration_seconds": self.duration_seconds,
                "truncated": self.truncated,
                "source": self.source,
            },
            self.agent,
        )


class EventBus:
    """Append-only event stream for product runtime observability."""

    def __init__(
        self,
        events_path: str | Path | None = None,
        *,
        listener: Callable[[SessionEvent], None] | None = None,
    ):
        self.events_path = Path(events_path) if events_path is not None else None
        self.listener = listener
        self.events: list[SessionEvent] = []
        self._sequence = 0
        if self.events_path is not None:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        event_type: str,
        *,
        agent: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> SessionEvent:
        self._sequence += 1
        event = SessionEvent(
            sequence=self._sequence,
            timestamp=time.time(),
            type=event_type,
            agent=agent,
            payload=payload or {},
        )
        self.events.append(event)
        if self.events_path is not None:
            with self.events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        if self.listener is not None:
            try:
                self.listener(event)
            except Exception as exc:
                log.debug("EventBus listener error: %s", exc)

        return event

    def emit_event(self, structured_event: StructuredEvent) -> SessionEvent:
        event = structured_event.to_event()
        return self.emit(
            event.type,
            agent=event.agent,
            payload=event.payload,
        )


def _tool_status_from_ok(ok: bool | None) -> str:
    if ok is True:
        return "success"
    if ok is False:
        return "failed"
    return "unknown"


def _ok_from_tool_status(status: str) -> bool | None:
    if status == "success":
        return True
    if status == "failed":
        return False
    return None


def classify_tool_failure(tool_result: Any) -> str:
    """Classify failed tool results into the phase-two taxonomy."""
    status = str(getattr(tool_result, "status", "") or "")
    if status and status != "failed":
        return "unknown"

    metadata = getattr(tool_result, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    source = str(metadata.get("status_source", "") or "").lower()
    output = str(getattr(tool_result, "output", "") or "")
    error = str(getattr(tool_result, "error", "") or "")
    combined = f"{output}\n{error}".lower()

    # Canonical failure taxonomy (ToolFailure) takes precedence; the event
    # category enum is kept stable for existing consumers.
    canonical_category = metadata.get("failure_category")
    if isinstance(canonical_category, str) and canonical_category:
        kind = str(metadata.get("failure_kind", "") or "")
        if canonical_category == "invalid_call":
            return "validation_error"
        if canonical_category == "policy":
            return "user_cancelled" if kind == "approval_denied" else "tool_error"
        if canonical_category == "resource":
            return "runtime_error"
        if canonical_category == "execution":
            return "runtime_error" if source in {"runtime", "exception", "shell"} else "tool_error"
        if canonical_category == "verification":
            return "runtime_error"

    if source == "validation" or "validation" in combined or "empty file path" in combined:
        return "validation_error"
    if source == "approval" or "approval_denied" in combined or "cancelled" in combined or "canceled" in combined:
        return "user_cancelled"
    if source in {"runtime", "exception", "shell"}:
        return "runtime_error"
    if source in {"native", "registry", "permission", "browser"}:
        return "tool_error"
    if "traceback" in combined or "exception" in combined or "command exited with code" in combined:
        return "runtime_error"
    if str(getattr(tool_result, "tool", "") or ""):
        return "unknown"
    return "unknown"
