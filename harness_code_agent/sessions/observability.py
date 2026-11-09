from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._event_helpers import changed_files as _changed_files
from ._event_helpers import event_type as _event_type
from ._event_helpers import payload as _payload

if TYPE_CHECKING:
    from .store import SessionStore


@dataclass
class TokenMetrics:
    llm_calls: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    cache_miss_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def cache_hit_ratio(self) -> float:
        if self.prompt_tokens <= 0:
            return 0.0
        return self.cached_tokens / self.prompt_tokens

    def add(self, other: TokenMetrics) -> None:
        self.llm_calls += other.llm_calls
        self.prompt_tokens += other.prompt_tokens
        self.cached_tokens += other.cached_tokens
        self.cache_miss_tokens += other.cache_miss_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cache_hit_ratio": self.cache_hit_ratio,
        }


@dataclass
class ToolBreakdown:
    calls: int = 0
    results: int = 0
    successes: int = 0
    failures: int = 0
    unknown: int = 0

    @property
    def pending_calls(self) -> int:
        return max(0, self.calls - self.results)

    def add(self, other: ToolBreakdown) -> None:
        self.calls += other.calls
        self.results += other.results
        self.successes += other.successes
        self.failures += other.failures
        self.unknown += other.unknown

    def to_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "results": self.results,
            "successes": self.successes,
            "failures": self.failures,
            "unknown": self.unknown,
            "pending_calls": self.pending_calls,
        }


@dataclass
class ToolMetrics:
    tool_calls: int = 0
    tool_results: int = 0
    successes: int = 0
    failures: int = 0
    unknown: int = 0
    by_tool: dict[str, ToolBreakdown] = field(default_factory=dict)

    @property
    def pending_calls(self) -> int:
        return sum(item.pending_calls for item in self.by_tool.values())

    @property
    def success_rate(self) -> float:
        if self.tool_results <= 0:
            return 0.0
        return self.successes / self.tool_results

    def add(self, other: ToolMetrics) -> None:
        self.tool_calls += other.tool_calls
        self.tool_results += other.tool_results
        self.successes += other.successes
        self.failures += other.failures
        self.unknown += other.unknown
        for tool, breakdown in other.by_tool.items():
            self.by_tool.setdefault(tool, ToolBreakdown()).add(breakdown)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "successes": self.successes,
            "failures": self.failures,
            "unknown": self.unknown,
            "pending_calls": self.pending_calls,
            "success_rate": self.success_rate,
            "by_tool": {
                name: breakdown.to_dict()
                for name, breakdown in sorted(self.by_tool.items())
            },
        }


@dataclass
class DistributionMetrics:
    values: list[int] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def min(self) -> int:
        return min(self.values) if self.values else 0

    @property
    def max(self) -> int:
        return max(self.values) if self.values else 0

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    @property
    def p50(self) -> int:
        return _percentile(self.values, 0.50)

    @property
    def p95(self) -> int:
        return _percentile(self.values, 0.95)

    @property
    def p99(self) -> int:
        return _percentile(self.values, 0.99)

    def add_value(self, value: float | None) -> None:
        if value is None:
            return
        try:
            numeric = round(float(value))
        except (TypeError, ValueError):
            return
        if numeric >= 0:
            self.values.append(numeric)

    def add(self, other: DistributionMetrics) -> None:
        self.values.extend(other.values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
        }


@dataclass
class PerformanceMetrics:
    llm_response_latency_ms: DistributionMetrics = field(default_factory=DistributionMetrics)
    llm_first_token_ms: DistributionMetrics = field(default_factory=DistributionMetrics)
    turn_duration_ms: DistributionMetrics = field(default_factory=DistributionMetrics)

    def add(self, other: PerformanceMetrics) -> None:
        self.llm_response_latency_ms.add(other.llm_response_latency_ms)
        self.llm_first_token_ms.add(other.llm_first_token_ms)
        self.turn_duration_ms.add(other.turn_duration_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_response_latency_ms": self.llm_response_latency_ms.to_dict(),
            "llm_first_token_ms": self.llm_first_token_ms.to_dict(),
            "turn_duration_ms": self.turn_duration_ms.to_dict(),
        }


@dataclass
class AuditMetrics:
    failures: int = 0
    failure_categories: Counter[str] = field(default_factory=Counter)
    fallbacks: int = 0
    latest_fallback: str = ""
    compactions_started: int = 0
    compactions_committed: int = 0
    tokens_saved: int = 0
    context_anxiety_observed: int = 0
    approvals_requested: int = 0
    approvals_approved: int = 0
    approvals_denied: int = 0
    changed_files: list[str] = field(default_factory=list)

    def add(self, other: AuditMetrics) -> None:
        self.failures += other.failures
        self.failure_categories.update(other.failure_categories)
        self.fallbacks += other.fallbacks
        if other.latest_fallback:
            self.latest_fallback = other.latest_fallback
        self.compactions_started += other.compactions_started
        self.compactions_committed += other.compactions_committed
        self.tokens_saved += other.tokens_saved
        self.context_anxiety_observed += other.context_anxiety_observed
        self.approvals_requested += other.approvals_requested
        self.approvals_approved += other.approvals_approved
        self.approvals_denied += other.approvals_denied
        for path in other.changed_files:
            if path not in self.changed_files:
                self.changed_files.append(path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "failures": self.failures,
            "failure_categories": dict(sorted(self.failure_categories.items())),
            "fallbacks": self.fallbacks,
            "latest_fallback": self.latest_fallback,
            "compactions_started": self.compactions_started,
            "compactions_committed": self.compactions_committed,
            "tokens_saved": self.tokens_saved,
            "context_anxiety_observed": self.context_anxiety_observed,
            "approvals_requested": self.approvals_requested,
            "approvals_approved": self.approvals_approved,
            "approvals_denied": self.approvals_denied,
            "changed_files": list(self.changed_files),
        }


@dataclass
class SessionObservability:
    session_id: str
    model: str = ""
    status: str = ""
    cwd: str = ""
    created_at: str = ""
    tokens: TokenMetrics = field(default_factory=TokenMetrics)
    tools: ToolMetrics = field(default_factory=ToolMetrics)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    audit: AuditMetrics = field(default_factory=AuditMetrics)
    recent_events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "model": self.model,
            "status": self.status,
            "cwd": self.cwd,
            "created_at": self.created_at,
            "tokens": self.tokens.to_dict(),
            "tools": self.tools.to_dict(),
            "performance": self.performance.to_dict(),
            "audit": self.audit.to_dict(),
            "recent_events": list(self.recent_events),
        }


@dataclass
class ProjectObservability:
    session_count: int = 0
    tokens: TokenMetrics = field(default_factory=TokenMetrics)
    tools: ToolMetrics = field(default_factory=ToolMetrics)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    audit: AuditMetrics = field(default_factory=AuditMetrics)
    sessions: list[SessionObservability] = field(default_factory=list)
    top_token_sessions: list[SessionObservability] = field(default_factory=list)
    top_failure_sessions: list[SessionObservability] = field(default_factory=list)
    low_cache_sessions: list[SessionObservability] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_count": self.session_count,
            "tokens": self.tokens.to_dict(),
            "tools": self.tools.to_dict(),
            "performance": self.performance.to_dict(),
            "audit": self.audit.to_dict(),
            "sessions": [session.to_dict() for session in self.sessions],
            "top_token_sessions": [session.to_dict() for session in self.top_token_sessions],
            "top_failure_sessions": [session.to_dict() for session in self.top_failure_sessions],
            "low_cache_sessions": [session.to_dict() for session in self.low_cache_sessions],
        }


@dataclass(frozen=True)
class ExportResult:
    markdown_path: Path
    json_path: Path


def build_session_observability(metadata: dict[str, Any], events: list[dict[str, Any]]) -> SessionObservability:
    metadata = metadata or {}
    events = events or []
    snapshot = SessionObservability(
        session_id=str(metadata.get("id", "")),
        model=str(metadata.get("model", "")),
        status=str(metadata.get("status", "")),
        cwd=str(metadata.get("cwd", "")),
        created_at=str(metadata.get("created_at", "")),
        recent_events=_recent_audit_events(events),
    )

    tool_breakdowns: defaultdict[str, ToolBreakdown] = defaultdict(ToolBreakdown)
    first_token_call_ids: set[str] = set()
    for event in events:
        event_name = _event_type(event)
        data = _payload(event)
        if event_name == "llm_usage":
            snapshot.tokens.llm_calls += 1
            snapshot.tokens.prompt_tokens += _int_value(data.get("prompt_tokens"))
            snapshot.tokens.cached_tokens += _int_value(data.get("cached_tokens"))
            snapshot.tokens.cache_miss_tokens += _int_value(data.get("cache_miss_tokens"))
            snapshot.tokens.completion_tokens += _int_value(data.get("completion_tokens"))
            snapshot.tokens.total_tokens += _int_value(data.get("total_tokens"))
        elif event_name == "tool_call":
            tool = _tool_name(data)
            snapshot.tools.tool_calls += 1
            tool_breakdowns[tool].calls += 1
        elif event_name == "tool_result":
            tool = _tool_name(data)
            status = str(data.get("status") or "unknown")
            snapshot.tools.tool_results += 1
            breakdown = tool_breakdowns[tool]
            breakdown.results += 1
            if status == "success":
                snapshot.tools.successes += 1
                breakdown.successes += 1
            elif status == "failed":
                snapshot.tools.failures += 1
                breakdown.failures += 1
            else:
                snapshot.tools.unknown += 1
                breakdown.unknown += 1
        elif event_name == "failure":
            snapshot.audit.failures += 1
            category = str(data.get("category") or "unknown")
            snapshot.audit.failure_categories[category] += 1
        elif event_name == "agent_fallback":
            snapshot.audit.fallbacks += 1
            snapshot.audit.latest_fallback = str(data.get("reason") or "unknown")
        elif event_name == "context_compaction_started":
            snapshot.audit.compactions_started += 1
        elif event_name == "context_compaction_committed":
            snapshot.audit.compactions_committed += 1
            snapshot.audit.tokens_saved += _int_value(data.get("tokens_saved"))
        elif event_name == "context_anxiety_observed":
            snapshot.audit.context_anxiety_observed += 1
        elif event_name == "llm_response_finished":
            snapshot.performance.llm_response_latency_ms.add_value(data.get("duration_ms"))
            call_id = str(data.get("call_id") or "")
            if call_id not in first_token_call_ids:
                snapshot.performance.llm_first_token_ms.add_value(data.get("first_token_ms"))
        elif event_name == "llm_first_token":
            snapshot.performance.llm_first_token_ms.add_value(data.get("elapsed_ms"))
            call_id = str(data.get("call_id") or "")
            if call_id:
                first_token_call_ids.add(call_id)
        elif event_name == "turn_finished":
            duration_seconds = data.get("duration_seconds")
            if duration_seconds is not None:
                try:
                    snapshot.performance.turn_duration_ms.add_value(float(duration_seconds) * 1000)
                except (TypeError, ValueError):
                    pass
        elif event_name == "approval_requested":
            snapshot.audit.approvals_requested += 1
        elif event_name == "approval_decided":
            if data.get("approved") is True:
                snapshot.audit.approvals_approved += 1
            else:
                snapshot.audit.approvals_denied += 1

    snapshot.tools.by_tool = dict(tool_breakdowns)
    snapshot.audit.changed_files = _changed_files(events)
    return snapshot


def build_project_observability(store: SessionStore) -> ProjectObservability:
    project = ProjectObservability()
    sessions: list[SessionObservability] = []
    for metadata in store.list_sessions():
        session_id = str(metadata.get("id", ""))
        if not session_id:
            continue
        try:
            events = store.read_events(session_id)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        snapshot = build_session_observability(metadata, events)
        sessions.append(snapshot)
        project.tokens.add(snapshot.tokens)
        project.tools.add(snapshot.tools)
        project.performance.add(snapshot.performance)
        project.audit.add(snapshot.audit)

    project.sessions = sessions
    project.session_count = len(sessions)
    project.top_token_sessions = sorted(
        sessions,
        key=lambda item: item.tokens.total_tokens,
        reverse=True,
    )[:5]
    project.top_failure_sessions = sorted(
        sessions,
        key=lambda item: item.audit.failures,
        reverse=True,
    )[:5]
    project.low_cache_sessions = sorted(
        [session for session in sessions if session.tokens.prompt_tokens > 0],
        key=lambda item: item.tokens.cache_hit_ratio,
    )[:5]
    return project


def format_session_observability(store: SessionStore, session_id: str) -> str:
    metadata = store.read_metadata(session_id)
    events = store.read_events(session_id)
    return render_session_observability(build_session_observability(metadata, events))


def format_project_observability(store: SessionStore) -> str:
    return render_project_observability(build_project_observability(store))


def render_session_observability(snapshot: SessionObservability) -> str:
    lines = [
        "Observability dashboard",
        f"session: {snapshot.session_id}",
        f"model: {snapshot.model}",
        f"status: {snapshot.status}",
        f"created_at: {snapshot.created_at}",
        "",
        _token_line(snapshot.tokens),
        _tool_line(snapshot.tools),
        _performance_line(snapshot.performance),
        _audit_line(snapshot.audit),
        "",
        "tool breakdown:",
    ]
    if snapshot.tools.by_tool:
        for tool, breakdown in sorted(snapshot.tools.by_tool.items()):
            lines.append(
                f"- {tool}: calls={breakdown.calls}, results={breakdown.results}, "
                f"success={breakdown.successes}, failed={breakdown.failures}, "
                f"unknown={breakdown.unknown}, pending={breakdown.pending_calls}"
            )
    else:
        lines.append("- none")

    lines.append("")
    lines.append("recent audit events:")
    lines.extend(f"- {item}" for item in snapshot.recent_events) if snapshot.recent_events else lines.append("- none")
    return "\n".join(lines)


def render_project_observability(snapshot: ProjectObservability) -> str:
    lines = [
        "Project observability",
        f"sessions: {snapshot.session_count}",
        "",
        _token_line(snapshot.tokens),
        _tool_line(snapshot.tools),
        _performance_line(snapshot.performance),
        _audit_line(snapshot.audit),
        "",
        "top token sessions:",
        *_session_rank_lines(snapshot.top_token_sessions, key="tokens"),
        "",
        "top failure sessions:",
        *_session_rank_lines(snapshot.top_failure_sessions, key="failures"),
        "",
        "low cache sessions:",
        *_session_rank_lines(snapshot.low_cache_sessions, key="cache"),
    ]
    return "\n".join(lines)


def export_observability_report(
    store: SessionStore,
    *,
    mode: str,
    session_id: str | None = None,
) -> ExportResult:
    mode = _normalize_mode(mode)
    if mode == "current":
        if not session_id:
            raise ValueError("session_id is required for current observability export")
        metadata = store.read_metadata(session_id)
        events = store.read_events(session_id)
        snapshot = build_session_observability(metadata, events)
        markdown = render_session_observability(snapshot)
        target_name = f"session-{_safe_filename(session_id)}"
        payload: dict[str, Any] = {"mode": mode, "snapshot": snapshot.to_dict()}
    else:
        project = build_project_observability(store)
        markdown = render_project_observability(project)
        target_name = "project"
        payload = {"mode": mode, "snapshot": project.to_dict()}

    report_dir = store.root / "reports" / "observability"
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = report_dir / f"{target_name}.md"
    json_path = report_dir / f"{target_name}.json"
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return ExportResult(markdown_path=markdown_path, json_path=json_path)


def format_export_result(result: ExportResult) -> str:
    return "\n".join([
        f"observability_export_markdown: {result.markdown_path}",
        f"observability_export_json: {result.json_path}",
    ])


def _normalize_mode(mode: str) -> str:
    normalized = (mode or "current").strip().lower()
    if normalized in {"current", "session"}:
        return "current"
    if normalized in {"project", "all"}:
        return "project"
    raise ValueError("mode must be current or project")


def _token_line(tokens: TokenMetrics) -> str:
    return (
        "tokens: "
        f"llm_calls={tokens.llm_calls}, "
        f"prompt={tokens.prompt_tokens}, "
        f"cached={tokens.cached_tokens}, "
        f"miss={tokens.cache_miss_tokens}, "
        f"completion={tokens.completion_tokens}, "
        f"total={tokens.total_tokens}, "
        f"cache hit ratio: {_percent(tokens.cache_hit_ratio)}"
    )


def _tool_line(tools: ToolMetrics) -> str:
    return (
        "tools: "
        f"calls={tools.tool_calls}, results={tools.tool_results}, "
        f"success={tools.successes}, failed={tools.failures}, unknown={tools.unknown}, "
        f"pending: {tools.pending_calls}, success rate: {_percent(tools.success_rate)}"
    )


def _performance_line(performance: PerformanceMetrics) -> str:
    return (
        "performance: "
        f"llm_response={_distribution_summary(performance.llm_response_latency_ms)}, "
        f"ttft={_distribution_summary(performance.llm_first_token_ms)}, "
        f"turn={_distribution_summary(performance.turn_duration_ms)}"
    )


def _audit_line(audit: AuditMetrics) -> str:
    categories = _format_counter(audit.failure_categories)
    return (
        "audit: "
        f"failures={audit.failures}, categories={categories}, "
        f"fallbacks={audit.fallbacks}, latest_fallback={audit.latest_fallback or 'none'}, "
        f"compactions={audit.compactions_committed} committed, "
        f"tokens_saved={audit.tokens_saved}, "
        f"context_anxiety={audit.context_anxiety_observed} observed, "
        f"approvals={audit.approvals_requested} requested/{audit.approvals_approved} approved/{audit.approvals_denied} denied, "
        f"changed_files={len(audit.changed_files)}"
    )


def _session_rank_lines(sessions: list[SessionObservability], *, key: str) -> list[str]:
    if not sessions:
        return ["- none"]
    lines = []
    for session in sessions:
        if key == "tokens":
            value = f"total_tokens={session.tokens.total_tokens}"
        elif key == "failures":
            value = f"failures={session.audit.failures}"
        else:
            value = f"cache_hit={_percent(session.tokens.cache_hit_ratio)}"
        lines.append(f"- {session.session_id}: {value}")
    return lines


def _recent_audit_events(events: list[dict[str, Any]], limit: int = 6) -> list[str]:
    interesting = {
        "llm_usage",
        "tool_result",
        "failure",
        "agent_fallback",
        "context_compaction_committed",
        "context_anxiety_observed",
        "approval_decided",
    }
    lines: list[str] = []
    for event in reversed(events):
        event_name = _event_type(event)
        if event_name not in interesting:
            continue
        data = _payload(event)
        sequence = event.get("sequence", "?")
        if event_name == "llm_usage":
            lines.append(f"#{sequence} llm_usage total_tokens={data.get('total_tokens')}")
        elif event_name == "tool_result":
            lines.append(f"#{sequence} tool_result {data.get('tool', 'unknown')} status={data.get('status', 'unknown')}")
        elif event_name == "failure":
            lines.append(f"#{sequence} failure category={data.get('category', 'unknown')}")
        elif event_name == "agent_fallback":
            lines.append(f"#{sequence} fallback reason={data.get('reason', 'unknown')}")
        elif event_name == "context_compaction_committed":
            lines.append(f"#{sequence} compaction tokens_saved={data.get('tokens_saved', 0)}")
        elif event_name == "context_anxiety_observed":
            lines.append(f"#{sequence} context_anxiety score={data.get('score', 0)}")
        elif event_name == "approval_decided":
            lines.append(f"#{sequence} approval approved={data.get('approved')}")
        if len(lines) >= limit:
            break
    lines.reverse()
    return lines


def _tool_name(data: dict[str, Any]) -> str:
    return str(data.get("tool") or "unknown")


def _int_value(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _distribution_summary(metrics: DistributionMetrics) -> str:
    if metrics.count <= 0:
        return "n=0"
    return f"n={metrics.count}, p50={metrics.p50}ms, p95={metrics.p95}ms, p99={metrics.p99}ms"


def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[idx]


def _format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counter.items()))


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value) or "unknown"
