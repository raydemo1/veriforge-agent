"""Agent runtime state containers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..runtime.todo_state import TodoList
from ..runtime.tool_failures import FailureTracker
from ..workspace.shell_jobs import ShellJobManager


@dataclass
class AgentFallbackState:
    total_tokens: int = 0
    llm_call_count: int = 0
    tool_call_count: int = 0
    budget_warnings: set[str] = field(default_factory=set)
    stop_requested: bool = False
    stop_reason: str = ""
    stop_limit_type: str = ""
    stop_used: int | None = None
    stop_limit: int | None = None
    stop_last_tool: str = ""
    stop_fingerprint_hash: str = ""
    recent_action_summary: list[str] = field(default_factory=list)
    fallback_event_emitted: bool = False

    def request_stop(
        self,
        *,
        reason: str,
        limit_type: str = "",
        used: int | None = None,
        limit: int | None = None,
        last_tool: str = "",
        fingerprint_hash: str = "",
        recent_action_summary: list[str] | None = None,
    ) -> None:
        if self.stop_requested:
            return
        self.stop_requested = True
        self.stop_reason = reason
        self.stop_limit_type = limit_type
        self.stop_used = used
        self.stop_limit = limit
        self.stop_last_tool = last_tool
        self.stop_fingerprint_hash = fingerprint_hash
        if recent_action_summary is not None:
            self.recent_action_summary = list(recent_action_summary)[-5:]

    def record_action(self, summary: str) -> None:
        summary = summary.strip()
        if not summary:
            return
        self.recent_action_summary.append(summary[:240])
        if len(self.recent_action_summary) > 5:
            self.recent_action_summary = self.recent_action_summary[-5:]


@dataclass
class AgentRuntimeState:
    shell_job_manager: ShellJobManager | None = None
    todo: TodoList | None = None
    fallback: AgentFallbackState = field(default_factory=AgentFallbackState)
    failures: FailureTracker = field(default_factory=FailureTracker)
    workspace_conflict_attempts: dict[str, int] = field(default_factory=dict)
    current_turn_start_index: int = 0
    session_id: str = "default"
    permission_mode: str = ""
    auto_compaction_turn_start_index: int = -1
    auto_compaction_suspended: bool = False
    context_anxiety_turn_start_index: int = -1
    event_bus: Any = None
    turn_id: str = ""
    current_user_task: str = ""
    epoch_id: str = ""
    observation_sequence: int = 0
    run_started: float | None = None
    time_budget: float | None = None
    pending_budget_warnings: set[str] = field(default_factory=set)
    verification: dict | None = None
    verification_stale: bool = False
    agent_config: Any = None
