"""Agent runtime state containers."""
from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, field
from typing import Any

from ..runtime.tool_failures import FailureTracker
from ..workspace.shell_jobs import ShellJobManager
from ..workspace.shell_session import PersistentShellSession
from .acceptance import AcceptanceState


@dataclass
class TaskBoard:
    original_task: str = ""
    goal: str = ""
    task_metadata: dict[str, Any] = field(default_factory=dict)
    steps: list[str] = field(default_factory=list)
    current_step: str = ""
    completed_steps: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    next_action: str = ""
    planning_mode: str = "unset"
    update_count: int = 0
    action_count: int = 0
    changed_files: list[str] = field(default_factory=list)
    requires_approval: bool = False
    requires_update: bool = False
    needs_final_update: bool = False
    replan_required: bool = False
    replan_reason: str = ""
    plan_revision: int = 0
    result_status: str = ""
    validation: str = ""
    remaining_issues: list[str] = field(default_factory=list)
    actions_since_progress: int = 0
    acceptance: AcceptanceState = field(default_factory=AcceptanceState)


@dataclass
class RecoveryState:
    mode: str = "NORMAL"
    failure_signature: str = ""
    repeat_count: int = 0
    replan_attempt_count: int = 0
    last_successful_action: str = ""
    last_verification_result: str = ""
    probe_in_flight: bool = False


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
class ExecutionFacts:
    sequence: int = 0
    last_business_edit_sequence: int = 0
    last_foreground_shell_sequence: int = 0
    last_foreground_shell_success: bool = False

    def record_result(
        self,
        tool_name: str,
        *,
        status: str,
        return_code: int | None,
        metadata: dict | None,
    ) -> None:
        self.sequence += 1
        metadata = dict(metadata or {})
        if tool_name in {"write_file", "apply_patch"} and status == "success":
            changes = metadata.get("file_changes")
            if isinstance(changes, list) and any(
                _is_business_path(change.get("path"))
                for change in changes
                if isinstance(change, dict)
            ):
                self.last_business_edit_sequence = self.sequence
        if (
            tool_name == "run_bash"
            and metadata.get("status_source") != "shell_job"
        ):
            self.last_foreground_shell_sequence = self.sequence
            self.last_foreground_shell_success = status == "success" and return_code == 0


def _is_business_path(path) -> bool:
    normalized = str(path or "").replace("\\", "/").lstrip("./")
    return bool(normalized) and not normalized.startswith((".harness/", "global_plan/"))


@dataclass
class AgentRuntimeState:
    active_shell_sessions: set[PersistentShellSession] = field(default_factory=set)
    _shell_sessions_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    shell_job_manager: ShellJobManager | None = None
    browser_job_id: str | None = None
    task_board: TaskBoard = field(default_factory=TaskBoard)
    recovery: RecoveryState = field(default_factory=RecoveryState)
    fallback: AgentFallbackState = field(default_factory=AgentFallbackState)
    execution_facts: ExecutionFacts = field(default_factory=ExecutionFacts)
    failures: FailureTracker = field(default_factory=FailureTracker)
    action_tool_count: int = 0
    current_turn_start_index: int = 0
    session_id: str = "default"
    permission_mode: str = ""
    auto_compaction_turn_start_index: int = -1
    auto_compaction_suspended: bool = False
    context_refill_streak: int = 0
    context_anxiety_turn_start_index: int = -1
    event_bus: Any = None

    def register_shell_session(self, session: PersistentShellSession) -> None:
        with self._shell_sessions_lock:
            self.active_shell_sessions.add(session)

    def unregister_shell_session(self, session: PersistentShellSession) -> None:
        with self._shell_sessions_lock:
            self.active_shell_sessions.discard(session)

    def interrupt_shell_sessions(self) -> bool:
        with self._shell_sessions_lock:
            sessions = tuple(self.active_shell_sessions)
        for session in sessions:
            with contextlib.suppress(Exception):
                session.interrupt()
        return bool(sessions)

    def close_shell_sessions(self) -> None:
        with self._shell_sessions_lock:
            sessions = tuple(self.active_shell_sessions)
            self.active_shell_sessions.clear()
        for session in sessions:
            with contextlib.suppress(Exception):
                session.close()
