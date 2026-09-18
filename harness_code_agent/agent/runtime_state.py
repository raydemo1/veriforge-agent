"""Agent runtime state containers."""
from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, field
from typing import Any

from ..runtime.tool_failures import FailureTracker
from ..workspace.shell_jobs import ShellJobManager
from ..workspace.shell_session import PersistentShellSession

TODO_STATUSES = ("pending", "in_progress", "completed", "cancelled")
MAX_TODO_ITEMS = 20


@dataclass(frozen=True)
class TodoItem:
    id: str
    text: str
    status: str = "pending"


@dataclass
class TodoList:
    """Agent-owned execution checklist.

    The list exists only once the agent creates it; a turn without a todo
    list simply has ``runtime_state.todo is None``.
    """

    items: list[TodoItem] = field(default_factory=list)
    revision: int = 0
    updated_at: str = ""
    next_seq: int = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "updated_at": self.updated_at,
            "items": [
                {"id": item.id, "text": item.text, "status": item.status}
                for item in self.items
            ],
        }

    def render(self) -> str:
        markers = {
            "completed": "[done]",
            "in_progress": "[in progress]",
            "pending": "[pending]",
            "cancelled": "[cancelled]",
        }
        return "\n".join(
            f"{markers.get(item.status, '[pending]')} {item.text}"
            for item in self.items
        )


def normalize_todo_items(
    raw_items: Any,
    *,
    next_seq: int,
) -> tuple[list[TodoItem], int]:
    """Validate model-provided todo items and assign ids where missing.

    Returns the normalized items plus the advanced id sequence counter.
    Raises ValueError with an agent-readable message on invalid input.
    """
    if not isinstance(raw_items, list):
        raise TypeError("items must be a list")
    if not raw_items:
        raise ValueError("items must contain at least one entry")
    if len(raw_items) > MAX_TODO_ITEMS:
        raise ValueError(f"a todo list allows at most {MAX_TODO_ITEMS} items")

    items: list[TodoItem] = []
    seen_ids: set[str] = set()
    seq = int(next_seq)
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise TypeError("each todo item must be an object")
        text = str(raw.get("text") or "").strip()
        if not text:
            raise ValueError("each todo item requires non-empty text")
        if len(text) > 500:
            raise ValueError("each todo item text must be at most 500 characters")
        status = str(raw.get("status") or "pending").strip().lower()
        if status not in TODO_STATUSES:
            raise ValueError(
                "todo item status must be one of: " + ", ".join(TODO_STATUSES)
            )
        raw_id = str(raw.get("id") or "").strip()
        if raw_id:
            item_id = raw_id[:60]
        else:
            seq += 1
            item_id = f"todo_{seq}"
        if item_id in seen_ids:
            raise ValueError(f"duplicate todo item id: {item_id}")
        seen_ids.add(item_id)
        items.append(TodoItem(id=item_id, text=text, status=status))
    return items, seq


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
    active_shell_sessions: set[PersistentShellSession] = field(default_factory=set)
    _shell_sessions_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    shell_job_manager: ShellJobManager | None = None
    browser_job_id: str | None = None
    todo: TodoList | None = None
    task_metadata: dict[str, Any] = field(default_factory=dict)
    fallback: AgentFallbackState = field(default_factory=AgentFallbackState)
    failures: FailureTracker = field(default_factory=FailureTracker)
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
