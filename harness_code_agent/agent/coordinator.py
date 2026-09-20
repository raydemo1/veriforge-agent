"""Session-scoped subagent lifecycle, steering, and isolated integration."""
from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import config
from ..runtime.execution_planner import acquire_concurrency, workspace_claim
from ..runtime.permission_middleware import PermissionMiddleware
from ..runtime.permissions import PermissionPolicy
from ..runtime.tool_context import ToolContext
from ..runtime.tool_registry import (
    TOOL_CAPABILITY_READONLY_AGENT,
    TOOL_CAPABILITY_WORKER_AGENT,
    ToolRegistry,
)
from ..workspace.service import WorkspaceService
from .cancellation import CancellationToken, CancelledError
from .change_proposal import ChangeProposalStore

AGENT_ROLES = {"explorer", "test_designer", "reviewer", "verifier", "worker"}
READ_ONLY_ROLES = AGENT_ROLES - {"worker"}
ACTIVE_STATES = {"queued", "running"}
TERMINAL_STATES = {"completed", "blocked", "failed", "interrupted", "closed"}
MAX_OPEN_AGENTS = max(1, int(config.MAX_OPEN_AGENTS))
MAX_CONCURRENT_AGENTS = max(1, int(config.MAX_CONCURRENT_AGENTS))


class SubagentCapacityError(ValueError):
    """Raised when the open-agent slot cap (not a policy rule) is reached.

    Subclasses ValueError so existing ``except ValueError`` callers keep
    working; spawn_agent distinguishes it to emit a resource-class failure.
    Note: the concurrency semaphore (MAX_CONCURRENT_AGENTS) blocks queued
    workers rather than raising; this error covers the hard open cap.
    """

@dataclass
class AgentRecord:
    id: str
    name: str
    role: str
    task: str
    expected_output: str
    allowed_paths: list[str]
    fork_turns: str | int
    model_intensity: str | None
    max_turns: int
    max_seconds: int
    state: str = "queued"
    summary: str = ""
    error: str | None = None
    proposal_id: str | None = None
    conversation: Any = None
    token: CancellationToken | None = None
    parent_token: CancellationToken | None = None
    followups: list[str] = field(default_factory=list)
    future: Any = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None


class AgentCoordinator:
    """Deep module for agent threads, inboxes, isolation, and change integration."""

    def __init__(
        self,
        tool_context: ToolContext,
        *,
        parent_messages: Callable[[], list[dict]] | None = None,
        max_concurrent: int = MAX_CONCURRENT_AGENTS,
    ) -> None:
        self.context = tool_context
        self._parent_messages = parent_messages or list
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(max_concurrent)), thread_name_prefix="hca-agent")
        self._condition = threading.Condition(threading.RLock())
        self._records: dict[str, AgentRecord] = {}
        self._names: dict[str, str] = {}
        self._version = 0
        self._closed = False
        self.changes = ChangeProposalStore()

    def spawn(
        self,
        *,
        name: str,
        role: str,
        task: str,
        expected_output: str = "",
        allowed_paths: list[str] | None = None,
        fork_turns: str | int = "none",
        model_intensity: str | None = None,
        max_turns: int = 6,
        max_seconds: int = 300,
        parent_cancellation_token: CancellationToken | None = None,
    ) -> dict:
        role = str(role or "").strip().lower()
        task = str(task or "").strip()
        clean_name = _agent_name(name)
        paths = [_normalize_rel(path) for path in allowed_paths or [] if _normalize_rel(path)]
        if role not in AGENT_ROLES:
            raise ValueError(f"unknown role: {role}")
        if not task:
            raise ValueError("task is required")
        if role == "worker" and not paths:
            raise ValueError("worker requires non-empty allowed_paths")
        if model_intensity is not None and model_intensity not in {"fast", "normal", "hard", "max"}:
            raise ValueError(f"unknown model_intensity: {model_intensity}")
        if (
            parent_cancellation_token is not None
            and parent_cancellation_token.is_cancelled
        ):
            raise ValueError("cannot spawn agent: the requesting turn is cancelled")
        normalized_fork = _normalize_fork_turns(fork_turns)
        with self._condition:
            self._ensure_open()
            if len(self._records) >= MAX_OPEN_AGENTS:
                raise SubagentCapacityError(
                    f"at most {MAX_OPEN_AGENTS} agent threads may remain open"
                )
            if clean_name in self._names:
                raise ValueError(f"agent name is already in use: {clean_name}")
            if role == "worker":
                self._check_worker_ownership(paths)
            agent_id = f"agent_{uuid.uuid4().hex[:16]}"
            record = AgentRecord(
                id=agent_id,
                name=clean_name,
                role=role,
                task=task,
                expected_output=str(expected_output or ""),
                allowed_paths=paths,
                fork_turns=normalized_fork,
                model_intensity=model_intensity,
                max_turns=max(1, min(20, int(max_turns))),
                max_seconds=max(30, min(1800, int(max_seconds))),
                parent_token=parent_cancellation_token,
            )
            self._records[agent_id] = record
            self._names[clean_name] = agent_id
            record.future = self._executor.submit(self._run_record, agent_id)
            self._changed_locked()
        self._emit(record, "agent_spawned")
        return self._snapshot(record)

    def send(self, agent_id: str, message: str) -> dict:
        record = self._record(agent_id)
        text = str(message or "").strip()
        if not text:
            raise ValueError("message is required")
        with self._condition:
            if record.state == "queued" or record.conversation is None:
                record.followups.append(f"[PARENT MESSAGE]\n{str(message).strip()}")
                self._changed_locked()
            elif record.state == "running":
                record.conversation.queue_message(text)
                self._changed_locked()
            else:
                raise ValueError("agent is not running; use followup_agent to start another turn")
        self._emit(record, "agent_message", message=text, mode="steer")
        return self._snapshot(record)

    def followup(self, agent_id: str, task: str) -> dict:
        text = str(task or "").strip()
        if not text:
            raise ValueError("follow-up task is required")
        record = self._record(agent_id)
        with self._condition:
            if record.state == "running" and record.conversation is not None:
                record.conversation.queue_message(text)
            elif record.state == "queued":
                record.followups.append(text)
            elif record.state in TERMINAL_STATES:
                if record.conversation is None:
                    record.followups.append(text)
                else:
                    record.conversation.add_user_turn(text)
                record.error = None
                record.state = "queued"
                record.future = self._executor.submit(self._run_record, record.id)
            else:
                raise ValueError(f"cannot follow up agent in state: {record.state}")
            self._changed_locked()
        self._emit(record, "agent_message", message=text, mode="followup")
        return self._snapshot(record)

    def wait(self, agent_ids: list[str] | None = None, *, timeout_seconds: float = 30) -> dict:
        with self._condition:
            selected = self._select_records(agent_ids)
            completed = lambda: self._closed or all(
                record.state in TERMINAL_STATES for record in selected
            )
            finished = completed() or self._condition.wait_for(
                completed,
                timeout=max(0.0, float(timeout_seconds)),
            )
            return {
                "agents": [self._snapshot(record) for record in selected],
                "timed_out": not finished,
            }

    def list(self, status: str = "") -> dict:
        with self._condition:
            records = list(self._records.values())
            if status:
                records = [record for record in records if record.state == status]
            return {"agents": [self._snapshot(record) for record in records]}

    def interrupt(self, agent_id: str) -> dict:
        record = self._record(agent_id)
        token = record.token
        if token is not None:
            token.cancel()
        return self._snapshot(record)

    def close_agent(self, agent_id: str, *, discard_changes: bool = False) -> dict:
        with self._condition:
            record = self._record(agent_id)
            if record.state in ACTIVE_STATES:
                raise ValueError("interrupt the running agent before closing it")
            previous = record.state
            record.state = "closing"
            self._records.pop(record.id, None)
            self._names.pop(record.name, None)
            self._changed_locked()
        error: Exception | None = None
        try:
            self.changes.close_agent(record.id, discard_changes=discard_changes)
        except Exception as exc:  # noqa: BLE001 - continue closing owned resources
            error = exc
        try:
            if record.conversation is not None:
                record.conversation.close()
        except Exception as exc:  # noqa: BLE001 - preserve the first cleanup failure
            error = error or exc
        with self._condition:
            record.state = "closed"
            self._changed_locked()
        self._emit(record, "agent_status", previous=previous, status="closed")
        if error is not None:
            raise error
        return {"agent_id": record.id, "status": "closed"}

    def has_active_agents(self) -> bool:
        with self._condition:
            return any(record.state in ACTIVE_STATES for record in self._records.values())

    def proposal_paths(self, proposal_id: str) -> list[str]:
        return self.changes.proposal_paths(proposal_id)

    def conflict_paths(self, conflict_id: str) -> list[str]:
        return self.changes.conflict_paths(conflict_id)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            records = list(self._records.values())
            for record in records:
                if record.token is not None:
                    record.token.cancel()
            self._changed_locked()
        self._executor.shutdown(wait=True, cancel_futures=True)
        for record in records:
            if record.conversation is not None:
                record.conversation.close()
        self.changes.close()

    def _run_record(self, agent_id: str) -> None:
        record = self._record(agent_id)
        with acquire_concurrency("subagent"):
            try:
                self._ensure_conversation(record)
                while True:
                    with self._condition:
                        if self._closed:
                            return
                        if record.parent_token is not None and record.parent_token.is_cancelled:
                            self._finish(record, "interrupted", error="Turn cancelled by user")
                            return
                        record.state = "running"
                        record.started_at = record.started_at or time.time()
                        record.ended_at = None
                        previous_token = record.token
                        record.token = CancellationToken(parent=record.parent_token)
                        if previous_token is not None:
                            previous_token.close()
                        token = record.token
                        self._changed_locked()
                    self._emit(record, "agent_status", status="running")
                    if record.conversation.messages[-1].get("role") != "user":
                        next_task = self._take_followup(record)
                        if next_task:
                            record.conversation.add_user_turn(next_task)
                    result = record.conversation.run_until_idle(cancellation_token=token)
                    record.summary = str(result or "")
                    with self._condition:
                        if record.conversation.has_queued_messages():
                            continue
                    if record.role == "worker":
                        proposal = self.changes.finalize(record.id, record.allowed_paths)
                        record.proposal_id = proposal.id
                        if proposal.invalid_reasons:
                            self._finish(record, "failed", error="; ".join(proposal.invalid_reasons))
                            return
                    followup = self._take_followup(record)
                    if followup:
                        record.conversation.add_user_turn(followup)
                        continue
                    self._finish(record, "completed")
                    return
            except CancelledError as exc:
                self._finish(record, "interrupted", error=str(exc))
            except Exception as exc:  # noqa: BLE001 - thread boundary records failures
                self._finish(record, "failed", error=f"{type(exc).__name__}: {exc}")

    def _ensure_conversation(self, record: AgentRecord) -> None:
        if record.conversation is not None:
            return
        from .conversation import Agent

        if record.role == "worker":
            root_claim = workspace_claim(self.context.workspace.root, ".", scope="global", access="read")
            with self.context.resource_coordinator.acquire((root_claim,)):
                sandbox = self.changes.create_sandbox(record.id, self.context.workspace.root)
            workspace = WorkspaceService(root=sandbox.workspace)
        else:
            workspace = self.context.workspace
        registry = self._role_registry(record.role)
        policy = PermissionPolicy.for_role(
            record.role,
            allowed_paths=record.allowed_paths,
            sandbox_mode="host",
        )
        sub_context = ToolContext(
            workspace=workspace,
            permission_policy=policy,
            event_bus=self.context.event_bus,
            session_id=self.context.session_id,
            tool_registry=registry,
            allowed_tool_permissions={"read", "network_read", "edit", "shell"},
        )
        sub_context.resource_coordinator = self.context.resource_coordinator
        sub_context.tool_tasks = self.context.tool_tasks
        agent = Agent(
            name=record.name,
            system_prompt=_role_prompt(record, self._parent_messages()),
            use_tools=True,
            tool_schemas=registry.schemas(),
            middlewares=[PermissionMiddleware(tool_context=sub_context, tool_registry=registry)],
            time_budget=float(record.max_seconds),
            tool_context=sub_context,
            model_intensity=record.model_intensity,
            max_iterations=record.max_turns,
        )
        record.conversation = agent.start_conversation(_task_prompt(record))
        with self._condition:
            pending = list(record.followups)
            record.followups.clear()
        for message in pending:
            record.conversation.queue_message(message)

    def _role_registry(self, role: str) -> ToolRegistry:
        # Depth limit: subagent registries are capability-filtered views of
        # the parent registry. spawn_agent carries no *_AGENT capability, so
        # workers/subagents cannot spawn further agents (hard depth = 1,
        # enforced structurally — there is no depth counter or state machine).
        source = self.context.tool_registry
        if source is None:
            raise RuntimeError("parent tool registry is unavailable")
        capability = TOOL_CAPABILITY_WORKER_AGENT if role == "worker" else TOOL_CAPABILITY_READONLY_AGENT
        registry = ToolRegistry()
        for spec in source.specs():
            if capability not in spec.capabilities:
                continue
            registry.register_spec(spec)
        return registry

    def _finish(self, record: AgentRecord, state: str, *, error: str | None = None) -> None:
        with self._condition:
            previous = record.state
            record.state = state
            record.error = error
            record.ended_at = time.time()
            token = record.token
            record.token = None
            self._changed_locked()
        # Detach the run token from the parent turn token outside the
        # coordinator lock so finished runs do not leak parent callbacks.
        if token is not None:
            token.close()
        self._emit(record, "agent_status", previous=previous, status=state)

    def _take_followup(self, record: AgentRecord) -> str | None:
        with self._condition:
            if not record.followups:
                return None
            return record.followups.pop(0)

    def _record(self, agent_id: str) -> AgentRecord:
        key = str(agent_id or "")
        with self._condition:
            resolved = self._names.get(key, key)
            record = self._records.get(resolved)
        if record is None:
            raise KeyError(f"unknown agent: {agent_id}")
        return record

    def _select_records(self, agent_ids: list[str] | None) -> list[AgentRecord]:
        if not agent_ids:
            return list(self._records.values())
        return [self._record(agent_id) for agent_id in agent_ids]

    def _check_worker_ownership(self, paths: list[str]) -> None:
        for record in self._records.values():
            if record.role != "worker":
                continue
            if any(_paths_overlap(left, right) for left in paths for right in record.allowed_paths):
                raise ValueError(f"worker paths overlap open agent {record.name}: {record.allowed_paths}")

    def _snapshot(self, record: AgentRecord) -> dict:
        duration = None
        if record.started_at is not None:
            duration = (record.ended_at or time.time()) - record.started_at
        return {
            "agent_id": record.id,
            "name": record.name,
            "role": record.role,
            "status": record.state,
            "summary": record.summary,
            "error": record.error,
            "proposal_id": record.proposal_id,
            "allowed_paths": list(record.allowed_paths),
            "duration_seconds": round(duration, 3) if duration is not None else None,
        }

    def _emit(self, record: AgentRecord, event_type: str, **extra) -> None:
        payload = self._snapshot(record)
        payload.update(extra)
        self.context.event_bus.emit(event_type, agent=record.name, payload=payload)

    def _changed_locked(self) -> None:
        self._version += 1
        self._condition.notify_all()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("agent coordinator is closed")


def _role_prompt(record: AgentRecord, parent_messages: list[dict]) -> str:
    role_text = {
        "explorer": "Investigate code and return evidence. Do not modify files.",
        "test_designer": "Design focused tests and failure cases. Do not modify files.",
        "reviewer": "Review for correctness, regressions, security, and missing tests. Do not modify files.",
        "verifier": "Run focused verification and interpret results. Do not modify source files.",
        "worker": "Implement the assigned change in your isolated workspace and verify it.",
    }[record.role]
    system = ""
    if parent_messages and parent_messages[0].get("role") == "system":
        system = str(parent_messages[0].get("content") or "")
    inherited = _forked_context(parent_messages, record.fork_turns)
    return (
        f"You are the {record.name} subagent with role {record.role}. {role_text}\n"
        "Stay within the assigned task. The parent owns integration and final completion.\n"
        f"Project instructions inherited from the parent:\n{system}\n"
        f"Inherited recent context:\n{inherited}"
    )


def _task_prompt(record: AgentRecord) -> str:
    parts = [f"Task:\n{record.task}"]
    if record.expected_output:
        parts.append(f"Expected output:\n{record.expected_output}")
    if record.allowed_paths:
        parts.append("Writable paths:\n" + "\n".join(f"- {path}" for path in record.allowed_paths))
    return "\n\n".join(parts)


def _forked_context(messages: list[dict], fork_turns: str | int) -> str:
    if fork_turns == "none":
        return "none"
    visible = [message for message in messages if message.get("role") in {"user", "assistant"}]
    if isinstance(fork_turns, int):
        visible = visible[-(fork_turns * 2):]
    blocks = [f"{message.get('role')}: {str(message.get('content') or '')[:4000]}" for message in visible]
    return "\n\n".join(blocks)[:20_000] or "none"


def _normalize_fork_turns(value: str | int) -> str | int:
    if isinstance(value, int):
        if 1 <= value <= 5:
            return value
        raise ValueError("fork_turns integer must be between 1 and 5")
    text = str(value or "none").strip().lower()
    if text in {"none", "all"}:
        return text
    if text.isdigit() and 1 <= int(text) <= 5:
        return int(text)
    raise ValueError("fork_turns must be none, all, or an integer from 1 to 5")


def _agent_name(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value or "").strip())
    if not text:
        raise ValueError("agent name is required")
    return text[:48]


def _normalize_rel(value: str) -> str:
    raw = Path(str(value or "").replace("\\", "/"))
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"path must be workspace-relative: {value}")
    return raw.as_posix().removeprefix("./")


def _path_allowed(path: str, allowed: list[str]) -> bool:
    candidate = Path(path)
    for root in allowed:
        base = Path(root)
        if candidate == base:
            return True
        try:
            candidate.relative_to(base)
            return True
        except ValueError:
            continue
    return False


def _paths_overlap(left: str, right: str) -> bool:
    return _path_allowed(left, [right]) or _path_allowed(right, [left])
