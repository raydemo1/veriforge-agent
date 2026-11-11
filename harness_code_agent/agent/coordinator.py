"""Session-scoped subagent lifecycle, steering, and isolated integration."""
from __future__ import annotations

import threading
import time
import uuid
import weakref
from collections.abc import Callable
from concurrent.futures import wait
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import config
from ..runtime.agent_roles import AGENT_ROLE_NAMES, role_definition
from ..runtime.builtins.discovery import discovery_catalog_prompt
from ..runtime.errors import SubagentCapacityError
from ..runtime.execution_planner import workspace_claim
from ..runtime.mcp import PublishedMcpCatalog
from ..runtime.middleware.stack import build_subagent_middlewares
from ..runtime.output_artifacts import copy_artifact, file_digest
from ..runtime.permissions import PermissionPolicy
from ..runtime.tool_context import ToolContext
from ..runtime.tool_registry import (
    TOOL_CAPABILITY_READONLY_AGENT,
    TOOL_CAPABILITY_WORKER_AGENT,
    ToolRegistry,
)
from ..support.cancellation import CancellationToken, CancelledError
from ..workspace.paths import path_within
from ..workspace.service import WorkspaceService
from .change_proposal import ChangeProposalStore

AGENT_ROLES = AGENT_ROLE_NAMES
ACTIVE_STATES = {"queued", "running"}
TERMINAL_STATES = {"completed", "blocked", "failed", "interrupted", "closed"}
MAX_OPEN_AGENTS = max(1, int(config.MAX_OPEN_AGENTS))


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
    state: str = "queued"
    summary: str = ""
    error: str | None = None
    proposal_id: str | None = None
    conversation: Any = None
    token: CancellationToken | None = None
    followups: list[str] = field(default_factory=list)
    future: Any = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None
    output_artifacts: list[dict] = field(default_factory=list)
    saved_messages: list[dict] | None = None
    saved_shell_jobs: list[dict] = field(default_factory=list)


class AgentCoordinator:
    """Deep module for agent threads, inboxes, isolation, and change integration."""

    def __init__(
        self,
        tool_context: ToolContext,
        *,
        parent_messages: Callable[[], list[dict]] | None = None,
        parent_system_prompt: Callable[[], str] | None = None,
        parent_message_sink: Callable[[str, str], None] | None = None,
    ) -> None:
        self.context = tool_context
        self._parent_messages = parent_messages or list
        self._parent_system_prompt = parent_system_prompt or (lambda: "")
        # Delivers (text, tag) into the main agent conversation queue. The
        # topology is fixed Main <-> Child: this is the only upward path,
        # there is no child-to-child addressing.
        self._parent_message_sink = parent_message_sink
        self._resources = tool_context.project_resources
        self._resource_owner = weakref.finalize(self, self._resources.hold())
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
            )
            if parent_cancellation_token is not None:
                parent_cancellation_token.check()
            self._records[agent_id] = record
            self._names[clean_name] = agent_id
            record.future = self._submit_record(record)
            self._changed_locked()
        self._emit(record, "agent_spawned")
        return self._snapshot(record)

    def send(self, agent_id: str, message: str) -> dict:
        record = self._record(agent_id)
        text = str(message or "").strip()
        if not text:
            raise ValueError("message is required")
        self._materialize_outputs(record, text)
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

    def send_to_parent(self, agent_id: str, message: str) -> dict:
        record = self._record(agent_id)
        text = str(message or "").strip()
        if not text:
            raise ValueError("message is required")
        sink = self._parent_message_sink
        if sink is None:
            raise ValueError("no parent agent is available")
        # Queue into the main conversation; injection happens at the main
        # agent's next safe boundary. No event is delivered to any sibling.
        sink(text, f"SUBAGENT MESSAGE from {record.name}")
        self._emit(record, "agent_parent_message", message=text)
        return self._snapshot(record)

    def followup(self, agent_id: str, task: str) -> dict:
        text = str(task or "").strip()
        if not text:
            raise ValueError("follow-up task is required")
        record = self._record(agent_id)
        self._materialize_outputs(record, text)
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
                record.future = self._submit_record(record)
            else:
                raise ValueError(f"cannot follow up agent in state: {record.state}")
            self._changed_locked()
        self._emit(record, "agent_message", message=text, mode="followup")
        return self._snapshot(record)

    def wait(self, agent_ids: list[str] | None = None, *, timeout_seconds: float = 30) -> dict:
        with self._condition:
            selected = self._select_records(agent_ids)
            def completed():
                return self._closed or all(record.state in TERMINAL_STATES for record in selected)
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
        with self._condition:
            record = self._record(agent_id)
            if record.token is not None:
                record.token.cancel()
            if record.future is not None and record.future.cancel():
                self._finish(record, "interrupted", error="Agent cancelled before execution")
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
            if record.conversation is not None:
                record.conversation.close()
            self._publish_outputs(record)
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

    _SAVED_FIELDS = ("id", "name", "role", "task", "expected_output", "allowed_paths", "fork_turns",
                     "model_intensity", "state", "summary", "error", "proposal_id", "created_at",
                     "started_at", "ended_at", "output_artifacts")

    def export_state(self) -> list[dict]:
        """Finished agents, kept across a runtime suspension."""
        with self._condition:
            if any(record.state in ACTIVE_STATES for record in self._records.values()):
                raise RuntimeError("cannot export running agents")
            return [dict({name: getattr(record, name) for name in self._SAVED_FIELDS},
                         messages=(record.conversation.messages if record.conversation is not None
                                   else record.saved_messages),
                         shell_jobs=(record.conversation.runtime_state.shell_job_manager.export_jobs()
                                     if record.conversation is not None else record.saved_shell_jobs))
                    for record in self._records.values()]

    def import_state(self, entries: list[dict]) -> None:
        with self._condition:
            self._ensure_open()
            for entry in entries:
                record = AgentRecord(**{name: entry[name] for name in self._SAVED_FIELDS},
                                     saved_messages=entry["messages"], saved_shell_jobs=entry["shell_jobs"])
                self._records[record.id] = record
                self._names[record.name] = record.id
            self._changed_locked()

    def has_active_agents(self) -> bool:
        with self._condition:
            return any(record.state in ACTIVE_STATES for record in self._records.values())

    def proposal_paths(self, proposal_id: str) -> list[str]:
        return self.changes.proposal_paths(proposal_id)

    def conflict_paths(self, conflict_id: str) -> list[str]:
        return self.changes.conflict_paths(conflict_id)

    def close(self, *, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            self._closed = True
            records = list(self._records.values())
            for record in records:
                if record.token is not None:
                    record.token.cancel()
            self._changed_locked()
        for record in records:
            if record.future is not None:
                record.future.cancel()
        futures = [record.future for record in records
                   if record.future is not None and not record.future.done()]
        if futures:
            wait(futures, timeout=max(0.0, deadline - time.monotonic()))
        if any(not future.done() for future in futures):
            raise RuntimeError("Agent executions are still running; agent resources retained")
        if not self.context.tool_tasks.wait_idle(timeout=max(0.0, deadline - time.monotonic())):
            raise RuntimeError("Foreground tools are still running; agent resources retained")
        for record in records:
            if record.conversation is not None:
                record.conversation.close()
            self._publish_outputs(record)
        self.changes.close()
        self._resource_owner()

    def _submit_record(self, record):
        record.token = CancellationToken()
        return self._resources.submit("agents", self._run_record, record.id,
                                      session_id=self.context.session_id, label=record.name)

    def _run_record(self, agent_id: str) -> None:
        record = self._record(agent_id)
        try:
            record.token.check()
            self._ensure_conversation(record)
            while True:
                with self._condition:
                    if self._closed:
                        return
                    record.state = "running"
                    record.started_at = record.started_at or time.time()
                    record.ended_at = None
                    token = record.token
                    token.check()
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
        parent_sync = self.context.sync_mcp_catalog
        published = parent_sync() if parent_sync is not None else None
        registry = self._role_registry(record.role)

        def sync_mcp_catalog():
            nonlocal published
            latest = parent_sync()
            if latest is not published:
                self._refresh_child_mcp(agent, sub_context, latest)
                published = latest
            return published

        def discover_mcp(server_name):
            failures = self.context.discover_mcp_tools(server_name)
            if parent_sync is None:
                self._refresh_child_mcp(agent, sub_context)
            else:
                sync_mcp_catalog()
            return failures

        policy = PermissionPolicy.delegated_from(
            self.context.permission_policy,
            record.role,
            allowed_paths=record.allowed_paths,
        )
        sub_context = ToolContext(
            workspace=workspace,
            project_resources=self.context.project_resources,
            ui_request_id=self.context.ui_request_id,
            discover_mcp_tools=discover_mcp if self.context.discover_mcp_tools is not None else None,
            sync_mcp_catalog=sync_mcp_catalog if parent_sync is not None else None,
            mcp_discovery_statuses=self.context.mcp_discovery_statuses,
            mcp_discovery_scopes=self.context.mcp_discovery_scopes,
            permission_policy=policy,
            event_bus=self.context.event_bus,
            session_id=self.context.session_id,
            session_root=self.context.session_root,
            model_selection=config.ModelSelection(model=self.context.model_selection.model, reasoning_effort=self.context.model_selection.reasoning_effort),
            memory_use_enabled=self.context.memory_use_enabled,
            memory_auto_extract_enabled=self.context.memory_auto_extract_enabled,
            memory_service=self.context.memory_service,
            tool_registry=registry,
            tool_capability=self._role_capability(record.role),
            allowed_tool_permissions={"read", "network_read", "edit", "shell"},
            agent_id=record.id,
            agent_coordinator=self,
        )
        sub_context.resource_coordinator = self.context.resource_coordinator
        sub_context.tool_tasks = self.context.tool_tasks
        parent_messages = self._parent_messages()
        role_prompt = _role_prompt(record, parent_messages, self._parent_system_prompt())
        inherited = role_prompt + "\n" + _task_prompt(record)
        for reference in self.context.tool_output_artifacts.copy().values():
            if reference["path"] in inherited:
                if workspace.root != self.context.workspace.root:
                    copy_artifact(self.context.workspace.root, workspace.root, reference)
                sub_context.tool_output_artifacts[reference["path"]] = dict(reference)
        agent = Agent(
            name=record.name,
            system_prompt=role_prompt,
            use_tools=True,
            tool_schemas=[
                spec.schema for spec in registry.specs() if spec.disclosure == "core"
            ],
            middlewares=build_subagent_middlewares(
                tool_context=sub_context,
                tool_registry=registry,
            ),
            tool_context=sub_context,
            model_intensity=record.model_intensity,
        )
        if sub_context.mcp_discovery_scopes is not None:
            agent.set_tool_discovery_catalog(discovery_catalog_prompt(registry, sub_context.mcp_discovery_scopes()))
        record.conversation = agent.start_conversation(_task_prompt(record))
        saved, record.saved_messages = record.saved_messages, None
        if saved:
            # Restored after a suspension: continue the earlier conversation,
            # and take follow-ups as new turns like a live finished agent does.
            record.conversation._replace_messages(saved, reason="resume")
            record.conversation.runtime_state.shell_job_manager.import_jobs(record.saved_shell_jobs)
            record.saved_shell_jobs = []
        with self._condition:
            pending = list(record.followups)
            record.followups.clear()
        for message in pending:
            self._materialize_outputs(record, message)
            if saved:
                record.conversation.add_user_turn(message)
            else:
                record.conversation.queue_message(message)

    def _materialize_outputs(self, record: AgentRecord, text: str) -> None:
        if record.conversation is None or record.role != "worker":
            return
        sandbox = self.changes.sandbox_for(record.id)
        if sandbox is None:
            return
        for reference in self.context.tool_output_artifacts.copy().values():
            if reference["path"] in text:
                copy_artifact(self.context.workspace.root, sandbox.workspace, reference)
                record.conversation.agent.tool_context.tool_output_artifacts[reference["path"]] = dict(reference)

    def _refresh_child_mcp(self, agent, context: ToolContext, catalog: PublishedMcpCatalog | None = None) -> None:
        registry = context.tool_registry
        previous = registry.specs()
        specs = catalog.specs if catalog is not None else self.context.tool_registry.specs()
        registry.replace_mcp_specs([
            spec for spec in specs
            if spec.origin_server is not None and context.tool_capability in spec.capabilities
        ])
        invalidated = context.invalidate_mcp_tools(previous, statuses=catalog.statuses if catalog is not None else None)
        if invalidated:
            agent.update_tool_schemas([
                schema for schema in agent.tool_schemas
                if schema["function"]["name"] not in invalidated
            ])
        if context.mcp_discovery_scopes is not None:
            agent.set_tool_discovery_catalog(discovery_catalog_prompt(registry, context.mcp_discovery_scopes()))

    @staticmethod
    def _role_capability(role: str) -> str:
        return TOOL_CAPABILITY_WORKER_AGENT if role == "worker" else TOOL_CAPABILITY_READONLY_AGENT

    def _role_registry(self, role: str) -> ToolRegistry:
        # Depth limit: subagent registries are capability-filtered views of
        # the parent registry. spawn_agent carries no *_AGENT capability, so
        # workers/subagents cannot spawn further agents (hard depth = 1,
        # enforced structurally — there is no depth counter or state machine).
        source = self.context.tool_registry
        if source is None:
            raise RuntimeError("parent tool registry is unavailable")
        capability = self._role_capability(role)
        registry = ToolRegistry()
        for spec in source.specs():
            if capability not in spec.capabilities:
                continue
            # Deferred specs stay in the registry so child agents can discover
            # them through search_tools, but they are not part of the initial
            # schema set (see ``_ensure_conversation``).
            registry.register_spec(spec)
        return registry

    def _finish(self, record: AgentRecord, state: str, *, error: str | None = None) -> None:
        try:
            self._publish_outputs(record)
        except (OSError, ValueError) as exc:
            state, error = "failed", f"Could not preserve tool output artifacts: {exc}"
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

    def _publish_outputs(self, record: AgentRecord) -> None:
        if record.conversation is None:
            return
        sandbox = self.changes.sandbox_for(record.id)
        from ..workspace.paths import observation_dir_for

        source_root = sandbox.workspace if sandbox is not None else self.context.workspace.root
        directory = observation_dir_for(source_root, self.context.session_id, record.id)
        if not directory.resolve().is_relative_to(source_root.resolve()):
            raise ValueError("Tool output directory escapes its workspace")
        manager = getattr(getattr(record.conversation, "runtime_state", None), "shell_job_manager", None)
        jobs = manager.list_jobs() if manager is not None else []
        for path in directory.glob("*.txt"):
            if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Tool output artifact escapes its directory")
            output = next((job.output for job in jobs if job.output.path == path), None)
            with output._lock if output is not None else nullcontext():
                reference = {"path": path.relative_to(source_root).as_posix(), "sha256": file_digest(path),
                             "bytes": path.stat().st_size, "append_only": output is not None}
                copy_artifact(source_root, self.context.workspace.root, reference)
                self.context.register_output_artifact(reference)
        record.output_artifacts = [reference for reference in self.context.tool_output_artifacts.copy().values()
                                   if Path(reference["path"]).parent == directory.relative_to(source_root)]

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
            "task": record.task,
            "status": record.state,
            "summary": record.summary,
            "error": record.error,
            "proposal_id": record.proposal_id,
            "output_artifacts": list(record.output_artifacts),
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


def _role_prompt(record: AgentRecord, parent_messages: list[dict], system: str) -> str:
    role_text = role_definition(record.role).instruction
    inherited = _forked_context(parent_messages, record.fork_turns)
    return (
        f"You are the {record.name} subagent with role {record.role}. {role_text}\n"
        "Stay within the assigned task. The parent owns integration and final completion.\n"
        "Use send_parent_message only for: a finding that changes the parent's plan, a missing "
        "decision or constraint you cannot resolve, or a conflict / high-value risk. Do not send "
        "routine progress such as status percentages; return the final result through task completion.\n"
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
    visible = [message for message in messages if message.get("role") in {"user", "assistant", "tool"}]
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


def _paths_overlap(left: str, right: str) -> bool:
    return path_within(left, [right]) or path_within(right, [left])
