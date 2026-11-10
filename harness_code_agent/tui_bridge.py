"""NDJSON bridge between the OpenTUI process and the Python runtime.

The bridge deliberately owns no terminal state.  It translates the existing
``InteractiveSession`` event stream into a small, stable protocol so the
terminal renderer can be replaced without moving agent/runtime code into Bun.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config
from .attachments import (
    AttachmentError,
    ExternalPathConfirmationRequired,
    PreparedTurn,
    TurnSubmission,
    classify_attachment,
    model_input_mode,
)
from .core.interactive import InteractiveSession
from .runtime.approvals import ApprovalRequest, ApprovalResult
from .runtime.permissions import PermissionPolicy
from .runtime.questions import (
    QuestionOption,
    QuestionRequest,
    QuestionResult,
    question_result_from_option,
)
from .sessions.events import SessionEvent
from .sessions.observability import (
    export_observability_report,
    format_export_result,
    format_project_observability,
    format_session_observability,
)
from .support.cancellation import CancellationToken, CancelledError
from .tui.approval import ApprovalAllowlist, _persistent_prefix_for_request
from .tui.commands import default_command_registry
from .tui.completion import MentionIndex
from .tui.projection import SECTION_EVENTS, WorkStateProjection
from .tui.protocol import UI_PROTOCOL_VERSION, validate_ui_event
from .tui.state import SessionStatusSnapshot, TranscriptBlock, TuiState, _localize_error

log = logging.getLogger("harness.opentui")

_PERMISSION_COPY = {
    PermissionPolicy.WORKSPACE_WRITE: ("请求批准", "工作区内写入允许，风险操作询问，工作区外写入拒绝"),
    PermissionPolicy.LLM_AUTO: ("替我审批", "由模型判断风险并自动批准具体、范围明确的安全操作"),
    PermissionPolicy.DANGER_FULL_ACCESS: ("完全访问", "工作区内外修改均不询问；灾难性命令仍会被拦截"),
    PermissionPolicy.READ_ONLY: ("只读", "禁止一切修改与未知程序执行"),
}

_MODEL_COPY = {
    "deepseek-flash": ("DeepSeek V4.1 Flash", "通用快速模型，支持图片输入"),
    "deepseek-v4-pro": ("DeepSeek V4 Pro", "旗舰模型，适合复杂推理与编码"),
}

_EFFORT_COPY = {
    "low": ("低", "简单任务，响应更快"),
    "high": ("高", "默认强度，适合日常任务"),
    "max": ("最大", "最深推理，适合复杂场景"),
}


def _relative_age(value: Any) -> str:
    try:
        created = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        seconds = max(0, int((datetime.now(timezone.utc) - created.astimezone(timezone.utc)).total_seconds()))
    except (TypeError, ValueError):
        return "时间未知"
    if seconds < 60:
        return "刚刚"
    if seconds < 3600:
        return f"{seconds // 60} 分钟前"
    if seconds < 86400:
        return f"{seconds // 3600} 小时前"
    if seconds < 86400 * 30:
        return f"{seconds // 86400} 天前"
    return created.astimezone().strftime("%Y-%m-%d")


def _localize_observability(text: str) -> str:
    replacements = {
        "Observability dashboard": "可观测性面板",
        "Project observability": "项目可观测性",
        "session:": "会话：",
        "model:": "模型：",
        "status:": "状态：",
        "created_at:": "创建时间：",
        "tokens:": "令牌：",
        "tools:": "工具：",
        "performance:": "性能：",
        "audit:": "审计：",
        "tool breakdown:": "工具明细：",
        "recent audit events:": "最近审计事件：",
        "top token sessions:": "令牌消耗最多的会话：",
        "top failure sessions:": "失败最多的会话：",
        "low cache sessions:": "缓存命中率较低的会话：",
        "success rate:": "成功率：",
        "cache hit ratio:": "缓存命中率：",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _format_attachment_size(size: int) -> str:
    """Mirror the frontend formatBytes helper for transcript summaries."""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _user_transcript_body(text: str, attachments: tuple) -> str:
    """Build the user transcript row body, matching the optimistic client row."""
    lines = [text.strip()]
    for item in attachments:
        data = item.public_dict()
        lines.append(f"[{data['kind']}] {data['name']} ({_format_attachment_size(int(data['size']))})")
    return "\n".join(line for line in lines if line)


class BridgeInteractionProvider:
    """Deep interaction module shared by approvals and agent questions.

    Runtime worker threads block on this module while the bridge's stdin loop
    stays free to receive the matching ``resolve_interaction`` request.
    """

    def __init__(self, emit, *, project_root: Path) -> None:
        self._emit = emit
        self._allowlist = ApprovalAllowlist(project_root)
        self._lock = threading.Lock()
        self._counter = 0
        self._pending: dict[str, tuple[threading.Event, dict[str, Any]]] = {}
        self._presentation_lock = threading.Lock()
        self._closed = False

    def request(self, request: ApprovalRequest) -> ApprovalResult:
        if request.tool_name == "run_bash":
            rule = self._allowlist.match(str(request.args.get("command", "")))
            if rule is not None:
                return ApprovalResult(
                    True,
                    "approved by project allowlist",
                    {"ui": "opentui", "approval_source": "project_allowlist", "prefix": rule.get("prefix", [])},
                )
        interaction_id, result = self._open(
            "approval",
            {
                "toolName": request.tool_name,
                "args": request.args,
                "risk": request.risk,
                "reason": request.reason,
                "persistAvailable": _persistent_prefix_for_request(request) is not None,
            },
        )
        decision = str(result.get("decision") or "deny")
        approved = decision in {"approve", "persist"}
        if decision == "persist":
            prefix = _persistent_prefix_for_request(request)
            if prefix:
                self._allowlist.add_prefix_rule(prefix, command=str(request.args.get("command", "")))
        return ApprovalResult(
            approved,
            "approved in OpenTUI" if approved else "denied in OpenTUI",
            {"ui": "opentui", "interaction_id": interaction_id, "persisted": decision == "persist"},
        )

    def ask(self, request: QuestionRequest) -> QuestionResult:
        interaction_id, result = self._open(
            "question",
            {
                "question": request.question,
                "options": [option.to_dict() for option in request.options],
                "details": request.details,
            },
        )
        if bool(result.get("cancelled")):
            return QuestionResult(
                cancelled=True,
                reason="cancelled in OpenTUI",
                metadata={"ui": "opentui", "interaction_id": interaction_id},
            )
        try:
            selected_index = int(result.get("selectedIndex"))
        except (TypeError, ValueError):
            return QuestionResult(cancelled=True, reason="invalid OpenTUI question result")
        if not 0 <= selected_index < len(request.options):
            return QuestionResult(cancelled=True, reason="invalid OpenTUI question option")
        return question_result_from_option(
            request,
            selected_index,
            custom_text=str(result.get("customText") or "").strip(),
            metadata={"ui": "opentui", "interaction_id": interaction_id},
        )

    def _open(self, kind: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        from .support.cancellation import current_token
        token = current_token.get()
        while not self._presentation_lock.acquire(timeout=0.05):
            if token is not None:
                token.check()
            if self._closed:
                raise CancelledError("工作台正在关闭")
        try:
            if token is not None:
                token.check()
            if self._closed:
                raise CancelledError("工作台正在关闭")
            return self._open_presented(kind, payload)
        finally:
            self._presentation_lock.release()

    def _open_presented(self, kind: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        with self._lock:
            self._counter += 1
            interaction_id = f"interaction-{self._counter}"
            event = threading.Event()
            result: dict[str, Any] = {}
            self._pending[interaction_id] = (event, result)
        self._emit({"type": "interaction", "id": interaction_id, "kind": kind, "payload": payload})
        from .support.cancellation import current_token
        token = current_token.get()
        try:
            while not event.wait(timeout=0.05):
                if token is not None:
                    token.check()
            if token is not None:
                token.check()
        finally:
            with self._lock:
                self._pending.pop(interaction_id, None)
            self._emit({"type": "interaction_closed", "id": interaction_id})
        return interaction_id, result

    def resolve(self, interaction_id: str, result: dict[str, Any]) -> bool:
        with self._lock:
            pending = self._pending.get(interaction_id)
            if pending is None:
                return False
            event, target = pending
            target.update(result)
            event.set()
            return True

    def cancel_all(self) -> None:
        with self._lock:
            self._closed = True
            pending = list(self._pending.values())
        for event, result in pending:
            result["cancelled"] = True
            result.setdefault("decision", "deny")
            event.set()



class SessionBridge:
    """Own one Python session and serve request/event messages over stdio."""

    def __init__(self, *, cwd: Path,
                 session_id: str, state_root: Path, environment: str, emit, session_action,
                 project_resources=None) -> None:
        self.runtime_id = session_id
        self.environment = environment
        self.state_root = state_root
        self._emit_write = emit
        self._session_action = session_action
        self.project_resources = project_resources
        self._runtime_lock = threading.RLock()
        self._runtime_users = 0
        # Set while no suspension is closing the runtime; the close itself
        # runs outside _runtime_lock so requests are never stuck behind it.
        self._not_suspending = threading.Event()
        self._not_suspending.set()
        self._worker_lock = threading.Lock()
        self._suspend_retry_at = 0
        self._last_activity = time.monotonic()
        self.cwd = cwd.resolve()
        self._write_lock = threading.Lock()
        self._tasks: queue.Queue[PreparedTurn | Any | None] = queue.Queue()
        self.work = WorkStateProjection(self.cwd)
        self._rewind_pending = False
        self._active_token: CancellationToken | None = None
        self._active_lock = threading.Lock()
        self._stopping = threading.Event()
        self._closing = threading.Event()
        self._output_broken = threading.Event()
        self._session_lock = threading.Lock()
        self._session_generation = 0
        self._session: InteractiveSession | None = None
        self._session_thread = None
        self._worker = None
        self._session_error: str | None = None
        self._assistant_group_id: str | None = None
        self._assistant_group_counter = 0
        self._assistant_id: str | None = None
        self._assistant_text = ""
        self._assistant_counter = 0
        self.state = TuiState(
            snapshot=SessionStatusSnapshot(
                model=config.MODEL,
                provider=config.PROVIDER,
                permission_mode=os.environ.get("HARNESS_PERMISSION_MODE", "workspace-write"),
                session_id=None,
                cwd=self.cwd,
                status="starting",
            )
        )
        self._interactions = BridgeInteractionProvider(self._send_event, project_root=self.cwd)
        self._mention_index: MentionIndex | None = None
        from .sessions.store import SessionStore
        store = SessionStore(self.state_root)
        if not (store.sessions_dir / self.runtime_id / "session.json").exists():
            store.create(cwd=self.cwd, model=config.MODEL,
                         permission_mode=self.state.snapshot.permission_mode,
                         session_id=self.runtime_id,
                         environment=environment)
        metadata = store.read_metadata(self.runtime_id)
        self._model_selection = config.ModelSelection(**metadata.get("model_selection", {}))
        self.state.snapshot.session_id = self.runtime_id
        self.state.snapshot.permission_mode = metadata.get("permission_mode", self.state.snapshot.permission_mode)
        self.state.snapshot.status = "idle"
        items = self._history_items(self.runtime_id)
        self._emit_commands()
        self._send_event({"type": "session_reset", "snapshot": self._snapshot_payload(), "items": items})
        self._send_event({"type": "runtime_state", "state": "dormant"})
        self._send_event({"type": "progress", "status": "ready", "detail": ""})

    def _start_session_construction(self) -> None:
        with self._session_lock:
            self._session_generation += 1
            generation = self._session_generation
        thread = threading.Thread(
            target=self._construct_session,
            kwargs={"generation": generation},
            name="opentui-session",
            daemon=True,
        )
        self._session_thread = thread
        thread.start()

    def _is_current_generation(self, generation: int) -> bool:
        with self._session_lock:
            return generation == self._session_generation and not self._closing.is_set()

    def _construct_session(self, *, generation: int) -> None:
        session: InteractiveSession | None = None
        try:
            session = InteractiveSession(
                cwd=self.cwd,
                session_id=self.runtime_id,
                state_root=self.state_root,
                environment=self.environment,
                project_resources=self.project_resources,
                stream_sink=lambda text: (
                    self._stream_delta(text) if self._is_current_generation(generation) else None
                ),
                event_listener=lambda event: (
                    self._event_listener(event) if self._is_current_generation(generation) else None
                ),
                approval_provider=self._interactions,
                question_provider=self._interactions,
                output_sink=lambda text: (
                    self._notice("info", str(text)) if self._is_current_generation(generation) else None
                ),
                enable_turn_summary=False,
                startup_sink=lambda stage: (
                    self._startup_progress(stage) if self._is_current_generation(generation) else None
                ),
            )
            with self._session_lock:
                stale = generation != self._session_generation or self._closing.is_set()
                if not stale:
                    self._session = session
                    self._session_error = None
            if stale:
                session.close()
                return
            self._emit_commands()
            self._send_snapshot()
            self._send_work_state()
            self._send_event({"type": "progress", "status": "ready", "detail": "Python 会话已就绪。"})
            self._send_event({"type": "runtime_state", "state": "warm"})
        except Exception as exc:  # pragma: no cover - exercised by real startup failures
            error = _localize_error(exc, "会话启动失败，请稍后重试")
            with self._session_lock:
                current = generation == self._session_generation and not self._closing.is_set()
                if current:
                    self._session_error = error
            if current:
                # Startup failure surfaces through progress/failed (the
                # welcome row); an extra error notice would say it twice.
                self._send_event({"type": "progress", "status": "failed", "detail": error})
            log.debug("OpenTUI bridge session construction failed\n%s", traceback.format_exc())

    def _startup_progress(self, stage: str) -> None:
        self._send_event({"type": "progress", "status": stage, "detail": stage})

    def _emit_commands(self) -> None:
        try:
            registry = default_command_registry(skill_registry=self._session.skill_registry if self._session else None)
            commands = [
                {
                    "name": spec.name,
                    "category": spec.group,
                    "description": spec.description,
                }
                for spec in registry.candidates()
            ]
        except Exception as exc:
            log.debug("Failed to build slash command catalog: %s", exc)
            commands = []
        self._send_event({"type": "commands", "commands": commands})

    def _event_listener(self, event: SessionEvent) -> None:
        try:
            block = self.state.apply_event(event)
            changed = self.work.apply_event(event)
            if self._refresh_proposal_state():
                changed.add("changes")
            for key in changed:
                self._send_work_section(key)
            event_type = event.type
            if event_type == "turn_started":
                self._begin_assistant_group(int(event.payload.get("turn") or self.state.snapshot.turn))
                if not self._stopping.is_set():
                    self._send_event({"type": "turn_state", "state": "running"})
            elif event_type == "turn_finished":
                self._close_assistant_group({"completed": "success", "cancelled": "cancelled", "failed": "error"}.get(event.payload.get("status"), "success"))
            elif event_type == "recovery_point_created":
                self._send_event({"type": "recovery_available", "turn": event.payload["turn"], "pointId": event.payload["point_id"]})
                self._send_event({"type": "turn_state", "state": "idle"})
            elif event_type == "session_started":
                self._send_snapshot()
            elif event_type == "session_finished":
                self._close_assistant_group("success")
                self._send_event({"type": "turn_state", "state": "idle"})

            if event_type == "assistant_message":
                if self._assistant_id is not None:
                    # Streaming already created this child item; deltas own its body.
                    self._send_event(
                        {
                            "type": "transcript_update",
                            "id": self._assistant_id,
                            "body": str(event.payload.get("text") or ""),
                            "state": "success",
                        }
                    )
                    block = None
                self._reset_assistant_segment()
            elif event_type in {"tool_call", "tool_result"}:
                # A tool boundary ends the current model-text child, but not
                # the parent assistant group for this user turn.
                self._reset_assistant_segment("success")
            if block is not None and event_type != "user_input":
                self.state.add_block(block)
                self._send_transcript(block)
            if event_type not in {"assistant_message", "user_input"}:
                self._send_snapshot()
        except Exception as exc:  # listener errors must never break the runtime
            log.debug("OpenTUI event translation failed: %s", exc, exc_info=True)

    def _stream_delta(self, text: str) -> None:
        if not text or self._closing.is_set():
            return
        self._begin_assistant()
        self._assistant_text += text
        self._send_event({"type": "assistant_delta", "id": self._assistant_id, "text": text})

    def _begin_assistant(self) -> None:
        if self._assistant_id is None:
            self._assistant_counter += 1
            self._assistant_id = f"assistant-{self._assistant_counter}"
            self._assistant_text = ""
            item: dict[str, Any] = {
                "id": self._assistant_id,
                "kind": "assistant",
                "title": "助手",
                "body": "",
                "state": "running",
                "role": "message",
            }
            if self._assistant_group_id:
                item["parentId"] = self._assistant_group_id
            self._send_event({"type": "transcript", "item": item})

    def _begin_assistant_group(self, turn: int) -> None:
        if self._assistant_group_id is not None:
            # Already opened (e.g. pre-opened by _run_task before routing);
            # turn_started must not duplicate it.
            return
        self._assistant_group_counter += 1
        self._assistant_group_id = f"assistant-group-{turn}-{self._assistant_group_counter}"
        self._reset_assistant_segment()
        self._send_event(
            {
                "type": "transcript",
                "item": {
                    "id": self._assistant_group_id,
                    "kind": "assistant",
                    "title": "助手",
                    "body": "",
                    "state": "running",
                    "role": "group",
                    "turn": turn,
                },
            }
        )

    def _close_assistant_group(self, status: str) -> None:
        if self._assistant_group_id is None:
            return
        self._reset_assistant_segment(status)
        self._send_event(
            {
                "type": "transcript_update",
                "id": self._assistant_group_id,
                "body": "",
                "state": status,
            }
        )
        self._assistant_group_id = None
        self._reset_assistant_segment()

    def _reset_assistant_segment(self, status: str | None = None) -> None:
        if self._assistant_id is not None and status is not None:
            self._send_event(
                {
                    "type": "transcript_update",
                    "id": self._assistant_id,
                    "body": self._assistant_text,
                    "state": status,
                }
            )
        self._assistant_id = None
        self._assistant_text = ""

    def _finish_interrupted_assistant(self) -> None:
        if self._assistant_id is None:
            self._begin_assistant()
        body = self._assistant_text.rstrip()
        body = f"{body}\n\n已停止" if body else "已停止"
        self._send_event({
            "type": "transcript_update",
            "id": self._assistant_id,
            "body": body,
            "state": "failed",
        })
        self._reset_assistant_segment()
        self._close_assistant_group("failed")

    def _send_transcript(self, block: TranscriptBlock) -> None:
        self._send_event({"type": "transcript", "item": self._block_item(block)})

    def _block_item(self, block: TranscriptBlock) -> dict[str, Any]:
        kind_map = {
            "assistant": "assistant",
            "tool": "tool",
            "error": "error",
            "failure": "error",
            "plan": "plan",
            "user": "user",
            "file": "file",
            "thought": "thought",
            "agent": "agent",
        }
        kind = kind_map.get(block.kind, "status")
        state = {
            "running": "running",
            "success": "success",
            "failed": "failed",
            "blocked": "failed",
            "cancelled": "failed",
            "pending": "pending",
        }.get(block.status, "success")
        item = {
            "id": block.id or f"block-{self.state.snapshot.turn}-{len(self.state.blocks)}-{block.title}",
            "kind": kind,
            "title": block.title,
            "body": block.body,
            "state": state,
        }
        if block.turn is not None:
            item["turn"] = block.turn
        if block.direction:
            item["direction"] = block.direction
        if self._assistant_group_id:
            item["parentId"] = self._assistant_group_id
        return item

    def _send_snapshot(self) -> None:
        self._send_event({"type": "snapshot", "snapshot": self._snapshot_payload()})

    def _refresh_proposal_state(self) -> bool:
        session = self._session
        coordinator = getattr(getattr(session, "tool_context", None), "agent_coordinator", None)
        if coordinator is None:
            return False
        try:
            return self.work.merge_proposals(coordinator.changes.snapshot_proposals())
        except Exception:
            return False

    def _send_work_section(self, key: str) -> None:
        self._send_event({"type": SECTION_EVENTS[key], key: self.work.section(key)})

    def _send_work_state(self) -> None:
        self._refresh_proposal_state()
        for key in SECTION_EVENTS:
            self._send_work_section(key)

    def _snapshot_payload(self) -> dict[str, Any]:
        snapshot = self.state.snapshot
        context_percent = 99
        if snapshot.context_window_tokens:
            context_percent = max(
                0,
                min(100, round((1 - snapshot.context_tokens / snapshot.context_window_tokens) * 100)),
            )
        profile = config.resolve_model_profile(config.MODEL_INTENSITY,
                                               self._session.model_selection if self._session else self._model_selection)
        return {
            "permissionMode": snapshot.permission_mode,
            "model": profile.model,
            "reasoningEffort": profile.reasoning_effort,
            "provider": snapshot.provider,
            "contextPercent": context_percent,
            "contextTokens": snapshot.context_tokens,
            "contextWindowTokens": snapshot.context_window_tokens,
            "status": snapshot.status,
            "cwd": str(snapshot.cwd),
            "sessionId": snapshot.session_id,
            "dirtyCount": snapshot.dirty_count,
            "inputMode": model_input_mode(),
        }

    def _notice(self, level: str, text: str) -> None:
        self._send_event({"type": "notice", "level": level, "text": text})

    def _send_event(self, event: dict[str, Any]) -> None:
        if self._output_broken.is_set():
            return
        if self._closing.is_set() and event.get("type") != "shutdown":
            return
        validate_ui_event(event)
        self._write({"type": "event", "event": event})

    def _write(self, message: dict[str, Any]) -> None:
        self._emit_write(message)

    def _response(self, request_id: str, *, result: Any = None, error: str | None = None) -> None:
        message: dict[str, Any] = {"type": "response", "id": request_id, "ok": error is None}
        if error is None:
            message["result"] = result
        else:
            message["error"] = error
        self._write(message)

    def _enqueue_task(self, task):
        with self._worker_lock:
            if self._closing.is_set():
                raise RuntimeError("会话正在关闭")
            self._last_activity = time.monotonic()
            self._tasks.put(task)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._worker_loop, name="opentui-submit", daemon=True)
                self._worker.start()

    def _worker_loop(self) -> None:
        while not self._closing.is_set():
            with self._worker_lock:
                try:
                    task = self._tasks.get_nowait()
                except queue.Empty:
                    self._worker = None
                    return
                token = CancellationToken()
                with self._active_lock:
                    self._active_token = token
            self._assistant_id = None
            self._assistant_text = ""
            if self._assistant_group_id is not None:
                # A previous task crashed before turn_finished; close the
                # stale group so this turn's group can open cleanly.
                self._close_assistant_group("success")
            self._send_event({"type": "turn_state", "state": "running"})
            try:
                from .support.cancellation import cancellation_scope
                with cancellation_scope(token):
                    if self._session is not None:
                        while self._session.tool_context.tool_tasks.pending_foreground_count:
                            token.check()
                            self._closing.wait(.05)
                    if callable(task):
                        task = task()
                    if isinstance(task, PreparedTurn):
                        self._run_task(task, token)
            except CancelledError:
                self._finish_interrupted_assistant()
            except Exception as exc:  # runtime errors stay inside the protocol
                self._finish_interrupted_assistant()
                self._notice("error", _localize_error(exc, "回合执行失败，请稍后重试"))
                log.debug("OpenTUI submit failed\n%s", traceback.format_exc())
            finally:
                with self._active_lock:
                    self._active_token = None
                self._stopping.clear()
                self._send_event({"type": "turn_state", "state": "idle"})
                self._tasks.task_done()

    def _enqueue_external_submission(self, submission: TurnSubmission, request_id=None) -> None:
        """Queue approval-then-submit for a prompt referencing external files.

        Runs on the worker thread so the approval interaction can block while
        the stdin loop keeps processing resolve_interaction. On approval the
        real user transcript row is emitted (optimistic rows only exist for
        accepted submissions) and the prepared turn re-enters the normal task
        path; on denial nothing is submitted.
        """

        session = self._require_session()

        def task() -> None:
            if session is not self._session:
                return
            try:
                prepared = session.prepare_with_external_approval(submission)
                prepared = replace(prepared, ui_request_id=request_id)
            except AttachmentError as exc:
                self._notice("info", str(exc))
                return
            self._send_event({
                "type": "transcript",
                "item": {
                    "id": f"user-{uuid.uuid4().hex}",
                    "kind": "user",
                    "title": "你",
                    "body": _user_transcript_body(submission.text, prepared.attachments),
                },
            })
            self._enqueue_task(prepared)

        self._enqueue_task(task)

    def _run_task(self, task: PreparedTurn, token: CancellationToken) -> None:
        session = self._require_session()
        text = task.text
        session.tool_context.ui_request_id = task.ui_request_id
        session.conversation.trace._write("ui_request", {"ui_request_id": task.ui_request_id, "turn_id": None})
        registry = default_command_registry(skill_registry=session.skill_registry)
        if not task.attachments and text.lstrip().startswith("/") and not registry.is_agent_command(text):
            should_continue = session.handle_slash_command(text)
            result = getattr(session, "last_command_result", None)
            if result is not None:
                if getattr(result, "text", ""):
                    self._notice("info", str(result.text))
                action = getattr(result, "action", None)
                if action in {"mcp", "observe"}:
                    self._send_event({"type": "panel", "panel": self._panel(str(action))})
                elif action == "compact":
                    self._notice("info", session.compact_current_context())
                elif action == "fork":
                    self._session_action("fork_session", {})
            if not should_continue:
                self._closing.set()
            return
        # Open the assistant group before routing/preparation so the transcript
        # shows "助手" immediately; turn_started later finds it already open.
        self._begin_assistant_group(session.turn_count + 1)
        result = session.submit_prepared(task, cancellation_token=token)
        if getattr(result, "notice", ""):
            self._notice("info", str(result.notice))

    def _sessions_panel(self):
        return self._session_action("open_sessions", {})["panel"]

    def _panel(self, kind: str) -> dict[str, Any]:
        session = self._require_session()
        if kind == "permission":
            current = session.permission_mode
            options = []
            for mode in (
                PermissionPolicy.WORKSPACE_WRITE,
                PermissionPolicy.LLM_AUTO,
                PermissionPolicy.DANGER_FULL_ACCESS,
            ):
                label, description = _PERMISSION_COPY[mode]
                options.append({
                    "id": mode,
                    "label": label,
                    "description": description,
                    "tone": "success" if current == mode else ("danger" if mode == PermissionPolicy.DANGER_FULL_ACCESS else "default"),
                    "selected": current == mode,
                })
            return {
                "kind": "permission",
                "title": "审批模式",
                "options": options,
            }
        if kind == "model":
            current = config.resolve_model_profile(config.MODEL_INTENSITY, self._require_session().model_selection).model
            options = [
                {
                    "id": name,
                    "label": _MODEL_COPY.get(name, (name, ""))[0],
                    "description": _MODEL_COPY.get(name, (name, ""))[1],
                    "tone": "success" if current == name else "default",
                    "selected": current == name,
                }
                for name in config.AVAILABLE_MODELS
            ]
            return {"kind": "model", "title": "模型", "options": options}
        if kind == "effort":
            current = config.resolve_model_profile(config.MODEL_INTENSITY, self._require_session().model_selection).reasoning_effort
            options = [
                {
                    "id": effort,
                    "label": _EFFORT_COPY[effort][0],
                    "description": _EFFORT_COPY[effort][1],
                    "tone": "success" if current == effort else "default",
                    "selected": current == effort,
                }
                for effort in config.REASONING_EFFORTS
            ]
            return {"kind": "effort", "title": "推理强度", "options": options}

        if kind == "mcp":
            session.mcp_status()
            manager = session.mcp_manager
            statuses = getattr(manager, "statuses", {})
            names = set(manager.configured_server_names())
            names.update(name for name in statuses if name != "config")
            options = [
                {
                    "id": "add",
                    "label": "新增 MCP 服务",
                    "description": "/mcp add <名称> <URL> 或 /mcp add <名称> --command <命令>",
                },
                {"id": "reload", "label": "重新加载全部 MCP 服务"},
            ]
            for name in sorted(names):
                status = statuses.get(name)
                state = "已停用" if status is None else ("已连接" if status.state == "connected" else "连接失败")
                server = manager.config.servers.get(name)
                description = (server.description if server is not None else "") or state
                options.append({"id": f"reconnect:{name}", "label": f"重新连接 {name}", "description": state})
                options.append({"id": f"describe:{name}", "label": f"重新生成描述 {name}", "description": description})
                options.append({"id": f"remove:{name}", "label": f"移除 {name}"})
                options.append({"id": f"toggle:{name}", "label": f"切换启用 {name}"})
            options.append({"id": "open_config", "label": "显示 MCP 配置路径"})
            connected = sum(getattr(status, "state", "") == "connected" for status in statuses.values())
            return {
                "kind": "mcp",
                "title": "MCP 管理",
                "body": f"已连接 {connected} 个服务 · 已注册 {len(getattr(manager, 'tool_bindings', []))} 个工具",
                "options": options,
            }
        if kind == "observe":
            session_id = getattr(getattr(session, "session", None), "id", None)
            body = _localize_observability(format_session_observability(session.session_store, session_id)) if session_id else "当前还没有会话。"
            return {
                "kind": "observe",
                "title": "运行观察 · 当前会话",
                "body": body,
                "options": [
                    {"id": "observe:project", "label": "切换到项目概览"},
                    {"id": "observe:export-current", "label": "导出当前会话报告"},
                ],
            }
        if kind == "help":
            commands = default_command_registry(skill_registry=session.skill_registry).candidates()
            return {
                "kind": "help",
                "title": "快捷键与命令",
                "body": "Enter 提交  Shift+Enter 换行  Ctrl+C 关闭工作台  Ctrl+O 运行观察  Ctrl+P 打开审批模式\nAlt+←/→ 切换会话  Ctrl+N 新建  Ctrl+Shift+W 隐藏标签\n\n"
                + "\n".join(f"{item.usage:<18} {item.description}" for item in commands),
            }
        raise ValueError(f"unknown panel: {kind}")

    def _panel_action(self, panel: str, action: str) -> dict[str, Any]:
        if self._rewind_pending:
            raise ValueError("回撤正在进行，请稍候")
        session = self._require_session()
        if panel == "sessions":
            return self._session_action("switch_session", {"sessionId": action})
        if panel == "command":
            if action == "compact":
                message = session.compact_current_context()
            elif action == "fork":
                return self._session_action("fork_session", {})
            else:
                raise ValueError(f"unknown command action: {action}")
        elif panel == "permission":
            message = session.set_permission_mode(action)
        elif panel == "model":
            session.apply_model_override(model=action)
        elif panel == "effort":
            session.apply_model_override(reasoning_effort=action)

        elif panel == "mcp":
            name = action.split(":", 1)[1] if ":" in action else ""
            if action == "add":
                message = "在输入框执行 /mcp add <名称> <URL> 或 /mcp add <名称> --command <命令>"
            elif action == "reload":
                message = session.reload_mcp()
            elif action.startswith("reconnect:"):
                message = session.reload_mcp_server(name)
            elif action.startswith("describe:"):
                message = session.refresh_mcp_server_description(name, refresh=True)
            elif action.startswith("remove:"):
                message = session.remove_mcp_server(name)
            elif action.startswith("toggle:"):
                message = session.toggle_mcp_server(name)
            elif action == "open_config":
                message = f"MCP 配置：{session.mcp_manager.config.path}"
            else:
                raise ValueError(f"unknown MCP action: {action}")
        elif panel == "observe":
            session_id = getattr(getattr(session, "session", None), "id", None)
            if action == "observe:project":
                return {"ok": True, "panel": {"kind": "observe", "title": "运行观察 · 项目概览", "body": _localize_observability(format_project_observability(session.session_store)), "options": [{"id": "observe:current", "label": "切换到当前会话"}, {"id": "observe:export-project", "label": "导出项目报告"}]}}
            if action == "observe:current":
                return {"ok": True, "panel": self._panel("observe")}
            if action == "observe:export-current":
                message = format_export_result(export_observability_report(session.session_store, mode="current", session_id=session_id))
            elif action == "observe:export-project":
                message = format_export_result(export_observability_report(session.session_store, mode="project"))
            else:
                raise ValueError(f"unknown observe action: {action}")
        else:
            raise ValueError(f"unknown panel action: {panel}")
        self._send_snapshot()
        if panel in {"permission", "model", "effort"}:
            return {"ok": True}
        return {"ok": True, "message": str(message or "")}

    def _workbench_action(self, params: dict[str, Any]) -> dict[str, Any]:
        # Workbench is observation-only: these ops are pure inspections.
        # Proposal integration/conflict resolution are owned by the main
        # agent, and the plan decision is a question interaction.
        session = self._require_session()
        coordinator = getattr(session.tool_context, "agent_coordinator", None)
        op = str(params.get("op") or "")

        if coordinator is None:
            raise RuntimeError("当前会话没有多代理协调器")
        changes = coordinator.changes
        proposal_id = str(params.get("proposalId") or "")
        if op == "read_proposal":
            result = changes.read_changes(proposal_id)
            return {"ok": True, "content": result["content"], "paths": result["paths"], "totalChars": result["total_chars"]}
        if op == "read_conflict":
            result = changes.read_conflicts(str(params.get("conflictId") or ""))
            return {"ok": True, "content": result["content"], "paths": result["paths"], "totalChars": result["total_chars"]}
        raise ValueError(f"unknown workbench action: {op}")

    def _history_items(self, session_id: str) -> list[dict[str, Any]]:
        from .sessions.store import SessionStore
        store = SessionStore(self.state_root)
        metadata = store.read_metadata(session_id)
        replay = TuiState(snapshot=SessionStatusSnapshot(
            model=self.state.snapshot.model,
            provider=self.state.snapshot.provider,
            permission_mode=metadata.get("permission_mode", "workspace-write"),
            session_id=session_id, cwd=self.cwd,
        ))
        items = []
        self.work = WorkStateProjection(self.cwd)
        group = None
        owned_points = {}
        for key in ("checkpoint", "branch_origin"):
            point = metadata.get(key)
            if point and point["session_id"] == session_id:
                owned_points[point["id"]] = point
        for event in store.read_history_events(session_id):
            if event["type"] in {"recovery_point_created", "session_rewound"}:
                point_id = event["payload"]["point_id"]
                path = self.cwd / ".harness" / "recovery" / "points" / f"{point_id}.json"
                if path.is_file():
                    point = json.loads(path.read_text(encoding="utf-8"))
                    if point["session_id"] == session_id:
                        owned_points[point_id] = point
        boundary = metadata.get("history_boundary")
        for index, event in enumerate(store.read_history_events(session_id, event_count=boundary["event_count"] if boundary else 0)):
            if event["type"] == "recovery_point_created" and event["payload"]["point_id"] not in owned_points:
                continue
            if event["type"] == "turn_started":
                turn = event["payload"]["turn"]
                group = {"id": f"history-group-{turn}", "kind": "assistant", "title": "助手", "body": "", "state": "success", "role": "group", "turn": turn}
                items.append(group)
            block = replay.apply_event(event)
            self.work.apply_event(event)
            if block is not None:
                replay.add_block(block)
                item = self._block_item(block)
                item["id"] = f"history-{index}-{block.kind}"
                item.pop("parentId", None)
                if group is not None and item["kind"] != "user":
                    item["parentId"] = group["id"]
                items.append(item)
            if event["type"] == "recovery_point_created":
                turn = event["payload"]["turn"]
                for item in items:
                    if item.get("role") == "group" and item.get("turn") == turn:
                        item["recoveryPointId"] = event["payload"]["point_id"]
            if event["type"] == "turn_finished":
                group = None
        for point_id, point in owned_points.items():
            if point["turn"] == 0:
                items.insert(0, {"id": "recovery-origin", "kind": "status", "title": "会话开始", "body": "", "turn": 0, "recoveryPointId": point_id})
            for item in items:
                if item.get("role") == "group" and item.get("turn") == point["turn"]:
                    item["recoveryPointId"] = point_id
        self.state = replay
        self.state.snapshot.session_id = session_id
        self.state.snapshot.permission_mode = metadata.get("permission_mode", "workspace-write")
        self.state.snapshot.status = "idle"
        self._assistant_group_id = None
        self._assistant_id = None
        return items

    def _queue_rewind(self, point_id: str) -> dict[str, Any]:
        session = self._require_session()
        with self._active_lock:
            own_request = threading.current_thread() is self._worker
            if ((self._active_token is not None and not own_request) or self._rewind_pending
                    or self._tasks.unfinished_tasks > int(own_request) or self._stopping.is_set()):
                raise ValueError("请等待当前操作结束，再回撤")
            if point_id not in session.recovery_points():
                raise ValueError("回撤点已失效，请重新选择")
            self._rewind_pending = True
            def rewind():
                try:
                    if session.rewind_to_point(point_id):
                        items = self._history_items(session.session.id)
                        self._send_event({"type": "session_reset", "snapshot": self._snapshot_payload(), "items": items})
                        self._send_work_state()
                        destination = "会话开始时" if session.turn_count == 0 else f"第 {session.turn_count} 轮结束时"
                        self._notice("info", f"已回撤到{destination}")
                finally:
                    self._rewind_pending = False
            self._enqueue_task(rewind)
        return {"ok": True, "queued": True}

    def _new_session(self):
        return self._session_action("new_session", {})

    def resource_state(self):
        session = self._session
        with self._active_lock:
            turn = self._active_token is not None
        pending = self._tasks.unfinished_tasks
        interactions = bool(self._interactions._pending)
        jobs, agents = [], []
        tools = 0
        pending_changes = False
        if session is not None:
            tools = session.tool_context.tool_tasks.pending_count
            managers = [session.conversation.runtime_state.shell_job_manager]
            coordinator = session.tool_context.agent_coordinator
            if coordinator is not None:
                pending_changes = coordinator.changes.has_pending_changes()
                with coordinator._condition:
                    records = list(coordinator._records.values())
                agents = [record.id for record in records if record.future is not None and not record.future.done()]
                for record in records:
                    if record.conversation is not None:
                        managers.append(record.conversation.runtime_state.shell_job_manager)
            for manager in managers:
                if manager is not None:
                    jobs.extend(manager.running_jobs())
        constructing = self._session_thread is not None and self._session_thread.is_alive()
        resources = self.project_resources or (session.project_resources if session is not None else None)
        executing = resources.snapshot() if resources is not None else {"active": [], "waiting": []}
        occupied = any(entry["sessionId"] == self.runtime_id for entry in [*executing["active"], *executing["waiting"]])
        active = bool(turn or pending or interactions or tools or jobs or agents or constructing or occupied
                      or pending_changes or self._runtime_users)
        return {"state": "active" if active else "warm" if session is not None else "dormant",
                    "turn": turn, "queue": max(0, pending - int(turn)), "approvals": int(interactions),
                    "tools": tools, "agents": len(agents),
                    "jobs": [{"id": job.job_id, "pid": job.pid, "status": job.status, "command": job.command,
                               "background": job.background} for job in jobs]}

    @contextmanager
    def turns_paused(self):
        """Hold off new turns; fails if one is already executing."""
        session = self._session
        if session is None:
            yield
            return
        with session._execution_condition:
            if session._turn_thread is not None:
                raise ValueError("当前回合仍在执行，请等待回合结束或先停止，再创建隔离工作区")
            yield

    def suspend_if_idle(self, now, idle_seconds):
        with self._runtime_lock:
            state = self.resource_state()["state"]
            if state == "active":
                self._last_activity = now
                return False
            if state != "warm" or now - self._last_activity < idle_seconds or now < self._suspend_retry_at:
                return False
            session = self._session
            self._model_selection = session.model_selection
            self._not_suspending.clear()
        try:
            try:
                session.close(suspend=True)
            except Exception as exc:
                with self._runtime_lock:
                    self._suspend_retry_at = now + idle_seconds
                if session._closing:
                    self._notice("error", f"自动休眠未完成，部分资源仍在释放，稍后自动重试：{exc}")
                else:
                    self._notice("error", f"自动休眠未完成，运行时已保留：{exc}")
                log.exception("Session %s suspension failed", self.runtime_id)
                return False
            with self._runtime_lock, self._session_lock:
                self._session_generation += 1
                self._session = None
                self._session_thread = None
                self._session_error = None
                self._mention_index = None
        finally:
            self._not_suspending.set()
        self._send_event({"type": "runtime_state", "state": "dormant"})
        return True

    @contextmanager
    def _runtime_use(self, *, wake=True):
        # The lock is only ever held briefly, so waiting for it never stalls
        # the stdin loop; a runtime that is dormant, starting or suspending
        # is handed to the worker instead.
        reserved = False
        with self._runtime_lock:
            if self._closing.is_set():
                raise RuntimeError("会话正在关闭")
            constructing = self._session_thread is not None and self._session_thread.is_alive()
            if wake or (self._session is not None and not constructing and self._not_suspending.is_set()):
                self._runtime_users += 1
                reserved = True
        if not reserved:
            yield None
            return
        try:
            yield self._require_session()
        finally:
            with self._runtime_lock:
                self._runtime_users -= 1
                self._last_activity = time.monotonic()

    def _require_session(self) -> InteractiveSession:
        from .support.cancellation import current_token
        token = current_token.get()
        while True:
            while not self._not_suspending.wait(.05):
                if token is not None:
                    token.check()
                if self._closing.is_set():
                    raise RuntimeError("会话正在关闭")
            with self._runtime_lock:
                if not self._not_suspending.is_set():
                    continue
                if token is not None:
                    token.check()
                if self._closing.is_set():
                    raise RuntimeError("会话正在关闭")
                thread = self._session_thread
                if self._session is not None and (thread is None or not thread.is_alive()
                                                 or thread is threading.current_thread()):
                    self._last_activity = time.monotonic()
                    return self._session
                if thread is None or not thread.is_alive():
                    self._session_error = None
                    self._send_event({"type": "progress", "status": "starting", "detail": "正在唤醒会话…"})
                    self._start_session_construction()
                    thread = self._session_thread
                break
        deadline = time.monotonic() + 30
        while thread.is_alive() and time.monotonic() < deadline:
            if token is not None:
                token.check()
            if self._closing.is_set():
                raise CancelledError("会话正在关闭")
            thread.join(.05)
        with self._runtime_lock:
            if self._closing.is_set():
                raise RuntimeError("会话正在关闭")
            if thread.is_alive() or self._session is None:
                raise RuntimeError(self._session_error or "会话尚未准备好，请稍候")
            self._last_activity = time.monotonic()
            return self._session

    def _action(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        changes_runtime = name in {"toggle_permission", "rewind"} or (
            name == "panel_action" and params.get("panel") not in {"observe", "sessions"}
            and not (params.get("panel") == "mcp" and params.get("action") in {"add", "open_config"}))
        if changes_runtime:
            with self._active_lock:
                if self._active_token is not None and threading.current_thread() is not self._worker:
                    raise RuntimeError("当前回合仍在执行，暂时无法修改运行时")
            session = self._require_session()
            if session.tool_context.tool_tasks.pending_count:
                raise RuntimeError("前台工具尚未退出，暂时无法修改运行时")
        if name == "rewind":
            return self._queue_rewind(str(params.get("pointId") or ""))
        if name == "open_sessions":
            return {"ok": True, "panel": self._sessions_panel()}
        if name == "new_session":
            return self._new_session()
        if name == "open_panel":
            return {"ok": True, "panel": self._panel(str(params.get("panel") or ""))}
        if name == "panel_action":
            return self._panel_action(str(params.get("panel") or ""), str(params.get("action") or ""))
        if name == "workbench_action":
            return self._workbench_action(params)
        if name == "toggle_permission":
            self._require_session().toggle_permission_mode()
            self._send_snapshot()
            return {"ok": True}
        if name == "complete_mention":
            session = self._require_session()
            if self._mention_index is None:
                self._mention_index = MentionIndex(self.cwd, session.session_store)
            candidates = self._mention_index.candidates(
                str(params.get("prefix") or ""), limit=30
            )
            mode = model_input_mode()
            filtered = []
            for item in candidates:
                if item.kind != "file":
                    filtered.append(item)
                    continue
                name = item.insert_text.removeprefix("file:")
                try:
                    kind, _mime_type = classify_attachment(name, mimetypes.guess_type(name)[0])
                except AttachmentError:
                    continue
                if mode == "text" and kind == "image":
                    continue
                filtered.append(item)
            candidates = filtered
            return {"ok": True, "candidates": [{"insertText": item.insert_text, "display": item.display, "description": item.description, "kind": item.kind} for item in candidates]}
        if name == "stage_attachments":
            session = self._require_session()
            attachments = []
            paths = params.get("paths") or []
            if not isinstance(paths, list):
                raise AttachmentError("文件路径格式错误")
            source = str(params.get("source") or "picker")
            for path in paths:
                attachments.append(session.stage_attachment_path(str(path), source=source))
            clipboard = params.get("clipboard")
            if clipboard is not None:
                if not isinstance(clipboard, dict):
                    raise AttachmentError("剪贴板附件格式错误")
                try:
                    data = base64.b64decode(str(clipboard.get("dataBase64") or ""), validate=True)
                except ValueError as exc:
                    raise AttachmentError("剪贴板图片数据无效") from exc
                attachments.append(session.stage_attachment_bytes(
                    data,
                    name=str(clipboard.get("name") or "clipboard.png"),
                    mime_type=str(clipboard.get("mimeType") or "application/octet-stream"),
                    source="clipboard",
                ))
            return {"ok": True, "attachments": [item.public_dict() for item in attachments]}
        if name == "remove_attachment":
            removed = self._require_session().remove_attachment(str(params.get("attachmentId") or ""))
            return {"ok": removed}
        raise ValueError("不支持的操作，请重试")

    def _handle_request(self, message: dict[str, Any]) -> bool:
        if message.get("method") in {"cancel", "resolve_interaction"}:
            return self._dispatch_request(message)
        with self._runtime_use(wake=False) as session:
            if session is not None:
                return self._dispatch_request(message)
        def request():
            try:
                with self._runtime_use():
                    return self._dispatch_request(message)
            except Exception as exc:
                self._response(str(message.get("id") or ""), error=_localize_error(exc))
        self._enqueue_task(request)
        return True

    def _dispatch_request(self, message: dict[str, Any]) -> bool | PreparedTurn:
        request_id = str(message.get("id") or "")
        method = str(message.get("method") or "")
        params = message.get("params") or {}
        if not isinstance(params, dict):
            self._response(request_id, error="请求格式错误，请重试")
            return True
        if method == "submit":
            text = str(params.get("text") or "").strip()
            attachment_ids = params.get("attachmentIds") or []
            authorized_paths = params.get("authorizedPaths") or []
            if not isinstance(attachment_ids, list) or not isinstance(authorized_paths, list):
                self._response(request_id, error="附件参数格式错误，请重试")
            elif not text and not attachment_ids:
                self._response(request_id, error="请输入任务或添加附件后再提交")
            elif self._rewind_pending:
                self._response(request_id, error="回撤正在进行，请稍候")
            elif self._stopping.is_set():
                self._response(request_id, error="当前回合正在停止，请稍候")
            else:
                submission = TurnSubmission(
                    text=text,
                    attachment_ids=tuple(str(item) for item in attachment_ids),
                    authorized_paths=tuple(str(item) for item in authorized_paths),
                )
                try:
                    prepared = replace(self._require_session().prepare_submission(submission), ui_request_id=request_id)
                except ExternalPathConfirmationRequired:
                    # The single decision surface handles this as an approval
                    # interaction. It must be requested on the worker thread:
                    # the worker blocks waiting for the reply while this stdin
                    # loop stays free to receive resolve_interaction. The
                    # client keeps its draft (accepted: false) and shows no
                    # custom confirmation UI.
                    self._enqueue_external_submission(submission, request_id)
                    with self._active_lock:
                        active = self._active_token is not None
                    if active:
                        self._send_event({"type": "turn_state", "state": "queued", "queueDepth": self._tasks.qsize()})
                    self._response(request_id, result={"accepted": False})
                    return True
                except Exception as exc:
                    self._response(request_id, error=_localize_error(exc, "附件校验失败，请重试"))
                    return True
                self._send_event({"type": "transcript", "item": {
                    "id": f"user-{uuid.uuid4().hex}", "kind": "user", "title": "你",
                    "body": _user_transcript_body(submission.text, prepared.attachments),
                }})
                if threading.current_thread() is self._worker:
                    self._response(request_id, result={
                        "accepted": True, "attachments": [item.public_dict() for item in prepared.attachments],
                    })
                    return prepared
                self._enqueue_task(prepared)
                with self._active_lock:
                    active = self._active_token is not None
                if active:
                    self._send_event({"type": "turn_state", "state": "queued", "queueDepth": self._tasks.qsize()})
                self._response(request_id, result={
                    "accepted": True,
                    "attachments": [item.public_dict() for item in prepared.attachments],
                })
            return True
        if method == "cancel":
            with self._active_lock:
                token = self._active_token
            self._stopping.set()
            if token is not None:
                token.cancel()
            discarded = 0
            if token is not None:
                self._send_event({"type": "turn_state", "state": "cancelling", "queueDepth": self._tasks.qsize()})
            else:
                self._stopping.clear()
            self._response(
                request_id,
                result={"cancelled": token is not None, "discardedQueued": discarded},
            )
            return True
        if method == "action":
            name = str(params.get("name") or "")
            action_params = params.get("params")
            if not isinstance(action_params, dict):
                action_params = {}
            try:
                result = self._action(name, action_params)
            except Exception as exc:
                self._response(request_id, error=_localize_error(exc))
            else:
                self._response(request_id, result=result)
            return True
        if method == "resolve_interaction":
            interaction_id = str(params.get("id") or "")
            result = params.get("result")
            if not isinstance(result, dict):
                self._response(request_id, error="操作结果格式错误，请重试")
                self._send_event({"type": "interaction_closed", "id": interaction_id})
            elif not self._interactions.resolve(interaction_id, result):
                self._response(request_id, error="交互已失效，请重新操作")
                self._send_event({"type": "interaction_closed", "id": interaction_id})
            else:
                self._response(request_id, result={"resolved": True})
                self._send_event({"type": "interaction_closed", "id": interaction_id})
            return True
        self._response(request_id, error="不支持的请求，请重试")
        return True

    def _discard_queued_tasks(self) -> int:
        discarded = 0
        while True:
            try:
                task = self._tasks.get_nowait()
            except queue.Empty:
                break
            discarded += 1
            if callable(task) and self._rewind_pending:
                self._rewind_pending = False
            self._tasks.task_done()
        return discarded

    def close(self) -> None:
        self._closing.set()
        with self._session_lock:
            self._session_generation += 1
            session = self._session
        with self._active_lock:
            token = self._active_token
        if token is not None:
            token.cancel()
        self._interactions.cancel_all()
        self._discard_queued_tasks()
        if self._session_thread is not None and self._session_thread is not threading.current_thread():
            self._session_thread.join(timeout=10)
        if self._session_thread is not None and self._session_thread.is_alive():
            raise RuntimeError("会话初始化尚未退出，资源已保留")
        session = session or self._session
        if session is not None:
            session.close()
        if self._worker is not None and self._worker is not threading.current_thread():
            self._worker.join(timeout=2)
        if self._worker is not None and self._worker.is_alive():
            raise RuntimeError("会话执行线程尚未退出，资源已保留")
        if session is None:
            from .sessions.store import SessionStore
            SessionStore(self.state_root).update_status(self.runtime_id, "closed")
        self._send_event({"type": "shutdown", "reason": "session closed"})


class BridgeServer:
    def __init__(self, *, cwd: Path):
        from .core.session_runtime_registry import SessionRuntimeRegistry
        self.cwd = cwd.resolve()
        self._write_lock = threading.Lock()
        self._output_broken = threading.Event()
        self._closed = False
        self._closing = False
        self._control_lock = threading.Lock()
        self._controls = {}
        self.registry = SessionRuntimeRegistry(self.cwd, self._create_host, self._event)
        self.registry.create()
        self.registry.start_maintenance()

    def _write(self, message):
        line = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        with self._write_lock:
            if self._output_broken.is_set():
                return
            try:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
            except (BrokenPipeError, OSError):
                self._output_broken.set()

    def _event(self, event):
        if event["type"] not in {"sessions", "shutdown", "session_view", "resources"} and (
            not event.get("sessionId") or not isinstance(event.get("sequence"), int)
        ):
            raise ValueError("会话事件缺少寻址或序列")
        self._write({"type": "event", "event": event})

    def _response(self, request_id, result=None, error=None):
        response = {"type": "response", "id": request_id, "ok": error is None}
        response["result" if error is None else "error"] = result if error is None else str(error)
        self._write(response)

    def _create_host(self, record):
        def send(message):
            if message["type"] == "event":
                self.registry.publish(record.id, message["event"])
            else:
                self._write(message)
        return SessionBridge(cwd=record.cwd, session_id=record.id,
                             state_root=self.registry.store.root, environment=record.environment, emit=send,
                             session_action=lambda name, params: self._action(record.id, name, params),
                             project_resources=self.registry.resources)

    def _sessions_panel(self):
        active = {entry.id: entry for entry in self.registry._records.values()}
        options = [{"id": "action:new_session", "label": "新建会话"},
                   {"id": "action:fork_session", "label": "从稳定历史创建分支"},
                   {"id": "action:fork_worktree", "label": "在隔离工作区中继续"}]
        if self.registry.active_id and self.registry.get(self.registry.active_id).environment == "worktree":
            options.append({"id": "action:apply_worktree", "label": "应用更改到 Local"})
        for metadata in self.registry.store.list_sessions():
            session_id = metadata["id"]
            entry = active.get(session_id)
            options.append({"id": session_id, "label": entry.title if entry else metadata.get("title", "历史会话"),
                            "description": (entry.status if entry else "已保存") + " · " + _relative_age(metadata["created_at"]),
                            "selected": session_id == self.registry.active_id})
        return {"kind": "sessions", "title": "会话", "searchable": True, "options": options}

    def _action(self, session_id, name, params):
        if name == "watch_resources":
            self.registry.watch_resources = bool(params.get("enabled"))
            return {"ok": True, "resources": self.registry.resource_snapshot()}
        if name == "new_session":
            return {"ok": True, "sessionId": self.registry.create()}
        if name == "open_sessions" or (name == "open_panel" and params.get("panel") == "sessions"):
            return {"ok": True, "panel": self._sessions_panel()}
        if name == "panel_action" and params.get("panel") == "sessions":
            action = params["action"]
            if action.startswith("action:"):
                return self._action(session_id, action.split(":", 1)[1], {})
            name, params = "switch_session", {"sessionId": action}
        if name == "switch_session":
            target = params["sessionId"]
            self.registry.resume(target)
            return {"ok": True, "sessionId": target}
        if name == "hide_session":
            self.registry.hide(params.get("sessionId", session_id))
            if self.registry.active_id is None:
                self.registry.create()
            return {"ok": True}
        if name in {"fork_session", "fork_worktree"}:
            target = self.registry.fork(session_id, worktree=name == "fork_worktree")
            return {"ok": True, "sessionId": target}
        if name == "apply_worktree":
            record = self.registry.get(session_id)
            if record.environment != "worktree":
                raise ValueError("当前会话不在 Worktree 中")
            record.host._enqueue_task(lambda: self._apply_worktree(record))
            return {"ok": True, "message": "已排队应用到 Local"}
        if name == "sync_session":
            return {"ok": True, "sessionId": session_id, "view": self.registry.snapshot(session_id)}
        host = self.registry.get(session_id).host
        with host._runtime_use():
            return host._action(name, params)

    def _apply_worktree(self, record):
        from .support.cancellation import current_token
        from .workspace.worktrees import (
            WorktreeMergeConflict,
            WorktreeService,
            prepare_merge_review,
        )
        service = WorktreeService(self.cwd)
        session = record.host._require_session()
        if session.tool_context.agent_coordinator.has_active_agents():
            raise RuntimeError("请先停止当前会话的后台任务，再应用更改")
        originals = {}
        for attempt in range(3):
            try:
                paths = service.apply(record.id)
                record.host._notice("info", f"已应用 {len(paths)} 个文件到 Local")
                return
            except WorktreeMergeConflict as exc:
                for name, versions in exc.files.items():
                    originals.setdefault(name, versions)
                if attempt == 2:
                    directory = record.cwd / ".harness" / "apply-conflicts" / uuid.uuid4().hex
                    details, approved = prepare_merge_review(directory, exc.files, originals)
                    answer = session.question_provider.ask(QuestionRequest(
                        question="Agent 已完成两轮协调。请审阅合并结果，确认后才会写入 Local。",
                        details=f"完整版本与差异已保存：{directory}\n\n{details}",
                        options=[QuestionOption("保留现场", "keep"), QuestionOption("确认应用以上合并结果", "apply")],
                        session_id=record.id, agent_name="main_agent"))
                    if not answer.cancelled and answer.value == "apply":
                        paths = service.apply(record.id, approved=approved)
                        record.host._notice("info", f"已应用确认的合并结果：{len(paths)} 个文件")
                    return
                directory = record.cwd / ".harness" / "apply-conflicts" / uuid.uuid4().hex
                descriptions = []
                for name, versions in exc.files.items():
                    descriptions.append(f"- {name}")
                    for kind, data in versions.items():
                        path = directory / kind / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        if data is not None:
                            path.write_bytes(data)
                        else:
                            descriptions.append(f"  {kind} 版本不存在")
                prompt = ("用户要求将当前 Worktree 的任务成果应用到 Local，但存在文件冲突。"
                          f"这是第 {attempt + 1}/2 轮自动协调。仅在当前任务范围内协调以下文件，保留两边意图；"
                          "读取 base、local、task 三个版本后，将合并结果写入当前 Worktree 的原文件。"
                          "不要修改 Local；无法判断时询问用户，不要猜测。\n"
                          f"版本目录：{directory}\n" + "\n".join(descriptions))
                session.submit(prompt, cancellation_token=current_token.get())

    def _handle_request(self, message):
        request_id = message.get("id", "")
        method = message.get("method")
        params = message.get("params") or {}
        try:
            if method == "initialize":
                if params.get("protocolVersion") != UI_PROTOCOL_VERSION:
                    raise ValueError("TUI 协议版本不匹配")
                self.registry.announce()
                for record in list(self.registry._records.values()):
                    self._event({"type": "session_view", "sessionId": record.id, "view": self.registry.snapshot(record.id)})
                self._response(request_id, {"protocolVersion": UI_PROTOCOL_VERSION, "status": "ready", "activeSessionId": self.registry.active_id})
            elif method == "shutdown":
                self.close()
                self._response(request_id, {"closed": True})
                return False
            else:
                session_id = message.get("sessionId")
                if not session_id:
                    raise ValueError("请求缺少 sessionId")
                if method == "action":
                    name, action_params = params["name"], params.get("params") or {}
                    slow_action = name in {"fork_session", "fork_worktree", "apply_worktree", "panel_action", "workbench_action"} or (
                        name == "open_panel" and action_params.get("panel") == "mcp")
                    global_action = name in {"watch_resources", "new_session", "open_sessions", "switch_session",
                                             "hide_session", "sync_session"} or (
                        name == "open_panel" and action_params.get("panel") == "sessions")
                    if slow_action:
                        self._start_control(request_id, session_id, name, action_params)
                    elif global_action:
                        self._response(request_id, self._action(session_id, name, action_params))
                    else:
                        self.registry.get(session_id).host._handle_request(message)
                else:
                    self.registry.get(session_id).host._handle_request(message)
        except Exception as exc:
            self._response(request_id, error=exc)
        return True

    def _start_control(self, request_id, session_id, name, params):
        from .support.cancellation import cancellation_scope
        token = CancellationToken()
        def run():
            try:
                with cancellation_scope(token):
                    self._response(request_id, self._action(session_id, name, params))
            except Exception as exc:
                self._response(request_id, error=exc)
            finally:
                with self._control_lock:
                    self._controls.pop(threading.current_thread(), None)
        with self._control_lock:
            if self._closing or len(self._controls) >= 4:
                raise RuntimeError("工作台正在关闭或等待中的工作区操作过多")
            thread = threading.Thread(target=run, name="workbench-control", daemon=True)
            self._controls[thread] = token
            thread.start()

    def close(self):
        if self._closed:
            return
        with self._control_lock:
            self._closing = True
            controls = list(self._controls.items())
        for _, token in controls:
            token.cancel()
        for thread, _ in controls:
            thread.join(timeout=2)
        self.registry.close()
        self._closed = True
        self._event({"type": "shutdown", "reason": "工作台已关闭"})

    def run(self):
        try:
            for line in sys.stdin:
                try:
                    message = json.loads(line)
                    if message.get("type") != "request":
                        raise ValueError("请求格式错误")
                    if not self._handle_request(message):
                        break
                except (ValueError, TypeError) as exc:
                    self._response("", error=exc)
        finally:
            self.close()
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VeriForge OpenTUI NDJSON bridge")
    parser.add_argument("--cwd", default=".")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = BridgeServer(cwd=Path(args.cwd))
    return server.run()


if __name__ == "__main__":  # pragma: no cover - exercised by Bun subprocess
    raise SystemExit(main())
