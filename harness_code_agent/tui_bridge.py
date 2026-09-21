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
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config
from .agent.cancellation import CancellationToken, CancelledError
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
from .tui.approval import ApprovalAllowlist, _persistent_prefix_for_request
from .tui.commands import default_command_registry
from .tui.completion import MentionIndex, mention_candidates
from .tui.protocol import UI_PROTOCOL_VERSION, validate_ui_event
from .tui.state import SessionStatusSnapshot, TranscriptBlock, TuiState, _localize_error

log = logging.getLogger("harness.opentui")

_PROFILE_COPY = {
    "general": ("通用", "问答、分析与轻量只读任务"),
    "coding-agent": ("编码", "修改代码、运行测试并完成验证"),
    "plan": ("规划", "调查现状并形成决策完整的实施方案"),
    "app-builder": ("应用构建", "端到端构建并验证应用界面"),
    "review": ("审查", "只读检查代码并优先报告问题"),
}

_PERMISSION_COPY = {
    PermissionPolicy.WORKSPACE_WRITE: ("请求批准", "工作区内写入允许，风险操作询问，工作区外写入拒绝"),
    PermissionPolicy.LLM_AUTO: ("替我审批", "由模型判断风险并自动批准具体、范围明确的安全操作"),
    PermissionPolicy.DANGER_FULL_ACCESS: ("完全访问", "工作区内外修改均不询问；灾难性命令（如 rm -rf /、git reset --hard）仍会被拦截"),
    PermissionPolicy.READ_ONLY: ("只读", "禁止一切修改与未知程序执行"),
}

_MODEL_COPY = {
    "deepseek-v4-flash": ("DeepSeek V4 Flash", "通用快速模型，适合日常任务"),
    "deepseek-v4-flash-vision-exp": ("DeepSeek V4 Flash Vision", "实验性多模态模型，额外支持图片输入"),
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
        "profile:": "配置：",
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
        with self._lock:
            self._counter += 1
            interaction_id = f"interaction-{self._counter}"
            event = threading.Event()
            result: dict[str, Any] = {}
            self._pending[interaction_id] = (event, result)
        self._emit({"type": "interaction", "id": interaction_id, "kind": kind, "payload": payload})
        event.wait()
        with self._lock:
            self._pending.pop(interaction_id, None)
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
            pending = list(self._pending.values())
        for event, result in pending:
            result["cancelled"] = True
            result.setdefault("decision", "deny")
            event.set()



class BridgeServer:
    """Own one Python session and serve request/event messages over stdio."""

    def __init__(self, *, cwd: Path, profile_name: str, profile_explicit: bool) -> None:
        self.cwd = cwd.resolve()
        self.profile_name = profile_name
        self.profile_explicit = profile_explicit
        self._write_lock = threading.Lock()
        self._tasks: queue.Queue[PreparedTurn | None] = queue.Queue()
        self._active_token: CancellationToken | None = None
        self._active_lock = threading.Lock()
        self._stopping = threading.Event()
        self._closing = threading.Event()
        self._output_broken = threading.Event()
        self._session_lock = threading.Lock()
        self._session_generation = 0
        self._session: InteractiveSession | None = None
        self._session_error: str | None = None
        self._assistant_group_id: str | None = None
        self._assistant_group_counter = 0
        self._assistant_id: str | None = None
        self._assistant_text = ""
        self._assistant_counter = 0
        self.state = TuiState(
            snapshot=SessionStatusSnapshot(
                profile=profile_name,
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
        self._send_event({"type": "progress", "status": "starting", "detail": "正在准备 Python 会话…"})
        self._start_session_construction()
        self._worker = threading.Thread(target=self._worker_loop, name="opentui-submit", daemon=True)
        self._worker.start()

    def _start_session_construction(self, *, reset_ui: bool = False) -> None:
        with self._session_lock:
            self._session_generation += 1
            generation = self._session_generation
        thread = threading.Thread(
            target=self._construct_session,
            kwargs={"generation": generation, "reset_ui": reset_ui},
            name="opentui-session",
            daemon=True,
        )
        self._session_thread = thread
        thread.start()

    def _is_current_generation(self, generation: int) -> bool:
        with self._session_lock:
            return generation == self._session_generation and not self._closing.is_set()

    def _construct_session(self, *, generation: int, reset_ui: bool = False) -> None:
        session: InteractiveSession | None = None
        try:
            session = InteractiveSession(
                cwd=self.cwd,
                profile_name=self.profile_name,
                profile_explicit=self.profile_explicit,
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
            if reset_ui:
                self._send_event({"type": "session_reset", "snapshot": self._snapshot_payload(), "items": []})
            self._send_event({"type": "progress", "status": "ready", "detail": "Python 会话已就绪。"})
            self._start_mcp_warm(session)
        except Exception as exc:  # pragma: no cover - exercised by real startup failures
            error = _localize_error(exc, "会话启动失败，请稍后重试")
            with self._session_lock:
                current = generation == self._session_generation and not self._closing.is_set()
                if current:
                    self._session_error = error
            if current:
                self._notice("error", error)
                self._send_event({"type": "progress", "status": "failed", "detail": error})
            log.debug("OpenTUI bridge session construction failed\n%s", traceback.format_exc())

    def _startup_progress(self, stage: str) -> None:
        self._send_event({"type": "progress", "status": stage, "detail": stage})

    def _start_mcp_warm(self, session: InteractiveSession) -> None:
        """Connect configured MCP servers in the background after ready."""

        def _warm() -> None:
            try:
                session.warm_mcp_tools()
            except Exception:  # warm failures are invisible to the user path
                log.debug("Background MCP warm failed", exc_info=True)

        threading.Thread(target=_warm, name="opentui-mcp-warm", daemon=True).start()

    def _emit_commands(self) -> None:
        if self._session is None:
            return
        try:
            registry = default_command_registry(skill_registry=self._session.skill_registry)
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
            event_type = event.type
            if event_type == "turn_started":
                self._begin_assistant_group(int(event.payload.get("turn") or self.state.snapshot.turn))
                if not self._stopping.is_set():
                    self._send_event({"type": "turn_state", "state": "running"})
            elif event_type == "turn_finished":
                self._close_assistant_group("success")
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
            "todo": "todo",
            "user": "user",
            "file": "file",
            "thought": "thought",
            "profile": "profile",
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
        if block.direction:
            item["direction"] = block.direction
        if self._assistant_group_id:
            item["parentId"] = self._assistant_group_id
        return item

    def _send_snapshot(self) -> None:
        self._send_event({"type": "snapshot", "snapshot": self._snapshot_payload()})

    def _snapshot_payload(self) -> dict[str, Any]:
        snapshot = self.state.snapshot
        context_percent = 99
        if snapshot.context_window_tokens:
            context_percent = max(
                0,
                min(100, round((1 - snapshot.context_tokens / snapshot.context_window_tokens) * 100)),
            )
        profile = config.resolve_model_profile(config.MODEL_INTENSITY)
        return {
            "profile": snapshot.profile,
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
            "routingMode": getattr(self._session, "display_routing_mode", "auto"),
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
        if self._output_broken.is_set():
            return
        line = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        with self._write_lock:
            try:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
            except (BrokenPipeError, OSError):
                self._output_broken.set()
                log.debug("OpenTUI output pipe closed", exc_info=True)

    def _response(self, request_id: str, *, result: Any = None, error: str | None = None) -> None:
        message: dict[str, Any] = {"type": "response", "id": request_id, "ok": error is None}
        if error is None:
            message["result"] = result
        else:
            message["error"] = error
        self._write(message)

    def _worker_loop(self) -> None:
        while not self._closing.is_set():
            task = self._tasks.get()
            if task is None:
                self._tasks.task_done()
                return
            token = CancellationToken()
            with self._active_lock:
                self._active_token = token
            self._assistant_id = None
            self._assistant_text = ""
            self._send_event({"type": "turn_state", "state": "running"})
            try:
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
                if self._stopping.is_set():
                    self._discard_queued_tasks()
                    self._stopping.clear()
                self._send_event({"type": "turn_state", "state": "idle"})
                self._tasks.task_done()

    def _run_task(self, task: PreparedTurn, token: CancellationToken) -> None:
        session = self._session
        if session is None:
            if self._session_thread.is_alive():
                self._session_thread.join(timeout=30)
            session = self._session
        if session is None:
            raise RuntimeError(self._session_error or "会话尚未准备好，请稍候")
        text = task.text
        registry = default_command_registry(skill_registry=session.skill_registry)
        if not task.attachments and text.lstrip().startswith("/") and not registry.is_agent_command(text):
            should_continue = session.handle_slash_command(text)
            result = getattr(session, "last_command_result", None)
            if result is not None:
                if getattr(result, "text", ""):
                    self._notice("info", str(result.text))
                action = getattr(result, "action", None)
                if action in {"profile", "checkpoint", "mcp", "observe"}:
                    self._send_event({"type": "panel", "panel": self._panel(str(action))})
                elif action == "compact":
                    self._notice("info", session.compact_current_context())
                elif action == "fork":
                    self._notice("info", session.fork_current_session())
            if not should_continue:
                self._closing.set()
            return
        result = session.submit_prepared(task, cancellation_token=token)
        if getattr(result, "notice", ""):
            self._notice("info", str(result.notice))
        checkpoint = str(getattr(result, "checkpoint", "") or "").strip()
        if checkpoint and checkpoint not in {
            "no changes to checkpoint",
            "checkpoint auto off",
            "checkpoint cadence skipped",
        }:
            self._notice("info", checkpoint)

    def _sessions_panel(self) -> dict[str, Any]:
        session = self._require_session()
        options = []
        current_session_id = session.session_id
        for metadata in session.session_store.list_sessions():
            session_id = str(metadata.get("id") or "").strip()
            if not session_id:
                continue
            if session_id == current_session_id:
                # Resuming the active session would fork it onto itself; the
                # current session has no place in the resume list.
                continue
            preview = ""
            try:
                for event in reversed(session.session_store.read_events(session_id)):
                    if event.get("type") == "user_input":
                        preview = str((event.get("payload") or {}).get("text") or "").replace("\n", " ").strip()
                        break
            except (OSError, ValueError, KeyError, TypeError):
                log.debug("Could not read session preview", exc_info=True)
            if not preview:
                continue
            options.append(
                {
                    "id": session_id,
                    "label": preview[:72] + ("…" if len(preview) > 72 else ""),
                    "description": " · ".join((
                        _PROFILE_COPY.get(str(metadata.get("profile") or "general"), (str(metadata.get("profile") or "general"), ""))[0],
                        _relative_age(metadata.get("created_at")),
                    )),
                }
            )
        return {
            "kind": "sessions",
            "title": "历史会话",
            "options": options,
            "searchable": True,
            "body": "" if options else "还没有包含用户任务的历史会话。",
        }

    def _panel(self, kind: str) -> dict[str, Any]:
        session = self._require_session()
        if kind == "profile":
            from .profiles import list_profiles

            current = "auto" if session.display_routing_mode == "auto" else session.display_profile
            options = [
                {
                    "id": "auto",
                    "label": "自动路由",
                    "description": "根据当前任务选择工作模式",
                    "tone": "success" if current == "auto" else "default",
                    "selected": current == "auto",
                }
            ]
            options.extend({
                "id": item["name"],
                "label": _PROFILE_COPY.get(item["name"], (item["name"], item["description"]))[0],
                "description": _PROFILE_COPY.get(item["name"], (item["name"], item["description"]))[1],
                "tone": "success" if current == item["name"] else "default",
                "selected": current == item["name"],
            } for item in list_profiles())
            body = ""
            if session.display_routing_mode == "auto":
                active_label = _PROFILE_COPY.get(session.display_profile, (session.display_profile, ""))[0]
                body = f"自动路由 ✓ · 当前：{active_label}"
            return {"kind": "profile", "title": "工作模式", "options": options, "body": body}
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
            current = config.resolve_model_profile(config.MODEL_INTENSITY).model
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
            current = config.resolve_model_profile(config.MODEL_INTENSITY).reasoning_effort
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
        if kind == "checkpoint":
            checkpoint = session.checkpoint
            return {
                "kind": "checkpoint",
                "title": "检查点",
                "body": f"当前：自动{'开启' if checkpoint.auto else '关闭'} · 每 {checkpoint.every_turns} 轮",
                "options": [
                    {"id": "create", "label": "立即创建检查点"},
                    {"id": "auto_on", "label": "开启自动检查点"},
                    {"id": "auto_off", "label": "关闭自动检查点"},
                    {"id": "every_turn", "label": "每轮创建"},
                ],
            }
        if kind == "mcp":
            session.mcp_status()
            manager = session.mcp_manager
            statuses = getattr(manager, "statuses", {})
            names = set(manager.configured_server_names())
            names.update(name for name in statuses if name != "config")
            options = [{"id": "reload", "label": "重新加载全部 MCP 服务"}]
            for name in sorted(names):
                status = statuses.get(name)
                state = "已停用" if status is None else ("已连接" if status.state == "connected" else "连接失败")
                options.append({"id": f"reconnect:{name}", "label": f"重新连接 {name}", "description": state})
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
                "body": "Enter 提交  Shift+Enter 换行  Ctrl+C 取消/退出  Ctrl+O 运行观察  Ctrl+P 打开审批模式\n\n"
                + "\n".join(f"{item.usage:<18} {item.description}" for item in commands),
            }
        raise ValueError(f"unknown panel: {kind}")

    def _panel_action(self, panel: str, action: str) -> dict[str, Any]:
        session = self._require_session()
        if panel == "sessions":
            session.resume_from_session(action)
            items = self._history_items(action)
            self._send_event({"type": "session_reset", "snapshot": self._snapshot_payload(), "items": items})
            return {"ok": True, "message": "已从历史会话创建分支"}
        if panel == "command":
            if action == "compact":
                message = session.compact_current_context()
            elif action == "fork":
                message = session.fork_current_session()
            else:
                raise ValueError(f"unknown command action: {action}")
        elif panel == "profile":
            message = session.enable_auto_profile_routing() if action == "auto" else session.switch_profile(action)
        elif panel == "permission":
            message = session.set_permission_mode(action)
        elif panel == "model":
            session.apply_model_override(model=action)
        elif panel == "effort":
            session.apply_model_override(reasoning_effort=action)
        elif panel == "checkpoint":
            if action == "create":
                message = session.create_checkpoint(manual=True)
            elif action == "auto_on":
                result = session.set_auto_checkpoint(True)
                message = "自动检查点已开启" if session.checkpoint.auto else result
            elif action == "auto_off":
                session.set_auto_checkpoint(False)
                message = "自动检查点已关闭"
            elif action == "every_turn":
                result = session.set_auto_checkpoint(True)
                if session.checkpoint.auto:
                    session.checkpoint.every_turns = 1
                    message = "检查点频率已设为每轮"
                else:
                    message = result
            else:
                raise ValueError(f"unknown checkpoint action: {action}")
        elif panel == "mcp":
            name = action.split(":", 1)[1] if ":" in action else ""
            if action == "reload":
                message = session.reload_mcp()
            elif action.startswith("reconnect:"):
                message = session.reload_mcp_server(name)
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
        if panel in {"profile", "permission", "model", "effort"}:
            return {"ok": True}
        return {"ok": True, "message": str(message or "")}

    def _history_items(self, session_id: str) -> list[dict[str, Any]]:
        session = self._require_session()
        replay = TuiState(snapshot=SessionStatusSnapshot(
            profile=self.state.snapshot.profile,
            model=self.state.snapshot.model,
            provider=self.state.snapshot.provider,
            permission_mode=self.state.snapshot.permission_mode,
            session_id=self.state.snapshot.session_id,
            cwd=self.cwd,
        ))
        items = []
        for event in session.session_store.read_events(session_id):
            block = replay.apply_event(event)
            if block is not None:
                replay.add_block(block)
                items.append(self._block_item(block))
        return items

    def _new_session(self) -> dict[str, Any]:
        if self._active_token is not None:
            raise RuntimeError("当前回合仍在执行，暂时无法新建会话")
        if self._session_thread.is_alive():
            raise RuntimeError("新会话正在启动，请稍候")
        old_session = self._session
        if old_session is not None:
            old_session.close()
        self._session = None
        self._session_error = None
        self._assistant_id = None
        self.state = TuiState(snapshot=SessionStatusSnapshot(
            profile=self.profile_name,
            model=config.MODEL,
            provider=config.PROVIDER,
            permission_mode=os.environ.get("HARNESS_PERMISSION_MODE", "workspace-write"),
            session_id=None,
            cwd=self.cwd,
            status="starting",
        ))
        self._send_event({"type": "progress", "status": "starting", "detail": "正在创建新会话…"})
        self._start_session_construction(reset_ui=True)
        return {"ok": True, "message": "新会话正在启动…"}

    def _require_session(self) -> InteractiveSession:
        if self._session is None:
            raise RuntimeError(self._session_error or "会话尚未准备好，请稍候")
        return self._session

    def _action(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        if name == "open_sessions":
            return {"ok": True, "panel": self._sessions_panel()}
        if name == "new_session":
            return self._new_session()
        if name == "open_panel":
            return {"ok": True, "panel": self._panel(str(params.get("panel") or ""))}
        if name == "panel_action":
            return self._panel_action(str(params.get("panel") or ""), str(params.get("action") or ""))
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
        request_id = str(message.get("id") or "")
        method = str(message.get("method") or "")
        params = message.get("params") or {}
        if not isinstance(params, dict):
            self._response(request_id, error="请求格式错误，请重试")
            return True
        if method == "initialize":
            client_version = params.get("protocolVersion")
            if client_version != UI_PROTOCOL_VERSION:
                self._response(
                    request_id,
                    error=(
                        f"TUI 协议版本不兼容：客户端 {client_version!r}，"
                        f"服务端 {UI_PROTOCOL_VERSION}"
                    ),
                )
                return True
            if self._session_error:
                self._response(request_id, error=self._session_error)
            else:
                self._response(
                    request_id,
                    result={
                        "protocolVersion": UI_PROTOCOL_VERSION,
                        "cwd": str(self.cwd),
                        "status": "ready" if self._session is not None else "starting",
                    },
                )
            return True
        if method == "submit":
            text = str(params.get("text") or "").strip()
            attachment_ids = params.get("attachmentIds") or []
            authorized_paths = params.get("authorizedPaths") or []
            if not isinstance(attachment_ids, list) or not isinstance(authorized_paths, list):
                self._response(request_id, error="附件参数格式错误，请重试")
            elif not text and not attachment_ids:
                self._response(request_id, error="请输入任务或添加附件后再提交")
            elif self._stopping.is_set():
                self._response(request_id, error="当前回合正在停止，请稍候")
            else:
                try:
                    prepared = self._require_session().prepare_submission(TurnSubmission(
                        text=text,
                        attachment_ids=tuple(str(item) for item in attachment_ids),
                        authorized_paths=tuple(str(item) for item in authorized_paths),
                    ))
                except ExternalPathConfirmationRequired as exc:
                    self._response(request_id, result={
                        "accepted": False,
                        "confirmation": {"kind": "external_paths", "paths": exc.paths},
                    })
                    return True
                except Exception as exc:
                    self._response(request_id, error=_localize_error(exc, "附件校验失败，请重试"))
                    return True
                self._tasks.put(prepared)
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
            discarded = self._discard_queued_tasks()
            if self._session is not None:
                try:
                    self._session.interrupt_current_shell()
                except Exception:
                    log.debug("Failed to interrupt current shell", exc_info=True)
            if token is not None or discarded:
                self._send_event({"type": "turn_state", "state": "cancelling", "queueDepth": 0})
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
        if method == "shutdown":
            self._response(request_id, result={"closing": True})
            self.close()
            return False
        self._response(request_id, error="不支持的请求，请重试")
        return True

    def _discard_queued_tasks(self) -> int:
        discarded = 0
        while True:
            try:
                task = self._tasks.get_nowait()
            except queue.Empty:
                break
            if task is None:
                self._tasks.task_done()
                self._tasks.put(None)
                break
            discarded += 1
            self._tasks.task_done()
        return discarded

    def close(self) -> None:
        if self._closing.is_set():
            return
        self._closing.set()
        with self._session_lock:
            self._session_generation += 1
            session = self._session
            self._session = None
            session_thread = self._session_thread
        with self._active_lock:
            token = self._active_token
        if token is not None:
            token.cancel()
        self._interactions.cancel_all()
        self._tasks.put(None)
        if session is not None:
            try:
                session.close()
            except Exception:
                log.debug("Failed to close OpenTUI session", exc_info=True)
        if session_thread is not threading.current_thread():
            session_thread.join(timeout=2)
        if self._worker is not threading.current_thread():
            self._worker.join(timeout=2)
        self._send_event({"type": "shutdown", "reason": "bridge closed"})

    def run(self) -> int:
        for raw_line in sys.stdin:
            try:
                message = json.loads(raw_line)
            except json.JSONDecodeError:
                self._notice("error", "收到无法解析的桥接消息。")
                continue
            if not isinstance(message, dict) or message.get("type") != "request":
                self._notice("error", "桥接消息缺少 request 类型。")
                continue
            if not self._handle_request(message):
                return 0
        self.close()
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VeriForge OpenTUI NDJSON bridge")
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--profile", default="general")
    parser.add_argument("--profile-explicit", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = BridgeServer(
        cwd=Path(args.cwd),
        profile_name=args.profile,
        profile_explicit=args.profile_explicit,
    )
    return server.run()


if __name__ == "__main__":  # pragma: no cover - exercised by Bun subprocess
    raise SystemExit(main())
