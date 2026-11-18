from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..runtime.tool_failures import FailureVisibility


@dataclass
class SessionStatusSnapshot:
    model: str
    provider: str
    permission_mode: str
    session_id: str | None
    cwd: Path
    turn: int = 0
    running_tool: str = ""
    status: str = "idle"
    dirty_count: int = 0
    context_tokens: int = 0
    context_window_tokens: int = 0


@dataclass
class TranscriptBlock:
    kind: str
    title: str
    body: str = ""
    status: str = ""
    turn: int | None = None
    id: str | None = None
    direction: str = ""
    #: Set for rows that belong to one inline subagent timeline.
    parent_id: str | None = None
    #: ``group`` marks the collapsed subagent row that owns ``parent_id`` rows.
    role: str = ""


_MAIN_AGENT_NAMES = (None, "", "main", "main_agent")
_AGENT_EVENTS = {"agent_spawned", "agent_status", "agent_message", "agent_parent_message"}
_AGENT_TERMINAL_STATUSES = {"success", "failed", "interrupted"}
_AGENT_STATUS_LABELS = {
    "queued": "等待中",
    "running": "运行中",
    "completed": "已完成",
    "failed": "失败",
    "blocked": "受阻",
    "interrupted": "已中断",
}


@dataclass
class TuiState:
    snapshot: SessionStatusSnapshot
    blocks: list[TranscriptBlock] = field(default_factory=list)
    #: (scope, tool) -> call blocks without a tool_call_id (older journals).
    active_tool_blocks: dict[tuple[str, str], list[TranscriptBlock]] = field(default_factory=dict)
    #: (scope, tool_call_id) -> call block; the primary pairing key.
    active_tool_calls: dict[tuple[str, str], TranscriptBlock] = field(default_factory=dict)
    tool_call_counter: int = 0
    active_thought_block: TranscriptBlock | None = None
    #: agent block id -> running thought row for that subagent.
    agent_thoughts: dict[str, TranscriptBlock] = field(default_factory=dict)
    #: stable agent id (or name for old journals) -> inline agent block.
    agent_blocks: dict[str, TranscriptBlock] = field(default_factory=dict)
    agent_keys_by_name: dict[str, str] = field(default_factory=dict)
    #: Parent rows whose visible summary/state changed; drained by the bridge.
    pending_agent_updates: list[TranscriptBlock] = field(default_factory=list)
    thought_counter: int = 0
    block_counter: int = 0

    def drain_agent_updates(self) -> list[TranscriptBlock]:
        updates, self.pending_agent_updates = self.pending_agent_updates, []
        return updates

    def apply_event(self, event: Any) -> TranscriptBlock | None:
        data = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        event_type = str(data.get("type", ""))
        payload = data.get("payload") or {}
        agent = data.get("agent")
        if event_type in _AGENT_EVENTS:
            return self._apply_agent_event(event_type, payload, agent)
        if str(payload.get("agent_id") or "") or agent not in _MAIN_AGENT_NAMES:
            return self._apply_child_event(event_type, payload, agent)
        return self._apply_main_event(event_type, payload)

    def _apply_main_event(self, event_type: str, payload: dict[str, Any]) -> TranscriptBlock | None:
        if event_type == "session_started":
            self.snapshot.status = "ready"
            if payload.get("session_id"):
                self.snapshot.session_id = str(payload.get("session_id"))
            return None
        if event_type == "user_input":
            self.snapshot.turn = int(payload.get("turn") or self.snapshot.turn)
            body = str(payload.get("text", ""))
            attachments = payload.get("attachments") or []
            summaries = [
                f"[{item.get('kind', 'file')}] {item.get('name', '附件')} ({format_attachment_size(item.get('size', 0))})"
                for item in attachments if isinstance(item, dict)
            ]
            if summaries:
                body = "\n".join(filter(None, [body, *summaries]))
            return TranscriptBlock("user", f"第 {self.snapshot.turn} 回合", body, turn=self.snapshot.turn)
        if event_type == "turn_started":
            self.snapshot.status = "thinking"
            self.snapshot.turn = int(payload.get("turn") or self.snapshot.turn)
            return None
        if event_type == "assistant_message":
            self.snapshot.status = "idle"
            turn = _payload_turn(payload, self.snapshot.turn)
            return TranscriptBlock("assistant", "助手", str(payload.get("text", "")), turn=turn)
        if event_type == "llm_usage":
            prompt_tokens = payload.get("prompt_tokens")
            if isinstance(prompt_tokens, int) and prompt_tokens > 0:
                self.snapshot.context_tokens = prompt_tokens
                self.snapshot.context_window_tokens = _context_window_tokens()
            return None
        if event_type == "tool_call":
            return self._apply_tool_call(payload)
        if event_type == "tool_result":
            return self._apply_tool_result(payload)
        if event_type == "middleware_activity":
            return self._apply_middleware_activity(payload)
        if event_type == "file_change":
            self.snapshot.dirty_count += 1
            operation = str(payload.get("operation") or "changed")
            path = str(payload.get("path") or "")
            verb = _FILE_OPERATION_LABELS.get(operation, operation)
            diff = str(payload.get("diff") or "")
            additions, deletions = _file_change_counts(
                diff,
                payload.get("additions"),
                payload.get("deletions"),
            )
            title = f"{verb} {path}  +{additions}  -{deletions}"
            return TranscriptBlock("file", title, diff, "changed", turn=self.snapshot.turn)
        if event_type == "failure":
            if _is_protocol_failure(payload):
                return None
            self.snapshot.status = "needs attention"
            return self._apply_failure(payload)
        if event_type == "agent_budget_warning":
            return None
        if event_type == "agent_fallback":
            self.snapshot.status = "blocked"
            self.snapshot.running_tool = ""
            reason = str(payload.get("reason") or "stopped")
            if payload.get("limit_type") == "schema_failures":
                return None
            if reason == "approval_denied":
                # The user made this decision; the "操作已拒绝" tool result
                # already says so — an extra "代理已停止" block would read
                # like a system malfunction.
                return None
            display_reason = reason.replace("_", " ")
            body = f"已停止：{_FAILURE_LABELS.get(display_reason, '任务未完成')}"
            last_tool = str(payload.get("last_tool") or "")
            if last_tool:
                body += f"（最后工具：{last_tool}）"
            return TranscriptBlock("failure", "代理已停止", body, "blocked", turn=self.snapshot.turn)
        if event_type == "approval_requested":
            return None
        if event_type == "approval_decided":
            return None
        if event_type == "plan_executed":
            return None
        if event_type == "permission_mode_switched":
            self.snapshot.permission_mode = str(payload.get("permission_mode") or self.snapshot.permission_mode)
            return None
        if event_type == "plan_ready":
            self.snapshot.status = "plan ready"
            path = str(payload.get("plan_path") or "global_plan/current/plan.md")
            revision = payload.get("plan_revision")
            suffix = f" rev {revision}" if revision is not None else ""
            return TranscriptBlock(
                "plan",
                "计划已准备",
                f"{path}{suffix}",
                "pending",
            )
        if event_type == "context_compaction_started":
            self.snapshot.status = "compacting context"
            return None
        if event_type == "context_compaction_committed":
            self.snapshot.status = "idle"
            tokens_saved = payload.get("tokens_saved", 0)
            body = f"已节省约 {tokens_saved} 个 token" if tokens_saved else "上下文已压缩"
            return TranscriptBlock("status", "上下文已压缩", body, "success", turn=self.snapshot.turn)
        if event_type == "context_anxiety_observed":
            return None
        if event_type == "turn_summary":
            self.snapshot.status = "idle"
            return None
        if event_type == "turn_finished":
            self.snapshot.status = "idle"
            self.snapshot.running_tool = ""
            self._clear_scope("main")
            return None
        if event_type == "session_finished":
            self.snapshot.status = str(payload.get("status") or "closed")
            self.active_tool_blocks.clear()
            self.active_tool_calls.clear()
            self.active_thought_block = None
            self.agent_thoughts.clear()
            return None
        if event_type == "thought_started":
            self.snapshot.status = "thinking"
            self.thought_counter += 1
            self.active_thought_block = TranscriptBlock(
                "thought",
                "正在思考",
                "",
                "running",
                turn=self.snapshot.turn,
                id=f"thought-{self.snapshot.turn}-{self.thought_counter}",
            )
            return self.active_thought_block
        if event_type == "thought_finished":
            self.snapshot.status = "running"
            duration = payload.get("duration_seconds", 0)
            body = _format_elapsed(duration)
            if payload.get("truncated"):
                body += "  内容已截断"
            if self.active_thought_block is not None:
                self.active_thought_block.title = "思考"
                self.active_thought_block.body = body
                self.active_thought_block.status = "success"
                block = self.active_thought_block
                self.active_thought_block = None
                return block
            return TranscriptBlock("thought", "思考", body, "success", turn=self.snapshot.turn)
        return None

    def add_block(self, block: TranscriptBlock | None) -> None:
        if block is None:
            return
        if not block.id:
            self.block_counter += 1
            block.id = f"block-{block.turn or self.snapshot.turn}-{self.block_counter}"
        if block.id and any(existing.id == block.id for existing in self.blocks):
            return
        self.blocks.append(block)

    def _apply_tool_call(
        self,
        payload: dict[str, Any],
        *,
        scope: str = "main",
        parent: TranscriptBlock | None = None,
    ) -> TranscriptBlock | None:
        if payload.get("failure_visibility") == FailureVisibility.PROTOCOL:
            return None
        tool = str(payload.get("tool", "tool"))
        if scope == "main":
            self.snapshot.running_tool = tool
            self.snapshot.status = "tool"
        if tool in _HIDDEN_TOOL_CALLS:
            return None
        title = _tool_call_title(tool, payload.get("args"))
        self.tool_call_counter += 1
        block = TranscriptBlock(
            "tool",
            title,
            "",
            "running",
            turn=self.snapshot.turn,
            id=f"tool-{self.snapshot.turn}-{self.tool_call_counter}"
            if scope == "main"
            else f"{scope}-tool-{self.tool_call_counter}",
            parent_id=parent.id if parent is not None else None,
        )
        call_id = str(payload.get("tool_call_id") or "")
        if call_id:
            self.active_tool_calls[(scope, call_id)] = block
        self.active_tool_blocks.setdefault((scope, tool), []).append(block)
        if parent is not None:
            self._set_agent_activity(parent, title)
        return block

    def _apply_tool_result(
        self,
        payload: dict[str, Any],
        *,
        scope: str = "main",
        parent: TranscriptBlock | None = None,
    ) -> TranscriptBlock | None:
        tool = str(payload.get("tool", "tool"))
        call_id = str(payload.get("tool_call_id") or "")
        if _is_protocol_failure(payload):
            # Older persisted calls may already have created a visible row.
            block = self._take_active_tool_block(tool, scope=scope, tool_call_id=call_id)
            if block is not None:
                if scope == "main":
                    if self.snapshot.running_tool == tool:
                        self.snapshot.running_tool = ""
                    self.snapshot.status = "running"
                block.status = "failed"
            return block
        if scope == "main" and self.snapshot.running_tool == tool:
            self.snapshot.running_tool = ""
        status = str(payload.get("status", "unknown"))
        if tool == "ask_user":
            return None
        if tool == "update_todo" and status == "success":
            # Task detail lives in the Workbench Tasks section; the transcript
            # only keeps the Work Strip counter.
            if scope == "main":
                self.snapshot.status = "running"
            return None
        if scope == "main":
            self.snapshot.status = "running"
        raw_metadata = payload.get("metadata")
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        # Approval/policy rejections collapse to a single short line in the
        # UI; the model still receives the full error text for recovery.
        rejection = {
            "approval": "操作已拒绝",
            "tool_policy": "该执行方式受限",
            "permission": "操作被安全策略拦截",
        }.get(str(metadata.get("status_source") or ""))
        return_code = payload.get("return_code")
        # run_bash reports status="success" even on non-zero exit; treat a bad
        # exit as failure so the title/color match reality, and keep the raw
        # output tail (pytest failures, ModuleNotFoundError, ...).
        bash_failed = tool == "bash" and return_code not in (None, 0)
        if bash_failed:
            status = "failed"
        if rejection:
            # A policy interception is an intermediate self-correction step, so
            # it is not shown to the user at all. Approval denials and
            # permission bans stay visible.
            self._take_active_tool_block(tool, scope=scope, tool_call_id=call_id)
            if str(metadata.get("status_source")) == "tool_policy":
                return None
            body = rejection
        else:
            error = str(payload.get("error") or "")
            parts = []
            if return_code not in (None, 0):
                parts.append(f"退出码 {return_code}")
            body = "  ".join(parts)
            if error:
                localized_error = _localize_error(error)
                body += f"\n{localized_error}" if body else localized_error
                raw_tail = _bounded_tail(error, 600)
                if raw_tail and raw_tail not in body:
                    body += f"\n{raw_tail}"
        if bash_failed:
            output_tail = _bounded_tail(str(payload.get("output") or ""), 1200)
            if output_tail and output_tail not in body:
                body += f"\n{output_tail}" if body else output_tail
        block = self._take_active_tool_block(tool, scope=scope, tool_call_id=call_id)
        if block is not None:
            block.title = _tool_result_title_with_detail(tool, status, block.title)
            block.body = body
            block.status = status
            if parent is not None:
                self._set_agent_activity(parent, _tool_result_title(tool, status))
            return block
        if parent is not None:
            self._set_agent_activity(parent, _tool_result_title(tool, status))
        return TranscriptBlock(
            "tool",
            _tool_result_title(tool, status),
            body,
            status,
            turn=self.snapshot.turn,
            parent_id=parent.id if parent is not None else None,
        )

    def _take_active_tool_block(
        self,
        tool: str,
        *,
        scope: str = "main",
        tool_call_id: str = "",
    ) -> TranscriptBlock | None:
        if tool_call_id:
            block = self.active_tool_calls.pop((scope, tool_call_id), None)
            if block is not None:
                self._forget_queued_tool(scope, tool, block)
                return block
        blocks = self.active_tool_blocks.get((scope, tool))
        if not blocks:
            return None
        block = blocks.pop(0)
        if not blocks:
            self.active_tool_blocks.pop((scope, tool), None)
        return block

    def _forget_queued_tool(self, scope: str, tool: str, block: TranscriptBlock) -> None:
        blocks = self.active_tool_blocks.get((scope, tool))
        if not blocks:
            return
        remaining = [item for item in blocks if item is not block]
        if remaining:
            self.active_tool_blocks[(scope, tool)] = remaining
        else:
            self.active_tool_blocks.pop((scope, tool), None)

    def _clear_scope(self, scope: str) -> None:
        for key in [key for key in self.active_tool_blocks if key[0] == scope]:
            self.active_tool_blocks.pop(key, None)
        for key in [key for key in self.active_tool_calls if key[0] == scope]:
            self.active_tool_calls.pop(key, None)
        if scope == "main":
            self.active_thought_block = None
        else:
            self.agent_thoughts.pop(scope, None)

    # ------------------------------------------------------------------
    # Subagent inline timelines
    # ------------------------------------------------------------------

    def _apply_child_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        agent_name: str | None,
    ) -> TranscriptBlock | None:
        block = self._agent_block(payload, agent_name)
        if event_type == "thought_started":
            self.thought_counter += 1
            thought = TranscriptBlock(
                "thought",
                "正在思考",
                "",
                "running",
                turn=self.snapshot.turn,
                id=f"{block.id}-thought-{self.thought_counter}",
                parent_id=block.id,
            )
            self.agent_thoughts[block.id] = thought
            self._set_agent_activity(block, "思考中")
            return thought
        if event_type == "thought_finished":
            duration = payload.get("duration_seconds", 0)
            body = _format_elapsed(duration)
            if payload.get("truncated"):
                body += "  内容已截断"
            thought = self.agent_thoughts.pop(block.id, None)
            if thought is not None:
                thought.title = "思考"
                thought.body = body
                thought.status = "success"
                return thought
            return TranscriptBlock(
                "thought",
                "思考",
                body,
                "success",
                turn=self.snapshot.turn,
                parent_id=block.id,
            )
        if event_type == "tool_call":
            return self._apply_tool_call(payload, scope=block.id or "", parent=block)
        if event_type == "tool_result":
            return self._apply_tool_result(payload, scope=block.id or "", parent=block)
        if event_type == "failure":
            if _is_protocol_failure(payload) or payload.get("tool"):
                return None
            message = _localize_error(payload.get("message")) if payload.get("message") else ""
            body = str(message or "")
            self._set_agent_activity(block, body or "执行失败")
            return TranscriptBlock(
                "failure",
                "错误",
                body,
                "failed",
                turn=self.snapshot.turn,
                parent_id=block.id,
            )
        if event_type == "agent_fallback":
            reason = str(payload.get("reason") or "stopped")
            if payload.get("limit_type") == "schema_failures" or reason == "approval_denied":
                return None
            display_reason = reason.replace("_", " ")
            body = f"已停止：{_FAILURE_LABELS.get(display_reason, '任务未完成')}"
            last_tool = str(payload.get("last_tool") or "")
            if last_tool:
                body += f"（最后工具：{last_tool}）"
            self._set_agent_activity(block, body)
            return TranscriptBlock(
                "failure",
                "代理已停止",
                body,
                "blocked",
                turn=self.snapshot.turn,
                parent_id=block.id,
            )
        if event_type == "file_change":
            self.snapshot.dirty_count += 1
            operation = str(payload.get("operation") or "changed")
            path = str(payload.get("path") or "")
            verb = _FILE_OPERATION_LABELS.get(operation, operation)
            diff = str(payload.get("diff") or "")
            additions, deletions = _file_change_counts(
                diff,
                payload.get("additions"),
                payload.get("deletions"),
            )
            title = f"{verb} {path}  +{additions}  -{deletions}"
            self._set_agent_activity(block, title)
            return TranscriptBlock(
                "file",
                title,
                diff,
                "changed",
                turn=self.snapshot.turn,
                parent_id=block.id,
            )
        # Usage, compaction and turn bookkeeping stay out of the inline
        # timeline; they describe the agent, not its work.
        return None

    def _apply_agent_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        agent_name: str | None,
    ) -> TranscriptBlock | None:
        raw_status = str(payload.get("status") or "")
        if event_type == "agent_status" and raw_status == "closed":
            # Closing adds no outcome; the final inline state stays visible.
            return None
        block = self._agent_block(payload, agent_name)
        if event_type == "agent_spawned":
            block.title = str(payload.get("name") or payload.get("agent_name") or block.title)
            # A very fast worker may finish before its spawn event reaches the
            # shared bus; never reopen a terminal record.
            if block.status not in _AGENT_TERMINAL_STATUSES:
                block.status = "running"
            task = str(payload.get("task") or "").strip()
            if task and block.body in {"", "运行中", "等待中"}:
                block.body = _short_line(task, 120)
            self._touch_agent(block)
            return block
        if event_type == "agent_status":
            mapped = _agent_ui_status(raw_status)
            if mapped is not None:
                block.status = mapped
            summary = str(payload.get("summary") or "").strip()
            error = str(payload.get("error") or "").strip()
            if raw_status == "completed":
                block.body = _short_line(summary or "已完成", 160)
            elif raw_status in {"failed", "blocked", "interrupted"}:
                block.body = _short_line(error or summary or _AGENT_STATUS_LABELS[raw_status], 160)
            elif raw_status in {"queued", "running"} and not summary:
                block.body = _AGENT_STATUS_LABELS[raw_status]
            self._touch_agent(block)
            return block
        # agent_message / agent_parent_message: a message row inside the
        # timeline, never a new top-level "已补充"/"已上报" block.
        mapped = _agent_ui_status(raw_status)
        if mapped in {"running", "pending"}:
            block.status = mapped
        message = str(payload.get("message") or "")
        outbound = event_type == "agent_message"
        self._set_agent_activity(block, "继续任务" if outbound else "上报消息")
        return TranscriptBlock(
            "agent",
            "已补充" if outbound else "上报",
            message,
            "running",
            turn=self.snapshot.turn,
            parent_id=block.id,
            direction="out" if outbound else "in",
        )

    def _agent_block(
        self,
        payload: dict[str, Any],
        agent_name: str | None,
    ) -> TranscriptBlock:
        raw_id = str(payload.get("agent_id") or "")
        name = str(
            payload.get("name")
            or payload.get("agent_name")
            or (agent_name if agent_name not in _MAIN_AGENT_NAMES else "")
            or "子代理"
        )
        key = raw_id or self.agent_keys_by_name.get(name) or name
        block = self.agent_blocks.get(key)
        if block is None:
            block = TranscriptBlock(
                "agent",
                name,
                _short_line(str(payload.get("task") or ""), 120) or "运行中",
                "running",
                turn=self.snapshot.turn,
                id=f"agent-{key}",
                role="group",
            )
            self.agent_blocks[key] = block
        self.agent_keys_by_name[name] = key
        return block

    def _set_agent_activity(self, block: TranscriptBlock, text: str) -> None:
        if block.status in _AGENT_TERMINAL_STATUSES:
            return
        body = _short_line(text, 160)
        if not body or body == block.body:
            return
        block.body = body
        self._touch_agent(block)

    def _touch_agent(self, block: TranscriptBlock) -> None:
        if all(item is not block for item in self.pending_agent_updates):
            self.pending_agent_updates.append(block)

    def _apply_middleware_activity(self, payload: dict[str, Any]) -> None:
        return None

    def _apply_failure(self, payload: dict[str, Any]) -> TranscriptBlock | None:
        tool = str(payload.get("tool") or "")
        if tool:
            # The tool_result event is already the user-facing record for a
            # failed tool; rendering the failure too would duplicate it.
            return None
        category = str(payload.get("category") or "error").replace("_", " ")
        message = _localize_error(payload.get("message")) if payload.get("message") else ""
        category_label = _FAILURE_LABELS.get(category, "执行失败")
        body = f"{category_label}：{message}" if message else category_label
        return TranscriptBlock("failure", "错误", body, "failed", turn=self.snapshot.turn)


_FILE_OPERATION_LABELS = {
    "write": "文件已写入",
    "write_file": "文件已写入",
    "edit": "已编辑",
    "apply_patch": "已编辑",
    "delete": "已删除",
    "create": "已创建",
    "rename": "已重命名",
    "changed": "已变更",
}


def _is_protocol_failure(payload: dict[str, Any]) -> bool:
    metadata = payload.get("metadata")
    return isinstance(metadata, dict) and metadata.get("failure_visibility") == FailureVisibility.PROTOCOL


def format_attachment_size(value: Any) -> str:
    """Human-readable size, matching the frontend formatBytes helper."""
    try:
        size = max(0, int(value))
    except (TypeError, ValueError):
        size = 0
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"

_FAILURE_LABELS = {
    "stopped": "已停止",
    "time budget exhausted": "超出时间预算",
    "runtime error": "运行时错误",
    "api error": "模型接口错误",
    "error": "错误",
}


_ERROR_PREFIX = re.compile(r"^(?:(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Error|Exception|Failure)|Error):\s*", re.IGNORECASE)
_CONTEXT_PREFIX = re.compile(r"^(?:回合失败|任务失败|操作失败|错误)[：:]\s*")


def _localize_error(value: Any, fallback: str = "操作失败，请稍后重试") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    lowered = text.lower()
    if "cannot start a new session while a turn is running" in lowered:
        return "当前回合仍在执行，暂时无法新建会话"
    if "current session is still starting" in lowered:
        return "新会话正在启动，请稍候"
    if "current turn is stopping" in lowered:
        return "当前回合正在停止，请稍候"
    if "empty task" in lowered:
        return "请输入任务后再提交"
    if "python session is not ready" in lowered:
        return "会话尚未准备好，请稍候"
    if "python session failed to start" in lowered:
        return "会话启动失败，请稍后重试"
    if "api error" in lowered or "model error" in lowered:
        return "模型请求失败，请稍后重试"
    if "command not found" in lowered:
        return "找不到该命令"
    if "modulenotfounderror" in lowered or "module not found" in lowered:
        return "缺少运行依赖，请检查环境"
    if "no such file or directory" in lowered or "filenotfounderror" in lowered:
        return "找不到指定文件或目录"
    if "not a directory" in lowered or "isadirectoryerror" in lowered or "notadirectoryerror" in lowered:
        return "文件路径无效"
    if "permission denied" in lowered or "permissionerror" in lowered:
        return "没有权限执行该操作"
    if "timed out" in lowered or "timeout" in lowered or "timeouterror" in lowered:
        return "操作超时，请稍后重试"
    if "connection" in lowered or "network" in lowered:
        return "连接服务失败，请稍后重试"
    if "invalid json" in lowered or "jsondecode" in lowered or "jsondecodeerror" in lowered:
        return "返回数据格式异常，请稍后重试"
    if "calledprocesserror" in lowered or "exit code" in lowered or "command failed" in lowered:
        return "命令执行失败，请检查命令后重试"
    if "assertionerror" in lowered or "verification failed" in lowered:
        return "结果校验失败，请检查后重试"
    if "cancelled" in lowered or "canceled" in lowered:
        return "操作已取消"
    if "unknown " in lowered or "unsupported " in lowered:
        return "不支持的操作，请重试"
    if lowered.startswith("[blocked]") or " blocked" in lowered:
        return "操作被安全策略拦截"
    if lowered.startswith("[error]"):
        return "操作执行失败，请稍后重试"
    if "interaction not found" in lowered:
        return "交互已失效，请重新操作"
    text = _CONTEXT_PREFIX.sub("", text).strip()
    while _ERROR_PREFIX.match(text):
        text = _ERROR_PREFIX.sub("", text, count=1).strip()
    if any("\u4e00" <= char <= "\u9fff" for char in text) and not re.search(r"[A-Za-z]{2,}", text):
        return text or fallback
    return fallback

# Tools whose call titles are reduced to their single most useful argument.
_TOOL_PRIMARY_ARGS = {
    "bash": "command",
    "read_file": "path",
    "write_file": "path",
    "apply_patch": "path",
    "spawn_agent": "task",
    "send_agent_message": "message",
    "followup_agent": "task",
    "close_agent": "agent_id",
    "read_skill_file": "path",
    "web_search": "query",
    "web_fetch": "url",
}

_HIDDEN_TOOL_CALLS = {"ask_user", "update_todo"}

_TOOL_DISPLAY_LABELS = {
    "read_file": "阅读文件",
    "write_file": "写入文件",
    "apply_patch": "修改文件",
    "bash": "执行命令",
    "spawn_agent": "启动子代理",
    "send_agent_message": "补充子代理消息",
    "followup_agent": "继续子代理任务",
    "wait_agents": "等待子代理",
    "interrupt_agent": "中断子代理",
    "close_agent": "关闭子代理",
    "web_search": "搜索网页",
    "web_fetch": "阅读网页",
    "read_skill_file": "阅读技能说明",
    "search_tools": "查找工具",
    "memory_search": "搜索记忆",
    "memory_read": "阅读记忆",
    "memory_write": "保存记忆",
    "memory_validate": "验证记忆",
    "memory_forget": "遗忘记忆",
    "list_shell_jobs": "查看后台任务",
    "read_shell_output": "读取后台输出",
    "stop_shell_job": "停止后台任务",
}

_TOOL_RESULT_LABELS = {
    "read_file": "文件已阅读",
    "write_file": "文件已写入",
    "apply_patch": "文件已修改",
    "bash": "命令已执行",
    "spawn_agent": "子代理已启动",
    "send_agent_message": "消息已补充",
    "followup_agent": "后续任务已发送",
    "wait_agents": "子代理状态已更新",
    "interrupt_agent": "子代理已中断",
    "close_agent": "子代理已关闭",
    "web_search": "网页已搜索",
    "web_fetch": "网页已阅读",
    "read_skill_file": "技能说明已阅读",
    "search_tools": "工具已查找",
    "memory_search": "记忆已搜索",
    "memory_read": "记忆已阅读",
    "memory_write": "记忆已保存",
    "memory_validate": "记忆已验证",
    "memory_forget": "记忆已遗忘",
    "list_shell_jobs": "后台任务已查看",
    "read_shell_output": "后台输出已读取",
    "stop_shell_job": "后台任务已停止",
}


def _agent_ui_status(status: str) -> str | None:
    return {
        "queued": "pending",
        "running": "running",
        "completed": "success",
        "failed": "failed",
        "blocked": "failed",
        "interrupted": "interrupted",
    }.get(status)


def _short_line(text: Any, limit: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) > limit:
        return value[: max(0, limit - 1)] + "…"
    return value


def _payload_turn(payload: dict[str, Any], default: int) -> int:
    try:
        return int(payload.get("turn") or default)
    except (TypeError, ValueError):
        return default


def _tool_call_title(tool: str, args: Any) -> str:
    label = _tool_display_label(tool)
    primary_key = _tool_primary_arg(tool, args)
    if primary_key and isinstance(args, dict) and args.get(primary_key):
        value = str(args[primary_key]).replace("\n", " ")
        if len(value) > 160:
            value = value[:157] + "..."
        return f"{label}  {value}"
    return label


def _tool_result_title(tool: str, status: str) -> str:
    if status == "success":
        result_label = _TOOL_RESULT_LABELS.get(tool)
        if result_label:
            return result_label
        return f"{_tool_display_label(tool)}已完成"
    if tool == "bash":
        return "命令执行失败"
    return f"{_tool_display_label(tool)}失败"


def _tool_result_title_with_detail(tool: str, status: str, call_title: str) -> str:
    result_title = _tool_result_title(tool, status)
    call_label = _tool_display_label(tool)
    prefix = f"{call_label}  "
    if call_title.startswith(prefix):
        detail = call_title[len(prefix):].strip()
        if detail:
            return f"{result_title}  {detail}"
    return result_title


def _mcp_tool_identity(tool: str) -> tuple[str, str] | None:
    if not tool.startswith("mcp__"):
        return None
    identity = tool[len("mcp__"):]
    server, separator, tool_name = identity.partition("__")
    if not separator or not server or not tool_name:
        return None
    return server, tool_name


def _tool_display_label(tool: str) -> str:
    label = _TOOL_DISPLAY_LABELS.get(tool)
    if label:
        return label
    identity = _mcp_tool_identity(tool)
    if identity is None:
        return tool
    server, tool_name = identity
    combined = f"{server} {tool_name}".lower()
    if "exa" in combined and "search" in combined:
        return "语义搜索"
    if "exa" in combined and ("fetch" in combined or "read" in combined):
        return "Exa 阅读网页"
    return f"MCP 工具  {server}/{tool_name}"


def _tool_primary_arg(tool: str, args: Any) -> str | None:
    primary_key = _TOOL_PRIMARY_ARGS.get(tool)
    if primary_key:
        return primary_key
    if not isinstance(args, dict):
        return None
    identity = _mcp_tool_identity(tool)
    if identity is None:
        return None
    _, tool_name = identity
    lowered = tool_name.lower()
    if "search" in lowered and args.get("query"):
        return "query"
    if ("fetch" in lowered or "read" in lowered) and args.get("urls"):
        return "urls"
    if args.get("query"):
        return "query"
    return None


def _file_change_counts(diff: str, additions: Any, deletions: Any) -> tuple[int, int]:
    """Use event metadata when available; fall back to parsing older events."""
    try:
        added = int(additions) if additions is not None else None
    except (TypeError, ValueError):
        added = None
    try:
        deleted = int(deletions) if deletions is not None else None
    except (TypeError, ValueError):
        deleted = None
    if added is not None and deleted is not None:
        return max(0, added), max(0, deleted)
    lines = diff.splitlines()
    added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++ "))
    deleted = sum(1 for line in lines if line.startswith("-") and not line.startswith("--- "))
    return added, deleted


def _format_elapsed(seconds: float) -> str:
    if seconds < 1.0:
        return f"{int(seconds * 1000)}ms"
    return f"{seconds:.1f}s"


def _context_window_tokens() -> int:
    try:
        from ..config import CONTEXT_WINDOW_TOKENS

        return int(CONTEXT_WINDOW_TOKENS)
    except (ImportError, TypeError, ValueError):
        return 200_000


def _bounded_tail(text: str, limit: int) -> str:
    text = text.strip("\n")
    if len(text) <= limit:
        return text
    return "…" + text[-limit:]
