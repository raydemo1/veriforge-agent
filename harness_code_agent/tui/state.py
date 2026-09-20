from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SessionStatusSnapshot:
    profile: str
    model: str
    provider: str
    permission_mode: str
    session_id: str | None
    cwd: Path
    turn: int = 0
    pending_plan: bool = False
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


@dataclass(frozen=True)
class TodoStep:
    text: str
    status: str


@dataclass
class TuiState:
    snapshot: SessionStatusSnapshot
    blocks: list[TranscriptBlock] = field(default_factory=list)
    todo_steps: list[TodoStep] = field(default_factory=list)
    active_tool_blocks: dict[str, list[TranscriptBlock]] = field(default_factory=dict)
    tool_call_counter: int = 0
    active_thought_block: TranscriptBlock | None = None
    thought_counter: int = 0
    block_counter: int = 0

    def apply_event(self, event: Any) -> TranscriptBlock | None:
        data = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        event_type = str(data.get("type", ""))
        payload = data.get("payload") or {}

        if event_type == "session_started":
            self.snapshot.status = "ready"
            self.snapshot.profile = str(payload.get("profile") or self.snapshot.profile)
            if payload.get("session_id"):
                self.snapshot.session_id = str(payload.get("session_id"))
            return None
        if event_type == "user_input":
            self.snapshot.turn = int(payload.get("turn") or self.snapshot.turn)
            body = str(payload.get("text", ""))
            attachments = payload.get("attachments") or []
            summaries = [
                f"[{item.get('kind', 'file')}] {item.get('name', '附件')} ({_format_attachment_size(item.get('size', 0))})"
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
        if event_type == "tool_call":
            return self._apply_tool_call(payload)
        if event_type == "tool_result":
            return self._apply_tool_result(payload)
        if event_type == "agent_spawned":
            name = str(payload.get("name") or payload.get("agent_id") or "子代理")
            role = str(payload.get("role") or "")
            return TranscriptBlock("agent", f"子代理已启动  {name}", role, "running", turn=self.snapshot.turn)
        if event_type == "agent_message":
            name = str(payload.get("name") or payload.get("agent_id") or "子代理")
            return TranscriptBlock("agent", f"已补充消息  {name}", "", "running", turn=self.snapshot.turn)
        if event_type == "agent_status":
            name = str(payload.get("name") or payload.get("agent_id") or "子代理")
            status = str(payload.get("status") or "unknown")
            labels = {"running": "运行中", "completed": "已完成", "failed": "失败", "blocked": "受阻", "interrupted": "已中断", "closed": "已关闭"}
            body = str(payload.get("error") or payload.get("proposal_id") or "")
            return TranscriptBlock("agent", f"子代理{name}：{labels.get(status, status)}", body, status, turn=self.snapshot.turn)
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
            self.snapshot.status = "needs attention"
            return self._apply_failure(payload)
        if event_type == "agent_budget_warning":
            return None
        if event_type == "agent_fallback":
            self.snapshot.status = "blocked"
            self.snapshot.running_tool = ""
            reason = str(payload.get("reason") or "stopped").replace("_", " ")
            body = f"已停止：{_FAILURE_LABELS.get(reason, '任务未完成')}"
            last_tool = str(payload.get("last_tool") or "")
            if last_tool:
                body += f"（最后工具：{last_tool}）"
            return TranscriptBlock("failure", "代理已停止", body, "blocked", turn=self.snapshot.turn)
        if event_type == "approval_requested":
            return None
        if event_type == "approval_decided":
            return None
        if event_type == "profile_switched":
            self.snapshot.profile = str(payload.get("profile") or self.snapshot.profile)
            self.snapshot.pending_plan = False
            return None
        if event_type == "profile_route_decision":
            if not bool(payload.get("switched")):
                return None
            profile = str(payload.get("profile") or "")
            body = f"已切换到 {profile}" if profile else "工作模式已切换"
            return TranscriptBlock("profile", "工作模式", body, turn=self.snapshot.turn)
        if event_type == "permission_mode_switched":
            self.snapshot.permission_mode = str(payload.get("permission_mode") or self.snapshot.permission_mode)
            return None
        if event_type == "plan_ready":
            self.snapshot.pending_plan = True
            self.snapshot.status = "plan ready"
            path = str(payload.get("plan_path") or "global_plan/current/plan.md")
            revision = payload.get("plan_revision")
            suffix = f" rev {revision}" if revision is not None else ""
            return TranscriptBlock(
                "plan",
                "计划已准备",
                f"{path}{suffix}\n[计划已准备]  [如需修改，请描述要调整的内容]",
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
            self.active_tool_blocks.clear()
            self.active_thought_block = None
            return None
        if event_type == "session_finished":
            self.snapshot.status = str(payload.get("status") or "closed")
            self.active_tool_blocks.clear()
            self.active_thought_block = None
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

    def _apply_tool_call(self, payload: dict[str, Any]) -> TranscriptBlock | None:
        tool = str(payload.get("tool", "tool"))
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
            id=f"tool-{self.snapshot.turn}-{self.tool_call_counter}",
        )
        self.active_tool_blocks.setdefault(tool, []).append(block)
        return block

    def _apply_tool_result(self, payload: dict[str, Any]) -> TranscriptBlock | None:
        tool = str(payload.get("tool", "tool"))
        if self.snapshot.running_tool == tool:
            self.snapshot.running_tool = ""
        status = str(payload.get("status", "unknown"))
        if tool == "ask_user":
            return None
        if tool == "update_todo" and status == "success":
            self._update_todo_steps_from_metadata(payload.get("metadata"))
            self.snapshot.status = "running"
            return TranscriptBlock(
                "todo",
                "待办",
                _format_todo_steps(self.todo_steps),
                "updated",
                turn=self.snapshot.turn,
            )
        self.snapshot.status = "running"
        error = str(payload.get("error") or "")
        return_code = payload.get("return_code")
        parts = []
        if return_code not in (None, 0):
            parts.append(f"退出码 {return_code}")
        body = "  ".join(parts)
        if error:
            localized_error = _localize_error(error)
            body += f"\n{localized_error}" if body else localized_error
        block = self._take_active_tool_block(tool)
        if block is not None:
            block.title = _tool_result_title_with_detail(tool, status, block.title)
            block.body = body
            block.status = status
            return block
        return TranscriptBlock("tool", _tool_result_title(tool, status), body, status, turn=self.snapshot.turn)

    def _take_active_tool_block(self, tool: str) -> TranscriptBlock | None:
        blocks = self.active_tool_blocks.get(tool)
        if not blocks:
            return None
        block = blocks.pop(0)
        if not blocks:
            self.active_tool_blocks.pop(tool, None)
        return block

    def _apply_middleware_activity(self, payload: dict[str, Any]) -> None:
        return None

    def _apply_failure(self, payload: dict[str, Any]) -> TranscriptBlock | None:
        category = str(payload.get("category") or "error").replace("_", " ")
        message = _localize_error(payload.get("message")) if payload.get("message") else ""
        tool = str(payload.get("tool") or "")
        category_label = _FAILURE_LABELS.get(category, "执行失败")
        body = f"{category_label}：{message}" if message else category_label
        if tool:
            body += f"\n执行工具：{tool}"
        return TranscriptBlock("failure", "错误", body, "failed", turn=self.snapshot.turn)

    def _update_todo_steps_from_metadata(self, metadata: Any) -> None:
        if not isinstance(metadata, dict):
            return
        todo_state = metadata.get("todo_state")
        if not isinstance(todo_state, dict):
            return
        raw_items = todo_state.get("items")
        if not isinstance(raw_items, list):
            return

        todo_steps: list[TodoStep] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text") or "").strip()
            if not text:
                continue
            status = str(raw.get("status") or "pending").strip()
            todo_steps.append(TodoStep(text, status))
        self.todo_steps = todo_steps


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


def _format_attachment_size(value: Any) -> str:
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
    "run_bash": "command",
    "read_file": "path",
    "write_file": "path",
    "apply_patch": "path",
    "edit_file": "path",
    "list_files": "directory",
    "search_files": "query",
    "spawn_agent": "task",
    "send_agent_message": "message",
    "followup_agent": "task",
    "read_agent_changes": "proposal_id",
    "apply_agent_changes": "proposal_id",
    "read_agent_conflicts": "conflict_id",
    "resolve_agent_conflicts": "conflict_id",
    "close_agent": "agent_id",
    "browser_test": "url",
    "read_skill_file": "path",
    "repo_search": "query",
    "web_search": "query",
    "web_fetch": "url",
}

_HIDDEN_TOOL_CALLS = {"ask_user", "update_todo"}

_TOOL_DISPLAY_LABELS = {
    "list_files": "查看目录",
    "read_file": "阅读文件",
    "write_file": "写入文件",
    "apply_patch": "修改文件",
    "edit_file": "修改文件",
    "delete_file": "删除文件",
    "rename_file": "重命名文件",
    "repo_search": "搜索代码",
    "search_files": "搜索文件",
    "run_bash": "执行命令",
    "spawn_agent": "启动子代理",
    "send_agent_message": "补充子代理消息",
    "followup_agent": "继续子代理任务",
    "wait_agents": "等待子代理",
    "list_agents": "查看子代理",
    "interrupt_agent": "中断子代理",
    "read_agent_changes": "查看子代理改动",
    "apply_agent_changes": "应用子代理改动",
    "read_agent_conflicts": "查看合并冲突",
    "resolve_agent_conflicts": "解决合并冲突",
    "close_agent": "关闭子代理",
    "web_search": "搜索网页",
    "web_fetch": "阅读网页",
    "browser_test": "测试页面",
    "read_skill_file": "阅读技能说明",
    "tool_search": "查找工具",
    "memory_search": "搜索记忆",
    "memory_read": "阅读记忆",
    "memory_write": "保存记忆",
    "memory_validate": "验证记忆",
    "memory_forget": "遗忘记忆",
    "list_shell_jobs": "查看后台任务",
    "read_shell_output": "读取后台输出",
    "stop_shell_job": "停止后台任务",
    "stop_dev_server": "停止开发服务",
}

_TOOL_RESULT_LABELS = {
    "list_files": "目录已查看",
    "read_file": "文件已阅读",
    "write_file": "文件已写入",
    "apply_patch": "文件已修改",
    "edit_file": "文件已修改",
    "delete_file": "文件已删除",
    "rename_file": "文件已重命名",
    "repo_search": "代码已搜索",
    "search_files": "文件已搜索",
    "run_bash": "命令已执行",
    "spawn_agent": "子代理已启动",
    "send_agent_message": "消息已补充",
    "followup_agent": "后续任务已发送",
    "wait_agents": "子代理状态已更新",
    "list_agents": "子代理已查看",
    "interrupt_agent": "子代理已中断",
    "read_agent_changes": "子代理改动已读取",
    "apply_agent_changes": "子代理改动已应用",
    "read_agent_conflicts": "合并冲突已读取",
    "resolve_agent_conflicts": "合并冲突已解决",
    "close_agent": "子代理已关闭",
    "web_search": "网页已搜索",
    "web_fetch": "网页已阅读",
    "browser_test": "页面已测试",
    "read_skill_file": "技能说明已阅读",
    "tool_search": "工具已查找",
    "memory_search": "记忆已搜索",
    "memory_read": "记忆已阅读",
    "memory_write": "记忆已保存",
    "memory_validate": "记忆已验证",
    "memory_forget": "记忆已遗忘",
    "list_shell_jobs": "后台任务已查看",
    "read_shell_output": "后台输出已读取",
    "stop_shell_job": "后台任务已停止",
    "stop_dev_server": "开发服务已停止",
}


def _format_todo_steps(steps: list[TodoStep]) -> str:
    markers = {
        "completed": "✓",
        "in_progress": "›",
        "pending": "○",
        "cancelled": "—",
    }
    return "\n".join(f"{markers.get(step.status, '○')} {step.text}" for step in steps)


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
