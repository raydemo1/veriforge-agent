"""Agent and conversation loop implementation."""
from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
import weakref
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from ..runtime.arg_preview import safe_args_preview
from ..runtime.builtins.registry import TOOL_SCHEMAS
from ..runtime.tool_context import ToolContext
from ..runtime.tool_result import ToolResult
from ..runtime.tool_runner import finalize_intercepted_tool_result
from ..workspace.shell_jobs import ShellJobManager
from . import context
from .cancellation import CancelledError
from .compaction import CompactionGate, compaction_action, get_thresholds
from .llm_channel import (
    LlmChannel,
    LlmRequestTimeoutError,
    LlmStreamTimeoutError,
    llm_call_simple,
)
from .observations import FactTracker, ObservationStore
from .providers import ProviderAdapter, current_adapter, get_client
from .runtime_state import (
    AgentFallbackState,
    AgentRuntimeState,
)
from .session_emitter import SessionEmitter
from .tool_executor import ToolExecutor
from .trace import TraceWriter
from .turn_controller import TurnController
from .utils import (
    _prompt_cache_key,
    _short_hash,
    capture_prompt_cache_shape,
    compare_prompt_cache_shapes,
)

log = logging.getLogger("harness")
DYNAMIC_CONTEXT_MARKER = "[HARNESS_DYNAMIC_CONTEXT:"




class Agent:
    """
    A single agent with a system prompt and tool access.

    This is the 'managed agent loop' from the architecture:
    - while loop with llm.call(prompt)
    - tool execution
    - context lifecycle (lightweight compaction / handoff reset)

    Skills are handled via progressive disclosure:
    - Model-invoked descriptions are baked into the system prompt.
    - The agent loads relevant SKILL.md files and disclosed references on demand.
    - User-invoked skills stay out of the catalog and enter only through `/name`.
    """

    def __init__(self, name: str, system_prompt: str, use_tools: bool = True,
                 extra_tool_schemas: list[dict] | None = None,
                 middlewares: list | None = None,
                 time_budget: float | None = None,
                 tool_schemas: list[dict] | None = None,
                 tool_context: ToolContext | None = None,
                 stream_callback=None,
                 prompt_cache_identity: dict[str, str] | None = None,
                 model_intensity: str | None = None,
                 max_iterations: int | None = None):
        self.name = name
        self.system_prompt = system_prompt
        self.use_tools = use_tools
        self.extra_tool_schemas = extra_tool_schemas or []
        self.middlewares = middlewares or []  # list[AgentMiddleware]
        self.time_budget = time_budget
        self.tool_schemas = tool_schemas
        self.allowed_tool_names = _tool_names_from_schemas(tool_schemas) if tool_schemas is not None else None
        self.tool_context = tool_context
        self.stream_callback = stream_callback
        self.prompt_cache_identity = prompt_cache_identity
        self.model_intensity = model_intensity
        self.max_iterations = max_iterations
        self.current_task_metadata: dict = {}
        self._conversations: weakref.WeakSet[AgentConversation] = weakref.WeakSet()

    def _create_runtime_state(self, task: str) -> AgentRuntimeState:
        workspace = self.tool_context.workspace.root if self.tool_context is not None else Path.cwd()
        return AgentRuntimeState(
            task_metadata=dict(self.current_task_metadata or {}),
            shell_job_manager=ShellJobManager(workspace),
        )

    def run(self, task: str) -> str:
        """
        Execute the agent loop until the model stops calling tools
        or we hit the iteration limit.

        Returns the final assistant text response.
        Writes a JSONL trace file to {WORKSPACE}/.harness/traces/trace_{name}.jsonl
        """
        conversation = self.start_conversation(task)
        try:
            return conversation.run_until_idle()
        finally:
            conversation.close()

    def update_tool_schemas(self, tool_schemas: list[dict]) -> None:
        """Hot-reload tool schemas (e.g. after MCP reconnect).

        Updates the schema list, allowed name set, and invalidates any
        cached prompt-cache key on running conversations only when schemas change.
        """
        if tool_schemas == self.tool_schemas:
            self.allowed_tool_names = _tool_names_from_schemas(tool_schemas)
            return
        self.tool_schemas = tool_schemas
        self.allowed_tool_names = _tool_names_from_schemas(tool_schemas)
        for conversation in list(self._conversations):
            conversation._cached_prompt_cache_key = None

    def start_conversation(self, initial_task: str | None = None) -> AgentConversation:
        conversation = AgentConversation(self, initial_task)
        self._conversations.add(conversation)
        return conversation


class AgentConversation:
    """Reusable live conversation for interactive CLI sessions."""

    def __init__(self, agent: Agent, initial_task: str | None = None):
        self.agent = agent
        self.trace = TraceWriter(
            agent.name,
            workspace=(agent.tool_context.workspace.root if agent.tool_context is not None else Path.cwd()),
        )
        self.runtime_state = agent._create_runtime_state(initial_task or "")
        if agent.tool_context is not None and agent.tool_context.session_id:
            self.runtime_state.session_id = agent.tool_context.session_id
        if agent.tool_context is not None and agent.tool_context.permission_policy is not None:
            self.runtime_state.permission_mode = agent.tool_context.permission_policy.mode
        self.messages: list[dict] = [{"role": "system", "content": agent.system_prompt}]
        self.client = get_client()
        self._client_needs_refresh = False
        self.provider: ProviderAdapter = current_adapter()
        self.emitter = SessionEmitter(None, agent.name, self.provider.name)
        self.llm = LlmChannel(self)
        self.last_text = ""
        self.last_run_streamed_text = False
        self._closed = False
        self._queued_messages: list[str] = []
        self._queued_messages_lock = threading.Lock()
        self._iteration_offset = 0
        self.compaction_gate = CompactionGate()
        self.event_bus = agent.tool_context.event_bus if agent.tool_context is not None else None
        self.runtime_state.event_bus = self.event_bus
        self.emitter.event_bus = self.event_bus
        self.fact_tracker = FactTracker()
        self.observation_store = ObservationStore(self._observation_dir())
        self._cached_prompt_cache_key: str | None = None
        self._log_rewrite_version = 0
        self._last_prompt_cache_shape = None
        self._pending_prompt_cache_shape = None
        self._run_conversation_start_middlewares()
        if initial_task is not None:
            self.add_user_turn(initial_task)

    def _observation_dir(self) -> Path:
        from .observations import observation_dir_for

        return observation_dir_for(
            self._workspace_root(),
            getattr(self.runtime_state, "session_id", None),
        )

    def _build_prompt(self) -> list[dict]:
        """Build the messages list that will be sent to the LLM.

        The API prompt is the durable conversation log. Dynamic fact changes
        are represented as appended messages, not as a regenerated prelude.
        The current todo list is exposed as a short transient reminder so the
        model always knows what remains; it is never written to history.
        """
        todo = self.runtime_state.todo
        if todo is not None and todo.items:
            reminder = (
                "Current todo (state reminder; continue your work, "
                "do not acknowledge this message):\n" + todo.render()
            )
            return [*self.messages, {"role": "user", "content": reminder}]
        return self.messages

    def rebind_agent(self, agent: Agent) -> None:
        """Keep this conversation while changing the active profile runtime.

        Profile switching is a session-level policy change, not a new chat.
        The message history and runtime state stay intact; only the system
        contract, tool policy and middleware set come from the new agent.
        """
        if agent is self.agent:
            return
        previous = self.agent
        conversations = getattr(previous, "_conversations", None)
        if conversations is not None:
            conversations.discard(self)
        new_conversations = getattr(agent, "_conversations", None)
        if new_conversations is not None:
            new_conversations.add(self)
        self.agent = agent
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = {"role": "system", "content": agent.system_prompt}
        self._cached_prompt_cache_key = None
        self._last_prompt_cache_shape = None
        self._pending_prompt_cache_shape = None

    def compact_now(self) -> str:
        """Compact the current conversation on explicit user request.

        Unlike automatic compaction this does not wait for a token threshold;
        it still uses the existing summarizer and persistence/event pipeline.
        """
        if len(self.messages) <= 2:
            return "当前对话内容还不足以压缩"
        messages_before = len(self.messages)
        token_count_before = context.count_request_tokens(
            self.messages,
            tool_schemas=_tool_schemas_for_agent(self.agent),
        )
        self._strip_dynamic_context_messages()
        summarized = context.summarize_older_conversation(
            self.messages,
            llm_call_simple,
            current_turn_start_index=max(1, self.runtime_state.current_turn_start_index),
        )
        if summarized == self.messages:
            return "当前对话暂时无需压缩"
        self._replace_messages(summarized)
        self.compaction_gate.mark_compacted()
        summary_text = _first_compacted_summary(summarized)
        self._emit_compaction_committed(
            messages_before=messages_before,
            token_count_before=token_count_before,
            summary_chars=len(summary_text),
            summary_text=summary_text,
            phase="manual",
        )
        self._refresh_dynamic_context_after_compaction(phase="manual")
        return f"已压缩对话：{messages_before} 条消息 → {len(self.messages)} 条"

    def add_user_turn(
        self,
        task: str | list[dict],
        *,
        task_text: str | None = None,
    ) -> None:
        logical_task = task_text if task_text is not None else str(task)
        self.runtime_state.current_turn_start_index = len(self.messages)
        self.runtime_state.todo = None
        self.runtime_state.task_metadata = dict(self.agent.current_task_metadata or {})
        self.runtime_state.fallback = AgentFallbackState()
        self.runtime_state.auto_compaction_turn_start_index = -1
        self.runtime_state.auto_compaction_suspended = False
        self.runtime_state.context_anxiety_turn_start_index = -1
        for mw in self.agent.middlewares:
            mw.begin_turn(
                logical_task,
                self.messages,
                runtime_state=self.runtime_state,
                agent_name=self.agent.name,
            )
        self._append_message({"role": "user", "content": task})

    def queue_message(self, message: str) -> None:
        """Queue steering text for the next valid model-request boundary."""
        text = str(message or "").strip()
        if not text:
            raise ValueError("queued message must not be empty")
        with self._queued_messages_lock:
            self._queued_messages.append(text)

    def has_queued_messages(self) -> bool:
        with self._queued_messages_lock:
            return bool(self._queued_messages)

    def _drain_queued_messages(self) -> bool:
        with self._queued_messages_lock:
            pending = self._queued_messages
            self._queued_messages = []
        for message in pending:
            self._append_message({
                "role": "user",
                "content": f"[PARENT STEERING MESSAGE]\n{message}",
            })
            self.trace.middleware_inject("AgentInbox", "safe_boundary", message)
        return bool(pending)

    def _append_message(self, message: dict) -> None:
        self.messages.append(message)
        self.compaction_gate.bump_revision()

    def _run_conversation_start_middlewares(self) -> None:
        for mw in self.agent.middlewares:
            on_start = getattr(mw, "on_conversation_start", None)
            if on_start is None:
                continue
            injected = on_start(
                self.messages,
                runtime_state=self.runtime_state,
                agent_name=self.agent.name,
            ) or []
            for message in injected:
                self._append_message(message)
                self.trace.middleware_inject(
                    type(mw).__name__,
                    "on_conversation_start",
                    str(message.get("content") or ""),
                )

    def _replace_messages(self, messages: list[dict]) -> None:
        self.messages = list(messages)
        self._log_rewrite_version += 1
        self.compaction_gate.bump_revision()

    def _strip_dynamic_context_messages(self) -> bool:
        kept: list[dict] = []
        removed_before_turn = 0
        changed = False
        turn_start = self.runtime_state.current_turn_start_index
        for idx, message in enumerate(self.messages):
            content = _safe_message_content(message)
            if content.startswith(DYNAMIC_CONTEXT_MARKER):
                changed = True
                if idx < turn_start:
                    removed_before_turn += 1
                continue
            kept.append(message)
        if not changed:
            return False
        self.messages = kept
        self.runtime_state.current_turn_start_index = max(1, turn_start - removed_before_turn)
        self.compaction_gate.bump_revision()
        return True

    def _inject_dynamic_context_after_system(self, injected: list[dict], *, source: str) -> None:
        if not injected:
            return
        insert_at = 1 if self.messages and self.messages[0].get("role") == "system" else 0
        for offset, message in enumerate(injected):
            self.messages.insert(insert_at + offset, message)
            self.trace.middleware_inject(
                type(source).__name__ if not isinstance(source, str) else source,
                "on_context_compacted",
                str(message.get("content") or ""),
            )
        if self.runtime_state.current_turn_start_index >= insert_at:
            self.runtime_state.current_turn_start_index += len(injected)
        self.compaction_gate.bump_revision()

    def _refresh_dynamic_context_after_compaction(self, *, phase: str) -> None:
        self._strip_dynamic_context_messages()
        for mw in self.agent.middlewares:
            on_compacted = getattr(mw, "on_context_compacted", None)
            if on_compacted is None:
                continue
            injected = on_compacted(
                self.messages,
                runtime_state=self.runtime_state,
                agent_name=self.agent.name,
                phase=phase,
            ) or []
            self._inject_dynamic_context_after_system(injected, source=type(mw).__name__)

    def _session_compacted_dir(self) -> Path | None:
        session_id = getattr(self.runtime_state, "session_id", None)
        if not session_id or session_id == "default":
            return None
        safe_session_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(session_id))
        root = self._workspace_root()
        return root / ".harness" / "sessions" / safe_session_id / "compacted"

    def _persist_compacted_summary(self, summary: str, *, phase: str) -> None:
        text = (summary or "").strip()
        if not text:
            return
        compacted_dir = self._session_compacted_dir()
        if compacted_dir is None:
            return
        try:
            history_dir = compacted_dir / "history"
            history_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
            safe_phase = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in phase) or "compact"
            body = f"# Compacted Context\n\nphase: {phase}\ncreated_at: {datetime.now(timezone.utc).isoformat()}\n\n{text}\n"
            history_path = history_dir / f"{stamp}-{safe_phase}.md"
            history_path.write_text(body, encoding="utf-8")
            (compacted_dir / "latest.md").write_text(body, encoding="utf-8")
        except OSError as exc:
            log.debug("Failed to persist compacted summary: %s", exc)

    def _emit_compaction_committed(self, *, messages_before: int, token_count_before: int,
                                   summary_chars: int = 0, summary_text: str = "",
                                   phase: str = "compact") -> None:
        if summary_text:
            self._persist_compacted_summary(summary_text, phase=phase)
        if self.emitter.event_bus is None:
            return
        tokens_after = context.count_request_tokens(
            self.messages,
            tool_schemas=_tool_schemas_for_agent(self.agent),
        )
        self.emitter.emit_compaction_committed(
            messages_before=messages_before,
            messages_after=len(self.messages),
            tokens_saved=max(0, token_count_before - tokens_after),
            summary_chars=summary_chars,
        )

    def _maybe_auto_compact(self, agent: Agent, *, token_count: int, thresholds) -> None:
        state = self.runtime_state
        if state.auto_compaction_suspended:
            self.trace.context_event("auto_compaction_suspended", f"tokens={token_count}")
            return
        if state.auto_compaction_turn_start_index == state.current_turn_start_index:
            state.auto_compaction_suspended = True
            self.trace.context_event("auto_compaction_suspended", "already attempted this turn")
            return
        if not self.compaction_gate.can_compact(coalesce_seconds=0):
            return

        state.auto_compaction_turn_start_index = state.current_turn_start_index
        messages_before = len(self.messages)
        token_count_before = token_count
        self._strip_dynamic_context_messages()

        # Layer 1 — summarize older conversation via lightweight LLM call
        self.emitter.emit_compaction_started(
            token_count=token_count,
            threshold=thresholds.compact,
            phase="summarizing_history",
        )
        summarized = context.summarize_older_conversation(
            self.messages,
            llm_call_simple,
            current_turn_start_index=state.current_turn_start_index,
        )
        summary_chars = 0
        summary_text = ""
        if summarized != self.messages:
            summary_text = _first_compacted_summary(summarized)
            summary_chars = len(summary_text)
            self._replace_messages(summarized)
            self.compaction_gate.mark_compacted()
            self._emit_compaction_committed(
                messages_before=messages_before,
                token_count_before=token_count_before,
                summary_chars=summary_chars,
                summary_text=summary_text,
                phase="summarizing_history",
            )
            self._refresh_dynamic_context_after_compaction(phase="summarizing_history")

        tokens_after_summary = context.count_request_tokens(
            self.messages,
            tool_schemas=_tool_schemas_for_agent(agent),
        )
        if tokens_after_summary < thresholds.compact:
            state.context_refill_streak = 0
            return

        # Layer 2 — reset context from a persisted handoff document
        state.context_refill_streak += 1
        state.auto_compaction_suspended = True
        self.emitter.emit_compaction_started(
            token_count=tokens_after_summary,
            threshold=thresholds.compact,
            phase="auto_compaction_suspended",
        )
        self.trace.context_event("auto_compaction_suspended", f"streak={state.context_refill_streak}")
        if state.context_refill_streak >= 2:
            self._handoff_reset_context(agent, token_count=tokens_after_summary, thresholds=thresholds)

    def _handoff_reset_context(self, agent: Agent, *, token_count: int, thresholds) -> None:
        self.emitter.emit_compaction_started(
            token_count=token_count,
            threshold=thresholds.compact,
            phase="handoff_reset",
        )
        messages_before = len(self.messages)
        system_prompt = (
            self.messages[0].get("content", "")
            if self.messages and self.messages[0].get("role") == "system"
            else agent.system_prompt
        )
        workspace = str(self._workspace_root())
        handoff, handoff_path = context.create_handoff_reset(
            self.messages,
            self._working_context_state(),
            llm_call_simple,
            session_id=self.runtime_state.session_id,
            profile=agent.name,
            workspace=workspace,
            max_turns=5,
        )
        reset_messages = context.restore_from_handoff_reset(handoff, system_prompt, handoff_path)
        self._replace_messages(reset_messages)
        self.runtime_state.current_turn_start_index = max(1, len(self.messages) - 1)
        self.compaction_gate.mark_compacted()
        self.runtime_state.context_refill_streak = 0
        self.runtime_state.auto_compaction_suspended = True
        self.runtime_state.auto_compaction_turn_start_index = -1
        self.runtime_state.context_anxiety_turn_start_index = -1
        self.runtime_state.fallback = AgentFallbackState()
        summary_text = _first_compacted_summary(self.messages)
        self._emit_compaction_committed(
            messages_before=messages_before,
            token_count_before=token_count,
            summary_chars=len(summary_text),
            summary_text=summary_text,
            phase="handoff_reset",
        )
        self._refresh_dynamic_context_after_compaction(phase="handoff_reset")

    def _working_context_state(self) -> dict:
        recent_errors, failed_commands = self._recent_error_state()
        files = self._changed_workspace_files()
        observed_files = [
            key.removeprefix("file:")
            for observation in self.observation_store.observations
            for key in observation.resource_keys
            if key.startswith("file:")
        ]
        files_touched = list(dict.fromkeys([*files, *observed_files]))
        return {
            "current_user_task": self._latest_user_message(),
            "current_todo": self._current_todo_text(),
            "changed_files": files,
            "files_touched": files_touched,
            "recent_errors": recent_errors,
            "failed_commands": failed_commands,
            "active_constraints": self._active_constraints(),
            "latest_checkpoint_summary": self._latest_checkpoint_summary(),
            "next_recommended_action": "continue from the active task and verify the next smallest change",
        }

    def _changed_workspace_files(self) -> list[str]:
        tool_context = getattr(self.agent, "tool_context", None)
        workspace = getattr(tool_context, "workspace", None) if tool_context is not None else None
        if workspace is None:
            return []
        journal = getattr(workspace, "change_journal", None)
        if journal is not None:
            return [str(path) for path in journal.changed_paths()]
        return [str(path) for path in getattr(workspace, "changed_files", [])]

    def _latest_user_message(self) -> str:
        for msg in reversed(self.messages):
            if msg.get("role") == "user" and msg.get("content"):
                return _safe_message_content(msg)
        return ""

    def _current_todo_text(self) -> str:
        todo = self.runtime_state.todo
        if todo is None or not todo.items:
            return "none"
        return todo.render()

    def _recent_error_state(self) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        failed_commands: list[str] = []
        for msg in self.messages[-30:]:
            content = _safe_message_content(msg)
            lowered = content.lower()
            if "[error]" in lowered or "failed" in lowered or "traceback" in lowered:
                errors.append(content[:500])
            if msg.get("role") == "tool" and (
                "return_code:" in lowered
                or "status: failed" in lowered
                or "[error]" in lowered
                or ("[exit_code:" in lowered and "[exit_code: 0]" not in lowered)
            ):
                failed_commands.append(content[:300])
        return errors[-5:], failed_commands[-5:]

    def _active_constraints(self) -> list[str]:
        constraints = []
        for msg in self.messages[-20:]:
            content = _safe_message_content(msg)
            lowered = content.lower()
            if "constraint" in lowered or "不要" in content or "must" in lowered or "only" in lowered:
                constraints.append(content[:300])
        return constraints[-5:]

    def _latest_checkpoint_summary(self) -> str:
        root = self._workspace_root()
        for rel in ("progress.md", ".harness/checkpoints/latest.md"):
            path = root / rel
            if path.exists() and path.is_file():
                try:
                    text = path.read_text(encoding="utf-8", errors="replace").strip()
                    if text:
                        return text[:4_000]
                except OSError:
                    continue
        return "none"

    def submit(
        self,
        task: str | list[dict],
        cancellation_token=None,
        *,
        task_text: str | None = None,
    ) -> str:
        self.add_user_turn(task, task_text=task_text)
        return self.run_until_idle(cancellation_token=cancellation_token)

    def next_call_id(self) -> str:
        return f"{self.agent.name}-{time.time_ns()}"

    def _check_cancelled(self, cancellation_token) -> None:
        if cancellation_token is not None:
            cancellation_token.check()

    def _workspace_root(self) -> Path:
        if self.agent.tool_context is not None:
            return self.agent.tool_context.workspace.root
        return Path.cwd()

    def refresh_client(self) -> None:
        with contextlib.suppress(Exception):
            self.client.close()
        self.client = get_client()
        self._client_needs_refresh = False

    def record_llm_usage(self, usage: dict | None, prompt_cache_key: str | None) -> None:
        fallback = self.runtime_state.fallback
        fallback.llm_call_count += 1
        if usage:
            total_tokens = usage.get("total_tokens")
            if total_tokens is not None:
                try:
                    fallback.total_tokens += int(total_tokens)
                except (TypeError, ValueError):
                    pass
            self._maybe_emit_budget_warning(
                "total_tokens",
                fallback.total_tokens,
                config.MAX_AGENT_TOTAL_TOKENS,
            )
        if not usage or self.event_bus is None:
            return
        from ..sessions.events import LlmUsageEvent

        key_hash = _short_hash(prompt_cache_key) if prompt_cache_key else None
        cache_hit_tokens = int(usage.get("cache_hit_tokens") or usage.get("cached_tokens") or 0)
        cache_miss_tokens = int(usage.get("cache_miss_tokens") or 0)
        cache_shape = self._pending_prompt_cache_shape or capture_prompt_cache_shape(
            self.agent,
            _tool_schemas_for_agent(self.agent),
            log_rewrite_version=self._log_rewrite_version,
        )
        cache_diagnostics = compare_prompt_cache_shapes(
            self._last_prompt_cache_shape,
            cache_shape,
            usage,
        )
        self._last_prompt_cache_shape = cache_shape
        self._pending_prompt_cache_shape = None
        self.event_bus.emit_event(
            LlmUsageEvent(
                provider=getattr(self.provider, "name", "unknown"),
                model=config.MODEL,
                prompt_tokens=usage.get("prompt_tokens"),
                cached_tokens=cache_hit_tokens,
                cache_hit_tokens=cache_hit_tokens,
                cache_miss_tokens=cache_miss_tokens,
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                prompt_cache_key_hash=key_hash,
                cache_diagnostics=cache_diagnostics,
                agent=self.agent.name,
            ).to_event()
        )

    def _limit_enabled(self, limit: float | None) -> bool:
        try:
            return limit is not None and float(limit) > 0
        except (TypeError, ValueError):
            return False

    def _maybe_emit_budget_warning(self, limit_type: str, used: int, limit: int) -> None:
        if not self._limit_enabled(limit):
            return
        fallback = self.runtime_state.fallback
        if limit_type in fallback.budget_warnings:
            return
        fraction = config.AGENT_BUDGET_WARN_FRACTION
        if fraction <= 0:
            return
        threshold = max(1, int(float(limit) * min(fraction, 1.0)))
        if used < threshold:
            return
        fallback.budget_warnings.add(limit_type)
        if self.event_bus is None:
            return
        from ..sessions.events import AgentBudgetWarningEvent

        self.event_bus.emit_event(
            AgentBudgetWarningEvent(
                limit_type=limit_type,
                used=int(used),
                limit=int(limit),
                fraction=min(float(used) / float(limit), 999.0),
                agent=self.agent.name,
            ).to_event()
        )

    def _request_token_budget_stop_if_needed(self) -> bool:
        limit = config.MAX_AGENT_TOTAL_TOKENS
        fallback = self.runtime_state.fallback
        if not self._limit_enabled(limit) or fallback.total_tokens <= limit:
            return False
        fallback.request_stop(
            reason="token_budget_exceeded",
            limit_type="total_tokens",
            used=fallback.total_tokens,
            limit=limit,
            recent_action_summary=fallback.recent_action_summary,
        )
        return True

    def _record_tool_call_budget(self, tool_name: str, tool_args: dict) -> bool:
        fallback = self.runtime_state.fallback
        limit = config.MAX_AGENT_TOOL_CALLS
        if self._limit_enabled(limit) and fallback.tool_call_count >= limit:
            fallback.request_stop(
                reason="tool_call_budget_exceeded",
                limit_type="tool_calls",
                used=fallback.tool_call_count,
                limit=limit,
                last_tool=tool_name,
                recent_action_summary=fallback.recent_action_summary,
            )
            return False
        fallback.tool_call_count += 1
        fallback.record_action(_safe_tool_summary(tool_name, tool_args))
        self._maybe_emit_budget_warning("tool_calls", fallback.tool_call_count, limit)
        return True

    def _fallback_text(self) -> str:
        fallback = self.runtime_state.fallback
        details = [f"Agent fallback triggered: {fallback.stop_reason or 'unknown'}."]
        if fallback.stop_limit_type:
            used = "unknown" if fallback.stop_used is None else str(fallback.stop_used)
            limit = "unknown" if fallback.stop_limit is None else str(fallback.stop_limit)
            details.append(f"{fallback.stop_limit_type}: {used}/{limit}.")
        if fallback.stop_last_tool:
            details.append(f"Last tool: {fallback.stop_last_tool}.")
        details.append("The current turn was stopped to prevent runaway execution; inspect the session events for details.")
        return " ".join(details)

    def _append_blocked_tool_results(self, tool_calls: list, reason: str) -> None:
        status_source = "budget" if "budget" in reason else "fallback"
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_arguments = tc["function"].get("arguments") or "{}"
            try:
                fn_args = json.loads(fn_arguments)
            except json.JSONDecodeError:
                fn_args = {}
            output = f"[blocked] Agent fallback triggered ({reason}); tool was not executed."
            tool_result = finalize_intercepted_tool_result(
                ToolResult(
                    tool=fn_name,
                    status="failed",
                    output=output,
                    error=output.removeprefix("[blocked] "),
                    metadata={"status_source": status_source, "fallback_reason": reason},
                ),
                arguments=fn_args,
                tool_context=self.agent.tool_context,
                agent_name=self.agent.name,
            )
            result = tool_result.to_text()
            self.trace.tool_call(fn_name, fn_args, result)
            self._append_message({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    def run_until_idle(self, cancellation_token=None) -> str:
        agent = self.agent
        self.last_run_streamed_text = False
        run_started = time.monotonic()
        turn_controller = TurnController(self)

        iteration_limit = agent.max_iterations or config.MAX_AGENT_ITERATIONS
        for local_iteration in range(1, iteration_limit + 1):
            iteration = self._iteration_offset + local_iteration

            # --- Cancellation check ---
            self._check_cancelled(cancellation_token)
            self._drain_queued_messages()

            # --- Time budget check ---
            budget_decision = turn_controller.check_time_budget(
                run_started=run_started,
                iteration=iteration,
            )
            if budget_decision is not None:
                break

            # --- Middleware: per-iteration hooks ---
            for mw in agent.middlewares:
                inject = mw.per_iteration(
                    iteration,
                    self.messages,
                    runtime_state=self.runtime_state,
                    agent_name=agent.name,
                )
                if inject:
                    self._append_message({"role": "user", "content": inject})
                    self.trace.middleware_inject(type(mw).__name__, "per_iteration", inject)

            # --- Context lifecycle check ---
            tool_schemas = _tool_schemas_for_agent(agent)
            token_count = context.count_request_tokens(self.messages, tool_schemas=tool_schemas)
            log.info(f"[{agent.name}] iteration={iteration}  tokens~{token_count}")
            self.trace.iteration(iteration, token_count)

            thresholds = get_thresholds()
            action = compaction_action(token_count, thresholds)
            anxiety = context.detect_anxiety(self.messages)
            if anxiety and self.runtime_state.context_anxiety_turn_start_index != self.runtime_state.current_turn_start_index:
                self.runtime_state.context_anxiety_turn_start_index = self.runtime_state.current_turn_start_index
                score = getattr(anxiety, "score", 0)
                self.trace.context_event("context_anxiety", f"tokens={token_count} score={score}")
                self.emitter.emit_context_anxiety_observed(
                    token_count=token_count,
                    threshold=thresholds.compact,
                    signal=anxiety,
                )
            if action == "auto_compact":
                self._maybe_auto_compact(agent, token_count=token_count, thresholds=thresholds)

            # --- Build prompt and LLM call ---
            prompt_messages = self._build_prompt()
            profile = config.resolve_model_profile(agent.model_intensity or config.MODEL_INTENSITY)
            chat_args = {
                "profile": profile,
                "messages": prompt_messages,
                "max_tokens": 32768,
            }
            if agent.use_tools:
                chat_args["tools"] = tool_schemas
                chat_args["tool_choice"] = "auto"
            else:
                tool_schemas = None
            self._pending_prompt_cache_shape = capture_prompt_cache_shape(
                agent,
                tool_schemas,
                log_rewrite_version=self._log_rewrite_version,
            )
            if self.provider.supports_prompt_cache_key:
                if self._cached_prompt_cache_key is None:
                    self._cached_prompt_cache_key = _prompt_cache_key(agent, tool_schemas)
                chat_args["prompt_cache_key"] = self._cached_prompt_cache_key
            kwargs = self.provider.chat_kwargs(**chat_args)

            try:
                completion = self.llm.request_assistant_message(kwargs, cancellation_token=cancellation_token)
            except CancelledError:
                raise
            except (LlmStreamTimeoutError, LlmRequestTimeoutError) as e:
                self.trace.error("api_timeout", str(e))
                self.trace.finish("llm_timeout", iteration)
                raise
            except Exception as e:
                # The channel owns the bounded retry budget; once it raises,
                # the request has failed — record it and end the turn.
                log.error(f"[{agent.name}] LLM request failed: {e}")
                self.trace.error("api_error", str(e))
                self.trace.finish("api_error", iteration)
                break

            assistant_msg, finish_reason = completion
            content = assistant_msg.get("content")
            tool_calls = assistant_msg.get("tool_calls") or []

            self._check_cancelled(cancellation_token)

            # --- Append assistant message to history ---
            self._append_message(assistant_msg)

            # --- Trace the LLM response ---
            self.trace.llm_response(content, tool_calls, finish_reason)

            # --- If model produced text, capture it ---
            if content:
                self.last_text = content
                log.info(f"[{agent.name}] assistant: {content[:200]}...")

            # --- If no tool calls, check pre-exit middlewares ---
            if not tool_calls:
                decision = turn_controller.after_no_tool_calls(iteration=iteration)
                if decision.continue_loop:
                    continue
                break

            if self._request_token_budget_stop_if_needed():
                self.emitter.emit_agent_fallback(self.runtime_state.fallback)
                self.last_text = self._fallback_text()
                self._append_blocked_tool_results(tool_calls, self.runtime_state.fallback.stop_reason)
                self.trace.finish("agent_fallback", iteration)
                break

            # --- Execute tool calls ---
            stop_after_tool_loop = ToolExecutor(
                self,
                cancellation_token=cancellation_token,
            ).execute(tool_calls)

            if stop_after_tool_loop:
                break

            decision = turn_controller.after_tool_calls(
                finish_reason=finish_reason,
                iteration=iteration,
            )
            if not decision.continue_loop:
                break

        else:
            turn_controller.finish_max_iterations(iteration_limit=iteration_limit)

        self._iteration_offset += local_iteration
        return self.last_text

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for mw in self.agent.middlewares:
            on_close = getattr(mw, "on_conversation_close", None)
            if on_close is None:
                continue
            on_close(
                self.messages,
                runtime_state=self.runtime_state,
                agent_name=self.agent.name,
            )
        self.runtime_state.close_shell_sessions()
        if self.runtime_state.shell_job_manager is not None:
            self.runtime_state.shell_job_manager.close()
        with contextlib.suppress(Exception):
            self.client.close()


def _safe_tool_summary(tool_name: str, tool_args: dict) -> str:
    return f"{tool_name}({_safe_args_preview(tool_args)})"


def _safe_message_content(message: dict) -> str:
    content = message.get("content") or ""
    if isinstance(content, list):
        return context._messages_to_text([message])
    return str(content)


def _safe_args_preview(tool_args: dict) -> str:
    """Backward-compatible wrapper (200-char limit)."""
    return safe_args_preview(tool_args, max_chars=200)


def _tool_schemas_for_agent(agent: Agent) -> list[dict] | None:
    if not agent.use_tools:
        return None
    if agent.tool_schemas is not None:
        return agent.tool_schemas
    return TOOL_SCHEMAS + agent.extra_tool_schemas


def _first_compacted_summary(messages: list[dict]) -> str:
    for message in messages:
        content = str(message.get("content") or "")
        if not (
            content.startswith(("[COMPACTED CONTEXT", "[HANDOFF RESET]"))
        ):
            continue
        _header, _sep, body = content.partition("\n")
        return (body or content).strip()
    return ""


def _tool_names_from_schemas(tool_schemas: list[dict] | None) -> set[str]:
    names: set[str] = set()
    for schema in tool_schemas or []:
        function = schema.get("function") if isinstance(schema, dict) else None
        if isinstance(function, dict) and function.get("name"):
            names.add(str(function["name"]))
    return names


def _assistant_message_from_response(msg) -> dict:
    return current_adapter().assistant_message_from_response(msg)
