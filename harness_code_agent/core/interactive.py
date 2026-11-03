from __future__ import annotations

from functools import wraps
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from ..agent.cancellation import current_token
from ..agent.cancellation import CancellationToken, CancelledError, cancellation_scope
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .. import config
from ..agent.conversation import AgentConversation
from ..agent.coordinator import AgentCoordinator
from ..agent.prompts import GlobalRulesDoc, PromptPrefixBuilder
from ..attachments import (
    Attachment,
    AttachmentError,
    AttachmentManager,
    ExternalPathConfirmationRequired,
    PreparedTurn,
    TurnSubmission,
    build_model_content,
    model_input_mode,
)
from ..memory import EvidenceReference, MemoryService, MemoryWriteCommand
from ..memory.background import (
    begin_memory_turn,
    close_memory_session,
    finish_memory_turn,
    open_memory_session,
    start_memory_worker,
)
from ..profiles import get_profile
from ..profiles.base import BaseProfile
from ..profiles.router import (
    ROUTE_ACTION_SWITCH_PROFILE,
    ROUTING_MODE_AUTO,
    ROUTING_MODE_PINNED,
    TURN_MODE_DIRECT_ANSWER,
    JevRouteClassifier,
    RouteDecision,
    route_profile_for_turn,
)
from ..runtime.approvals import (
    ApprovalProvider,
    ApprovalRequest,
    ConsoleApprovalProvider,
    LlmAutoApprovalProvider,
)
from ..runtime.builtins.registry import BUILTIN_TOOL_REGISTRY
from ..runtime.lifecycle import LifecycleScope
from ..runtime.mcp import McpClientManager, McpConfigError, build_server_config
from ..runtime.middleware.loader import load_user_middlewares
from ..runtime.middleware.stack import build_main_agent_middlewares
from ..runtime.permissions import PermissionPolicy
from ..runtime.questions import (
    ConsoleQuestionProvider,
    QuestionOption,
    QuestionProvider,
    QuestionRequest,
)
from ..runtime.recovery import RecoveryService, atomic_json
from ..runtime.tool_context import ToolContext
from ..runtime.tool_registry import tool_schemas_for_profile
from ..sessions.events import (
    AssistantMessageEvent,
    FinalReportEvent,
    SessionFinishedEvent,
    TurnSummaryEvent,
    UserInputEvent,
)
from ..sessions.journal import SessionJournal
from ..sessions.report import build_final_report
from ..sessions.store import Session, SessionStore
from ..sessions.turn_summary import generate_turn_summary, should_summarize_turn
from ..skills import SkillRegistry
from ..workspace.service import WorkspaceService
from ..workspace.shell_session import (
    validate_shell_configuration,
)
from .mentions import (
    ResolvedMention,
    render_mention_context,
    resolve_mentions,
)

PRODUCT_DEFAULT_PROFILE = "general"
DIRECT_ANSWER_TURN_INSTRUCTION = (
    "Turn handling instruction: answer this turn directly and briefly. "
    "Do not create or edit files, run commands, launch browsers, or continue prior implementation work "
    "unless the user explicitly asks for that in this turn."
)
TURN_INLINE_CHAR_LIMIT = 40_000
TURN_EXCERPT_CHARS = 2_000
log = logging.getLogger("harness")


@dataclass
class TurnResult:
    text: str
    notice: str = ""
    streamed: bool = False


@dataclass
class ProfileSwitchEvent:
    previous: str
    current: str
    reason: str


def runtime_change(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self._execution_condition:
            if self._closed or self._closing:
                raise RuntimeError("会话正在关闭")
            if self._turn_thread is not None and self._turn_thread != threading.get_ident():
                raise RuntimeError("当前回合仍在执行，暂时无法修改运行时")
            if self.tool_context is not None and self.tool_context.tool_tasks.pending_count:
                raise RuntimeError("前台工具尚未退出，暂时无法修改运行时")
            return method(self, *args, **kwargs)
    return guarded


class InteractiveSession:
    def __init__(
        self,
        *,
        cwd: str | Path,
        profile_name: str = PRODUCT_DEFAULT_PROFILE,
        stream_sink: Callable[[str], None] | None = None,
        event_listener: Callable[[object], None] | None = None,
        approval_provider: ApprovalProvider | None = None,
        question_provider: QuestionProvider | None = None,
        output_sink: Callable[[str], None] | None = None,
        stream_callback=None,
        profile_explicit: bool | None = None,
        enable_turn_summary: bool = True,
        startup_sink: Callable[[str], None] | None = None,
        session_id: str | None = None,
        state_root: str | Path | None = None,
        environment: str = "local",
    ):
        self.cwd = Path(cwd).resolve()
        self._requested_session_id = session_id
        self.environment = environment
        self.state_root = Path(state_root).resolve() if state_root else self.cwd / ".harness"
        self.model_selection = config.ModelSelection()
        self.lifecycle = LifecycleScope()
        self._execution_condition = threading.Condition(threading.RLock())
        self._closing = False
        self._turn_thread = None
        self._turn_token = None
        self._startup_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="session-startup")
        self.lifecycle.register("startup_executor", lambda: self._startup_executor.shutdown(wait=True), order=50)
        self._mcp_future = None
        try:
            self.profile_router = JevRouteClassifier()
            self.lifecycle.register("profile_router", self.profile_router.close, order=40)
            self.startup_sink = startup_sink
            self._report_startup("checking workspace")
            validate_shell_configuration()
            self.stream_sink = stream_sink or stream_callback
            self.event_listener = event_listener
            self.permission_mode = os.environ.get("HARNESS_PERMISSION_MODE", "workspace-write")
            PermissionPolicy(mode=self.permission_mode)
            self._manual_approval_provider = approval_provider or ConsoleApprovalProvider()
            self.approval_provider = self._approval_provider_for_mode(self.permission_mode)
            self.question_provider = question_provider or ConsoleQuestionProvider()
            self.output_sink = output_sink or print
            self.enable_turn_summary = enable_turn_summary
            self.memory_use_enabled = os.environ.get("HARNESS_MEMORY_DISABLED", "").lower() not in {"1", "true", "yes", "on"}
            self.memory_auto_extract_enabled = os.environ.get("HARNESS_MEMORY_GENERATION_DISABLED", "").lower() not in {"1", "true", "yes", "on"}
            self.memory_service = MemoryService(self.cwd)
            self.recovery = RecoveryService(self.cwd)
            self._report_startup("loading skills")
            self.skill_registry = SkillRegistry()
            self._slash_registry = None
            self.last_command_result = None
            self.session_store = SessionStore(self.state_root)
            self.session_store.root.mkdir(parents=True, exist_ok=True)
            saved = None
            if session_id and (self.session_store.sessions_dir / session_id / "session.json").exists():
                saved = self.session_store.read_metadata(session_id)
                profile_name = saved["profile"]
                profile_explicit = saved["routing_mode"] == "pinned"
                self.permission_mode = saved["permission_mode"]
                self.approval_provider = self._approval_provider_for_mode(self.permission_mode)
                self.model_selection.select(**saved.get("model_selection", {}))
            inferred_explicit = profile_name != PRODUCT_DEFAULT_PROFILE
            self.profile_explicit = inferred_explicit if profile_explicit is None else profile_explicit
            self.routing_mode = ROUTING_MODE_PINNED if self.profile_explicit else ROUTING_MODE_AUTO
            self._pending_profile_name = profile_name
            self._profile_source = "explicit" if self.profile_explicit else "default"
            self.profile = get_profile(self._pending_profile_name)
            self.session: Session | None = None
            self.attachment_manager: AttachmentManager | None = None
            self.event_bus = None
            self.tool_context: ToolContext | None = None
            self.tool_registry = None
            self.mcp_manager = None
            self._mcp_tools_loaded = False
            self._mcp_load_lock = threading.Lock()
            self._mcp_connection_revision = 0
            self.agent = None
            self.conversation: AgentConversation | None = None
            self.turn_count = 0
            self.pending_plan_markdown: str | None = None
            self.pending_plan_revision = 0
            self.last_user_task: str = ""
            self.last_assistant_text: str = ""
            self.profile_history: list[ProfileSwitchEvent] = []
            self._resolved_task_timeout: float | None = None
            self._closed = False
            self._close_lock = threading.Lock()
            # User middlewares are loaded exactly once per session; profile
            # switches reuse the same instances.
            self.user_middlewares = load_user_middlewares()
            self._report_startup("connecting tools")
            self._bind_profile(self._pending_profile_name, source=self._profile_source)
            if saved:
                self._restore_history()
            if self.memory_auto_extract_enabled:
                open_memory_session(self.cwd, self.session.id, self.session.journal_path)
            self._report_startup("ready")
        except BaseException:
            for error in self.lifecycle.close():
                log.warning("Failed to close session resource %s: %s", error.name, error.error)
            session = getattr(self, "session", None)
            if session is not None:
                try:
                    self.session_store.update_status(session.id, "failed")
                except Exception as exc:
                    log.warning("Failed to update session status for %s: %s", session.id, exc)
            raise

    def _report_startup(self, stage: str) -> None:
        if self.startup_sink is not None:
            self.startup_sink(stage)

    @property
    def is_bound(self) -> bool:
        return self.session is not None and self.conversation is not None

    @property
    def session_id(self) -> str | None:
        return self.session.id if self.session is not None else None

    @property
    def display_profile(self) -> str:
        return self.profile.name()

    @property
    def display_routing_mode(self) -> str:
        return self.routing_mode

    def _build_agent(self, profile: BaseProfile, tool_context=None):
        from ..agent.conversation import Agent

        context = tool_context or self.tool_context
        if context is None:
            raise RuntimeError("Cannot build agent before the session is bound.")
        cfg = profile.main_agent()
        context.allowed_tool_permissions = set(cfg.allowed_tool_permissions)
        context.blocked_tool_names = set(cfg.blocked_tool_names)
        harness_rules = _load_harness_rules(self.cwd)
        catalog = self.skill_registry.catalog_listing()
        prefix = PromptPrefixBuilder().build(
            profile_prompt=cfg.system_prompt,
            global_rules_docs=[harness_rules] if harness_rules is not None else [],
            skill_catalog=catalog,
        )
        middlewares = build_main_agent_middlewares(
            user_middlewares=self.user_middlewares,
            tool_context=context,
            tool_registry=self.tool_registry,
            workspace=self.cwd,
        )
        return Agent(
            "main_agent",
            prefix.content,
            use_tools=True,
            tool_schemas=self._tool_schemas_for_agent_config(cfg, tool_context=context),
            middlewares=middlewares,
            time_budget=cfg.time_budget,
            tool_context=context,
            stream_callback=self.stream_sink,
            prompt_cache_identity=prefix.cache_identity,
        )

    def _load_mcp_tools(self) -> None:
        if self.tool_registry is None or self.mcp_manager is None:
            self._prepare_tool_registry()
        self.mcp_manager.connect_all()
        snapshot = self.mcp_manager.catalog_snapshot()
        self.mcp_manager.register_tools(self.tool_registry, snapshot=snapshot)
        self._mcp_connection_revision = snapshot.revision
        self._mcp_tools_loaded = True

    def _prepare_tool_registry(self) -> None:
        self._mcp_future = None
        self.tool_registry = BUILTIN_TOOL_REGISTRY.copy()
        self.mcp_manager = McpClientManager.from_workspace(self.cwd)
        manager = self.mcp_manager
        self.lifecycle.register(f"mcp:{id(manager)}", manager.close, order=10)
        self._mcp_tools_loaded = False

    def _wait_session_future(self, future):
        token = current_token.get()
        while True:
            if token is not None:
                token.check()
            try:
                return future.result(timeout=0.05)
            except FutureTimeout:
                if future.done():
                    return future.result()

    def _ensure_mcp_tools_loaded(self) -> None:
        with self._execution_condition:
            if self._closed or self._closing:
                raise RuntimeError("会话正在关闭")
            if self._mcp_future is None:
                self._mcp_future = self._startup_executor.submit(self._initialize_mcp_tools)
                self.tool_context.tool_tasks.track(self._mcp_future, foreground=False)
            future = self._mcp_future
        self._wait_session_future(future)

    def _initialize_mcp_tools(self) -> None:
        if self._mcp_tools_loaded:
            return
        with self._mcp_load_lock:
            if self._mcp_tools_loaded:
                return
            self._report_startup("connecting external tools")
            self._load_mcp_tools()
            self._refresh_agent_tool_schemas()
            self._refresh_mcp_catalog()
            self._report_startup("external tools ready")

    def warm_mcp_tools(self) -> None:
        """Connect configured MCP servers after the core session is usable."""
        self._ensure_mcp_tools_loaded()

    def _tool_schemas_for_agent_config(self, cfg, *, update_context: bool = True, tool_context=None) -> list[dict]:
        context = tool_context or self.tool_context
        core_schemas = tool_schemas_for_profile(
            allowed_permissions=cfg.allowed_tool_permissions,
            exclude_names=cfg.blocked_tool_names,
            registry=self.tool_registry,
        )
        discovered = set()
        if context is not None and update_context:
            context.allowed_tool_permissions = set(cfg.allowed_tool_permissions)
            context.blocked_tool_names = set(cfg.blocked_tool_names)
            discovered = set(context.discovered_tool_names)
        if not discovered:
            if update_context and context is not None:
                # The callable view must not outlive the discovery it came
                # from; recompute it even when nothing is discovered.
                context.revealed_tool_names = set()
            return core_schemas
        revealed_schemas = tool_schemas_for_profile(
            allowed_permissions=cfg.allowed_tool_permissions,
            include_names=discovered,
            exclude_names=cfg.blocked_tool_names,
            registry=self.tool_registry,
            disclosure={"deferred"},
        )
        if update_context and context is not None:
            # A profile switch only recomputes the callable view; discovery is
            # history and must survive, otherwise switching away and back would
            # make the repeat-query guard permanently block remounting.
            context.revealed_tool_names = {
                schema.get("function", {}).get("name")
                for schema in revealed_schemas
                if isinstance(schema, dict)
            }
        known = {
            schema.get("function", {}).get("name")
            for schema in core_schemas
            if isinstance(schema, dict)
        }
        return core_schemas + [
            schema
            for schema in revealed_schemas
            if schema.get("function", {}).get("name") not in known
        ]

    def _refresh_agent_tool_schemas(self) -> None:
        if self.agent is not None:
            self.agent.update_tool_schemas(self._tool_schemas_for_agent_config(self.profile.main_agent()))

    def _refresh_mcp_catalog(self) -> None:
        if self.mcp_manager is not None and self.agent is not None:
            self.agent.set_mcp_catalog(self.mcp_manager.catalog_prompt())

    def _apply_profile_task_timeout(self, user_prompt: str) -> None:
        if self.agent is None:
            return
        metadata = self._resolve_profile_task_metadata(user_prompt)
        self.agent.current_task_metadata = metadata
        resolver = getattr(self.profile, "resolve_task_timeout", None)
        if resolver is None:
            timeout = metadata.get("agent_timeout_sec") if metadata else None
        else:
            try:
                timeout = resolver(user_prompt)
            except Exception as exc:
                log.debug("Failed to resolve task timeout for profile %s: %s", self.profile.name(), exc)
                timeout = None
            if timeout is None and metadata:
                timeout = metadata.get("agent_timeout_sec")
        if metadata and self.event_bus is not None:
            payload = {
                "profile": self.profile.name(),
                "task_metadata": metadata,
            }
            self.event_bus.emit("task_metadata_resolved", agent="main_agent", payload=payload)
        if timeout is None:
            return
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            return
        if timeout <= 0:
            return
        if self._resolved_task_timeout == timeout:
            return
        self._resolved_task_timeout = timeout
        self.agent.time_budget = timeout
        if self.event_bus is not None:
            self.event_bus.emit(
                "task_timeout_resolved",
                agent="main_agent",
                payload={
                    "profile": self.profile.name(),
                    "timeout_seconds": timeout,
                },
            )

    def _resolve_profile_task_metadata(self, user_prompt: str) -> dict:
        resolver = getattr(self.profile, "resolve_task_metadata", None)
        if resolver is None:
            return {}
        try:
            metadata = resolver(user_prompt)
        except Exception as exc:
            log.debug("Failed to resolve task metadata for profile %s: %s", self.profile.name(), exc)
            return {}
        if not isinstance(metadata, dict):
            return {}
        resolved = dict(metadata)
        resolved["permission_mode"] = self.permission_mode
        return resolved

    def format_task(self, user_prompt: str) -> str:
        return f"Task:\n{user_prompt}"

    def prepare_submission(self, submission: TurnSubmission | str) -> PreparedTurn:
        if isinstance(submission, str):
            submission = TurnSubmission(submission)
        if self.attachment_manager is None:
            self.ensure_profile_bound_for_first_task(submission.text)
        if self.attachment_manager is None:
            raise RuntimeError("附件模块尚未准备好，请稍候")
        return replace(self.attachment_manager.prepare(submission), session_id=self.session.id)

    def stage_attachment_path(self, path: str | Path, *, source: str = "picker") -> Attachment:
        if self.attachment_manager is None:
            self.ensure_profile_bound_for_first_task("")
        if self.attachment_manager is None:
            raise RuntimeError("附件模块尚未准备好，请稍候")
        return self.attachment_manager.stage_path(
            path,
            source=source,  # type: ignore[arg-type]
            copy_to_session=None,
        )

    def stage_attachment_bytes(
        self,
        data: bytes,
        *,
        name: str,
        mime_type: str,
        source: str = "clipboard",
    ) -> Attachment:
        if self.attachment_manager is None:
            self.ensure_profile_bound_for_first_task("")
        if self.attachment_manager is None:
            raise RuntimeError("附件模块尚未准备好，请稍候")
        return self.attachment_manager.stage_bytes(
            data,
            name=name,
            mime_type=mime_type,
            source=source,  # type: ignore[arg-type]
        )

    def remove_attachment(self, attachment_id: str) -> bool:
        return bool(self.attachment_manager and self.attachment_manager.remove(attachment_id))

    def submit(self, user_prompt: str | TurnSubmission, cancellation_token=None) -> TurnResult:
        submission = user_prompt if isinstance(user_prompt, TurnSubmission) else TurnSubmission(user_prompt)
        prepared = self.prepare_with_external_approval(submission)
        return self.submit_prepared(prepared, cancellation_token=cancellation_token)

    def prepare_with_external_approval(self, submission: TurnSubmission) -> PreparedTurn:
        """Prepare a turn, routing external-path reads through the approval surface."""
        try:
            return self.prepare_submission(submission)
        except ExternalPathConfirmationRequired as exc:
            approval = self.approval_provider.request(ApprovalRequest(
                tool_name="read_external_attachment",
                args={"paths": exc.paths},
                risk="读取工作区外文件并复制到当前会话附件缓存",
                reason="用户提示词中引用了工作区外的文件",
                agent_name="main_agent",
                session_id=getattr(self.session, "id", None),
            ))
            if not approval.approved:
                raise AttachmentError("已拒绝读取工作区外文件，本次任务未提交")
            return self.prepare_submission(replace(
                submission,
                authorized_paths=tuple(exc.paths),
            ))

    @contextmanager
    def _turn_scope(self, cancellation_token=None):
        token = cancellation_token or current_token.get() or CancellationToken()
        if self._turn_thread == threading.get_ident():
            token.check()
            yield token
            return
        with self._execution_condition:
            if self._closed or self._closing or self._turn_thread is not None:
                raise RuntimeError("会话正在执行或关闭")
            if self.tool_context.tool_tasks.pending_foreground_count:
                raise RuntimeError("前台工具尚未退出，暂时无法启动新回合")
            self._turn_thread = threading.get_ident()
            self._turn_token = token
        try:
            token.check()
            self.turn_count += 1
            started = time.monotonic()
            status, reason = "completed", None
            try:
                self.recovery.begin_turn(
                    session_id=self.session.id, turn=self.turn_count - 1,
                    journal_sequence=self.conversation.journal.sequence,
                    event_count=len(self.session_store.read_events(self.session.id)),
                    profile=self.profile.name(), permission_mode=self.permission_mode,
                    routing_mode=self.routing_mode,
                    model_selection=dict(model=self.model_selection.model, reasoning_effort=self.model_selection.reasoning_effort),
                )
                self._announce_recovery_origin()
                self.event_bus.emit("turn_started", agent="main_agent", payload={"turn": self.turn_count})
                with cancellation_scope(token):
                    token.check()
                    yield token
                    token.check()
            except BaseException as exc:
                status = "cancelled" if isinstance(exc, CancelledError) else "failed"
                reason = str(exc)
                raise
            finally:
                self.event_bus.emit("turn_finished", agent="main_agent", payload={
                    "turn": self.turn_count, "status": status, "reason": reason,
                    "duration_seconds": time.monotonic() - started,
                    "pending_foreground_tools": self.tool_context.tool_tasks.pending_count,
                })
                if status == "completed":
                    self._finish_recovery_turn()
                metadata = self.session_store.read_metadata(self.session.id)
                metadata["title"] = metadata.get("title") or (self.last_user_task.splitlines()[0][:48] if self.last_user_task else "新会话")
                metadata["history_boundary"] = {
                    "turn": self.turn_count, "journal_sequence": self.conversation.journal.sequence,
                    "event_count": len(self.session_store.read_events(self.session.id)),
                    "change_cursor": self.tool_context.workspace.access.cursor(),
                }
                atomic_json(self.session.metadata_path, metadata)
        finally:
            with self._execution_condition:
                self._turn_thread = None
                self._turn_token = None
                self._execution_condition.notify_all()

    def submit_prepared(self, prepared: PreparedTurn, cancellation_token=None) -> TurnResult:
        with self._turn_scope(cancellation_token) as token:
            return self._submit_prepared(prepared, token)

    def _submit_prepared(self, prepared: PreparedTurn, cancellation_token) -> TurnResult:
        if prepared.session_id != self.session.id:
            raise ValueError("待执行消息不属于当前会话")
        user_prompt = prepared.text
        self._sync_mcp_connections()
        self._refresh_skill_catalog()
        self.ensure_profile_bound_for_first_task(user_prompt)
        cancellation_token.check()
        skill_invocation = self.skill_registry.build_user_invocation(user_prompt)
        if skill_invocation is not None:
            result = self._run_turn(
                user_prompt,
                cancellation_token=cancellation_token,
                turn_instruction=skill_invocation.prompt,
                attachments=prepared.attachments,
            )
        elif self.pending_plan_markdown and self.profile.name() == "plan":
            # Text entry after deferring the question still works: an
            # explicit confirmation executes; anything else is revision
            # feedback, which produces a fresh ready plan -> ask again.
            if _is_plan_execution_confirmation(user_prompt):
                result = self.execute_pending_plan(attachments=prepared.attachments)
            else:
                result = self.revise_pending_plan(user_prompt, attachments=prepared.attachments)
        else:
            route_decision = self._maybe_auto_route_profile(user_prompt)
            turn_instruction = _turn_instruction_for_route(route_decision)
            result = self._run_turn(
                user_prompt,
                cancellation_token=cancellation_token,
                turn_instruction=turn_instruction,
                attachments=prepared.attachments,
            )
        # Explicit plan mode is the one case where runtime cannot continue
        # without a user decision. Route it through the single interaction
        # surface (question), never through Workbench buttons.
        return self._settle_ready_plan(result)

    def ensure_profile_bound_for_first_task(self, user_prompt: str) -> None:
        if self.is_bound:
            return
        self._bind_profile(self._pending_profile_name, source=self._profile_source)

    def _refresh_skill_catalog(self) -> None:
        """Rebuild the prompt and Slash registry when the catalog changed on disk."""
        if not self.skill_registry.refresh_if_changed():
            return
        self._slash_registry = None
        if self.is_bound and self.profile is not None:
            self._activate_profile(self.profile.name())

    def _maybe_auto_route_profile(self, user_prompt: str) -> RouteDecision | None:
        if (
            not self.is_bound
            or self.event_bus is None
            or self.routing_mode != ROUTING_MODE_AUTO
        ):
            return None
        current = self.profile.name()
        future = self._startup_executor.submit(route_profile_for_turn,
            user_prompt,
            current_profile=current,
            routing_mode=self.routing_mode,
            previous_user_task=self.last_user_task,
            previous_assistant_text=self.last_assistant_text,
            llm_classifier=self.profile_router,
        )
        self.tool_context.tool_tasks.track(future)
        decision = self._wait_session_future(future)
        switched = decision.action == ROUTE_ACTION_SWITCH_PROFILE and decision.profile_name != current
        self.event_bus.emit(
            "profile_route_decision",
            agent="main_agent",
            payload={
                "profile": decision.profile_name,
                "current_profile": current,
                "matched_profile": decision.matched_profile,
                "action": decision.action,
                "turn_mode": decision.turn_mode,
                "confidence": decision.confidence,
                "margin": decision.margin,
                "reason": decision.reason,
                "fallback_used": decision.fallback_used,
                "fallback_reason": decision.fallback_reason,
                "elapsed_ms": round(float(getattr(decision, "elapsed_ms", 0.0)), 1),
                "source": decision.source,
                "switched": switched,
                "routing_mode": self.routing_mode,
                "decisive_signal": getattr(decision, "decisive_signal", ""),
                "probabilities": decision.probabilities,
                "llm_called": getattr(decision, "llm_called", False),
                "llm_confidence": getattr(decision, "llm_confidence", 0.0),
                "llm_provider": getattr(decision, "llm_provider", ""),
                "llm_model": getattr(decision, "llm_model", ""),
                "failure_type": getattr(decision, "failure_type", ""),
            },
        )
        if switched:
            self._switch_profile(decision.profile_name, reason="auto route")
        return decision

    def _bind_profile(
        self,
        profile_name: str,
        *,
        source: str,
    ) -> None:
        if self.is_bound:
            return
        self.profile = get_profile(profile_name)
        self._pending_profile_name = self.profile.name()
        self._profile_source = source
        if self._requested_session_id and (self.session_store.sessions_dir / self._requested_session_id / "session.json").exists():
            self.session = self.session_store.load(self._requested_session_id)
        else:
            self.session = self.session_store.create(
                profile=self.profile.name(),
                cwd=self.cwd,
                model=config.MODEL,
                permission_mode=self.permission_mode,
                profile_source=source,
                session_id=self._requested_session_id,
            )
        self.attachment_manager = AttachmentManager(
            self.cwd,
            self.session.root,
            model_input_mode(),
        )
        metadata = self.session_store.read_metadata(self.session.id)
        metadata["routing_mode"] = self.routing_mode
        metadata["environment"] = self.environment
        self.session.metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.event_bus = self.session_store.event_bus(self.session, listener=self.event_listener)
        self._prepare_tool_registry()
        self.tool_context = ToolContext(
            workspace=WorkspaceService(
                root=self.cwd,
                snapshots_dir=self.session.snapshots_dir,
                session_id=self.session.id,
            ),
            permission_policy=self._effective_permission_policy(),
            event_bus=self.event_bus,
            session_id=self.session.id,
            session_root=self.session.root,
            model_selection=self.model_selection,
            memory_use_enabled=self.memory_use_enabled,
            memory_auto_extract_enabled=self.memory_auto_extract_enabled,
            memory_service=self.memory_service,
            approval_provider=self.approval_provider,
            question_provider=self.question_provider,
            tool_registry=self.tool_registry,
        )
        self.lifecycle.register(
            "tool_tasks",
            lambda: self.tool_context.tool_tasks.close(timeout=1.0),
            order=20,
        )
        self.tool_context.agent_coordinator = AgentCoordinator(
            self.tool_context,
            parent_messages=lambda: (
                list(self.conversation.messages) if self.conversation is not None else []
            ),
            parent_system_prompt=lambda: self.conversation.agent.system_prompt if self.conversation is not None else "",
            parent_message_sink=self._queue_parent_message,
        )
        self.lifecycle.register(
            "agent_coordinator",
            lambda: self.tool_context.agent_coordinator.close(),
            order=30,
        )
        self._activate_profile(self.profile.name())
        self.event_bus.emit(
            "session_started",
            agent="main_agent",
            payload={
                "session_id": self.session.id,
                "profile": self.profile.name(),
                "profile_source": source,
                "workspace": str(self.cwd),
                "interactive": True,
            },
        )

    def _effective_permission_policy(self) -> PermissionPolicy:
        """Session permission mode tightened by the active profile preset."""
        preset = self.profile.main_agent().permission_preset
        return PermissionPolicy(mode=self.permission_mode).restricted_by(preset)

    def _queue_parent_message(self, text: str, tag: str) -> None:
        # Child -> Main upward channel. The message is queued on the main
        # conversation and injected at the next safe boundary of its loop.
        conversation = getattr(self, "conversation", None)
        if conversation is None:
            raise RuntimeError("main conversation is unavailable")
        conversation.queue_message(text, tag=tag)

    def _activate_profile(self, profile_name: str) -> bool:
        profile = get_profile(profile_name)
        agent = self._build_agent(profile)
        created = self.conversation is None
        if created:
            self.conversation = agent.start_conversation()
            self.lifecycle.register("conversation", lambda: self.conversation.close(), order=40)
        else:
            self.conversation.rebind_agent(agent)
        self.profile, self.agent = profile, agent
        self._refresh_mcp_catalog()
        self.tool_context.permission_policy = self._effective_permission_policy()
        return created

    def _run_turn(self, user_prompt: str, cancellation_token=None,
                  turn_instruction: str | None = None, attachments: tuple[Attachment, ...] = ()) -> TurnResult:
        with self._turn_scope(cancellation_token) as token:
            return self._run_tracked_turn(user_prompt, token, turn_instruction, attachments)

    def _run_tracked_turn(
        self,
        user_prompt: str,
        cancellation_token=None,
        turn_instruction: str | None = None,
        attachments: tuple[Attachment, ...] = (),
    ) -> TurnResult:
        cancellation_token = cancellation_token or current_token.get()
        if cancellation_token is not None:
            cancellation_token.check()
        session_id = self.session.id
        tracking = self.memory_auto_extract_enabled
        completed = False
        if tracking:
            try:
                self.memory_service.extraction.register(session_id, self.session.journal_path)
                begin_memory_turn(self.cwd, session_id)
            except (OSError, ValueError, sqlite3.Error) as exc:
                log.debug("Failed to defer memory extraction: %s", exc)
        try:
            result = self._run_turn_impl(user_prompt, cancellation_token, turn_instruction, attachments)
            completed = True
            return result
        finally:
            if tracking:
                try:
                    journal = self.session.journal_path if completed and self.memory_auto_extract_enabled else None
                    finish_memory_turn(self.cwd, session_id, journal)
                except (OSError, ValueError, sqlite3.Error) as exc:
                    log.debug("Failed to schedule memory extraction: %s", exc)

    def _run_turn_impl(
        self,
        user_prompt: str,
        cancellation_token=None,
        turn_instruction: str | None = None,
        attachments: tuple[Attachment, ...] = (),
    ) -> TurnResult:
        turn_started_at = time.time()
        self._apply_profile_task_timeout(user_prompt)
        resolved = resolve_mentions(
            user_prompt,
            workspace_root=self.cwd,
            session_store=self.session_store,
        )
        resolved = [item for item in resolved if item.kind != "file"]
        prompt_for_model = self._externalize_large_turn_text(
            "prompt",
            user_prompt,
            intro="The full user prompt was too large to inline.",
        )
        resolved_for_model = self._externalize_large_mentions(resolved)
        prompt_with_mentions = _format_turn_with_mentions(prompt_for_model, resolved_for_model)
        if turn_instruction:
            turn_instruction = self._externalize_large_turn_text(
                "turn-instruction",
                turn_instruction,
                intro="The turn instruction was too large to inline.",
            )
            prompt_with_mentions = f"{turn_instruction}\n\nUser request:\n{prompt_with_mentions}"
        task = self.format_task(prompt_with_mentions)
        model_content = build_model_content(
            task,
            attachments,
            text_transform=lambda attachment, content: self._externalize_large_turn_text(
                f"attachment-{attachment.id}",
                content,
                intro=f"The full contents of attachment {attachment.name!r} were too large to inline.",
            ),
        )
        attachment_metadata = [item.public_dict() for item in attachments]
        self.event_bus.emit_event(
            UserInputEvent(
                text=user_prompt,
                turn=self.turn_count,
                mentions=[item.raw for item in resolved],
                attachments=attachment_metadata,
            ).to_event()
        )
        # MCP cold start follows turn_started so the UI shows the assistant
        # immediately while tools are loading.
        self._ensure_mcp_tools_loaded()
        if cancellation_token is not None:
            cancellation_token.check()
        turn_event_start = len(getattr(self.event_bus, "events", []))
        source = self.conversation.journal.append_user_input(
            user_prompt, turn_id=f"{self.session.id}:{self.turn_count}",
            memory_generation=self.memory_service.extraction.generation,
            auto_extract=self.memory_auto_extract_enabled)
        self.tool_context.memory_journal_path = str(self.session.journal_path)
        self.tool_context.memory_user_sequence = source.sequence
        text = self.conversation.submit(
            model_content,
            task_text=user_prompt,
            cancellation_token=cancellation_token,
        )
        if self.memory_auto_extract_enabled:
            verification = getattr(self.conversation.runtime_state, "verification", None)
            if (verification and verification.get("turn_id") == self.conversation.runtime_state.turn_id
                    and not getattr(self.conversation.runtime_state, "verification_stale", True)):
                checks = [c for c in verification.get("checks", []) if c.get("status") in {"passed", "failed"}]
                if checks:
                    self.conversation.journal.append_verification(
                        summary=json.dumps(checks, ensure_ascii=False),
                        outcome="failed" if any(c["status"] == "failed" for c in checks) else "passed",
                        tool_call_id=f"native-checks:{self.session.id}:{self.turn_count}",
                        memory_generation=self.memory_service.extraction.generation,
                        files=[{"path": path, "fingerprint": fingerprint}
                               for path, fingerprint in verification.get("fingerprints", {}).items()
                               if path in verification.get("files", [])],
                    )
        if cancellation_token is not None and cancellation_token.is_cancelled:
            from ..agent.cancellation import CancelledError
            raise CancelledError("Turn cancelled by user")
        streamed = bool(getattr(self.conversation, "last_run_streamed_text", False))
        self.event_bus.emit_event(
            AssistantMessageEvent(
                text=text,
                turn=self.turn_count,
                streamed=streamed,
            ).to_event()
        )
        self.last_user_task = user_prompt
        self.last_assistant_text = text
        notice = self._capture_plan_handoff(text)
        duration_seconds = time.time() - turn_started_at
        self._maybe_emit_turn_summary(
            user_prompt=user_prompt,
            assistant_text=text,
            turn_event_start=turn_event_start,
            duration_seconds=duration_seconds,
        )
        return TurnResult(text=text, notice=notice, streamed=streamed)

    def _finish_recovery_turn(self) -> None:
        if self.tool_context.tool_tasks.pending_foreground_count:
            self.event_bus.emit("recovery_point_skipped", agent="main_agent",
                                payload={"turn": self.turn_count, "reason": "active_workspace_tasks"})
            return
        point = self.recovery.finish_turn(
            session_id=self.session.id,
            turn=self.turn_count,
            journal_sequence=self.conversation.journal.sequence,
            event_count=len(self.session_store.read_events(self.session.id)),
        )
        metadata = self.session_store.read_metadata(self.session.id)
        point.update(profile=self.profile.name(), permission_mode=self.permission_mode,
                     routing_mode=self.routing_mode,
                     model_selection=dict(model=self.model_selection.model, reasoning_effort=self.model_selection.reasoning_effort))
        atomic_json(self.recovery.store.root / "points" / f"{point['id']}.json", point)
        metadata["checkpoint"] = point
        atomic_json(self.session.metadata_path, metadata)
        self._announce_recovery_origin()
        self.event_bus.emit("recovery_point_created", agent="main_agent", payload={"turn": self.turn_count, "point_id": point["id"]})

    def _maybe_emit_turn_summary(
        self,
        *,
        user_prompt: str,
        assistant_text: str,
        turn_event_start: int,
        duration_seconds: float,
    ) -> None:
        if self.event_bus is None or self.event_listener is None:
            return
        if not self.enable_turn_summary:
            return
        events = [
            event.to_dict() if hasattr(event, "to_dict") else dict(event)
            for event in getattr(self.event_bus, "events", [])[turn_event_start:]
        ]
        if not should_summarize_turn(
            events,
            profile_name=self.profile.name(),
            duration_seconds=duration_seconds,
        ):
            return
        summary = generate_turn_summary(
            events,
            user_prompt=user_prompt,
            assistant_text=assistant_text,
        )
        self.event_bus.emit_event(
            TurnSummaryEvent(
                turn=self.turn_count,
                summary=summary.summary,
                duration_seconds=duration_seconds,
                tool_counts=summary.tool_counts,
                changed_files=summary.changed_files,
                generated_by=summary.generated_by,
            ).to_event()
        )

    def execute_pending_plan(self, *, attachments: tuple[Attachment, ...] = ()) -> TurnResult:
        if not self.pending_plan_markdown:
            raise ValueError("No pending plan to execute. Switch to /plan and create a plan first.")
        plan_markdown = self.pending_plan_markdown
        self.pending_plan_markdown = None
        self.pending_plan_revision = 0
        self._switch_profile(
            "coding-agent",
            reason="execute approved plan",
        )
        task = (
            "Execute the approved implementation plan below in coding-agent mode.\n\n"
            "Use the plan as the source of truth, but still inspect the repository, "
            "make the smallest appropriate code/test changes, and run verification before stopping.\n\n"
            f"Approved plan:\n{plan_markdown}"
        )
        return self._run_turn(task, attachments=attachments)

    def revise_pending_plan(
        self,
        feedback: str,
        *,
        attachments: tuple[Attachment, ...] = (),
    ) -> TurnResult:
        if not self.pending_plan_markdown:
            raise ValueError("No pending plan to revise. Switch to /plan and create a plan first.")
        feedback = feedback.strip()
        if not feedback and attachments:
            feedback = "Revise the plan using the attached files as additional feedback and context."
        elif not feedback:
            raise ValueError("Provide feedback for the pending plan, or say 'continue' to execute it.")
        return self._run_turn(
            "Revise the previous Markdown plan using this user feedback. "
            "Return the complete updated plan in the required structured Markdown format.\n\n"
            f"User feedback:\n{feedback}",
            attachments=attachments,
        )

    def _settle_ready_plan(self, first_result: TurnResult) -> TurnResult:
        """Ask the plan-mode decision question until execute/defer.

        Runs after the planning turn has closed, so execute/revise start
        fresh turns. "revise" loops: the revised plan asks again. Defer (or
        a cancelled/non-interactive question) keeps the pending plan, which
        the user can later execute with continue/继续 or revise by typing
        feedback in the composer.
        """
        result = first_result
        while self.pending_plan_markdown and self.profile.name() == "plan":
            answer = self.question_provider.ask(QuestionRequest(
                question="计划已经准备好，下一步？",
                options=[
                    QuestionOption(
                        label="执行计划",
                        value="execute",
                        description="切换到 coding-agent 并执行该计划",
                    ),
                    QuestionOption(
                        label="修改计划",
                        value="revise",
                        description="在输入框中说明要调整的内容",
                        is_other=True,
                    ),
                    QuestionOption(
                        label="暂时不执行",
                        value="defer",
                        description="保留计划，稍后可继续",
                    ),
                ],
                agent_name=self.agent.name if self.agent else None,
                session_id=self.session_id,
            ))
            if answer.cancelled or answer.value == "defer":
                return result
            if answer.value == "execute":
                return self.execute_pending_plan()
            if answer.value == "revise":
                feedback = (answer.custom_text or "").strip()
                if not feedback:
                    return result
                result = self.revise_pending_plan(feedback)
                continue
            return result
        return result

    def _capture_plan_handoff(self, text: str) -> str:
        if self.profile.name() != "plan" or not text.strip():
            return ""
        self.pending_plan_markdown = text.strip()
        self.pending_plan_revision += 1
        plan_path = self._write_pending_plan_artifact(self.pending_plan_markdown)
        self.event_bus.emit(
            "plan_ready",
            agent="main_agent",
            payload={
                "profile": self.profile.name(),
                "plan_path": str(plan_path.relative_to(self.cwd)),
                "plan_revision": self.pending_plan_revision,
                "plan_markdown": self.pending_plan_markdown,
                "approval_source": "/plan",
            },
        )
        return (
            f"计划已写入 `{plan_path.relative_to(self.cwd)}`，请选择下一步。"
            "非交互入口可回复 continue/继续 执行，或直接发送修改意见。"
        )

    def _write_pending_plan_artifact(self, plan_markdown: str) -> Path:
        plan_path = self.cwd / ".harness" / "sessions" / self.session.id / "plan.md"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(plan_markdown.rstrip() + "\n", encoding="utf-8")
        return plan_path

    @runtime_change
    def _switch_profile(
        self,
        profile_name: str,
        *,
        reason: str = "profile picker",
    ) -> None:
        previous = self.profile.name()
        if previous == profile_name:
            return
        if not self.is_bound:
            self.profile = get_profile(profile_name)
            self._pending_profile_name = self.profile.name()
            self._profile_source = "explicit"
            return
        self._activate_profile(
            profile_name,
        )
        self.profile_history.append(ProfileSwitchEvent(
            previous=previous,
            current=self.profile.name(),
            reason=reason,
        ))
        if self.session is not None:
            self.session_store.update_profile(
                self.session.id,
                self.profile.name(),
                profile_source=reason,
            )
        self.event_bus.emit(
            "profile_switched",
            agent="main_agent",
            payload={
                "previous_profile": previous,
                "profile": self.profile.name(),
                "reason": reason,
            },
        )

    def handle_slash_command(self, line: str) -> bool:
        from ..tui.commands import default_command_registry

        if self._slash_registry is None:
            self._slash_registry = default_command_registry(skill_registry=self.skill_registry)
        result = self._slash_registry.execute(line, self)
        self.last_command_result = result
        if result.text:
            self.output_sink(result.text)
        return result.should_continue

    def context_status(self) -> str:
        if self.conversation is None or self.agent is None:
            return "当前还没有活动会话"
        from ..agent.context_manager import ContextManager

        breakdown = ContextManager.breakdown(
            self.conversation.messages,
            self.agent.tool_schemas if self.agent.use_tools else None,
        )
        labels = {"system": "系统指令", "summary": "工作摘要", "recent": "近期消息", "memory": "长期记忆", "tools": "工具定义"}
        lines = [f"上下文估算：{sum(breakdown.values())} tokens"]
        lines.extend(f"- {labels[key]}：{value}" for key, value in breakdown.items())
        return "\n".join(lines)

    def memory_command(self, args: list[str]) -> str:
        service = self.memory_service
        if not args or args == ["status"]:
            counts = {scope: len(store.records()) for scope, store in service.stores.items()}
            return (
                f"memory use: {'on' if self.memory_use_enabled else 'off'}; "
                f"generate: {'on' if self.memory_auto_extract_enabled else 'off'}; "
                f"project={counts['project']}; user={counts['user']}"
            )
        if len(args) == 2 and args[0] in {"use", "generate"} and args[1] in {"on", "off"}:
            enabled = args[1] == "on"
            if args[0] == "use":
                self.memory_use_enabled = enabled
                if self.tool_context is not None:
                    self.tool_context.memory_use_enabled = enabled
            else:
                self.memory_auto_extract_enabled = enabled
                if self.tool_context is not None:
                    self.tool_context.memory_auto_extract_enabled = enabled
                if enabled:
                    service.extraction.resume(self.session.id)
                    open_memory_session(self.cwd, self.session.id, self.session.journal_path)
                elif self.session is not None:
                    service.extraction.cancel(self.session.id)
            suffix = "；启动索引将在下次新会话刷新" if args[0] == "use" else ""
            return f"memory {args[0]}: {args[1]}{suffix}"
        if args[0] == "topics":
            return json.dumps(service.topics(args[1] if len(args) > 1 else "project"), ensure_ascii=False, indent=2)
        if args[0] == "history" and len(args) == 2:
            scope, memory_id = _parse_memory_ref(args[1])
            return json.dumps(service.read(memory_id, scope, history=True), ensure_ascii=False, indent=2)
        if args[0] == "restore" and len(args) == 4:
            scope, memory_id = _parse_memory_ref(args[1])
            record = service.restore(memory_id, int(args[2]), int(args[3]), scope)
            return f"已恢复 {scope}:{record.id} v{record.version}"
        if args == ["retry"]:
            project = service.stores["project"]
            with project.connect(write=True) as db:
                for table in ("extraction_tasks", "consolidation_tasks"):
                    db.execute(f"UPDATE {table} SET status='pending',failures=0,available_at=0,error_type=NULL "
                               "WHERE namespace=? AND status IN ('failed','blocked')", (project.namespace,))
                db.execute("UPDATE vector_outbox SET failures=0,available_at=0,error_type=NULL WHERE memory_id IN "
                           "(SELECT id FROM memories WHERE namespace IN (?, 'user'))", (project.namespace,))
            start_memory_worker(self.cwd)
            return "已重新排队失败的记忆任务"
        if args[0] == "search" and len(args) > 1:
            return service.format_hits(service.search(" ".join(args[1:]))) or "没有找到相关记忆"
        if args[0] in {"show", "forget", "validate"} and len(args) >= 2:
            scope, memory_id = _parse_memory_ref(args[1])
            if args[0] == "show":
                doc = service.read(memory_id, scope=scope)
                return json.dumps(doc, ensure_ascii=False, indent=2)
            if len(args) != 3:
                raise ValueError(f"用法：/memory {args[0]} [project:|user:]<id> <version>")
            version = int(args[2])
            if args[0] == "forget":
                service.forget(memory_id, version, scope=scope)
                return f"已遗忘 {scope}:{memory_id}，正文已删除"
            doc = service.validate(memory_id, version, self._memory_command_evidence(args), scope=scope)
            return f"已确认 {scope}:{doc.id} v{doc.version}"
        if args[0] == "edit" and len(args) >= 4:
            scope, memory_id = _parse_memory_ref(args[1])
            version = int(args[2])
            from dataclasses import asdict
            current = asdict(service.stores[scope].read(memory_id))
            doc = service.write(MemoryWriteCommand(
                topic=current["topic"], fact_key=current["fact_key"], kind=current["kind"],
                evidence=self._memory_command_evidence(args),
                body=" ".join(args[3:]),
                scope=scope,
                applicability=current["applicability"],
                memory_id=current["id"],
                expected_version=version,
            ))
            return f"已编辑 {scope}:{doc.id} v{doc.version}"
        raise ValueError(
            "用法：/memory [status|topics [scope]|history <id>|restore <id> <revision> <version>|retry|"
            "search <query>|show <id>|edit <id> <version> <body>|forget <id> <version>|"
            "validate <id> <version>|use on|off|generate on|off]"
        )

    def _memory_command_evidence(self, args):
        text = "/memory " + " ".join(args)
        source = self.conversation.journal.append_user_input(
            text, turn_id=f"memory-command:{time.time_ns()}",
            memory_generation=self.memory_service.extraction.generation)
        return [EvidenceReference(self.session.id, source.sequence, "user_input", hashlib.sha256(text.encode()).hexdigest(),
                                  end_offset=len(text), source_namespace=self.memory_service.stores["project"].namespace, observed_at=source.timestamp)]

    def submit_skill_command(self, line: str) -> TurnResult:
        if self.skill_registry.build_user_invocation(line) is None:
            raise ValueError(f"Unknown user skill command: {line}")
        return self.submit(line)

    @runtime_change
    def switch_profile(self, profile_name: str) -> str:
        previous = self.profile.name()
        self.pending_plan_markdown = None
        self.pending_plan_revision = 0
        self.routing_mode = ROUTING_MODE_PINNED
        self._switch_profile(profile_name)
        current = self.profile.name()
        if self.session is not None:
            self.session_store.update_routing_mode(self.session.id, self.routing_mode)
        if current == previous:
            return f"profile already active: {current}"
        if not self.is_bound:
            return f"profile selected: {current}"
        return f"profile switched: {previous} -> {current}"

    @runtime_change
    def enable_auto_profile_routing(self) -> str:
        previous_mode = self.routing_mode
        self.routing_mode = ROUTING_MODE_AUTO
        if self.session is not None:
            self.session_store.update_routing_mode(self.session.id, self.routing_mode)
        if previous_mode == ROUTING_MODE_AUTO:
            return f"auto profile routing already active: {self.profile.name()}"
        return f"auto profile routing enabled: {self.profile.name()}"

    @runtime_change
    def set_permission_mode(self, permission_mode: str) -> str:
        PermissionPolicy(mode=permission_mode)
        previous = self.permission_mode
        if previous == permission_mode:
            return f"permission mode already active: {permission_mode}"
        self.permission_mode = permission_mode
        self.approval_provider = self._approval_provider_for_mode(permission_mode)
        if self.tool_context is not None:
            self.tool_context.permission_policy = self._effective_permission_policy()
            self.tool_context.approval_provider = self.approval_provider
        if self.conversation is not None and getattr(self.conversation, "runtime_state", None) is not None:
            self.conversation.runtime_state.permission_mode = permission_mode
        if self.session is not None:
            self.session_store.update_permission_mode(self.session.id, permission_mode)
        if self.event_bus is not None:
            self.event_bus.emit(
                "permission_mode_switched",
                agent="main_agent",
                payload={
                    "previous_permission_mode": previous,
                    "permission_mode": permission_mode,
                },
            )
        return f"permission mode switched: {previous} -> {permission_mode}"

    @runtime_change
    def apply_model_override(
        self,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        """Apply a TUI model/effort selection and drop the cached prompt key.

        The cache key embeds the runtime-resolved model, so a model switch
        must invalidate the conversation-level key; effort-only changes also
        invalidate, and the key is simply rebuilt identically next turn.
        """
        self.model_selection.select(model=model, reasoning_effort=reasoning_effort)
        metadata = self.session_store.read_metadata(self.session.id)
        metadata["model_selection"] = dict(model=self.model_selection.model, reasoning_effort=self.model_selection.reasoning_effort)
        metadata["model"] = config.resolve_model_profile(config.MODEL_INTENSITY, self.model_selection).model
        atomic_json(self.session.metadata_path, metadata)
        if self.conversation is not None:
            self.conversation._cached_prompt_cache_key = None

    def toggle_permission_mode(self) -> str:
        modes = [
            PermissionPolicy.READ_ONLY,
            PermissionPolicy.WORKSPACE_WRITE,
            PermissionPolicy.LLM_AUTO,
            PermissionPolicy.DANGER_FULL_ACCESS,
        ]
        try:
            index = modes.index(self.permission_mode)
        except ValueError:
            index = 0
        next_mode = modes[(index + 1) % len(modes)]
        return self.set_permission_mode(next_mode)

    def _approval_provider_for_mode(self, permission_mode: str) -> ApprovalProvider:
        if permission_mode == PermissionPolicy.LLM_AUTO:
            return LlmAutoApprovalProvider()
        return self._manual_approval_provider

    def mcp_status(self) -> str:
        self._ensure_mcp_tools_loaded()
        return self.mcp_manager.status_report()

    def mcp_list(self) -> str:
        self._ensure_mcp_tools_loaded()
        return self.mcp_manager.tools_report()

    @runtime_change
    def reload_mcp(self) -> str:
        self._ensure_mcp_tools_loaded()
        self.mcp_manager.close()
        self._prepare_tool_registry()
        self._ensure_mcp_tools_loaded()
        if self.tool_context is not None:
            self.tool_context.tool_registry = self.tool_registry
        self._refresh_agent_tool_schemas()
        self._refresh_mcp_catalog()
        if self.event_bus is not None:
            self.event_bus.emit(
                "mcp_reloaded",
                agent="main_agent",
                payload={"tool_count": len(getattr(self.mcp_manager, "tool_bindings", []))},
            )
        return "MCP reloaded\n" + self.mcp_manager.status_report()

    @runtime_change
    def reload_mcp_server(self, server_name: str) -> str:
        """Reconnect one configured MCP server and refresh active tool schemas."""
        self._ensure_mcp_tools_loaded()
        result = self.mcp_manager.reload_server(server_name)
        self._rebuild_mcp_state()
        return result

    @runtime_change
    def register_mcp_server(
        self,
        *,
        name: str,
        url: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        headers: dict[str, str] | None = None,
        permission: str | None = None,
        title: str = "",
        description: str = "",
    ) -> str:
        """Register one MCP server, connect it, and generate a missing description."""
        self._ensure_mcp_tools_loaded()
        server = build_server_config(
            name=name,
            url=url,
            command=command,
            args=args,
            env=env,
            cwd=cwd,
            headers=headers,
            permission=permission,
            title=title,
            description=description,
        )
        registration = self.mcp_manager.register_server(server)
        self._rebuild_mcp_state()
        verb = "已更新" if registration.replaced else "已注册"
        if registration.state == "connected":
            lines = [f"MCP 服务{verb}：{name}（{registration.tool_count} 个工具，权限 {server.permission}）"]
            if not registration.server.description:
                try:
                    lines.append(self.mcp_manager.generate_server_description(name))
                except McpConfigError as exc:
                    lines.append(str(exc))
                else:
                    self._refresh_mcp_catalog()
            return "\n".join(lines)
        lines = [
            f"MCP 服务{verb}但连接失败：{name}",
            registration.error or "未知错误",
            "记录已保存；修复配置后可重新连接并生成描述",
        ]
        return "\n".join(lines)

    @runtime_change
    def remove_mcp_server(self, server_name: str) -> str:
        """Remove one registered MCP server and refresh active tool schemas."""
        self._ensure_mcp_tools_loaded()
        result = self.mcp_manager.remove_server(server_name)
        self._rebuild_mcp_state()
        return result

    @runtime_change
    def refresh_mcp_server_description(self, server_name: str, *, refresh: bool = True) -> str:
        """Regenerate one server description from its current tool catalog."""
        self._ensure_mcp_tools_loaded()
        result = self.mcp_manager.generate_server_description(server_name, refresh=refresh)
        self._refresh_mcp_catalog()
        return result

    def _rebuild_mcp_state(self) -> None:
        if self.tool_registry is not None and self.mcp_manager is not None:
            snapshot = self.mcp_manager.catalog_snapshot()
            self._mcp_connection_revision = snapshot.revision
            self.tool_registry = BUILTIN_TOOL_REGISTRY.copy()
            self.mcp_manager.register_tools(self.tool_registry, snapshot=snapshot)
        if self.tool_context is not None:
            self.tool_context.tool_registry = self.tool_registry
            # Discovery history survives profile switches, but a tool that no
            # longer exists in the rebuilt catalog is dropped: it is gone, not
            # merely disallowed by the current profile.
            existing = {spec.name for spec in self.tool_registry.specs()}
            removed = self.tool_context.discovered_tool_names - existing
            self.tool_context.discovered_tool_names &= existing
            self.tool_context.revealed_tool_names &= self.tool_context.discovered_tool_names
            if removed:
                # Results from the old catalog can no longer justify skipping
                # a repeat query; a reconnect with the same names must allow
                # rediscovery.
                self.tool_context.search_tools_queries.clear()
                self.tool_context.search_tools_no_progress = 0
        self._refresh_agent_tool_schemas()
        self._refresh_mcp_catalog()

    def _sync_mcp_connections(self) -> None:
        """Refresh registry and schemas when a worker connection changed.

        Workers exit outside the user-visible commands; this runs only at the
        safe boundary before a turn, never from the worker thread.
        """
        manager = self.mcp_manager
        if manager is None:
            return
        if manager.connection_revision == self._mcp_connection_revision:
            return
        self._rebuild_mcp_state()

    @runtime_change
    def toggle_mcp_server(self, server_name: str) -> str:
        """Toggle one MCP config entry and rebuild the active manager."""
        self._ensure_mcp_tools_loaded()
        names = set(self.mcp_manager.configured_server_names())
        if server_name not in names:
            return f"MCP 服务不存在：{server_name}"
        enabled = server_name not in self.mcp_manager.config.servers
        result = self.mcp_manager.set_server_enabled(server_name, enabled)
        refreshed = self.reload_mcp()
        return f"{result}\n{refreshed}"

    @runtime_change
    def compact_current_context(self) -> str:
        """Run explicit compaction against the single shared conversation."""
        self._ensure_mcp_tools_loaded()
        if self.conversation is None:
            return "当前还没有可压缩的对话"
        return self.conversation.compact_now()

    @runtime_change
    def _restore_history(self) -> None:
        session_id = self.session.id
        metadata = self.session_store.read_metadata(session_id)
        point = metadata.get("history_boundary") or metadata.get("checkpoint")
        sequence = point["journal_sequence"] if point else 0
        events = self.session_store.read_history_events(session_id, event_count=point["event_count"] if point else 0)
        messages = SessionJournal(self.session.journal_path).recovery_messages(
            self.agent.full_system_prompt, through_sequence=sequence)
        self.conversation._replace_messages(messages, reason="resume")
        self.turn_count = point["turn"] if point else 0
        self.last_user_task = next((e["payload"].get("text", "") for e in reversed(events) if e["type"] == "user_input"), "")
        self.last_assistant_text = next((e["payload"].get("text", "") for e in reversed(events) if e["type"] == "assistant_message"), "")
        self._restore_pending_plan(events)
        todo = self._restore_todo(self.conversation.runtime_state, events)
        if todo is not None:
            atomic_json(self.session.root / "todo" / "state.json", todo)
        self.conversation.last_text = self.last_assistant_text
        self.recovery.origin = next((p for p in self.recovery_points().values() if p["turn"] == 0), None)
        self.event_bus.emit("session_resumed", payload={"session_id": session_id})

    @staticmethod
    def _restore_todo(runtime_state, events):
        from ..agent.runtime_state import TodoItem, TodoList
        payload = next(((event.get("payload", {}).get("metadata") or {})["todo_state"]
                        for event in reversed(events) if event["type"] == "tool_result"
                        and event.get("agent") == "main_agent"
                        and "todo_state" in (event.get("payload", {}).get("metadata") or {})), None)
        runtime_state.todo = TodoList(items=[TodoItem(**item) for item in payload["items"]],
            revision=payload.get("revision", 0), updated_at=payload.get("updated_at", ""),
            next_seq=payload.get("next_seq", 0)) if payload is not None else None
        return payload

    def _restore_pending_plan(self, events: list[dict[str, Any]]) -> None:
        """Rebuild in-memory pending plan state from the forked journal.

        The latest ``plan_ready`` snapshot is pending unless execution
        already started afterwards (``profile_switched`` with the execute
        reason). Without this, the Workbench would offer "执行计划" for a
        resumed session while ``execute_pending_plan`` had nothing to run.
        """
        self.pending_plan_markdown = None
        self.pending_plan_revision = 0
        latest: dict[str, Any] | None = None
        executed = False
        for item in events:
            payload = item.get("payload") if isinstance(item, dict) else None
            if not isinstance(payload, dict):
                continue
            event_type = str(item.get("type") or "")
            if event_type == "plan_ready":
                latest = payload
                executed = False
            elif (
                event_type == "profile_switched"
                and str(payload.get("reason") or "") == "execute approved plan"
            ):
                executed = True
        if latest is None or executed:
            return
        markdown = latest.get("plan_markdown")
        if not isinstance(markdown, str) or not markdown.strip():
            raise ValueError("保存的计划缺少正文")
        if markdown.strip():
            self.pending_plan_markdown = markdown.strip()
            revision = latest.get("plan_revision")
            self.pending_plan_revision = int(revision) if isinstance(revision, int) else 1

    def _announce_recovery_origin(self) -> None:
        origin = self.recovery.origin
        metadata = self.session_store.read_metadata(self.session.id)
        if origin is not None and metadata.get("recovery_origin_point") != origin["id"]:
            metadata["recovery_origin_point"] = origin["id"]
            atomic_json(self.session.metadata_path, metadata)
            self.event_bus.emit("recovery_point_created", agent="main_agent", payload={"turn": 0, "point_id": origin["id"]})

    def recovery_points(self) -> dict[str, dict]:
        metadata = self.session_store.read_metadata(self.session.id)
        points = {}
        for key in ("checkpoint", "branch_origin"):
            point = metadata.get(key)
            if point and point["session_id"] == self.session.id:
                points[point["id"]] = point
        for event in self.session_store.read_history_events(self.session.id):
            if event["type"] in {"recovery_point_created", "session_rewound"}:
                point_id = event["payload"]["point_id"]
                if not (self.recovery.store.root / "points" / f"{point_id}.json").is_file():
                    continue
                point = self.recovery.store.read_point(point_id)
                if point["session_id"] == self.session.id:
                    points[point["id"]] = point
        return points

    @runtime_change
    def rewind_to_point(self, point_id: str) -> bool:
        point = self.recovery_points().get(point_id)
        if point is None:
            raise ValueError("回撤点不属于当前会话")
        self.tool_context.workspace.access.check_rewind(self.session.id, point["change_cursor"])
        if self.tool_context.agent_coordinator.has_active_agents():
            raise RuntimeError("请先停止当前会话的子代理，再执行回撤")
        background_note = ("后台 Shell 仍在运行，其直接写入可能与回撤发生冲突。"
                           if self.conversation.runtime_state.shell_job_manager.running_jobs() else "")
        answer = self.question_provider.ask(QuestionRequest(
            question=f"回撤到第 {point['turn']} 轮？仅撤销本会话之后可安全恢复的修改。{background_note}",
            options=[QuestionOption("取消", "cancel"), QuestionOption("确认回撤", "rewind")],
            session_id=self.session.id, agent_name="main_agent"))
        if answer.cancelled or answer.value != "rewind":
            return False
        from ..runtime.recovery import atomic_bytes
        from ..agent.history import ContextHistory
        staged_root = self.state_root / "staging" / ("rewind-" + uuid.uuid4().hex)
        target_profile = get_profile(point["profile"])
        context = replace(self.tool_context, session_root=staged_root,
                          model_selection=config.ModelSelection(**point["model_selection"]),
                          approval_provider=self._approval_provider_for_mode(point["permission_mode"]))
        context.permission_policy = PermissionPolicy(mode=point["permission_mode"]).restricted_by(target_profile.main_agent().permission_preset)
        agent = self._build_agent(target_profile, tool_context=context)
        replacement = agent.start_conversation()
        events = self.session_store.read_history_events(self.session.id, event_count=point["event_count"])
        messages = SessionJournal(self.session.journal_path).recovery_messages(
            agent.full_system_prompt, through_sequence=point["journal_sequence"])
        replacement._replace_messages(messages, reason="rewind_prepared")
        old = self.conversation
        previous_context = self.tool_context
        todo_path = self.session.root / "todo" / "state.json"
        paths = [self.session.journal_path, self.session.events_path, self.session.metadata_path, todo_path]
        saved = {path: path.read_bytes() if path.exists() else None for path in paths}
        todo_payload = self._restore_todo(replacement.runtime_state, events)
        previous_sequence = self.event_bus._sequence
        previous_events_length = len(self.event_bus.events)
        coordinator = AgentCoordinator(context, parent_messages=lambda: self.conversation.messages,
                                       parent_system_prompt=lambda: self.agent.system_prompt,
                                       parent_message_sink=self._queue_parent_message)
        context.agent_coordinator = coordinator
        try:
            with self.event_bus._lock, context.workspace.access.rewind_transaction(self.session.id, point["change_cursor"], participants=paths):
                replacement.journal = SessionJournal(self.session.journal_path)
                replacement.history = ContextHistory()
                replacement._replace_messages(messages, reason="rewind")
                self.event_bus.emit("session_rewound", payload={"point_id": point_id, "turn": point["turn"],
                                   "event_count": point["event_count"], "journal_sequence": replacement.journal.sequence})
                metadata = self.session_store.read_metadata(self.session.id)
                metadata.update(checkpoint=point, profile=target_profile.name(),
                                routing_mode=point["routing_mode"],
                                permission_mode=point["permission_mode"], model_selection=point["model_selection"],
                                history_boundary=dict(turn=point["turn"], journal_sequence=replacement.journal.sequence,
                                                      event_count=len(self.session_store.read_events(self.session.id)),
                                                      change_cursor=point["change_cursor"]))
                atomic_json(self.session.metadata_path, metadata)
                if todo_payload is not None:
                    atomic_json(todo_path, todo_payload)
                else:
                    todo_path.unlink(missing_ok=True)
        except BaseException:
            with self.event_bus._lock:
                for path, data in saved.items():
                    if data is None:
                        path.unlink(missing_ok=True)
                    else:
                        atomic_bytes(path, data)
                self.event_bus._sequence = previous_sequence
                del self.event_bus.events[previous_events_length:]
            replacement.close()
            coordinator.close()
            raise
        context.session_root = self.session.root
        self.conversation, self.agent, self.tool_context = replacement, agent, context
        self.profile = target_profile
        self.routing_mode = point["routing_mode"]
        self.model_selection = context.model_selection
        self.permission_mode = point["permission_mode"]
        self.approval_provider = context.approval_provider
        self.turn_count = point["turn"]
        if point["turn"] == 0:
            self.recovery.origin = point
        self._restore_pending_plan(events)
        self.last_user_task = next((e["payload"].get("text", "") for e in reversed(events) if e["type"] == "user_input"), "")
        self.last_assistant_text = next((e["payload"].get("text", "") for e in reversed(events) if e["type"] == "assistant_message"), "")
        self.conversation.last_text = self.last_assistant_text
        context.workspace._versions.clear()
        replacement.runtime_state.shell_job_manager.close()
        replacement.runtime_state.shell_job_manager = old.runtime_state.shell_job_manager
        replacement.runtime_state.browser_job_id = old.runtime_state.browser_job_id
        old.runtime_state.shell_job_manager = None
        previous_context.agent_coordinator.close()
        old.close()
        return True

    def _externalize_large_turn_text(self, label: str, text: str, *, intro: str) -> str:
        limit = _env_int("HARNESS_TURN_INLINE_CHAR_LIMIT", TURN_INLINE_CHAR_LIMIT)
        if limit <= 0 or len(text) <= limit:
            return text
        path = self._write_turn_input_file(label, text)
        digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        head, tail = _head_tail(text, TURN_EXCERPT_CHARS)
        return (
            "[EXTERNALIZED TURN CONTENT]\n"
            f"{intro}\n"
            f"path: {path}\n"
            f"chars: {len(text)}\n"
            f"sha256: {digest}\n"
            "Use read_file to inspect the full text if needed. Do not assume the omitted middle.\n\n"
            "Head excerpt:\n"
            "```text\n"
            f"{head}\n"
            "```\n\n"
            "Tail excerpt:\n"
            "```text\n"
            f"{tail}\n"
            "```"
        )

    def _externalize_large_mentions(self, resolved: list[ResolvedMention]) -> list[ResolvedMention]:
        limit = _env_int("HARNESS_TURN_INLINE_CHAR_LIMIT", TURN_INLINE_CHAR_LIMIT)
        if limit <= 0:
            return resolved
        updated: list[ResolvedMention] = []
        for idx, item in enumerate(resolved, start=1):
            content = item.content or ""
            if len(content) <= limit:
                updated.append(item)
                continue
            label = f"mention-{idx}"
            replacement = self._externalize_large_turn_text(
                label,
                content,
                intro=f"The resolved content for mention {item.raw} was too large to inline.",
            )
            metadata = dict(item.metadata)
            metadata["externalized_content"] = True
            updated.append(replace(item, content=replacement, metadata=metadata))
        return updated

    def _write_turn_input_file(self, label: str, text: str) -> Path:
        if self.session is None:
            raise RuntimeError("Cannot externalize turn input before the session is bound.")
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in label) or "input"
        inputs_dir = self.cwd / ".harness" / "sessions" / self.session.id / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        path = inputs_dir / f"turn-{self.turn_count + 1:04d}-{safe_label}.txt"
        suffix = 1
        while path.exists():
            path = inputs_dir / f"turn-{self.turn_count + 1:04d}-{safe_label}-{suffix}.txt"
            suffix += 1
        path.write_text(text, encoding="utf-8")
        return path

    def _append_conversation_message(self, message: dict) -> None:
        self.conversation._append_message(message)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            with self._execution_condition:
                self._closing = True
                if self._turn_token is not None:
                    self._turn_token.cancel()
                if not self._execution_condition.wait_for(lambda: self._turn_thread is None, timeout=10):
                    raise RuntimeError("活动回合尚未退出，资源未释放，请稍后重试关闭")
            self.tool_context.tool_tasks.cancel()
            self.tool_context.agent_coordinator.close()
            if not self.tool_context.tool_tasks.wait_idle(timeout=10):
                raise RuntimeError("前台工具尚未退出，资源未释放，请稍后重试关闭")
            self.conversation.runtime_state.shell_job_manager.close()
            self._closed = True
        for error in self.lifecycle.close():
            log.warning("Failed to close session resource %s: %s", error.name, error.error)
        if self.session is None or self.event_bus is None:
            return
        close_memory_session(self.cwd, self.session.id)
        try:
            metadata = self.session_store.read_metadata(self.session.id)
            events = self.session_store.read_events(self.session.id)
            self.event_bus.emit_event(
                FinalReportEvent(
                    **build_final_report(
                        metadata,
                        events,
                        status="closed",
                        reason="user_exit",
                        summary=self.last_assistant_text,
                    )
                ).to_event()
            )
        except Exception as exc:
            log.warning("Failed to write final report for session %s: %s", self.session.id, exc)
        self.event_bus.emit_event(
            SessionFinishedEvent(
                reason="user_exit",
                status="closed",
            ).to_event()
        )
        self.session_store.update_status(self.session.id, "closed")
        try:
            self.session_store.write_summary(self.session.id)
        except Exception as exc:
            log.warning("Failed to write summary for session %s: %s", self.session.id, exc)

def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _head_tail(text: str, chars: int) -> tuple[str, str]:
    if chars <= 0:
        return "", ""
    if len(text) <= chars * 2:
        return text, ""
    return text[:chars], text[-chars:]


def _parse_memory_ref(value: str) -> tuple[str, str]:
    if ":" not in value:
        return "project", value
    scope, memory_id = value.split(":", 1)
    if scope not in {"project", "user"} or not memory_id:
        raise ValueError("记忆引用必须是 <id>、project:<id> 或 user:<id>")
    return scope, memory_id


def _format_turn_with_mentions(
    user_text: str,
    resolved: list[ResolvedMention],
) -> str:
    parts = []
    mention_context = render_mention_context(resolved)
    if mention_context:
        parts.append(mention_context)
    if not parts:
        return user_text
    parts.append(f"User turn:\n{user_text}")
    return "\n\n".join(parts)


def _turn_instruction_for_route(decision: RouteDecision | None) -> str | None:
    if decision is None or decision.turn_mode != TURN_MODE_DIRECT_ANSWER:
        return None
    return DIRECT_ANSWER_TURN_INSTRUCTION


def _is_plan_execution_confirmation(text: str) -> bool:
    normalized = " ".join(text.strip().lower().split())
    return normalized in {
        "continue",
        "go ahead",
        "proceed",
        "execute",
        "implement",
        "start",
        "yes",
        "ok",
        "继续",
        "执行",
        "开始",
        "实施",
        "可以",
        "好",
        "好的",
        "按计划执行",
        "继续执行",
    }


def _load_harness_rules(workspace: Path) -> GlobalRulesDoc | None:
    path = workspace / "HARNESS.md"
    if not path.exists() or not path.is_file():
        return None
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return None
    return GlobalRulesDoc(source=str(path), content=content)

def print_turn_result(result: TurnResult) -> None:
    if result.streamed:
        print()
    elif result.text:
        print(result.text)
    if result.notice:
        print(result.notice)
