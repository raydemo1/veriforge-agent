from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

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
from ..memory import MemoryService, MemoryWriteCommand
from ..memory.background import start_memory_worker
from ..profiles import get_profile
from ..profiles.base import BaseProfile
from ..profiles.router import (
    ROUTE_ACTION_SWITCH_PROFILE,
    ROUTING_MODE_AUTO,
    ROUTING_MODE_PINNED,
    TURN_MODE_DIRECT_ANSWER,
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
from ..runtime.mcp import McpClientManager
from ..runtime.middleware.loader import load_user_middlewares
from ..runtime.middleware.stack import build_main_agent_middlewares
from ..runtime.permissions import PermissionPolicy
from ..runtime.questions import ConsoleQuestionProvider, QuestionProvider
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
class CheckpointConfig:
    auto: bool = False
    every_turns: int = 1


@dataclass
class TurnResult:
    text: str
    checkpoint: str
    notice: str = ""
    streamed: bool = False


@dataclass
class ProfileSwitchEvent:
    previous: str
    current: str
    reason: str


@dataclass
class ProfileRuntime:
    profile: BaseProfile
    agent: object
    conversation: AgentConversation


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
        allow_checkpoint_init_failure: bool = False,
        startup_sink: Callable[[str], None] | None = None,
    ):
        self.cwd = Path(cwd).resolve()
        self.lifecycle = LifecycleScope()
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
        self.checkpoint = CheckpointConfig()
        self._allow_checkpoint_init_failure = allow_checkpoint_init_failure
        self.checkpoint_init_error: str = ""
        self._report_startup("loading skills")
        self.skill_registry = SkillRegistry()
        self._slash_registry = None
        self.last_command_result = None
        self.session_store = SessionStore(self.cwd / ".harness")
        self.session_store.root.mkdir(parents=True, exist_ok=True)
        self.resume_session_id: str | None = None
        self.resume_context: str | None = None
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
        self.agent = None
        self.conversation: AgentConversation | None = None
        self.profile_runtimes: dict[str, ProfileRuntime] = {}
        self._active_profile_name: str | None = None
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
        if self.memory_auto_extract_enabled:
            start_memory_worker(self.cwd)
        self._report_startup("ready")

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

    def _build_agent(self, profile: BaseProfile):
        from ..agent.conversation import Agent

        if self.tool_context is None:
            raise RuntimeError("Cannot build agent before the session is bound.")
        cfg = profile.main_agent()
        self.tool_context.allowed_tool_permissions = set(cfg.allowed_tool_permissions)
        self.tool_context.blocked_tool_names = set(cfg.blocked_tool_names)
        harness_rules = _load_harness_rules(self.cwd)
        catalog = self.skill_registry.build_catalog_prompt()
        prefix = PromptPrefixBuilder().build(
            profile_prompt=cfg.system_prompt,
            global_rules_docs=[harness_rules] if harness_rules is not None else [],
            skill_catalog=catalog,
        )
        middlewares = build_main_agent_middlewares(
            user_middlewares=self.user_middlewares,
            tool_context=self.tool_context,
            tool_registry=self.tool_registry,
            workspace=self.cwd,
        )
        # The memory index is dynamic reference material: it stays separate
        # from the stable prefix so cache boundaries (and diagnostics) treat
        # it as the variable suffix it is.
        memory_index = None
        if self.memory_use_enabled:
            block = MemoryService(self.cwd).index_block()
            if "\n- [" in block:
                memory_index = block
        return Agent(
            "main_agent",
            prefix.content,
            use_tools=True,
            tool_schemas=self._tool_schemas_for_agent_config(cfg),
            middlewares=middlewares,
            time_budget=cfg.time_budget,
            tool_context=self.tool_context,
            stream_callback=self.stream_sink,
            prompt_cache_identity=prefix.cache_identity,
            memory_index=memory_index,
        )

    def _load_mcp_tools(self) -> None:
        if self.tool_registry is None or self.mcp_manager is None:
            self._prepare_tool_registry()
        self.mcp_manager.connect_all()
        self.mcp_manager.register_tools(self.tool_registry)
        self._mcp_tools_loaded = True

    def _prepare_tool_registry(self) -> None:
        self.tool_registry = BUILTIN_TOOL_REGISTRY.copy()
        self.mcp_manager = McpClientManager.from_workspace(self.cwd)
        manager = self.mcp_manager
        self.lifecycle.register(f"mcp:{id(manager)}", manager.close, order=10)
        self._mcp_tools_loaded = False

    def _ensure_mcp_tools_loaded(self) -> None:
        if self._mcp_tools_loaded:
            return
        with self._mcp_load_lock:
            if self._mcp_tools_loaded:
                return
            self._report_startup("connecting external tools")
            self._load_mcp_tools()
            self._refresh_agent_tool_schemas()
            self._report_startup("external tools ready")

    def warm_mcp_tools(self) -> None:
        """Connect configured MCP servers after the core session is usable."""
        self._ensure_mcp_tools_loaded()

    def _tool_schemas_for_agent_config(self, cfg, *, update_context: bool = True) -> list[dict]:
        core_schemas = tool_schemas_for_profile(
            allowed_permissions=cfg.allowed_tool_permissions,
            exclude_names=cfg.blocked_tool_names,
            registry=self.tool_registry,
        )
        revealed = set()
        if self.tool_context is not None and update_context:
            self.tool_context.allowed_tool_permissions = set(cfg.allowed_tool_permissions)
            self.tool_context.blocked_tool_names = set(cfg.blocked_tool_names)
            revealed = set(self.tool_context.revealed_tool_names)
        if not revealed:
            return core_schemas
        revealed_schemas = tool_schemas_for_profile(
            allowed_permissions=cfg.allowed_tool_permissions,
            include_names=revealed,
            exclude_names=cfg.blocked_tool_names,
            registry=self.tool_registry,
            disclosure={"deferred"},
        )
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
        if not self.profile_runtimes:
            return
        active_name = self.profile.name() if self.profile is not None else None
        for slot in self.profile_runtimes.values():
            cfg = slot.profile.main_agent()
            schemas = self._tool_schemas_for_agent_config(
                cfg,
                update_context=slot.profile.name() == active_name,
            )
            slot.agent.update_tool_schemas(schemas)

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
        return self.attachment_manager.prepare(submission)

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
        try:
            prepared = self.prepare_submission(submission)
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
            prepared = self.prepare_submission(replace(
                submission,
                authorized_paths=tuple(exc.paths),
            ))
        return self.submit_prepared(prepared, cancellation_token=cancellation_token)

    def submit_prepared(self, prepared: PreparedTurn, cancellation_token=None) -> TurnResult:
        user_prompt = prepared.text
        self.ensure_profile_bound_for_first_task(user_prompt)
        skill_invocation = self.skill_registry.build_user_invocation(user_prompt)
        if skill_invocation is not None:
            return self._submit_to_current_agent(
                user_prompt,
                cancellation_token=cancellation_token,
                turn_instruction=skill_invocation.prompt,
                attachments=prepared.attachments,
            )
        if self.pending_plan_markdown and self.profile.name() == "plan":
            if _is_plan_execution_confirmation(user_prompt):
                return self.execute_pending_plan(attachments=prepared.attachments)
            return self.revise_pending_plan(user_prompt, attachments=prepared.attachments)
        route_decision = self._maybe_auto_route_profile(user_prompt)
        turn_instruction = _turn_instruction_for_route(route_decision)
        return self._submit_to_current_agent(
            user_prompt,
            cancellation_token=cancellation_token,
            turn_instruction=turn_instruction,
            attachments=prepared.attachments,
        )

    def ensure_profile_bound_for_first_task(self, user_prompt: str) -> None:
        if self.is_bound:
            return
        self._bind_profile(self._pending_profile_name, source=self._profile_source)

    def _maybe_auto_route_profile(self, user_prompt: str) -> RouteDecision | None:
        if (
            not self.is_bound
            or self.event_bus is None
            or self.routing_mode != ROUTING_MODE_AUTO
        ):
            return None
        current = self.profile.name()
        decision = route_profile_for_turn(
            user_prompt,
            current_profile=current,
            routing_mode=self.routing_mode,
            previous_user_task=self.last_user_task,
            previous_assistant_text=self.last_assistant_text,
        )
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
                "local_candidate": getattr(decision, "local_candidate", ""),
                "local_confidence": getattr(decision, "local_confidence", 0.0),
                "local_margin": getattr(decision, "local_margin", 0.0),
                "llm_called": getattr(decision, "llm_called", False),
                "llm_confidence": getattr(decision, "llm_confidence", 0.0),
                "llm_provider": getattr(decision, "llm_provider", ""),
                "llm_model": getattr(decision, "llm_model", ""),
                "failure_type": getattr(decision, "failure_type", ""),
            },
        )
        if switched:
            if getattr(decision, "decisive_signal", "") == "explicit_mode":
                self.routing_mode = ROUTING_MODE_PINNED
                if self.session is not None:
                    self.session_store.update_routing_mode(
                        self.session.id,
                        self.routing_mode,
                    )
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
        self.session = self.session_store.create(
            profile=self.profile.name(),
            cwd=self.cwd,
            model=config.MODEL,
            permission_mode=self.permission_mode,
            resumed_from=self.resume_session_id,
            profile_source=source,
        )
        self.attachment_manager = AttachmentManager(
            self.cwd,
            self.session.root,
            model_input_mode(),
        )
        metadata = self.session_store.read_metadata(self.session.id)
        metadata["routing_mode"] = self.routing_mode
        self.session.metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.event_bus = self.session_store.event_bus(self.session, listener=self.event_listener)
        if self.checkpoint_init_error:
            metadata = self.session_store.read_metadata(self.session.id)
            metadata["checkpoint_status"] = "disabled"
            metadata["checkpoint_init_error"] = self.checkpoint_init_error
            self.session.metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self.event_bus.emit(
                "checkpoint_disabled",
                agent="main_agent",
                payload={
                    "reason": "git repository initialization failed",
                    "error": self.checkpoint_init_error,
                },
            )
        self._prepare_tool_registry()
        self.tool_context = ToolContext(
            workspace=WorkspaceService(
                root=self.cwd,
                snapshots_dir=self.session.snapshots_dir,
            ),
            permission_policy=self._effective_permission_policy(),
            event_bus=self.event_bus,
            session_id=self.session.id,
            memory_use_enabled=self.memory_use_enabled,
            memory_auto_extract_enabled=self.memory_auto_extract_enabled,
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
        )
        self.lifecycle.register(
            "agent_coordinator",
            self.tool_context.agent_coordinator.close,
            order=30,
        )
        self._activate_profile_runtime(self.profile.name())
        self.event_bus.emit(
            "session_started",
            agent="main_agent",
            payload={
                "session_id": self.session.id,
                "profile": self.profile.name(),
                "profile_source": source,
                "workspace": str(self.cwd),
                "resumed_from": self.resume_session_id,
                "interactive": True,
            },
        )
        if self.resume_session_id:
            self._restore_session_messages(self.resume_session_id)
        elif self.resume_context:
            self._append_conversation_message({
                "role": "user", "content": f"Resume context:\n{self.resume_context}",
            })

    def _effective_permission_policy(self) -> PermissionPolicy:
        """Session permission mode tightened by the active profile preset."""
        preset = self.profile.main_agent().permission_preset
        return PermissionPolicy(mode=self.permission_mode).restricted_by(preset)

    def _activate_profile_runtime(
        self,
        profile_name: str,
    ) -> bool:
        if self.session is None or self.event_bus is None or self.tool_context is None:
            raise RuntimeError("Cannot activate a profile runtime before the session is bound.")
        created = False
        profile = get_profile(profile_name)
        slot = self.profile_runtimes.get(profile_name)
        if slot is None:
            agent = self._build_agent(profile)
            if self.conversation is None:
                conversation = agent.start_conversation()
                self.lifecycle.register("conversation", conversation.close, order=40)
                conversation._event_bus = self.event_bus
                self.conversation = conversation
            else:
                conversation = self.conversation
                rebind_agent = getattr(conversation, "rebind_agent", None)
                if callable(rebind_agent):
                    rebind_agent(agent)
                else:
                    # Lightweight test doubles may only expose ``agent``.
                    # Keep one conversation object and update its active
                    # runtime instead of creating a second history lane.
                    if hasattr(conversation, "agent"):
                        conversation.agent = agent
            slot = ProfileRuntime(profile=profile, agent=agent, conversation=conversation)
            self.profile_runtimes[profile.name()] = slot
            created = True
        else:
            slot.agent = self._build_agent(profile)
            slot.conversation = self.conversation
            if self.conversation is not None:
                rebind_agent = getattr(self.conversation, "rebind_agent", None)
                if callable(rebind_agent):
                    rebind_agent(slot.agent)
                elif hasattr(self.conversation, "agent"):
                    self.conversation.agent = slot.agent
        self._active_profile_name = profile.name()
        self.profile = profile
        self.agent = slot.agent
        self.conversation = slot.conversation
        # Profile presets (e.g. review's no-workspace-writes) live on the
        # policy, so switching profiles must re-apply the preset immediately.
        self.tool_context.permission_policy = self._effective_permission_policy()
        # Profile handoff messages used to split the conversation and hide
        # history. Approved plans are now carried by the execution turn itself.
        return created

    def _submit_to_current_agent(
        self,
        user_prompt: str,
        cancellation_token=None,
        turn_instruction: str | None = None,
        attachments: tuple[Attachment, ...] = (),
    ) -> TurnResult:
        self._ensure_mcp_tools_loaded()
        turn_started_at = time.time()
        baseline = capture_git_baseline(self.cwd) if self.checkpoint.auto else None
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
        context_block = self._augment_user_prompt(
            prompt_for_model,
            mention_paths=_memory_mention_paths(resolved),
        )
        prompt_with_mentions = _format_turn_with_mentions_and_memory(
            prompt_for_model,
            resolved_for_model,
            context_block,
        )
        if turn_instruction:
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
        self.turn_count += 1
        self.event_bus.emit_event(
            UserInputEvent(
                text=user_prompt,
                turn=self.turn_count,
                mentions=[item.raw for item in resolved],
                attachments=attachment_metadata,
            ).to_event()
        )
        self.event_bus.emit(
            "turn_started",
            agent="main_agent",
            payload={
                "turn": self.turn_count,
                "mentions": [item.raw for item in resolved],
                "attachments": attachment_metadata,
            },
        )
        turn_event_start = len(getattr(self.event_bus, "events", []))
        text = self.conversation.submit(
            model_content,
            task_text=task,
            cancellation_token=cancellation_token,
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
        checkpoint = self._maybe_auto_checkpoint(
            baseline=baseline,
        )
        duration_seconds = time.time() - turn_started_at
        self._maybe_emit_turn_summary(
            user_prompt=user_prompt,
            assistant_text=text,
            checkpoint=checkpoint,
            turn_event_start=turn_event_start,
            duration_seconds=duration_seconds,
        )
        self.event_bus.emit(
            "turn_finished",
            agent="main_agent",
            payload={
                "turn": self.turn_count,
                "checkpoint": checkpoint,
                "duration_seconds": duration_seconds,
            },
        )
        return TurnResult(text=text, checkpoint=checkpoint, notice=notice, streamed=streamed)

    def _maybe_emit_turn_summary(
        self,
        *,
        user_prompt: str,
        assistant_text: str,
        checkpoint: str,
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
            checkpoint=checkpoint,
        )
        self.event_bus.emit_event(
            TurnSummaryEvent(
                turn=self.turn_count,
                summary=summary.summary,
                duration_seconds=duration_seconds,
                tool_counts=summary.tool_counts,
                changed_files=summary.changed_files,
                checkpoint=checkpoint,
                generated_by=summary.generated_by,
            ).to_event()
        )

    def interrupt_current_shell(self) -> bool:
        """Best-effort interrupt for a shell command owned by the active conversation."""
        if self.conversation is None:
            return False
        runtime_state = getattr(self.conversation, "runtime_state", None)
        interrupt = getattr(runtime_state, "interrupt_shell_sessions", None)
        if interrupt is None:
            return False
        try:
            return bool(interrupt())
        except Exception as exc:
            log.debug("Failed to interrupt active shell session: %s", exc)
            return False

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
        return self._submit_to_current_agent(task, attachments=attachments)

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
        return self._submit_to_current_agent(
            "Revise the previous Markdown plan using this user feedback. "
            "Return the complete updated plan in the required structured Markdown format.\n\n"
            f"User feedback:\n{feedback}",
            attachments=attachments,
        )

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
                "approval_source": "/plan",
            },
        )
        return (
            "计划已写入 `global_plan/current/plan.md`。"
            "在 TUI 中选择 `执行计划` 继续，或在 `修改计划` 输入框中输入修改理由。"
            "非 TUI 入口可回复 continue/继续 执行，其他文本会作为修改理由。"
        )

    def _write_pending_plan_artifact(self, plan_markdown: str) -> Path:
        plan_path = self.cwd / "global_plan" / "current" / "plan.md"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(plan_markdown.rstrip() + "\n", encoding="utf-8")
        return plan_path

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
        self._activate_profile_runtime(
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
            tool_schemas_for_profile(self.agent),
        )
        labels = {"system": "系统指令", "summary": "工作摘要", "recent": "近期消息", "memory": "长期记忆", "tools": "工具定义"}
        lines = [f"上下文估算：{sum(breakdown.values())} tokens"]
        lines.extend(f"- {labels[key]}：{value}" for key, value in breakdown.items())
        return "\n".join(lines)

    def memory_command(self, args: list[str]) -> str:
        service = MemoryService(self.cwd)
        if not args or args == ["status"]:
            counts = {scope: len(store.list_documents()) for scope, store in service.stores.items()}
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
                    # Kick a one-shot pass so jobs queued while generate was
                    # off are processed immediately.
                    start_memory_worker(self.cwd)
            suffix = "；启动索引将在下次新会话刷新" if args[0] == "use" else ""
            return f"memory {args[0]}: {args[1]}{suffix}"
        if args[0] == "search" and len(args) > 1:
            return service.format_hits(service.search(" ".join(args[1:]))) or "没有找到相关记忆"
        if args[0] in {"show", "forget", "validate"} and len(args) >= 2:
            scope, memory_id = _parse_memory_ref(args[1])
            if args[0] == "show":
                doc = service.read(memory_id, scope=scope)
                return json.dumps({**doc.metadata(), "body": doc.body}, ensure_ascii=False, indent=2)
            if len(args) != 3:
                raise ValueError(f"用法：/memory {args[0]} [project:|user:]<id> <version>")
            version = int(args[2])
            if args[0] == "forget":
                service.forget(memory_id, version, scope=scope)
                return f"已遗忘 {scope}:{memory_id}，正文已删除"
            doc = service.validate(memory_id, version, scope=scope)
            return f"已验证 {scope}:{doc.id} v{doc.version}"
        if args[0] == "edit" and len(args) >= 4:
            scope, memory_id = _parse_memory_ref(args[1])
            version = int(args[2])
            current = service.read(memory_id, scope=scope)
            doc = service.write(MemoryWriteCommand(
                topic=current.topic,
                body=" ".join(args[3:]),
                scope=scope,
                applicability=current.applicability,
                source_sessions=current.source_sessions,
                source_paths=current.source_paths,
                memory_id=current.id,
                expected_version=version,
            ))
            return f"已编辑 {scope}:{doc.id} v{doc.version}"
        raise ValueError(
            "用法：/memory [status|search <query>|show <id>|edit <id> <version> <body>|forget <id> <version>|"
            "validate <id> <version>|use on|off|generate on|off]"
        )

    def submit_skill_command(self, line: str) -> TurnResult:
        if self.skill_registry.build_user_invocation(line) is None:
            raise ValueError(f"Unknown user skill command: {line}")
        return self.submit(line)

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

    def enable_auto_profile_routing(self) -> str:
        previous_mode = self.routing_mode
        self.routing_mode = ROUTING_MODE_AUTO
        if self.session is not None:
            self.session_store.update_routing_mode(self.session.id, self.routing_mode)
        if previous_mode == ROUTING_MODE_AUTO:
            return f"auto profile routing already active: {self.profile.name()}"
        return f"auto profile routing enabled: {self.profile.name()}"

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
        config.set_model_override(model=model, reasoning_effort=reasoning_effort)
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

    def reload_mcp(self) -> str:
        self._ensure_mcp_tools_loaded()
        self.mcp_manager.close()
        self._prepare_tool_registry()
        self._ensure_mcp_tools_loaded()
        if self.tool_context is not None:
            self.tool_context.tool_registry = self.tool_registry
        self._refresh_agent_tool_schemas()
        if self.event_bus is not None:
            self.event_bus.emit(
                "mcp_reloaded",
                agent="main_agent",
                payload={"tool_count": len(getattr(self.mcp_manager, "tool_bindings", []))},
            )
        return "MCP reloaded\n" + self.mcp_manager.status_report()

    def reload_mcp_server(self, server_name: str) -> str:
        """Reconnect one configured MCP server and refresh active tool schemas."""
        self._ensure_mcp_tools_loaded()
        result = self.mcp_manager.reload_server(server_name)
        if self.tool_registry is not None:
            self.tool_registry = BUILTIN_TOOL_REGISTRY.copy()
            self.mcp_manager.register_tools(self.tool_registry)
        if self.tool_context is not None:
            self.tool_context.tool_registry = self.tool_registry
        self._refresh_agent_tool_schemas()
        return result

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

    def compact_current_context(self) -> str:
        """Run explicit compaction against the single shared conversation."""
        self._ensure_mcp_tools_loaded()
        if self.conversation is None:
            return "当前还没有可压缩的对话"
        return self.conversation.compact_now()

    def fork_current_session(self) -> str:
        """Create a durable branch and continue with the same live context."""
        if self.session is None or self.conversation is None:
            return "当前还没有可分支的会话"
        source = self.session
        branched = self.session_store.fork(source.id)
        self._activate_session(branched)
        self.session_store.update_profile(
            branched.id,
            self.profile.name(),
            profile_source=self._profile_source,
        )
        self.session_store.update_routing_mode(branched.id, self.routing_mode)
        self.event_bus.emit(
            "session_forked_active",
            agent="main_agent",
            payload={"source_session_id": source.id, "session_id": branched.id},
        )
        return f"已进入会话分支：{branched.id}"

    def resume_from_session(self, session_id: str) -> None:
        """Fork a history session and activate the fork.

        Resuming session X is exactly "fork X + make the new fork the active
        session". The current conversation is never mutated in place: the fork
        already carries a copy of the source journal, so recovered messages are
        not appended a second time.
        """
        if self.session is not None and session_id == self.session.id:
            return
        branched = self.session_store.fork(session_id)
        recovered = SessionJournal(branched.journal_path).recovery_messages(
            self.agent.full_system_prompt
        )
        self._activate_session(branched)
        self.conversation._replace_messages(recovered)
        # The fork inherits the source session's profile in metadata; bring it
        # in line with the profile the live runtime is actually using.
        self.session_store.update_profile(
            branched.id,
            self.profile.name(),
            profile_source=self._profile_source,
        )
        self.session_store.update_routing_mode(branched.id, self.routing_mode)

    def _activate_session(self, session: Session) -> None:
        """Rebind every live component onto an already-created session."""
        self.session = session
        self.attachment_manager = AttachmentManager(
            self.cwd,
            session.root,
            model_input_mode(),
        )
        self.event_bus = self.session_store.event_bus(
            session,
            listener=self.event_listener,
        )
        self.conversation.event_bus = self.event_bus
        self.conversation._event_bus = self.event_bus
        self.conversation.journal = SessionJournal(session.journal_path)
        emitter = getattr(self.conversation, "emitter", None)
        if emitter is not None:
            emitter.event_bus = self.event_bus
        runtime_state = getattr(self.conversation, "runtime_state", None)
        if runtime_state is not None:
            runtime_state.session_id = session.id
            runtime_state.event_bus = self.event_bus
        if self.tool_context is not None:
            self.tool_context.session_id = session.id
            self.tool_context.event_bus = self.event_bus

    def _restore_session_messages(self, session_id: str) -> None:
        if self.conversation is None or self.agent is None:
            return
        journal_path = self.session_store.sessions_dir / session_id / "journal.jsonl"
        if not journal_path.exists() or not journal_path.stat().st_size:
            self._append_conversation_message({
                "role": "user", "content": f"Resume context:\n{self.resume_context or ''}",
            })
            return
        recovered = SessionJournal(journal_path).recovery_messages(self.agent.full_system_prompt)
        self.conversation._replace_messages(recovered)
        if self.conversation.journal is not None:
            for message in recovered[1:]:
                self.conversation.journal.append_message(message)

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
        inputs_dir = self.session.root / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        path = inputs_dir / f"turn-{self.turn_count + 1:04d}-{safe_label}.txt"
        suffix = 1
        while path.exists():
            path = inputs_dir / f"turn-{self.turn_count + 1:04d}-{safe_label}-{suffix}.txt"
            suffix += 1
        path.write_text(text, encoding="utf-8")
        return path

    def _append_conversation_message(self, message: dict) -> None:
        append = getattr(self.conversation, "_append_message", None)
        if append is not None:
            append(message)
        else:
            self.conversation.messages.append(message)

    def _augment_user_prompt(self, user_prompt: str, *, mention_paths: list[str]) -> str:
        if self.agent is None or self.conversation is None or not getattr(self, "memory_use_enabled", True):
            return ""
        hits = MemoryService(self.cwd).search(user_prompt, scope="both", paths=mention_paths)
        return MemoryService.format_hits(hits)

    def _handle_checkpoint_command(self, args: list[str]) -> str:
        if not args:
            return self.create_checkpoint(manual=True)
        if args[:2] == ["auto", "on"]:
            return self.set_auto_checkpoint(True)
        if args[:2] == ["auto", "off"]:
            return self.set_auto_checkpoint(False)
        if args[:2] == ["every", "turn"]:
            self.checkpoint.every_turns = 1
            return "checkpoint cadence: every turn"
        if (
            args
            and args[0] == "every"
            and len(args) in (2, 3)
            and (len(args) == 2 or args[2] in ("turn", "turns"))
        ):
                try:
                    turns = int(args[1])
                except ValueError as e:
                    raise ValueError("Usage: /checkpoint every <N> turns") from e
                if turns < 1:
                    raise ValueError("Checkpoint cadence must be at least 1 turn")
                self.checkpoint.every_turns = turns
                return f"checkpoint cadence: every {turns} turns"

        if args == ["status"]:
            return (
                f"checkpoint auto: {'on' if self.checkpoint.auto else 'off'}; "
                f"cadence: every {self.checkpoint.every_turns} turn(s)"
            )
        raise ValueError("Usage: /checkpoint [auto on|auto off|every turn|every <N> turns|status]")

    def set_auto_checkpoint(self, enabled: bool) -> str:
        if enabled and not self._ensure_checkpoint_repository():
            return "checkpoint unavailable: git repository initialization failed"
        self.checkpoint.auto = enabled
        return f"checkpoint auto: {'on' if enabled else 'off'}"

    def _ensure_checkpoint_repository(self) -> bool:
        if (self.cwd / ".git").exists():
            return True
        self._report_startup("preparing checkpoints")
        try:
            _ensure_git_repository(self.cwd)
            self.checkpoint_init_error = ""
            return True
        except (OSError, subprocess.CalledProcessError) as exc:
            if not self._allow_checkpoint_init_failure:
                raise
            self.checkpoint.auto = False
            self.checkpoint_init_error = f"{type(exc).__name__}: {exc}"
            if self.session is not None and self.event_bus is not None:
                metadata = self.session_store.read_metadata(self.session.id)
                metadata["checkpoint_status"] = "disabled"
                metadata["checkpoint_init_error"] = self.checkpoint_init_error
                self.session.metadata_path.write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                self.event_bus.emit(
                    "checkpoint_disabled",
                    agent="main_agent",
                    payload={
                        "reason": "git repository initialization failed",
                        "error": self.checkpoint_init_error,
                    },
                )
            return False

    def _maybe_auto_checkpoint(
        self,
        *,
        baseline: GitBaseline | None,
    ) -> str:
        if not self.checkpoint.auto:
            return "checkpoint auto off"
        if baseline is None:
            return "checkpoint skipped: git status unavailable"
        if self.turn_count % self.checkpoint.every_turns != 0:
            return "checkpoint cadence skipped"
        if baseline.staged_paths:
            return "checkpoint skipped: staged changes existed before turn"
        return self.create_checkpoint(manual=False, baseline_dirty=set(baseline.dirty_paths))

    def create_checkpoint(
        self,
        *,
        manual: bool,
        baseline_dirty: set[str] | None = None,
    ) -> str:
        if self.session is None:
            return "checkpoint skipped: no active session"
        if not self._ensure_checkpoint_repository():
            return "checkpoint skipped: git repository unavailable"
        if self.checkpoint_init_error:
            return "checkpoint skipped: git repository unavailable"
        if not git_has_committable_changes(self.cwd):
            return "no changes to checkpoint"
        paths_to_add = None
        if not manual and baseline_dirty is not None:
            current_dirty = git_dirty_paths(self.cwd)
            paths_to_add = sorted(current_dirty - baseline_dirty)
            if not paths_to_add:
                return "no changes to checkpoint"
        if paths_to_add is None:
            git_add_runtime_excluded(self.cwd)
        else:
            git_add_paths(self.cwd, paths_to_add)
        if not git_has_staged_changes(self.cwd):
            return "no changes to checkpoint"
        detail = "manual" if manual else f"turn {self.turn_count}"
        message = f"checkpoint: {self.session.id} {detail}"
        subprocess.run(
            git_commit_command(message),
            cwd=self.cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=self.cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"checkpoint created: {rev}"

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        for error in self.lifecycle.close():
            log.warning("Failed to close session resource %s: %s", error.name, error.error)
        if self.session is None or self.event_bus is None:
            return
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
        if self.memory_auto_extract_enabled and self.session.journal_path.exists():
            try:
                service = MemoryService(self.cwd)
                service.stores["project"].enqueue_extraction(
                    self.session.id,
                    self.session.journal_path.stat().st_size,
                    self.session.journal_path,
                )
                # Enqueue happens on close; kick the one-shot worker so the
                # job actually runs in this process.
                start_memory_worker(self.cwd)
            except (OSError, ValueError) as exc:
                log.debug("Failed to enqueue memory extraction: %s", exc)


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


def _memory_mention_paths(resolved: list[ResolvedMention]) -> list[str]:
    paths: list[str] = []
    for item in resolved:
        if item.kind not in {"file", "directory"}:
            continue
        value = item.metadata.get("path") or item.resolved or item.target
        if value:
            paths.append(str(value))
    return paths


def _parse_memory_ref(value: str) -> tuple[str, str]:
    if ":" not in value:
        return "project", value
    scope, memory_id = value.split(":", 1)
    if scope not in {"project", "user"} or not memory_id:
        raise ValueError("记忆引用必须是 <id>、project:<id> 或 user:<id>")
    return scope, memory_id


def _format_turn_with_mentions_and_memory(
    user_text: str,
    resolved: list[ResolvedMention],
    memory_block: str,
) -> str:
    parts = []
    mention_context = render_mention_context(resolved)
    if mention_context:
        parts.append(mention_context)
    if memory_block:
        parts.append(memory_block)
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
    if result.checkpoint:
        print(result.checkpoint)

from .formatters import _build_resume_context
from .git_helpers import (
    GitBaseline,
    _ensure_git_repository,
    capture_git_baseline,
    git_add_paths,
    git_add_runtime_excluded,
    git_commit_command,
    git_dirty_paths,
    git_has_committable_changes,
    git_has_staged_changes,
)
