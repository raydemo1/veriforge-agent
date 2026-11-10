from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import ModelSelection
from ..sessions.events import EventBus
from ..workspace.service import WorkspaceService
from .approvals import ApprovalProvider, NoApprovalProvider
from .execution_planner import ResourceCoordinator
from .permissions import PermissionPolicy
from .project_resources import ProjectResourceBudget
from .questions import NoQuestionProvider, QuestionProvider
from .task_supervisor import ToolTaskSupervisor
from .tool_registry import TOOL_CAPABILITY_MAIN

if TYPE_CHECKING:
    from ..agent.coordinator import AgentCoordinator
    from .mcp import McpServerStatus, PublishedMcpCatalog
    from .tool_registry import ToolRegistry, ToolSpec


@dataclass
class ToolContext:
    workspace: WorkspaceService
    permission_policy: PermissionPolicy
    event_bus: EventBus
    session_id: str | None = None
    session_root: Path | None = None
    environment: str = "local"
    ui_request_id: str | None = None
    project_resources: ProjectResourceBudget | None = None
    discover_mcp_tools: Callable | None = field(default=None, repr=False)
    sync_mcp_catalog: Callable[[], PublishedMcpCatalog | None] | None = field(default=None, repr=False)
    mcp_discovery_statuses: Callable[[], dict[str, McpServerStatus]] | None = field(default=None, repr=False)
    mcp_discovery_scopes: Callable[[], dict[str, str]] | None = field(default=None, repr=False)
    model_selection: ModelSelection = field(default_factory=ModelSelection)
    memory_use_enabled: bool = True
    memory_auto_extract_enabled: bool = True
    memory_service: object | None = field(default=None, repr=False)
    memory_journal_path: str | None = None
    memory_user_sequence: int | None = None
    approval_provider: ApprovalProvider = field(default_factory=NoApprovalProvider)
    question_provider: QuestionProvider = field(default_factory=NoQuestionProvider)
    tool_registry: ToolRegistry | None = None
    tool_capability: str = TOOL_CAPABILITY_MAIN
    allowed_tool_permissions: set[str] | None = None
    blocked_tool_names: set[str] = field(default_factory=set)
    #: Deferred tools this session has discovered.  This is the historical
    #: record; ``revealed_tool_names`` is the current callable view (the
    #: intersection of discovered tools and the active profile's permissions).
    discovered_tool_names: set[str] = field(default_factory=set)
    revealed_tool_names: set[str] = field(default_factory=set)
    search_tools_queries: set[tuple[str, str, int, tuple[tuple[str, str], ...]]] = field(default_factory=set)
    search_tools_signature: tuple[str, ...] = ()
    search_tools_no_progress: int = 0
    resource_coordinator: ResourceCoordinator = field(default_factory=ResourceCoordinator)
    tool_tasks: ToolTaskSupervisor = field(default_factory=ToolTaskSupervisor)
    agent_coordinator: AgentCoordinator | None = None
    #: Set only on child-agent contexts; identifies the calling subagent.
    agent_id: str | None = None
    tool_output_artifacts: dict[str, dict] = field(default_factory=dict)
    tool_output_artifact_sink: Callable[[dict], None] | None = field(default=None, repr=False)
    _language_service: object | None = field(default=None, init=False, repr=False)
    _language_service_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self):
        if self.project_resources is None:
            self.project_resources = ProjectResourceBudget.for_project(self.workspace.root)

    def invalidate_mcp_tools(
        self, previous: list[ToolSpec], *, statuses: dict[str, McpServerStatus] | None = None,
    ) -> set[str]:
        if statuses is None:
            statuses = self.mcp_discovery_statuses() if self.mcp_discovery_statuses else {}
        invalidated = set()
        for old in previous:
            if old.origin_server is None:
                continue
            current = self.tool_registry.spec_for(old.name)
            status = statuses.get(old.origin_server)
            if (
                current is None
                or current.schema != old.schema
                or current.permission != old.permission
                or current.origin_server != old.origin_server
                or current.origin_tool != old.origin_tool
                or (status is not None and status.state in {"failed", "invalid", "disabled"})
            ):
                invalidated.add(old.name)
        self.discovered_tool_names.difference_update(invalidated)
        self.revealed_tool_names.difference_update(invalidated)
        if invalidated:
            self.search_tools_queries.clear()
            self.search_tools_no_progress = 0
        return invalidated

    def register_output_artifact(self, reference: dict) -> None:
        if self.tool_output_artifacts.get(reference["path"]) == reference:
            return
        self.tool_output_artifacts[reference["path"]] = dict(reference)
        if self.tool_output_artifact_sink is not None:
            self.tool_output_artifact_sink(reference)

    def get_language_service(self):
        from .language_service import LanguageService

        with self._language_service_lock:
            if self._language_service is None:
                self._language_service = LanguageService(self.workspace.root)
            return self._language_service

    def close_language_service(self) -> None:
        with self._language_service_lock:
            service = self._language_service
            self._language_service = None
        if service is not None:
            service.close()
