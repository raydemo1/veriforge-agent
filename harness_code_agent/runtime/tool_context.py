from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..sessions.events import EventBus
from ..workspace.service import WorkspaceService
from .approvals import ApprovalProvider, NoApprovalProvider
from .execution_planner import ResourceCoordinator
from .permissions import PermissionPolicy
from .questions import NoQuestionProvider, QuestionProvider
from .task_supervisor import ToolTaskSupervisor

if TYPE_CHECKING:
    from ..agent.coordinator import AgentCoordinator
    from .tool_registry import ToolRegistry


@dataclass
class ToolContext:
    workspace: WorkspaceService
    permission_policy: PermissionPolicy
    event_bus: EventBus
    session_id: str | None = None
    memory_use_enabled: bool = True
    memory_auto_extract_enabled: bool = True
    approval_provider: ApprovalProvider = field(default_factory=NoApprovalProvider)
    question_provider: QuestionProvider = field(default_factory=NoQuestionProvider)
    tool_registry: ToolRegistry | None = None
    allowed_tool_permissions: set[str] | None = None
    blocked_tool_names: set[str] = field(default_factory=set)
    revealed_tool_names: set[str] = field(default_factory=set)
    resource_coordinator: ResourceCoordinator = field(default_factory=ResourceCoordinator)
    tool_tasks: ToolTaskSupervisor = field(default_factory=ToolTaskSupervisor)
    agent_coordinator: AgentCoordinator | None = None
    #: Set only on child-agent contexts; identifies the calling subagent.
    agent_id: str | None = None
    _language_service: object | None = field(default=None, init=False, repr=False)
    _language_service_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

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
