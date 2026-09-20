from __future__ import annotations

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
    memory_generate_enabled: bool = True
    approval_provider: ApprovalProvider = field(default_factory=NoApprovalProvider)
    question_provider: QuestionProvider = field(default_factory=NoQuestionProvider)
    tool_registry: ToolRegistry | None = None
    allowed_tool_permissions: set[str] | None = None
    blocked_tool_names: set[str] = field(default_factory=set)
    revealed_tool_names: set[str] = field(default_factory=set)
    resource_coordinator: ResourceCoordinator = field(default_factory=ResourceCoordinator)
    tool_tasks: ToolTaskSupervisor = field(default_factory=ToolTaskSupervisor)
    agent_coordinator: AgentCoordinator | None = None
