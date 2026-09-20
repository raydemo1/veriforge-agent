"""Middleware stack assembly.

Single home for the exact middleware order used by the main agent and by
spawned subagents. The order is semantic (later middlewares wrap earlier
observations), so callers must construct stacks through these factories
rather than re-assembling the list ad hoc.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..permission_middleware import PermissionMiddleware
from .base import AgentMiddleware
from .tool_failure_policy import ToolFailurePolicyMiddleware
from .tool_guard import ToolGuardMiddleware
from .verification import StaticVerifierMiddleware


def build_main_agent_middlewares(
    *,
    agent_config: Any,
    tool_context: Any,
    tool_registry: Any,
    workspace: Path,
) -> list[AgentMiddleware]:
    """Build the main-agent middleware stack.

    Order mirrors the previous inline assembly in InteractiveSession:
    profile-provided middlewares run first, then structural guards,
    failure policy, permission enforcement, and finally static
    verification. Memory is not middleware: its index is injected into
    the system prompt by the interactive session.
    """
    middlewares: list[AgentMiddleware] = list(agent_config.middlewares)
    middlewares.append(ToolGuardMiddleware())
    middlewares.append(ToolFailurePolicyMiddleware(tool_registry=tool_registry))
    middlewares.append(
        PermissionMiddleware(
            tool_context=tool_context,
            tool_registry=tool_registry,
        )
    )
    middlewares.append(
        StaticVerifierMiddleware(workspace_root=str(workspace), workspace=tool_context.workspace)
    )
    return middlewares


def build_subagent_middlewares(
    *,
    tool_context: Any,
    tool_registry: Any,
) -> list[AgentMiddleware]:
    """Build the subagent middleware stack.

    Subagents intentionally run with permission enforcement only; they do
    not get the main agent's tool guard, failure policy, or verification
    middlewares.
    """
    return [PermissionMiddleware(tool_context=tool_context, tool_registry=tool_registry)]
