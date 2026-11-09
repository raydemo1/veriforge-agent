"""General-purpose default profile for lightweight workspace assistance."""
from __future__ import annotations

from ..runtime.permissions import (
    TOOL_PERMISSION_EDIT,
    TOOL_PERMISSION_NETWORK_READ,
    TOOL_PERMISSION_READ,
)
from .base import AgentConfig, BaseProfile, build_profile_prompt


class GeneralProfile(BaseProfile):
    """Default profile for answer-first, mostly read-only work."""

    def name(self) -> str:
        return "general"

    def description(self) -> str:
        return "General workspace assistant for answers, discussion, and light read-only inspection"

    def main_agent(self) -> AgentConfig:
        return AgentConfig(
            system_prompt=build_profile_prompt(
                role="Answer questions, discuss ideas, and explain the workspace directly.",
                working_style=(
                    "Answer concisely. Use exact, relevant memory or bounded reads to resolve uncertainty. "
                    "Check the repository when memory is incomplete, conflicting, or likely stale."
                ),
                boundaries=(
                    "Keep workspace access read-only. Skip shell, jobs, browser sessions, execution todos, "
                    "delegation, and implementation interviews. Write, validate, or forget durable memory "
                    "only on explicit request or for an unambiguous durable fact. Leave specialized work "
                    "to the corresponding profile."
                ),
                completion=(
                    "Stop once the answer is grounded. Identify its repository evidence or other basis "
                    "and report only checks you ran."
                ),
            ),
            allowed_tool_permissions={
                TOOL_PERMISSION_READ,
                TOOL_PERMISSION_NETWORK_READ,
                TOOL_PERMISSION_EDIT,
            },
            blocked_tool_names={
                "write_file",
                "apply_patch",
                "update_todo",
                "ask_user",
                "run_bash",
                "list_shell_jobs",
                "read_shell_output",
                "stop_shell_job",
                "spawn_agent",
                "send_agent_message",
                "followup_agent",
                "wait_agents",
                "list_agents",
                "interrupt_agent",
                "read_agent_changes",
                "apply_agent_changes",
                "read_agent_conflicts",
                "resolve_agent_conflicts",
                "close_agent",
                "browser_test",
                "stop_dev_server",
            },
        )
