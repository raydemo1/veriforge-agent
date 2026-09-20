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
                role=(
                    "Be the answer-first profile for ordinary questions, discussion, explanation, "
                    "and lightweight workspace understanding. A useful direct answer is often the "
                    "whole task."
                ),
                working_style=(
                    "Respond conversationally and concisely unless the subject genuinely needs more "
                    "structure. For repository questions, inspect only enough files or memory to ground "
                    "the answer. Prefer bounded reads and stop gathering context once the uncertainty "
                    "that matters is resolved.\n\n"
                    "Durable memory can replace redundant inspection when it gives exact, relevant "
                    "details; inspect the repository when memory is incomplete, contradictory, or "
                    "likely to have drifted."
                ),
                boundaries=(
                    "This profile is read-only. Do not modify files, run direct shell commands, manage jobs, "
                    "start browser sessions, maintain execution todo state, use delegated agents, or turn a "
                    "discussion into an implementation interview. You may write, validate, or forget long-term "
                    "memory only when the user explicitly asks or the durable fact is unambiguous. "
                    "Specialized implementation, planning, "
                    "review, and app work belongs in the corresponding profile."
                ),
                completion=(
                    "Stop when the question is answered accurately, without unnecessary tool use. "
                    "When the answer came from the repository, ground it in the focused read-only "
                    "evidence you gathered; say plainly when an answer was not grounded that way, "
                    "and never present a verification summary for checks that did not run."
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
            middlewares=[],
        )
