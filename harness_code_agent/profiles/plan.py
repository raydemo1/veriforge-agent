"""Planning profile with constrained planning-artifact writes."""
from __future__ import annotations

from ..runtime.permissions import (
    TOOL_PERMISSION_CONTROL,
    TOOL_PERMISSION_NETWORK_READ,
    TOOL_PERMISSION_READ,
)
from .base import AgentConfig, BaseProfile, build_profile_prompt


class PlanProfile(BaseProfile):
    def name(self) -> str:
        return "plan"

    def description(self) -> str:
        return "Investigate and produce a decision-complete implementation plan"

    def main_agent(self) -> AgentConfig:
        return AgentConfig(
            system_prompt=build_profile_prompt(
                role="Investigate the request and produce a decision-complete implementation plan.",
                working_style=(
                    "Resolve repository facts and constraints from files or authoritative references first. "
                    "Ask focused questions for material decisions that remain, grouping 2-3 related choices "
                    "when needed. Settle the goal, scope, approach, failure handling, and acceptance criteria."
                ),
                boundaries=(
                    "Keep the workspace read-only and skip shell, browser, and implementation actions. "
                    "Deliver plan.md through the interactive planning flow. Do not write files directly "
                    "or create status.md or final.md."
                ),
                completion=(
                    "Include a title, summary, implementation changes by subsystem, test plan, and explicit "
                    "assumptions. Name files, symbols, tests, and commands where useful. Publish after "
                    "resolving material user decisions."
                ),
            ),
            allowed_tool_permissions={
                TOOL_PERMISSION_READ,
                TOOL_PERMISSION_NETWORK_READ,
                TOOL_PERMISSION_CONTROL,
            },
            blocked_tool_names={"list_shell_jobs", "read_shell_output", "stop_shell_job"},
        )
