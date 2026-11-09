"""Review profile for non-mutating code review tasks."""
from __future__ import annotations

from ..runtime.permissions import (
    TOOL_PERMISSION_CONTROL,
    TOOL_PERMISSION_NETWORK_READ,
    TOOL_PERMISSION_READ,
    TOOL_PERMISSION_SHELL,
    PermissionPreset,
)
from .base import AgentConfig, BaseProfile, build_profile_prompt


class ReviewProfile(BaseProfile):
    def name(self) -> str:
        return "review"

    def description(self) -> str:
        return "Non-mutating code review mode with findings-first output"

    def main_agent(self) -> AgentConfig:
        return AgentConfig(
            system_prompt=build_profile_prompt(
                role="Independently review the requested code for actionable defects and risks.",
                working_style=(
                    "Trace the relevant diff and code paths. Use tests, browser or server checks, diagnostics, "
                    "and delegation when they improve the evidence. Prioritize correctness, security, data loss, "
                    "regressions, missing tests, and maintainability."
                ),
                boundaries=(
                    "Keep workspace files read-only. Leave repairs, planning, and implementation decisions "
                    "to the appropriate workflow. Skip execution todos."
                ),
                completion=(
                    "Lead with findings ordered by severity. Give each finding's location when available, "
                    "evidence, impact, and a concrete recommendation. If the review finds no actionable issues, "
                    "state the outcome and note residual risks, assumptions, or checks you skipped."
                ),
            ),
            allowed_tool_permissions={
                TOOL_PERMISSION_READ,
                TOOL_PERMISSION_NETWORK_READ,
                TOOL_PERMISSION_SHELL,
                TOOL_PERMISSION_CONTROL,
            },
            blocked_tool_names={
                "write_file",
                "apply_patch",
                "update_todo",
                "ask_user",
            },
            permission_preset=PermissionPreset.NO_WORKSPACE_WRITES,
        )
