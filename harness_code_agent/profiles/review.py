"""Review profile for non-mutating code review tasks."""
from __future__ import annotations

from typing import ClassVar

from ..runtime.middleware import AgentMiddleware
from ..runtime.permissions import (
    TOOL_PERMISSION_CONTROL,
    TOOL_PERMISSION_NETWORK_READ,
    TOOL_PERMISSION_READ,
    TOOL_PERMISSION_SHELL,
    is_workspace_write_command,
)
from ..runtime.tool_result import ToolResult
from .base import AgentConfig, BaseProfile, build_profile_prompt


class ReviewOnlyMiddleware(AgentMiddleware):
    """Keep review mode from editing the real workspace or updating todo state."""

    def __init__(self, *, profile_label: str = "review profile"):
        self._profile_label = profile_label

    _WRITE_OR_CONTROL_TOOLS: ClassVar[set] = {
        "write_file",
        "apply_patch",
        "update_todo",
        "ask_user",
    }

    def before_tool(
        self,
        tool_name: str,
        tool_args: dict,
        messages: list[dict],
        runtime_state=None,
        agent_name: str | None = None,
        permission_decision=None,
    ) -> ToolResult | None:
        if tool_name in self._WRITE_OR_CONTROL_TOOLS:
            return ToolResult(
                tool=tool_name,
                status="failed",
                output=f"[blocked] {self._profile_label} cannot modify workspace files, update the todo list, or ask the user.",
                error=f"{self._profile_label} cannot modify workspace files or update todo state",
                metadata={"status_source": self._profile_label.replace(" ", "_")},
            )
        if tool_name == "run_bash":
            command = str(tool_args.get("command", ""))
            if is_workspace_write_command(command):
                return ToolResult(
                    tool=tool_name,
                    status="failed",
                    output=f"[blocked] {self._profile_label} cannot run shell commands that directly write workspace files.",
                    error="direct workspace file writes are not allowed in review mode",
                    metadata={"status_source": self._profile_label.replace(" ", "_")},
                )
        return None


class ReviewProfile(BaseProfile):
    def name(self) -> str:
        return "review"

    def description(self) -> str:
        return "Non-mutating code review mode with findings-first output"

    def main_agent(self) -> AgentConfig:
        return AgentConfig(
            system_prompt=build_profile_prompt(
                role=(
                    "Act as an independent reviewer. Evaluate the requested code or changes for "
                    "actionable defects and risks rather than retelling the implementation."
                ),
                working_style=(
                    "Inspect the relevant diff, code paths, tests, browser behavior, and command output. Ground every "
                    "finding in observable evidence and prioritize correctness, security, data loss, "
                    "regressions, missing tests, and maintainability. Use delegation when a second "
                    "perspective or verification pass reduces blind spots.\n\n"
                    "Present findings first and order them by severity. Each finding should identify the "
                    "location when available, explain the evidence and impact, and give a concrete recommendation."
                ),
                boundaries=(
                    "Review mode is not repair mode or planning mode. Do not modify workspace files, "
                    "maintain an execution todo list, or ask the user to choose an implementation "
                    "direction. You may run tests, browser checks, server checks, and diagnostics, "
                    "but direct workspace writes remain blocked."
                ),
                completion=(
                    "Stop when the review surface has been examined deeply enough to support the findings. "
                    "If no actionable issue is found, say so plainly and identify residual risk, assumptions, "
                    "or tests that were not run."
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
            middlewares=[ReviewOnlyMiddleware()],
        )
