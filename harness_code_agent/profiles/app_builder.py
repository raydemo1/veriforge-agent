"""
App Builder profile for single-agent web app creation and browser verification.
"""
from __future__ import annotations

from ..tracking_policy import TASK_TRACKING_POLICY
from .base import (
    AgentConfig,
    BaseProfile,
    build_profile_prompt,
)

_APP_BUILDER_SYSTEM = build_profile_prompt(
    role="Build a complete working app for the intended audience and core interaction.",
    working_style=(
        "Choose the smallest suitable stack, reusing the repository's framework where present. "
        "Add a framework only when the behavior warrants it. Make deliberate choices about hierarchy, "
        "typography, color, spacing, and interaction.\n\n"
        f"{TASK_TRACKING_POLICY}\n\n"
        "Delegate focused investigation, design critique, or review when useful. Review and integrate "
        "worker proposals."
    ),
    boundaries=(
        "Deliver actual source changes and working flows. Remove stubs and placeholder behavior. "
        "Reuse the existing stack or platform features when they meet the requirement."
    ),
    completion=(
        "Run the app and exercise representative interactions on mobile and desktop. Resolve console errors "
        "and check semantic controls, labels, keyboard focus, and readable contrast. If browser checks are "
        "unavailable, run the strongest available build and static checks and report the limitation."
    ),
)


class AppBuilderProfile(BaseProfile):
    def name(self) -> str:
        return "app-builder"

    def description(self) -> str:
        return "Build complete web applications from a one-sentence prompt (Anthropic article scenario)"

    def main_agent(self) -> AgentConfig:
        return AgentConfig(
            system_prompt=_APP_BUILDER_SYSTEM,
            blocked_tool_names=set(),
        )
