"""
Coding Agent profile - product-oriented local coding assistant.

This profile is the default for the explicit `harness run` product command. It
keeps the single-owner main-agent model while sharing the same runtime
permissions, workspace snapshots, session events, and planning tools as the
terminal profile.
"""
from __future__ import annotations

from ..tracking_policy import TASK_TRACKING_POLICY
from .base import (
    AgentConfig,
    BaseProfile,
    build_profile_prompt,
)


class CodingAgentProfile(BaseProfile):
    def name(self) -> str:
        return "coding-agent"

    def description(self) -> str:
        return "Work in a local repository with sessions, permissions, tools, and verification"

    def main_agent(self) -> AgentConfig:
        return AgentConfig(
            system_prompt=build_profile_prompt(
                role="Implement the user's request in the repository and own the result.",
                working_style=(
                    "Inspect relevant code before editing. Reuse existing abstractions and make the "
                    "smallest complete change. Prefer code_intelligence when semantic navigation helps. "
                    "Reproduce bugs with tests before fixing them and cover changed behavior when the "
                    "repository has a suitable test seam.\n\n"
                    f"{TASK_TRACKING_POLICY}\n\n"
                    "Use repository tools to inspect and the shell to run code. Diagnose command failures "
                    "and adjust your approach. Review and apply worker proposals before integration."
                ),
                boundaries=(
                    "Preserve existing user changes. Keep changes in scope, avoid speculative extensions, "
                    "and avoid destructive Git operations unless the user requests them."
                ),
                completion=(
                    "Verify the requested behavior in proportion to risk. Fix in-scope failures and "
                    "report what changed, what you tested, and what remains unverified."
                ),
            ),
        )
