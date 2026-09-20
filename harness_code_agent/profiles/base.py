"""Base profile interface for main-agent task modes.

A Profile encapsulates everything scenario-specific:
  - The main-agent prompt
  - Extra tools for that agent
  - Task-specific timeout metadata

Configuration hierarchy (highest priority wins):
  1. Environment variables: PROFILE_<PROFILE_NAME>_<KEY> (e.g. PROFILE_TERMINAL_TASK_BUDGET=1200)
  2. ProfileConfig passed to constructor
  3. Profile subclass defaults
  4. BaseProfile defaults
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..runtime.permissions import (
    TOOL_PERMISSION_CONTROL,
    TOOL_PERMISSION_EDIT,
    TOOL_PERMISSION_NETWORK_READ,
    TOOL_PERMISSION_READ,
    TOOL_PERMISSION_SHELL,
    PermissionPreset,
)
from ..tracking_policy import TASK_TRACKING_POLICY

DEFAULT_PROFILE_TOOL_PERMISSIONS = {
    TOOL_PERMISSION_READ,
    TOOL_PERMISSION_NETWORK_READ,
    TOOL_PERMISSION_EDIT,
    TOOL_PERMISSION_CONTROL,
    TOOL_PERMISSION_SHELL,
}
DEFAULT_PROFILE_BLOCKED_TOOLS = {"browser_test", "stop_dev_server"}


def build_profile_prompt(
    *,
    role: str,
    working_style: str,
    boundaries: str,
    completion: str,
) -> str:
    """Render profile-local behavior in one predictable, readable shape."""
    return (
        f"## Role\n{role.strip()}\n\n"
        f"## Working Style\n{working_style.strip()}\n\n"
        f"## Boundaries\n{boundaries.strip()}\n\n"
        f"## Completion\n{completion.strip()}"
    )


@dataclass
class AgentConfig:
    """Configuration for the main agent."""
    system_prompt: str
    allowed_tool_permissions: set[str] = field(default_factory=lambda: set(DEFAULT_PROFILE_TOOL_PERMISSIONS))
    blocked_tool_names: set[str] = field(default_factory=lambda: set(DEFAULT_PROFILE_BLOCKED_TOOLS))
    time_budget: float | None = None  # seconds; None = no limit
    # Profile-scoped restriction layered on the session permission policy.
    # A preset can only tighten the user's session mode, never loosen it.
    permission_preset: PermissionPreset | None = None


@dataclass
class ProfileConfig:
    """
    Tunable parameters for a profile, separated from code.

    Only profiles that run under an external benchmark with a hard wall-clock
    limit (e.g. terminal/TB2) set ``task_budget``. Interactive product
    profiles leave it unset.
    """
    # --- Time budget (seconds) ---
    task_budget: float | None = None          # hard task timeout; None = no limit

    def _env_key(self, profile_name: str, field_name: str) -> str:
        """Build environment variable name: PROFILE_TERMINAL_TASK_BUDGET."""
        return f"PROFILE_{profile_name.upper().replace('-', '_')}_{field_name.upper()}"

    def resolve(self, field_name: str, profile_name: str, default):
        """
        Resolve a config value with priority: env var > explicit config > default.
        """
        # Check environment variable
        env_key = self._env_key(profile_name, field_name)
        env_val = os.environ.get(env_key)
        if env_val is not None:
            # Coerce to the type of default
            if isinstance(default, float):
                return float(env_val)
            elif isinstance(default, int):
                return int(env_val)
            return env_val

        # Check explicit config value
        config_val = getattr(self, field_name, None)
        if config_val is not None:
            return config_val

        return default


class BaseProfile(ABC):
    """
    Abstract base for task profiles.

    Subclass this to create a new scenario (app building, terminal tasks,
    review, etc.). The harness calls these methods to get
    scenario-specific configuration.

    Accepts an optional ProfileConfig for tunable parameters.
    Subclasses read config via self.cfg.resolve(field, profile_name, default).
    """

    def __init__(self, cfg: ProfileConfig | None = None):
        self.cfg = cfg or ProfileConfig()

    @abstractmethod
    def name(self) -> str:
        """Short identifier for this profile (e.g. 'app-builder', 'terminal')."""
        ...

    @abstractmethod
    def description(self) -> str:
        """One-line description shown in --help."""
        ...

    def main_agent(self) -> AgentConfig:
        """Config for the single owner agent that runs the full task loop."""
        prompt = build_profile_prompt(
            role=(
                "Own the complete task loop: understand the request, inspect the workspace, "
                "make the required changes, integrate useful delegated findings, verify the result, "
                "and decide when the work is complete."
            ),
            working_style=(
                "Begin from the task and current repository state. Use the planning policy below "
                "to match coordination overhead to risk, then follow existing project patterns and "
                "keep the implementation focused.\n\n"
                f"{TASK_TRACKING_POLICY}\n\n"
                "Use delegation only when independent investigation, test design, review, verification, "
                "or an isolated worker proposal would reduce risk or context load. Review worker changes "
                "before explicitly applying them. Long-running "
                "shell commands return job IDs; inspect and clean them up through the shell-job tools."
            ),
            boundaries=(
                "The task text is the source of truth. Delegation is evidence or an isolated "
                "proposal, not completed work. Keep integration, final verification, and the stop "
                "decision with the main agent."
            ),
            completion=(
                "Make any required code or test changes yourself; delegated findings or proposals are "
                "not completed work until you integrate them. Run concrete verification commands and "
                "read their output; if anything fails, diagnose the evidence and continue. Keep your "
                "todo list aligned with the remaining work, and mark the final items complete only "
                "after verification passes."
            ),
        )
        return AgentConfig(system_prompt=prompt)

    def resolve_task_timeout(self, user_prompt: str) -> float | None:
        """
        Resolve the actual timeout for a task based on the user prompt.

        Override in subclasses that have task-specific timeout metadata
        (e.g. terminal profile uses TB2 task.toml data).

        Returns timeout in seconds, or None to use the default budget.
        """
        return None

    def resolve_task_metadata(self, user_prompt: str) -> dict | None:
        """
        Resolve profile-specific task metadata.

        Profiles can expose benchmark/task metadata to runtime middleware without
        making those middleware depend on a concrete benchmark launcher.
        """
        return None
