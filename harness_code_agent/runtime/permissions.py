"""Permission policy: the single place that turns tool-call facts into decisions.

The analyzer (:mod:`runtime.shell_classification`) reports facts only; this
module maps those facts onto ``allow / ask / deny`` according to the active
:class:`~runtime.shell_classification.SandboxMode` / permission mode preset.

Catastrophic shell patterns (``rm -rf /``, ``git reset --hard`` ...) are
denied in EVERY mode, including full access.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from .shell_classification import (
    SandboxMode,
    ShellEffect,
    ShellTrait,
    TargetScope,
    analyze_shell_command,
    is_workspace_write_shell_command,
)

TOOL_PERMISSION_READ = "read"
TOOL_PERMISSION_NETWORK_READ = "network_read"
TOOL_PERMISSION_EDIT = "edit"
TOOL_PERMISSION_CONTROL = "control"
TOOL_PERMISSION_SHELL = "shell"
TOOL_PERMISSION_DANGEROUS = "dangerous"
VALID_TOOL_PERMISSIONS = {
    TOOL_PERMISSION_READ,
    TOOL_PERMISSION_NETWORK_READ,
    TOOL_PERMISSION_EDIT,
    TOOL_PERMISSION_CONTROL,
    TOOL_PERMISSION_SHELL,
    TOOL_PERMISSION_DANGEROUS,
}


class PermissionAction(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class ShellPermissionConfig:
    """Action for each discretionary shell effect bucket."""

    workspace_write: PermissionAction
    external_write: PermissionAction
    system_write: PermissionAction
    destructive_git: PermissionAction
    unknown_execution: PermissionAction


@dataclass
class PermissionDecision:
    action: str
    risk: str
    reason: str
    analysis: object | None = None

    @property
    def allowed(self) -> bool:
        return self.action == PermissionAction.ALLOW.value

    @property
    def requires_approval(self) -> bool:
        return self.action == PermissionAction.ASK.value


_READ_ONLY_ROLE_NAMES = frozenset(
    {"explorer", "test_designer", "reviewer", "verifier"}
)


class PermissionPolicy:
    """Runtime-enforced permission policy for tool calls."""

    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    LLM_AUTO = "llm-auto"
    DANGER_FULL_ACCESS = "danger-full-access"
    EVAL = "terminal-eval"
    VALID_MODES: ClassVar[set[str]] = {
        READ_ONLY,
        WORKSPACE_WRITE,
        LLM_AUTO,
        DANGER_FULL_ACCESS,
        EVAL,
    }

    _SHELL_PRESETS: ClassVar[dict[str, ShellPermissionConfig]] = {
        READ_ONLY: ShellPermissionConfig(
            PermissionAction.DENY,
            PermissionAction.DENY,
            PermissionAction.DENY,
            PermissionAction.DENY,
            PermissionAction.DENY,
        ),
        WORKSPACE_WRITE: ShellPermissionConfig(
            PermissionAction.ASK,
            PermissionAction.DENY,
            PermissionAction.DENY,
            PermissionAction.DENY,
            PermissionAction.ASK,
        ),
        LLM_AUTO: ShellPermissionConfig(
            PermissionAction.ASK,
            PermissionAction.DENY,
            PermissionAction.DENY,
            PermissionAction.DENY,
            PermissionAction.ASK,
        ),
        DANGER_FULL_ACCESS: ShellPermissionConfig(
            PermissionAction.ALLOW,
            PermissionAction.ALLOW,
            PermissionAction.ALLOW,
            PermissionAction.DENY,
            PermissionAction.ALLOW,
        ),
        EVAL: ShellPermissionConfig(
            PermissionAction.ALLOW,
            PermissionAction.DENY,
            PermissionAction.DENY,
            PermissionAction.DENY,
            PermissionAction.ALLOW,
        ),
    }

    def __init__(
        self,
        mode: str = WORKSPACE_WRITE,
        *,
        shell_config: ShellPermissionConfig | None = None,
        sandbox_mode: str | None = None,
        role: str | None = None,
        allowed_paths: tuple[str, ...] | list[str] | None = None,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown permission mode: {mode}")
        self.mode = mode
        self.role = role
        self.allowed_paths = tuple(allowed_paths or ())
        self.sandbox_mode = sandbox_mode or resolve_sandbox_mode()
        if self.sandbox_mode not in {m.value for m in SandboxMode}:
            self.sandbox_mode = SandboxMode.HOST.value
        # Backward compatibility: eval runs used to combine danger-full-access
        # with HCA_TERMINAL_EVAL_MODE=1 to relax container path handling.
        if mode == self.DANGER_FULL_ACCESS and _terminal_eval_enabled():
            self._terminal_eval = True
            shell_config = shell_config or self._SHELL_PRESETS[self.EVAL]
        else:
            self._terminal_eval = False
        self.shell_config = shell_config or self._SHELL_PRESETS[mode]

    # ------------------------------------------------------------------
    # Construction for delegated agents
    # ------------------------------------------------------------------

    @classmethod
    def for_role(
        cls,
        role: str,
        *,
        allowed_paths: tuple[str, ...] | list[str] | None = None,
        sandbox_mode: str | None = None,
    ) -> PermissionPolicy:
        """Policy for a delegated agent role.

        Read-only roles may only inspect/verify; workers may write inside their
        declared allowed_paths but never touch external/system state.
        """
        if role in _READ_ONLY_ROLE_NAMES:
            return cls(
                cls.READ_ONLY,
                role=role,
                sandbox_mode=sandbox_mode,
            )
        if role == "worker":
            return cls(
                cls.EVAL,
                shell_config=ShellPermissionConfig(
                    PermissionAction.ALLOW,
                    PermissionAction.DENY,
                    PermissionAction.DENY,
                    PermissionAction.DENY,
                    PermissionAction.DENY,
                ),
                role="worker",
                allowed_paths=allowed_paths,
                sandbox_mode=sandbox_mode,
            )
        raise ValueError(f"Unknown agent role: {role}")

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def decide_tool_call(
        self,
        tool_name: str,
        args: dict | None = None,
        tool_permission: str | None = None,
        workspace_root: str | None = None,
    ) -> PermissionDecision:
        args = args or {}
        permission = tool_permission or _builtin_permission(tool_name)

        if permission == TOOL_PERMISSION_SHELL:
            return self._decide_shell(tool_name, args, workspace_root)
        return self._decide_structured_tool(tool_name, args, permission)

    # ------------------------------------------------------------------
    # Shell
    # ------------------------------------------------------------------

    def _decide_shell(
        self, tool_name: str, args: dict, workspace_root: str | None
    ) -> PermissionDecision:
        command = str(args.get("command", ""))
        analysis = analyze_shell_command(
            command, workspace_root, self.sandbox_mode
        )
        effects = analysis.effects
        traits = analysis.traits
        cfg = self.shell_config

        # Catastrophic guardrail: denied in every mode, including full access.
        if ShellEffect.GIT_MUTATION in effects and ShellTrait.DESTRUCTIVE in traits:
            return PermissionDecision(
                PermissionAction.DENY.value,
                "shell_blocked",
                "destructive_git_operation",
                analysis=analysis,
            )
        writes = effects & {ShellEffect.WRITE, ShellEffect.DELETE}
        system_target = any(t.scope is TargetScope.SYSTEM for t in analysis.targets)
        if ShellTrait.DESTRUCTIVE in traits and system_target and (
            writes or ShellEffect.EXECUTE in effects
        ):
            return PermissionDecision(
                PermissionAction.DENY.value,
                "shell_blocked",
                "destructive_system_target",
                analysis=analysis,
            )

        # Scope-driven write/delete decisions.
        if writes:
            scopes = {t.scope for t in analysis.targets}
            if TargetScope.SYSTEM in scopes:
                action, reason = cfg.system_write, "shell command writes system paths"
            elif TargetScope.EXTERNAL in scopes or TargetScope.UNKNOWN in scopes:
                action, reason = cfg.external_write, "shell command mutates external paths"
            else:
                action, reason = cfg.workspace_write, "shell command writes inside the workspace"
        elif ShellEffect.GIT_MUTATION in effects:
            if ShellEffect.NETWORK in effects:
                action, reason = cfg.external_write, "shell command mutates git remote state"
            else:
                action, reason = cfg.workspace_write, "shell command mutates local git state"
        elif ShellTrait.UNKNOWN_EFFECT in traits:
            action, reason = (
                cfg.unknown_execution,
                "shell executes a program whose effects cannot be determined",
            )
        else:
            action, reason = (
                PermissionAction.ALLOW,
                f"{self.mode} mode allows read-only shell commands",
            )

        risk = "shell_safe" if action is PermissionAction.ALLOW else "shell_risky"
        return PermissionDecision(action.value, risk, reason, analysis=analysis)

    # ------------------------------------------------------------------
    # Structured (non-shell) tools
    # ------------------------------------------------------------------

    def _decide_structured_tool(
        self, tool_name: str, args: dict, permission: str | None
    ) -> PermissionDecision:
        if permission == TOOL_PERMISSION_READ:
            return PermissionDecision(
                PermissionAction.ALLOW.value, "read", f"{self.mode} mode allows read tools"
            )
        if permission == TOOL_PERMISSION_NETWORK_READ:
            return PermissionDecision(
                PermissionAction.ALLOW.value,
                "network_read",
                f"{self.mode} mode allows network read tools",
            )
        if permission == TOOL_PERMISSION_EDIT:
            blocked = self._edit_path_block(tool_name, args)
            if blocked is not None:
                return PermissionDecision(
                    PermissionAction.DENY.value, "edit", blocked
                )
            if self.mode == self.READ_ONLY:
                return PermissionDecision(
                    PermissionAction.DENY.value,
                    "edit",
                    "read-only mode does not allow file edits",
                )
            return PermissionDecision(
                PermissionAction.ALLOW.value, "edit", f"{self.mode} mode allows file edits"
            )
        if permission == TOOL_PERMISSION_CONTROL:
            return PermissionDecision(
                PermissionAction.ALLOW.value,
                "control",
                f"{self.mode} mode allows control tools",
            )
        if permission == TOOL_PERMISSION_DANGEROUS:
            if self.mode in {self.DANGER_FULL_ACCESS, self.EVAL} or self._terminal_eval:
                return PermissionDecision(
                    PermissionAction.ALLOW.value,
                    "dangerous",
                    f"{self.mode} mode allows this tool call",
                )
            if self.mode == self.READ_ONLY:
                return PermissionDecision(
                    PermissionAction.DENY.value,
                    "dangerous",
                    "read-only mode does not allow dangerous tools",
                )
            return PermissionDecision(
                PermissionAction.ASK.value,
                "dangerous",
                f"{self.mode} mode requires approval for dangerous tools",
            )
        # Unknown / undeclared tool.
        if self.mode in {self.DANGER_FULL_ACCESS, self.EVAL} or self._terminal_eval:
            return PermissionDecision(
                PermissionAction.ALLOW.value, "unknown", f"{self.mode} mode allows this tool call"
            )
        if self.mode == self.READ_ONLY:
            return PermissionDecision(
                PermissionAction.DENY.value,
                "unknown",
                "read-only mode does not allow undeclared tools",
            )
        return PermissionDecision(
            PermissionAction.ASK.value,
            "unknown",
            f"{self.mode} mode requires approval for undeclared tools",
        )

    def _edit_path_block(self, tool_name: str, args: dict) -> str | None:
        """Role / allowed_paths enforcement for structured edit tools."""
        if self.role in _READ_ONLY_ROLE_NAMES and tool_name in {"write_file", "apply_patch"}:
            return f"{self.role} role cannot modify the workspace"
        if self.role == "worker" and tool_name in {"write_file", "apply_patch"}:
            path = _normalize_rel_path(str(args.get("path") or ""))
            if not path or not _path_allowed(path, self.allowed_paths):
                return f"write path is outside allowed_paths: {path or '<empty>'}"
        return None


def is_workspace_write_command(command: str, workspace_root: str | None = None) -> bool:
    """Check only the workspace-write boundary, without a read command allowlist."""
    return is_workspace_write_shell_command(command, workspace_root)


def resolve_sandbox_mode() -> str:
    """Effective sandbox mode: explicit config wins; eval flag implies docker."""
    from .. import config

    if str(getattr(config, "SANDBOX_MODE", "host")).lower() == SandboxMode.DOCKER.value:
        return SandboxMode.DOCKER.value
    if _terminal_eval_enabled():
        return SandboxMode.DOCKER.value
    return SandboxMode.HOST.value


def _terminal_eval_enabled() -> bool:
    return (
        os.environ.get("HCA_TERMINAL_EVAL_MODE", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )


def _normalize_rel_path(value: str) -> str:
    value = str(value or "").replace("\\", "/").removeprefix("./")
    return value.rstrip("/") or "."


def _path_allowed(path: str, allowed: tuple[str, ...]) -> bool:
    if not allowed:
        return False
    needle = path.strip("/").split("/")
    for root in allowed:
        base = root.strip("/").split("/")
        if needle[: len(base)] == base:
            return True
    return False


def _builtin_permission(tool_name: str) -> str | None:
    from .builtins.registry import BUILTIN_TOOL_REGISTRY

    return BUILTIN_TOOL_REGISTRY.permission_for(tool_name)
