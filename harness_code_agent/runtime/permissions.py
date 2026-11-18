"""Permission policy: the single place that turns tool-call facts into decisions.

The analyzer (:mod:`runtime.shell_classification`) reports facts only; this
module maps those facts onto ``allow / ask / deny`` according to the active
permission mode preset.

Two rules shape everything here:

* Every fact contributes a candidate action, and the **strictest** candidate
  wins.  An unknown effect can therefore never be outvoted by the proven
  read/write half of the same command line.
* Only host-destructive shapes (system/disk destruction) are denied in EVERY
  mode.  Destructive git operations and network access are approval questions;
  unknown programs run inside the sandbox without manufacturing an approval.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields, replace
from enum import Enum
from pathlib import Path
from typing import ClassVar

from ..workspace.paths import is_same_or_child, normalize_rel_path, path_within
from .agent_roles import role_definition
from .shell_classification import (
    ShellEffect,
    ShellTarget,
    ShellTrait,
    TargetScope,
    analyze_shell_command,
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


SANDBOX_PERMISSION_USE_DEFAULT = "use_default"
SANDBOX_PERMISSION_ADDITIONAL = "with_additional_permissions"
_SANDBOX_PERMISSION_VALUES = {
    SANDBOX_PERMISSION_USE_DEFAULT,
    SANDBOX_PERMISSION_ADDITIONAL,
}


@dataclass(frozen=True)
class ShellAuthorization:
    """The exact execution capability one approved shell call may use.

    Produced only by the policy and the approval result; callers cannot widen
    it through ordinary tool arguments. Executors bind the sandbox to this
    snapshot instead of reading mutable session state.
    """

    mode: str  # ``read-only``, ``workspace-write`` or ``danger-full-access``
    writable_paths: tuple[str, ...] = ()
    network: bool = False
    fingerprint: str = ""


#: Order used to pick the strictest candidate for one command line.
_STRICTNESS = {
    PermissionAction.ALLOW: 0,
    PermissionAction.ASK: 1,
    PermissionAction.DENY: 2,
}


def _stricter(left: PermissionAction, right: PermissionAction) -> PermissionAction:
    """The more restrictive of two actions (``allow < ask < deny``)."""
    return left if _STRICTNESS[left] >= _STRICTNESS[right] else right


#: Targets whose removal takes the whole workspace with it.
_WHOLE_WORKSPACE_TARGETS = {".", "./", ".\\", "*", "./*", "./**", "**"}

#: Deny reasons that describe a catastrophic command shape (blocked in every
#: mode, never approvable).
_CATASTROPHIC_REASONS = {"destructive_system_target"}


@dataclass(frozen=True)
class ShellPermissionConfig:
    """Action for each discretionary shell effect bucket."""

    workspace_write: PermissionAction
    external_write: PermissionAction
    system_write: PermissionAction
    destructive_git: PermissionAction
    unknown_execution: PermissionAction
    verify_execution: PermissionAction = PermissionAction.DENY
    #: Ordinary network use (fetching a URL, a package index).
    network_read: PermissionAction = PermissionAction.DENY
    #: Sending local data out: uploads, POST/PUT bodies, forms.
    network_upload: PermissionAction = PermissionAction.DENY
    #: A delete that takes the whole workspace tree with it.
    broad_delete: PermissionAction = PermissionAction.DENY


@dataclass
class PermissionDecision:
    action: str
    risk: str
    reason: str
    analysis: object | None = None
    #: The request cannot be handed to the command reviewer at all (for
    #: example, credentials that could not be redacted safely). Ordinary risky
    #: commands do not set this: the reviewer is the decision-maker in
    #: ``llm-auto`` mode.
    requires_human: bool = False
    #: What exactly was decided. Execution re-checks it so an approval can
    #: never be replayed onto different arguments.
    fingerprint: str | None = None
    #: Whether the granted shell command may keep host network access. The
    #: executor binds the sandbox to this decision instead of re-analyzing the
    #: command text; the analyzer only supplied the facts behind it.
    network_access: bool = False
    #: The exact sandbox capability this call may use once approved.
    authorization: ShellAuthorization | None = None

    @property
    def allowed(self) -> bool:
        return self.action == PermissionAction.ALLOW.value

    @property
    def requires_approval(self) -> bool:
        return self.action == PermissionAction.ASK.value


#: Sandbox and approval policy shared by the two writing UI presets.
#:
#: The mode name is the grant: commands whose writes the analyzer can prove
#: stay inside the workspace run without another prompt. Anything it cannot
#: prove (unknown executables/scripts, dynamic targets), writes outside the
#: workspace, and system writes still ask/deny.
#:
#: ``workspace-write`` and ``llm-auto`` use this exact configuration; they
#: differ only in who answers an ``ask`` (see
#: ``core.interactive.InteractiveSession._approval_provider_for_mode``).
_WORKSPACE_WRITE_SHELL = ShellPermissionConfig(
    PermissionAction.ALLOW,
    PermissionAction.DENY,
    PermissionAction.DENY,
    # Destructive git operations are an approval question, not a permanent ban.
    PermissionAction.ASK,
    # An unknown program is bounded by the sandbox; semantic uncertainty alone
    # does not manufacture an approval request (Codex's on-request behaviour).
    PermissionAction.ALLOW,
    PermissionAction.ALLOW,
    # Network access is a separate grant: default workspace-write is offline.
    PermissionAction.ASK,
    PermissionAction.ASK,
    PermissionAction.ASK,
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
            PermissionAction.DENY,
            PermissionAction.ALLOW,
            PermissionAction.DENY,
            PermissionAction.DENY,
        ),
        WORKSPACE_WRITE: _WORKSPACE_WRITE_SHELL,
        # Same sandbox, same approval policy; only the approvals reviewer is
        # the command-review model instead of the user.
        LLM_AUTO: _WORKSPACE_WRITE_SHELL,
        DANGER_FULL_ACCESS: ShellPermissionConfig(
            PermissionAction.ALLOW,
            PermissionAction.ALLOW,
            PermissionAction.ALLOW,
            PermissionAction.ALLOW,
            PermissionAction.ALLOW,
            PermissionAction.ALLOW,
            PermissionAction.ALLOW,
            PermissionAction.ALLOW,
            # Full access is a standing grant: it answers allow or deny, never
            # "ask", so nothing is reviewed and nothing waits on a person.
            PermissionAction.ALLOW,
        ),
        EVAL: ShellPermissionConfig(
            PermissionAction.ALLOW,
            PermissionAction.DENY,
            PermissionAction.DENY,
            PermissionAction.ALLOW,
            PermissionAction.ALLOW,
            PermissionAction.ALLOW,
            PermissionAction.ALLOW,
            PermissionAction.ASK,
            PermissionAction.ASK,
        ),
    }

    def __init__(
        self,
        mode: str = WORKSPACE_WRITE,
        *,
        shell_config: ShellPermissionConfig | None = None,
        role: str | None = None,
        allowed_paths: tuple[str, ...] | list[str] | None = None,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown permission mode: {mode}")
        self.mode = mode
        self.role = role
        self.allowed_paths = tuple(allowed_paths or ())
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
    ) -> PermissionPolicy:
        """Policy for a delegated agent role.

        Read-only roles may only inspect/verify; workers may write inside their
        declared allowed_paths but never touch external/system state.
        """
        definition = role_definition(role)
        if definition.read_only:
            return cls(
                cls.READ_ONLY,
                role=role,
            )
        return cls(
            cls.EVAL,
            shell_config=ShellPermissionConfig(
                PermissionAction.ALLOW,
                PermissionAction.DENY,
                PermissionAction.DENY,
                PermissionAction.DENY,
                PermissionAction.DENY,
                PermissionAction.ALLOW,
                PermissionAction.ALLOW,
                PermissionAction.DENY,
            ),
            role=role,
            allowed_paths=allowed_paths,
        )

    @classmethod
    def delegated_from(
        cls,
        parent: PermissionPolicy | None,
        role: str,
        *,
        allowed_paths: tuple[str, ...] | list[str] | None = None,
    ) -> PermissionPolicy:
        """Policy for a delegated agent, never broader than its parent's.

        A read-only parent (an active planning flow) cannot grant write access
        by delegation, so worker roles inherit the read-only ceiling instead of
        their usual sandbox-write policy. Every other parent caps the delegated
        agent per shell bucket: the child keeps its own role restrictions, but
        no action may be more permissive than the parent's (a parent that must
        ask before network access cannot delegate that away).
        """
        if parent is not None and parent.mode == cls.READ_ONLY:
            return cls(cls.READ_ONLY, role=role, allowed_paths=allowed_paths)
        policy = cls.for_role(role, allowed_paths=allowed_paths)
        if parent is None:
            return policy
        own = policy.shell_config
        ceiling = parent.shell_config
        capped = ShellPermissionConfig(
            **{
                entry.name: _stricter(
                    getattr(own, entry.name), getattr(ceiling, entry.name)
                )
                for entry in fields(ShellPermissionConfig)
            }
        )
        return cls(
            policy.mode,
            shell_config=capped,
            role=role,
            allowed_paths=allowed_paths,
        )

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def decide_tool_call(
        self,
        tool_name: str,
        args: dict | None = None,
        tool_permission: str | None = None,
        workspace_root: str | None = None,
        *,
        shell_analysis=None,
    ) -> PermissionDecision:
        args = args or {}
        permission = tool_permission or _builtin_permission(tool_name)

        if permission == TOOL_PERMISSION_SHELL:
            return self._decide_shell(
                tool_name, args, workspace_root, shell_analysis=shell_analysis
            )
        return self._decide_structured_tool(tool_name, args, permission, workspace_root)

    def shell_analysis_root(self, workspace_root: str | None):
        """The root the shell analyzer must use for this policy."""
        return "/" if self.mode == self.EVAL and self.role is None else workspace_root

    # ------------------------------------------------------------------
    # Shell
    # ------------------------------------------------------------------

    def _decide_shell(
        self, tool_name: str, args: dict, workspace_root: str | None, *, shell_analysis=None
    ) -> PermissionDecision:
        command = str(args.get("command", ""))
        root = self.shell_analysis_root(workspace_root)
        analysis = (
            shell_analysis
            if shell_analysis is not None
            else analyze_shell_command(command, root)
        )
        effects = analysis.effects
        traits = analysis.traits
        if ShellTrait.ANALYSIS_FAILED in traits:
            return PermissionDecision(
                PermissionAction.DENY.value,
                "shell_risky",
                "shell analysis failed; command was not executed",
                analysis=analysis,
            )
        cfg = self.shell_config

        candidates: list[tuple[PermissionAction, str]] = []

        def add(action: PermissionAction, reason: str) -> None:
            candidates.append((action, reason))

        # Validated here (the candidate is added after the facts so it can
        # explain the final decision), because declared extra directories also
        # change which write targets are considered covered.
        authorization, permission_action, permission_reason = self._shell_authorization(args)
        extra_dirs = authorization.writable_paths if authorization is not None else ()

        writes = effects & {ShellEffect.WRITE, ShellEffect.DELETE}
        write_targets = analysis.write_targets
        write_scopes = {target.scope for target in write_targets}
        system_write = TargetScope.SYSTEM in write_scopes
        external_write = bool(write_scopes & {TargetScope.EXTERNAL, TargetScope.UNKNOWN})
        uncovered_external = any(
            target.scope in {TargetScope.EXTERNAL, TargetScope.UNKNOWN}
            and not _covered_by_extra(target, extra_dirs, root)
            for target in write_targets
        )

        # 1. Catastrophic guardrail: denied in every mode. The analyzer binds
        # this trait to the command that owns the system target, so a workspace
        # delete followed by reading /etc/hosts is not promoted to a system
        # deletion.
        if ShellEffect.GIT_MUTATION in effects and ShellTrait.DESTRUCTIVE in traits:
            add(cfg.destructive_git, "destructive_git_operation")
        if ShellTrait.CATASTROPHIC in traits:
            add(PermissionAction.DENY, "destructive_system_target")

        # 2. Write/delete scope.
        scope_action = PermissionAction.ALLOW
        if writes:
            if system_write:
                scope_action = cfg.system_write
                add(scope_action, "shell command writes system paths")
            elif external_write and uncovered_external:
                scope_action = cfg.external_write
                add(
                    scope_action,
                    "shell command mutates paths it cannot prove are inside the workspace",
                )
            elif write_scopes == {TargetScope.TEMP}:
                # The job manager points TEMP/TMP/TMPDIR at the per-call private
                # directory and grants write access there. It follows the same
                # write gate as the workspace: read-only mode still denies it
                # because the sandbox grants no temp directory in that mode.
                add(cfg.workspace_write, "shell command writes only its private temp directory")
            else:
                scope_action = cfg.workspace_write
                add(scope_action, "shell command writes inside the workspace")
        # A delegated worker may only write the directories it was assigned.
        worker_block = self._worker_write_block(analysis, root)
        if worker_block is not None:
            add(PermissionAction.DENY, worker_block)
        if (
            ShellEffect.GIT_MUTATION in effects
            and self.role is not None
            and self.allowed_paths
            and not path_within(".", self.allowed_paths)
        ):
            # A git mutation can touch any tracked file, so it is only inside a
            # worker's scope when that worker owns the whole workspace.
            add(PermissionAction.DENY, "git mutation can touch files outside allowed_paths")
        if ShellEffect.GIT_MUTATION in effects:
            git_write_targets = [t for t in write_targets if t.scope is not TargetScope.WORKSPACE]
            uncovered_git = any(
                not _covered_by_extra(target, extra_dirs, root)
                for target in git_write_targets
            )
            if TargetScope.SYSTEM in write_scopes:
                add(cfg.system_write, "shell command mutates a system git repository")
            elif uncovered_git:
                add(
                    cfg.external_write,
                    "shell command mutates a git repository outside the workspace",
                )
            elif ShellEffect.NETWORK in effects:
                add(cfg.network_upload, "shell command mutates git remote state")
            else:
                add(cfg.workspace_write, "shell command mutates local git state")

        # 3. A deletion this size is not recoverable by re-reading files, so it
        # follows its own bucket: ask in the modes that review, allow only in
        # full access. Which gate answers is the mode's business, not ours.
        if ShellEffect.DELETE in effects and _is_broad_delete(analysis):
            add(cfg.broad_delete, "shell command removes a whole workspace tree")

        # 4. Network is its own grant: reading a URL is one thing, moving local
        # bytes out is another, and the reviewer has to look at it either way.
        if ShellTrait.NETWORK_UPLOAD in traits:
            add(
                cfg.network_upload,
                "shell sends local data over the network (upload or POST body)",
            )
        elif ShellEffect.NETWORK in effects:
            add(cfg.network_read, "shell command uses the network")

        # 5. Unknown effects are never outvoted by a proven read/write half.
        if ShellTrait.UNKNOWN_EFFECT in traits:
            add(
                cfg.unknown_execution,
                "shell executes a program whose effects cannot be determined",
            )
        elif ShellTrait.VERIFY in traits:
            # Verification skips the repeated *command* approval only; it buys
            # no extra write, system or network capability.
            add(cfg.verify_execution, "shell runs project verification or a build task")

        # 6. Write scope declared by the caller.
        if args.get("write_paths"):
            write_decision = self._declared_write_scope(args, analysis, root)
            if write_decision is not None:
                add(write_decision[0], write_decision[1])

        # 7. A one-time capability request is its own reason: the approval (or
        # refusal) is about the requested permission, not only the command.
        if permission_action is not None:
            add(permission_action, permission_reason)

        if not candidates:
            add(PermissionAction.ALLOW, f"{self.mode} mode allows read-only shell commands")

        action = max(
            (candidate[0] for candidate in candidates), key=lambda item: _STRICTNESS[item]
        )
        reason = _reason_for(candidates, action)
        risk = "shell_safe" if action is PermissionAction.ALLOW else "shell_risky"
        if action is PermissionAction.DENY and reason in _CATASTROPHIC_REASONS:
            risk = "shell_blocked"
        # Network capability follows the decision, not a second reading of the
        # command: only a command that actually uses the network, or a call
        # that explicitly requested network access, keeps it.
        network_access = bool(
            ShellEffect.NETWORK in effects
            or (authorization is not None and authorization.network)
        )
        if authorization is not None:
            authorization = replace(authorization, network=network_access)
        return PermissionDecision(
            action.value,
            risk,
            reason,
            analysis=analysis,
            fingerprint=shell_call_fingerprint(args),
            network_access=network_access,
            authorization=authorization,
        )

    # -- one-time sandbox permissions -------------------------------------

    def _shell_authorization(self, args: dict):
        """Turn a per-call permission request into an approval decision.

        The request is data from the model; the policy validates it, the
        reviewer approves it, and the returned snapshot is the only thing the
        executor may act on. Requests the backend cannot honour are denied
        instead of being approved and silently ignored.
        """
        requested = str(
            args.get("sandbox_permissions") or SANDBOX_PERMISSION_USE_DEFAULT
        ).strip().lower()
        additional = args.get("additional_permissions") or {}
        default_mode = (
            "danger-full-access"
            if self.mode == self.DANGER_FULL_ACCESS
            else "read-only"
            if self.mode == self.READ_ONLY
            else "workspace-write"
        )
        base = ShellAuthorization(
            mode=default_mode, fingerprint=shell_call_fingerprint(args)
        )
        if requested not in _SANDBOX_PERMISSION_VALUES:
            return (
                base,
                PermissionAction.DENY,
                f"unsupported sandbox_permissions: {requested}",
            )
        if requested == SANDBOX_PERMISSION_USE_DEFAULT:
            return base, None, ""
        if self.mode == self.READ_ONLY:
            return (
                base,
                PermissionAction.DENY,
                "read-only mode cannot request additional permissions",
            )
        if self.role is not None:
            return (
                base,
                PermissionAction.DENY,
                "a delegated agent cannot request additional permissions",
            )
        if not isinstance(additional, dict):
            return (
                base,
                PermissionAction.DENY,
                "additional_permissions must be an object",
            )
        unknown = set(additional) - {"file_system", "network"}
        if unknown:
            return (
                base,
                PermissionAction.DENY,
                "unsupported permission type: " + ", ".join(sorted(unknown)),
            )
        write_dirs: tuple[str, ...] = ()
        file_system = additional.get("file_system") or {}
        if not isinstance(file_system, dict):
            return (
                base,
                PermissionAction.DENY,
                "additional_permissions.file_system must be an object",
            )
        file_unknown = set(file_system) - {"write"}
        if file_unknown:
            return (
                base,
                PermissionAction.DENY,
                "unsupported file_system permission: "
                + ", ".join(sorted(file_unknown)),
            )
        if "write" in file_system:
            write_dirs, error = _validated_extra_dirs(file_system["write"])
            if error is not None:
                return base, PermissionAction.DENY, error
        network = additional.get("network")
        if network is not None and not isinstance(network, bool):
            return (
                base,
                PermissionAction.DENY,
                "additional_permissions.network must be a boolean",
            )

        if self.mode == self.DANGER_FULL_ACCESS:
            # The session already has everything the request asks for.
            return (
                ShellAuthorization(
                    mode="danger-full-access",
                    writable_paths=write_dirs,
                    network=True,
                    fingerprint=base.fingerprint,
                ),
                None,
                "",
            )

        if not write_dirs and not network:
            return (
                base,
                PermissionAction.DENY,
                "additional_permissions requests nothing",
            )
        return (
            ShellAuthorization(
                mode="workspace-write",
                writable_paths=write_dirs,
                network=bool(network),
                fingerprint=base.fingerprint,
            ),
            PermissionAction.ASK,
            "shell requests additional permissions: "
            + _describe_extra_permissions(write_dirs, bool(network)),
        )

    # -- write_paths ------------------------------------------------------

    def _declared_write_scope(
        self, args: dict, analysis, workspace_root: str | None
    ) -> tuple[PermissionAction, str] | None:
        """Validate the caller's declared writable directories.

        ``read-only`` cannot buy write access by declaring them; they must be
        inside the workspace; and a write the analyzer already pinned to a path
        outside every declared directory is denied here instead of failing
        inside the sandbox after an approval round trip.
        """
        values = args.get("write_paths") or []
        if self.mode == self.DANGER_FULL_ACCESS:
            # write_paths narrows the sandbox; full access has no sandbox to
            # narrow, so it must not reject otherwise valid absolute paths.
            return None
        resolved = self._resolve_write_paths(values, workspace_root)
        if isinstance(resolved, str):
            return PermissionAction.DENY, resolved
        if self.mode == self.READ_ONLY:
            return PermissionAction.DENY, "read-only mode cannot grant shell write directories"
        if self.role is not None and self.allowed_paths and workspace_root is not None:
            root = _normalized(workspace_root)
            for directory in resolved:
                relative = _workspace_relative(directory, root)
                if relative is None or not path_within(relative, self.allowed_paths):
                    return (
                        PermissionAction.DENY,
                        f"write_paths is outside allowed_paths: {relative or directory}",
                    )
        uncovered = [
            target.raw
            for target in analysis.write_targets
            if target.scope is TargetScope.WORKSPACE
            and not _covered_by(target, resolved, workspace_root)
        ]
        if uncovered:
            return (
                PermissionAction.DENY,
                "write target is outside the declared write_paths: " + ", ".join(uncovered[:3]),
            )
        return None

    @staticmethod
    def _resolve_write_paths(values, workspace_root: str | None) -> tuple[str, ...] | str:
        if not isinstance(values, (list, tuple)):
            return "write_paths must be a list of workspace directories"
        if workspace_root is None:
            return "write_paths requires a known workspace root"
        root = _normalized(workspace_root)
        resolved: list[str] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                return "write_paths entries must be non-empty workspace paths"
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = Path(workspace_root) / candidate
            normalized = _normalized(str(candidate))
            if not is_same_or_child(normalized, root):
                return f"write path is outside the workspace: {value}"
            resolved.append(normalized)
        return tuple(resolved)

    def _worker_write_block(self, analysis, workspace_root: str | None) -> str | None:
        """A delegated worker's Shell writes stay inside its allowed_paths."""
        if self.role is None or not self.allowed_paths or workspace_root is None:
            return None
        root = _normalized(workspace_root)
        for target in analysis.write_targets:
            relative = _workspace_relative(target.raw, root)
            if relative is None:
                continue
            if not path_within(relative, self.allowed_paths):
                return f"write target is outside allowed_paths: {relative}"
        return None

    # ------------------------------------------------------------------
    # Structured (non-shell) tools
    # ------------------------------------------------------------------

    def _decide_structured_tool(
        self,
        tool_name: str,
        args: dict,
        permission: str | None,
        workspace_root: str | None = None,
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
            blocked = self._edit_path_block(tool_name, args, workspace_root)
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
            if self.mode in {self.DANGER_FULL_ACCESS, self.EVAL}:
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
        if self.mode in {self.DANGER_FULL_ACCESS, self.EVAL}:
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

    def _edit_path_block(
        self, tool_name: str, args: dict, workspace_root: str | None = None
    ) -> str | None:
        """Role / allowed_paths enforcement for structured edit tools."""
        if self.role is None or tool_name not in {"write_file", "apply_patch"}:
            return None
        definition = role_definition(self.role)
        if definition.read_only:
            return f"{self.role} role cannot modify the workspace"
        raw = str(args.get("path") or "")
        path = normalize_rel_path(raw)
        if not path_within(path, self.allowed_paths):
            return f"write path is outside allowed_paths: {path or '<empty>'}"
        if workspace_root is None:
            return None
        # The string check above accepts ``src/../other.txt`` and paths that
        # reach outside through a symlink or junction.  File tools resolve the
        # real path, so the same location must be checked against the scope.
        return _resolved_write_block(raw, workspace_root, self.allowed_paths)


def shell_call_fingerprint(args: dict) -> str:
    """Identity of a shell request: command, write scope and permissions."""
    command = str((args or {}).get("command") or "")
    write_paths = args.get("write_paths") or ()
    if isinstance(write_paths, (list, tuple)):
        scope = tuple(str(item) for item in write_paths)
    else:
        scope = (str(write_paths),)
    permissions = (
        str(
            (args or {}).get("sandbox_permissions")
            or SANDBOX_PERMISSION_USE_DEFAULT
        ),
        _canonical_permissions((args or {}).get("additional_permissions")),
    )
    return repr((command, scope, permissions))


def _canonical_permissions(value) -> str:
    if not isinstance(value, dict):
        return repr(value)
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return repr(value)


_FORBIDDEN_EXTRA_ROOTS = (
    "/etc", "/usr", "/bin", "/sbin", "/var", "/boot", "/sys", "/proc",
    "/lib", "/lib64", "/dev", "/root",
    "c:\\windows", "c:\\program files", "c:\\programdata",
)


def _validated_extra_dirs(values) -> tuple[tuple[str, ...], str | None]:
    if not isinstance(values, list) or not values:
        return (), "additional_permissions.file_system.write must be a non-empty list"
    from ..workspace.state_paths import veriforge_home

    state_home = Path(os.path.realpath(veriforge_home()))
    resolved: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            return (), "additional write paths must be non-empty strings"
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            return (), f"additional write path must be absolute: {value}"
        # Bind the approval to the real target, not to a symlink or junction
        # that can be retargeted between approval and execution.
        normalized = os.path.normpath(os.path.realpath(str(candidate)))
        if _forbidden_extra_root(normalized):
            return (), f"additional write path is a protected system location: {value}"
        target = Path(normalized)
        if target.is_relative_to(state_home) or state_home.is_relative_to(target):
            return (), f"additional write path overlaps the Agent state directory: {value}"
        if Path(normalized).exists() and not Path(normalized).is_dir():
            return (), f"additional write path must be a directory: {value}"
        resolved.append(normalized)
    return tuple(dict.fromkeys(resolved)), None


def revalidate_extra_dirs(values) -> tuple[str, ...]:
    """Re-check an approved grant immediately before the executor uses it.

    The stored paths are already real paths. If resolving them now yields a
    different target, a symlink or junction changed after approval, so the old
    grant is refused instead of being applied to an unapproved location.
    """
    stored = tuple(str(value) for value in values)
    rechecked, error = _validated_extra_dirs(list(stored))
    if error is not None:
        raise ValueError(error)
    if rechecked != stored:
        raise ValueError("approved write path changed after approval")
    return rechecked


def _forbidden_extra_root(normalized: str) -> bool:
    path = Path(normalized)
    try:
        if path == Path(path.anchor):
            return True
    except ValueError:
        return True
    low = os.path.normcase(normalized)
    for fragment in _FORBIDDEN_EXTRA_ROOTS:
        root = os.path.normcase(fragment)
        if low == root or low.startswith(root.rstrip("\\/") + os.sep):
            return True
    return False


def _describe_extra_permissions(write_dirs: tuple[str, ...], network: bool) -> str:
    parts: list[str] = []
    if write_dirs:
        shown = ", ".join(write_dirs[:3])
        if len(write_dirs) > 3:
            shown += ", ..."
        parts.append("write " + shown)
    if network:
        parts.append("network")
    return "; ".join(parts)


def _reason_for(candidates, action: PermissionAction) -> str:
    """Pick the reason that best explains the chosen action.

    Catastrophic guardrails win their name outright; otherwise the most
    specific candidate (the one added last) describes why.
    """
    matching = [reason for candidate_action, reason in candidates if candidate_action is action]
    for reason in matching:
        if reason in _CATASTROPHIC_REASONS:
            return reason
    return matching[-1]


def _is_broad_delete(analysis) -> bool:
    if not (ShellTrait.RECURSIVE in analysis.traits or ShellTrait.DESTRUCTIVE in analysis.traits):
        return False
    for target in analysis.write_targets:
        if target.scope is not TargetScope.WORKSPACE:
            continue
        if _bare_path(target.raw).lower() in _WHOLE_WORKSPACE_TARGETS:
            return True
    return False


def _covered_by_extra(target, extra_dirs, workspace_root: str | None) -> bool:
    """Whether a write target falls inside a one-time approved directory."""
    if not extra_dirs or workspace_root is None:
        return False
    if target.scope in {TargetScope.SYSTEM, TargetScope.UNKNOWN}:
        return False
    candidate = Path(target.raw)
    if not candidate.is_absolute():
        candidate = Path(workspace_root) / candidate
    normalized = _normalized(str(candidate))
    return any(
        is_same_or_child(normalized, _normalized(directory))
        for directory in extra_dirs
    )


def _resolved_write_block(
    raw: str, workspace_root: str, allowed_paths: tuple[str, ...]
) -> str | None:
    """Reject structured writes that land outside allowed_paths on disk."""
    root = Path(workspace_root).resolve()
    candidate = Path(str(raw or ""))
    target = candidate if candidate.is_absolute() else root / candidate
    try:
        resolved = target.resolve()
    except OSError as exc:
        return f"cannot resolve write path: {raw} ({exc})"
    resolved_key = _normalized(str(resolved))
    if not is_same_or_child(resolved_key, _normalized(str(root))):
        return f"write path escapes the workspace: {raw}"
    for allowed in allowed_paths:
        try:
            allowed_root = (root / allowed).resolve()
        except OSError:
            continue
        if is_same_or_child(resolved_key, _normalized(str(allowed_root))):
            return None
    return f"write path resolves outside allowed_paths: {raw}"


def _workspace_relative(raw: str, normalized_root: str) -> str | None:
    """A raw target's workspace-relative path, or None when it is outside."""
    candidate = Path(str(raw or ""))
    if not candidate.is_absolute():
        candidate = Path(normalized_root) / candidate
    normalized = _normalized(str(candidate))
    if not is_same_or_child(normalized, normalized_root):
        return None
    relative = os.path.relpath(normalized, normalized_root).replace("\\", "/")
    return "." if relative == "." else relative


def _bare_path(raw: str) -> str:
    return str(raw or "").strip().strip("\"'").replace("\\", "/")


def _covered_by(target: ShellTarget, directories: tuple[str, ...], workspace_root: str | None) -> bool:
    if workspace_root is None:
        return False
    candidate = Path(target.raw)
    if not candidate.is_absolute():
        candidate = Path(workspace_root) / candidate
    normalized = _normalized(str(candidate))
    return any(is_same_or_child(normalized, directory) for directory in directories)


def _normalized(value: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(str(value))))


def _builtin_permission(tool_name: str) -> str | None:
    from .builtins.registry import BUILTIN_TOOL_REGISTRY

    return BUILTIN_TOOL_REGISTRY.permission_for(tool_name)
