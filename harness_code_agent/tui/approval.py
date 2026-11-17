"""Workspace-state approval prefix rules shared by terminal frontends."""

from __future__ import annotations

import json
import posixpath
from datetime import UTC, datetime
from pathlib import Path

from ..runtime.approvals import ApprovalRequest
from ..runtime.shell_classification import (
    command_matches_prefix,
    command_matches_stage_prefixes,
    derive_persistent_stage_prefixes,
)
from ..workspace.paths import normalize_rel_path


class ApprovalAllowlist:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        from ..workspace.state_paths import ensure_workspace_state
        self.path = ensure_workspace_state(self.project_root) / "approval_allowlist.json"

    def add_prefix_rule(
        self,
        prefixes: list[list[str]],
        *,
        command: str,
        write_paths=(),
        sandbox_permissions: str = "use_default",
        additional_permissions=None,
        network_access: bool = False,
    ) -> None:
        clean = [
            [_normalize_token(token) for token in group if _normalize_token(token)]
            for group in prefixes
        ]
        clean = [group for group in clean if group]
        if not clean:
            return
        scope = _normalize_scope(write_paths)
        permissions = _permission_signature(
            sandbox_permissions, additional_permissions, network_access
        )
        data = self._read()
        rules = data.setdefault("rules", [])
        for rule in rules:
            if (
                rule.get("tool") == "bash"
                and rule.get("kind") == "stage_prefixes"
                and rule.get("prefixes") == clean
                and _normalize_scope(rule.get("write_paths")) == scope
                and (rule.get("permissions") or ["use_default", "{}", False])
                == permissions
            ):
                return
        rules.append(
            {
                "tool": "bash",
                "kind": "stage_prefixes",
                # A user-chosen scoped rule: the write scope, sandbox
                # permissions and network grant are all part of the identity.
                "scoped": True,
                "prefixes": clean,
                "write_paths": list(scope),
                "permissions": permissions,
                "command": command,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        self._write(data)

    def match(
        self,
        command: str,
        *,
        write_paths=(),
        sandbox_permissions: str = "use_default",
        additional_permissions=None,
        network_access: bool = False,
    ) -> dict | None:
        """A rule that still describes what this command will do.

        Matching re-parses the command, re-applies the safety rules and checks
        the requested write, sandbox and network scope, so neither a broad rule
        from an older version nor a wider capability request can reuse an
        earlier approval.
        """
        requested = _normalize_scope(write_paths)
        requested_permissions = _permission_signature(
            sandbox_permissions, additional_permissions, network_access
        )
        for rule in self._read().get("rules", []):
            if rule.get("tool") != "bash":
                continue
            if not _scope_covers(_normalize_scope(rule.get("write_paths")), requested):
                continue
            if (
                rule.get("permissions") or ["use_default", "{}", False]
            ) != requested_permissions:
                continue
            allow_scoped = bool(rule.get("scoped"))
            if rule.get("kind") == "stage_prefixes":
                prefixes = [
                    [str(token) for token in group]
                    for group in rule.get("prefixes", [])
                ]
                if prefixes and command_matches_stage_prefixes(
                    command, prefixes, allow_scoped=allow_scoped
                ):
                    return rule
            elif rule.get("kind") == "prefix":
                # Legacy single-prefix rule written by older versions.
                prefix = [str(token) for token in rule.get("prefix", [])]
                if prefix and command_matches_prefix(command, prefix):
                    return rule
        return None

    def matches(self, command: str, *, write_paths=(), **permissions) -> bool:
        return self.match(command, write_paths=write_paths, **permissions) is not None

    def _read(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "rules": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "rules": []}
        if not isinstance(data, dict):
            return {"version": 1, "rules": []}
        if not isinstance(data.get("rules"), list):
            data["rules"] = []
        data["version"] = 1
        return data

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def persistent_prefix_for_request(request: ApprovalRequest) -> list[list[str]] | None:
    """One prefix per command, or None when nothing stable can be persisted."""
    if request.tool_name != "bash":
        return None
    if request.requires_human:
        # A request that needed a human decision is not a routine operation.
        return None
    prefixes = derive_persistent_stage_prefixes(
        str(request.args.get("command", "")), allow_scoped=True
    )
    return prefixes if prefixes else None


def _permission_signature(
    sandbox_permissions: object, additional_permissions: object, network_access: bool
) -> list:
    """Identity of the capability a rule replays."""
    sandbox = str(sandbox_permissions or "use_default")
    try:
        additional = json.dumps(
            additional_permissions or {}, sort_keys=True, ensure_ascii=False
        )
    except TypeError:
        additional = repr(additional_permissions)
    return [sandbox, additional, bool(network_access)]


#: Backwards-compatible name used by older callers.
_persistent_prefix_for_request = persistent_prefix_for_request


def _normalize_token(token: object) -> str:
    return str(token).strip().strip("\"'").lower()


def _normalize_scope(values) -> tuple[str, ...]:
    """Normalized relative write directories of one approval.

    ``..`` segments are collapsed so `build/../src` cannot pass as a sub-path
    of a rule that granted `build`.
    """
    if values is None:
        return ()
    if isinstance(values, str):
        values = [values]
    return tuple(
        sorted({
            posixpath.normpath(normalize_rel_path(str(value)))
            for value in values
            if str(value).strip()
        })
    )


def _scope_covers(rule_scope: tuple[str, ...], requested: tuple[str, ...]) -> bool:
    """Whether the directories a rule granted include the ones now requested."""
    if not requested:
        return True
    return all(any(_within(path, root) for root in rule_scope) for path in requested)


def _within(path: str, root: str) -> bool:
    if root in {".", ""}:
        return True
    return path == root or path.startswith(root.rstrip("/") + "/")
