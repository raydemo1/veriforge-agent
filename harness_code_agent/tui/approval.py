"""Project-local approval prefix rules shared by terminal frontends."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..runtime.approvals import ApprovalRequest
from ..runtime.shell_classification import (
    command_matches_prefix,
    command_matches_stage_prefixes,
    derive_persistent_stage_prefixes,
)


class ApprovalAllowlist:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self.path = self.project_root / ".harness" / "approval_allowlist.json"

    def add_prefix_rule(self, prefixes: list[list[str]], *, command: str) -> None:
        clean = [
            [_normalize_token(token) for token in group if _normalize_token(token)]
            for group in prefixes
        ]
        clean = [group for group in clean if group]
        if not clean:
            return
        data = self._read()
        rules = data.setdefault("rules", [])
        for rule in rules:
            if (
                rule.get("tool") == "run_bash"
                and rule.get("kind") == "stage_prefixes"
                and rule.get("prefixes") == clean
            ):
                return
        rules.append(
            {
                "tool": "run_bash",
                "kind": "stage_prefixes",
                "prefixes": clean,
                "command": command,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._write(data)

    def match(self, command: str) -> dict | None:
        for rule in self._read().get("rules", []):
            if rule.get("tool") != "run_bash":
                continue
            if rule.get("kind") == "stage_prefixes":
                prefixes = [
                    [str(token) for token in group]
                    for group in rule.get("prefixes", [])
                ]
                if prefixes and command_matches_stage_prefixes(command, prefixes):
                    return rule
            elif rule.get("kind") == "prefix":
                # Legacy single-prefix rule written by older versions.
                prefix = [str(token) for token in rule.get("prefix", [])]
                if prefix and command_matches_prefix(command, prefix):
                    return rule
        return None

    def matches(self, command: str) -> bool:
        return self.match(command) is not None

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


def _persistent_prefix_for_request(request: ApprovalRequest) -> list[list[str]] | None:
    """One prefix per pipeline stage; None when no stable rule can persist."""
    if request.tool_name != "run_bash":
        return None
    prefixes = derive_persistent_stage_prefixes(str(request.args.get("command", "")))
    return prefixes if prefixes else None


def _normalize_token(token: object) -> str:
    return str(token).strip().strip("\"'").lower()
