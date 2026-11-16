"""Per-workspace Agent state outside the project tree.

The project directory stays a normal Git checkout: sessions, approvals,
recovery data, coordination databases, worktree metadata, observations and
traces live under ``~/.veriforge/workspaces/<name>-<hash>/`` instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

STATE_DIR_NAME = ".veriforge"
LEGACY_STATE_DIR_NAME = ".harness"
MIGRATION_MARKER = ".migrated-from-harness.json"

#: Files that are configuration (safe to copy forward). Sessions, traces,
#: observations, recovery snapshots, worktrees and approval rules are
#: deliberately not migrated.
MIGRATED_CONFIG_FILES = ("settings.json",)
MIGRATED_USER_CONFIG_FILES = ("middlewares.json",)

_user_config_migrated = False


def veriforge_home() -> Path:
    """The user-level Agent data directory."""
    global _user_config_migrated
    override = os.environ.get("VERIFORGE_HOME")
    if override:
        return Path(override).expanduser().resolve()
    home = (Path.home() / STATE_DIR_NAME).resolve()
    if not _user_config_migrated:
        _user_config_migrated = True
        _migrate_user_config(home)
    return home


def _migrate_user_config(home: Path) -> None:
    legacy = Path.home() / LEGACY_STATE_DIR_NAME
    if not legacy.is_dir():
        return
    for name in MIGRATED_USER_CONFIG_FILES:
        source = legacy / name
        target = home / name
        if source.is_file() and not target.exists():
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            except OSError:
                pass


def workspace_state_root(
    project_root: str | Path, *, state_root: str | Path | None = None
) -> Path:
    """Stable user-level state directory for one project path.

    The identity is derived from the normalized absolute project path, so
    reopening the same checkout always resolves to the same directory. An
    explicit ``state_root`` still wins for tests and evaluators.
    """
    if state_root is not None:
        return Path(state_root).expanduser().resolve()
    project = Path(project_root).expanduser().resolve()
    identity = os.path.normcase(str(project))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^a-zA-Z0-9_-]", "-", project.name)[:32] or "workspace"
    return veriforge_home() / "workspaces" / f"{slug}-{digest}"


def ensure_workspace_state(
    project_root: str | Path, *, state_root: str | Path | None = None
) -> Path:
    """Create the workspace state root and run the one-time legacy migration."""
    root = workspace_state_root(project_root, state_root=state_root)
    root.mkdir(parents=True, exist_ok=True)
    migrate_legacy_state(Path(project_root).expanduser().resolve(), root)
    return root


def legacy_state_root(project_root: str | Path) -> Path:
    return Path(project_root).expanduser().resolve() / LEGACY_STATE_DIR_NAME


def migrate_legacy_state(project: Path, state_root: Path) -> dict:
    """Copy forward only configuration; leave runtime history behind.

    Approval rules are not copied: they used to live in a directory the Agent
    could write, so silently promoting them to user-level grants would turn a
    possibly forged allowlist into trusted policy. The old directory is left
    untouched for the user to inspect or remove.
    """
    marker = state_root / MIGRATION_MARKER
    if marker.exists():
        try:
            return json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    legacy = legacy_state_root(project)
    report = {
        "legacy_root": str(legacy),
        "copied": [],
        "skipped": [
            "sessions",
            "attachments",
            "plans",
            "todos",
            "recovery",
            "worktrees",
            "observations",
            "traces",
            "approval_allowlist.json",
            "mcp.json",
        ],
    }
    if legacy.is_dir():
        for name in MIGRATED_CONFIG_FILES:
            source = legacy / name
            target = state_root / name
            if source.is_file() and not target.exists():
                shutil.copy2(source, target)
                report["copied"].append(name)
    marker.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
