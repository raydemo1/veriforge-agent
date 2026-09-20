"""Git helpers for interactive sessions and checkpoints."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .. import config
from ..sessions._event_helpers import is_ignored_changed_file

GIT_COMMIT_AUTHOR = ("Harness", "harness@example.invalid")
CHECKPOINT_EXCLUDES = [".harness", "global_plan", config.PROGRESS_FILE]
GIT_STATUS_TIMEOUT_SECONDS = 0.5


@dataclass(frozen=True)
class GitBaseline:
    dirty_paths: frozenset[str]
    staged_paths: frozenset[str]


def _ensure_git_repository(workspace: Path) -> None:
    if (workspace / ".git").exists():
        _git_add_runtime_exclude(workspace)
        return
    subprocess.run(["git", "init"], cwd=workspace, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    _git_add_runtime_exclude(workspace)
    subprocess.run(git_commit_command("init", allow_empty=True), cwd=workspace, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def _git_add_runtime_exclude(workspace: Path) -> None:
    info_exclude = workspace / ".git" / "info" / "exclude"
    if not (workspace / ".git").exists():
        return
    info_exclude.parent.mkdir(parents=True, exist_ok=True)
    existing = info_exclude.read_text(encoding="utf-8", errors="replace") if info_exclude.exists() else ""
    lines = set(existing.splitlines())
    additions = [item for item in [".harness/", "global_plan/", config.PROGRESS_FILE] if item not in lines]
    if additions:
        with info_exclude.open("a", encoding="utf-8") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            for item in additions:
                f.write(item + "\n")


def git_add_runtime_excluded(workspace: Path) -> None:
    if not _is_git_repository(workspace):
        return
    subprocess.run(
        ["git", "add", "-A", "--", "."],
        cwd=workspace,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    subprocess.run(
        ["git", "reset", "-q", "--", ".harness", "global_plan", config.PROGRESS_FILE],
        cwd=workspace,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def git_add_paths(workspace: Path, paths: list[str]) -> None:
    if not paths or not _is_git_repository(workspace):
        return
    subprocess.run(
        ["git", "add", "--", *paths],
        cwd=workspace,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def git_has_committable_changes(workspace: Path) -> bool:
    result = _run_git_status(workspace)
    if result is None:
        return False
    return bool(result.stdout.strip())


def git_dirty_paths(workspace: Path) -> set[str]:
    result = _run_git_status(workspace)
    if result is None:
        return set()
    return set(_parse_git_baseline(result.stdout).dirty_paths)


def capture_git_baseline(workspace: Path) -> GitBaseline | None:
    result = _run_git_status(workspace)
    if result is None:
        return None
    return _parse_git_baseline(result.stdout)


def _parse_git_baseline(output: str) -> GitBaseline:
    dirty_paths: set[str] = set()
    staged_paths: set[str] = set()
    for line in output.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[1]
        if not path or is_ignored_changed_file(path):
            continue
        dirty_paths.add(path)
        if line[0] not in {" ", "?"}:
            staged_paths.add(path)
    return GitBaseline(frozenset(dirty_paths), frozenset(staged_paths))


def git_has_staged_changes(workspace: Path) -> bool:
    if not _is_git_repository(workspace):
        return False
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=workspace,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 1


def _is_git_repository(workspace: Path) -> bool:
    return (workspace / ".git").exists()


def _run_git_status(workspace: Path) -> subprocess.CompletedProcess[str] | None:
    if not _is_git_repository(workspace):
        return None
    try:
        return subprocess.run(
            runtime_excluded_git_command("status", "--porcelain"),
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
            timeout=GIT_STATUS_TIMEOUT_SECONDS,
            **_background_git_options(),
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _background_git_options() -> dict:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    return {
        "stdin": subprocess.DEVNULL,
        "env": env,
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
    }


def runtime_excluded_git_command(*args: str) -> list[str]:
    command = ["git", *args, "--", "."]
    command.extend(f":(exclude){item}" for item in CHECKPOINT_EXCLUDES)
    return command


def git_commit_command(message: str, *, allow_empty: bool = False) -> list[str]:
    name, email = GIT_COMMIT_AUTHOR
    command = [
        "git",
        "-c",
        f"user.name={name}",
        "-c",
        f"user.email={email}",
        "commit",
        "-m",
        message,
    ]
    if allow_empty:
        command.append("--allow-empty")
    return command
