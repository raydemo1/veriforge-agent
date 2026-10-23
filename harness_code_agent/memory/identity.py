from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

GIT_IDENTITY_TIMEOUT_SECONDS = 0.5


def resolve_repo_key(workspace: Path) -> str:
    workspace = workspace.resolve()
    repo_name = workspace.name
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"], cwd=workspace,
            capture_output=True, text=True, check=True,
            timeout=GIT_IDENTITY_TIMEOUT_SECONDS, stdin=subprocess.DEVNULL,
            env=_noninteractive_git_env(), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        common = Path(proc.stdout.strip())
        common = common if common.is_absolute() else (workspace / common).resolve()
        basis = os.path.normcase(str(common))
        if common.name.lower() == ".git":
            repo_name = common.parent.name
    except (OSError, subprocess.SubprocessError):
        basis = os.path.normcase(str(workspace))
    digest = hashlib.sha256(basis.encode()).hexdigest()[:12]
    name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in repo_name)
    return f"{name or 'workspace'}-{digest}"


def _noninteractive_git_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    return env



def default_memory_root() -> Path:
    return Path(os.environ.get("HARNESS_MEMORY_ROOT", str(Path.home() / ".harness" / "memory-v3"))).expanduser().resolve()
