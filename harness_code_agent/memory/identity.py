from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from ..workspace.state_paths import veriforge_home

GIT_IDENTITY_TIMEOUT_SECONDS = 0.5


# Definite answers (a repository, or git saying "not a repository") per
# resolved workspace.  Every session start asks, so this saves a git process
# each time; timeouts and a missing git are retried instead of cached.
_REPO_KEYS: dict[Path, str] = {}


def resolve_repo_key(workspace: Path) -> str:
    workspace = workspace.resolve()
    cached = _REPO_KEYS.get(workspace)
    if cached is not None:
        return cached
    repo_name = workspace.name
    definite = True
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"], cwd=workspace,
            capture_output=True, encoding="utf-8", check=True,
            timeout=GIT_IDENTITY_TIMEOUT_SECONDS, stdin=subprocess.DEVNULL,
            env=_noninteractive_git_env(), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        common = Path(proc.stdout.strip())
        common = common if common.is_absolute() else (workspace / common).resolve()
        basis = os.path.normcase(str(common))
        if common.name.lower() == ".git":
            repo_name = common.parent.name
    except subprocess.CalledProcessError:
        basis = os.path.normcase(str(workspace))
    except (OSError, subprocess.SubprocessError):
        basis = os.path.normcase(str(workspace))
        definite = False
    digest = hashlib.sha256(basis.encode()).hexdigest()[:12]
    name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in repo_name)
    key = f"{name or 'workspace'}-{digest}"
    if definite:
        _REPO_KEYS[workspace] = key
    return key


def _noninteractive_git_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    return env



def default_memory_root() -> Path:
    return Path(
        os.environ.get("VERIFORGE_MEMORY_ROOT", str(veriforge_home() / "memory-v3"))
    ).expanduser().resolve()
