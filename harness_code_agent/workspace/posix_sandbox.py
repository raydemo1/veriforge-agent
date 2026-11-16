"""Filesystem confinement for POSIX shell jobs.

Linux uses bubblewrap: the whole filesystem is mounted read-only and only the
directories the caller granted are bound read-write. A foreground or background
command is granted the workspace itself; explicit ``write_paths`` narrow that
grant.

macOS has no supported backend here, so restricted shell execution is reported
as unavailable instead of silently running unconfined.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

BUBBLEWRAP = "bwrap"


def sandbox_backend() -> str | None:
    """The backend that can confine a shell job on this host, if any."""
    if sys.platform.startswith("linux") and shutil.which(BUBBLEWRAP):
        return "bubblewrap"
    return None


def unavailable_reason() -> str:
    if sys.platform == "darwin":
        return (
            "restricted shell execution is not implemented on macOS; "
            "no supported process sandbox backend is available"
        )
    if sys.platform.startswith("linux"):
        return (
            "restricted shell execution requires bubblewrap (bwrap) on Linux; "
            "install it or run in danger-full-access mode"
        )
    return f"restricted shell execution is not supported on {sys.platform}"


def wrap_command(
    command: str,
    *,
    root: str | Path,
    shell: str,
    write_dirs: tuple[Path, ...] = (),
    temp_dir: str | Path | None = None,
    allow_network: bool = True,
) -> list[str] | None:
    """Build the confined launch argv, or None when no backend is available.

    ``allow_network=False`` runs the command in its own network namespace with
    no host connectivity. Every granted directory is bound read-write as
    given: the sandbox enforces the granted scope, it does not redefine it.
    """
    backend = sandbox_backend()
    if backend is None:
        return None
    arguments = [
        shutil.which(BUBBLEWRAP),
        # One process tree, no shared namespaces, dies with its parent.
        "--die-with-parent",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--new-session",
    ]
    if not allow_network:
        arguments.append("--unshare-net")
    arguments += [
        # Read-only base system; the workspace is granted back below.
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
    ]
    if temp_dir is not None:
        arguments += ["--bind", str(temp_dir), str(temp_dir)]
    for directory in write_dirs:
        arguments += ["--bind", str(directory), str(directory)]
    arguments += ["--chdir", str(root), "--", shell, "-lc", command]
    return arguments
