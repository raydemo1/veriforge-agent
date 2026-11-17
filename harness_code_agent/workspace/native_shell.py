"""Native Windows write confinement and its declared project directories."""
from __future__ import annotations

import json
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path

from ..opentui_launcher import find_bun, frontend_directory
from ..runtime.execution_planner import workspace_claim


def helper_path():
    return frontend_directory() / "src" / "native-shell.ts"


def validate_native_shell():
    from .shell_environment import shell_executable

    if os.name != "nt":
        raise RuntimeError("The Windows file sandbox requires Windows")
    shell_executable()
    if find_bun() is None:
        raise RuntimeError("The native sandbox requires the VeriForge Bun runtime and PowerShell 7")
    package = helper_path().parents[1] / "node_modules" / "@deepseek-ai" / "dsh-sandbox-windows-acl" / "package.json"
    if not package.is_file():
        raise RuntimeError("Native sandbox dependencies are missing; run veriforge --setup-tui")


@lru_cache(maxsize=1)
def _current_user_sid(system32: Path) -> str:
    output = subprocess.check_output(
        [system32 / "whoami.exe", "/user", "/fo", "csv", "/nh"],
        timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    match = re.search(rb'"(S-1-\d+(?:-\d+)+)"\s*$', output)
    if match is None:
        raise RuntimeError("Could not identify the Windows account for the private shell directory")
    return match[1].decode("ascii")


def prepare_private_temp_dir(path: str) -> None:
    """Give the restricted job token a matching user ACE on its private temp directory."""
    system32 = Path(os.environ["SystemRoot"]) / "System32"
    sid = _current_user_sid(system32)
    # tempfile's OWNER RIGHTS ACE protects this directory but cannot satisfy
    # the restricted token's normal access check. Add only the current user;
    # AclSandbox still supplies a distinct, per-job SID for its second check.
    subprocess.run(
        [system32 / "icacls.exe", path, "/grant", f"*{sid}:(OI)(CI)F"],
        check=True,
        timeout=10,
        capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def write_directories(root, values):
    root = Path(root).resolve()
    if not isinstance(values, (list, tuple)) or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("write_paths must contain workspace directory paths")
    directories = set()
    for value in values:
        path = (root / value).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Shell writable directories must stay within the workspace")
        if path.exists() and not path.is_dir():
            raise ValueError(f"Shell write_paths requires directories, not files: {value}")
        directories.add(path)
    return tuple(sorted(path for path in directories if not any(parent != path and path.is_relative_to(parent) for parent in directories)))


def write_claims(root, values):
    return tuple(workspace_claim(root, path, scope="subtree", access="write")
                 for path in write_directories(root, values))


def launch_arguments(root, directories, temporary, command, extra_dirs=(), *, read_only=False):
    from .shell_environment import shell_executable

    validate_native_shell()
    policy = {
        "cwd": str(root),
        "write_paths": [str(path) for path in directories],
        # Approved per-call grants outside the workspace, never model-declared
        # through write_paths.
        "extra_write_paths": [str(path) for path in extra_dirs],
        "temp": str(temporary),
        "mode": "read-only" if read_only else "workspace-write",
    }
    return [find_bun(), "run", str(helper_path()), json.dumps(policy, ensure_ascii=False),
            shell_executable(), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
            _powershell_script(root, command)]


def direct_arguments(root, command):
    """Unconfined PowerShell launch for the explicit danger-full-access opt-in."""
    from .shell_environment import shell_executable

    return [shell_executable(), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
            _powershell_script(root, command)]


def _powershell_script(root, command):
    working_directory = str(root).replace("'", "''")
    # Format the command's success stream before testing LASTEXITCODE/$?.
    # Without Out-Default, object output (for example Select-Object results)
    # is discarded when the wrapped block exits through the explicit `exit`.
    return (
        "$ErrorActionPreference = 'Stop'; "
        f"Set-Location -LiteralPath '{working_directory}'; "
        "$global:__vf_ok = $null; $global:__vf_exit = $null; "
        f"& {{ {command}; $global:__vf_ok = $?; $global:__vf_exit = $global:LASTEXITCODE }} | Out-Default; "
        "if ($global:__vf_exit -is [int]) { exit $global:__vf_exit } "
        "elseif ($global:__vf_ok) { exit 0 } else { exit 1 }"
    )
