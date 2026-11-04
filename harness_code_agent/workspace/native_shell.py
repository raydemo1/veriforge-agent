"""Native Windows write confinement and its declared project directories."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..opentui_launcher import find_bun
from ..runtime.execution_planner import workspace_claim


def helper_path():
    return Path(__file__).resolve().parents[2] / "frontend" / "opentui" / "src" / "native-shell.ts"


def validate_native_shell():
    from .shell_environment import shell_executable

    if os.name != "nt":
        raise RuntimeError("The Windows file sandbox requires Windows")
    shell_executable()
    if find_bun() is None:
        raise RuntimeError("The native sandbox requires the VeriForge Bun runtime and PowerShell 7")
    package = helper_path().parents[1] / "node_modules" / "@deepseek-ai" / "dsh-sandbox-windows-acl" / "package.json"
    if not package.is_file():
        raise RuntimeError("Native sandbox dependencies are missing; run bun install in frontend/opentui")


def write_directories(root, values):
    root = Path(root).resolve()
    if not isinstance(values, (list, tuple)) or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("write_paths must contain workspace directory paths")
    directories = set()
    for value in values:
        path = (root / value).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Shell writable directories must stay within the workspace")
        if path.is_relative_to(root / ".harness"):
            raise ValueError("Shell cannot request the VeriForge state directory")
        if path.exists() and not path.is_dir():
            raise ValueError(f"Shell write_paths requires directories, not files: {value}")
        directories.add(path)
    return tuple(sorted(path for path in directories if not any(parent != path and path.is_relative_to(parent) for parent in directories)))


def write_claims(root, values):
    return tuple(workspace_claim(root, path, scope="subtree", access="write")
                 for path in write_directories(root, values))


def launch_arguments(root, directories, temporary, command):
    from .shell_environment import shell_executable

    validate_native_shell()
    working_directory = str(root).replace("'", "''")
    script = (
        "$ErrorActionPreference = 'Stop'; "
        f"Set-Location -LiteralPath '{working_directory}'; "
        f"& {{ {command} }}; "
        "if ($global:LASTEXITCODE -is [int]) { exit $global:LASTEXITCODE } "
        "elseif ($?) { exit 0 } else { exit 1 }"
    )
    policy = {"cwd": str(root), "write_paths": [str(path) for path in directories], "temp": str(temporary)}
    return [find_bun(), "run", str(helper_path()), json.dumps(policy, ensure_ascii=False),
            shell_executable(), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script]
