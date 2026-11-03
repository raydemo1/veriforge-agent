"""The platform shell used by the process supervisor."""
from __future__ import annotations

import os
import shutil


def shell_executable():
    executable = "pwsh" if os.name == "nt" else "bash"
    path = shutil.which(executable)
    if path is None:
        raise RuntimeError(f"Required platform shell not found: {executable}")
    return path


def validate_shell_configuration():
    shell_executable()
    if os.name == "nt":
        from .native_shell import validate_native_shell
        validate_native_shell()
