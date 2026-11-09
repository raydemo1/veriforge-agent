"""The platform shell used by the process supervisor."""
from __future__ import annotations

import functools
import os
import shutil


# ``shutil.which`` stats every PATH directory for every PATHEXT suffix — ~940
# stat calls on a machine with a long PATH, which measured 0.24s per lookup and
# was paid three times per session startup (shell validation, native shell
# validation, job launch). The resolved shell cannot change within a process,
# so resolve once.
@functools.lru_cache(maxsize=1)
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
