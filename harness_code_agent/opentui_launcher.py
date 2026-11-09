"""Process launcher for the Bun/OpenTUI frontend."""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_bun() -> str | None:
    """Find Bun without assuming it was installed system-wide."""
    configured = os.environ.get("VERIFORGE_BUN")
    if configured and Path(configured).is_file():
        return configured
    return _installed_bun()


# ``shutil.which`` costs one stat per PATH directory per PATHEXT suffix, and it
# runs on every session startup. The installed runtime does not move while the
# process lives, so probe the filesystem once; VERIFORGE_BUN is still read on
# every call.
@functools.lru_cache(maxsize=1)
def _installed_bun() -> str | None:
    on_path = shutil.which("bun")
    if on_path:
        return on_path
    for candidate in (
        Path.home() / ".bun" / "bin" / "bun.exe",
        Path.home() / ".bun" / "bin" / "bun",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


class OpenTuiApp:
    """Synchronous process wrapper for the Bun/OpenTUI frontend."""

    def __init__(
        self,
        *,
        cwd: str | Path,
        first_task: str = "",
        no_alt_screen: bool = False,
        theme: str = "auto",
        icons: str = "auto",
    ) -> None:
        self.cwd = Path(cwd).resolve()
        self.first_task = first_task
        self.no_alt_screen = no_alt_screen
        self.theme = theme
        self.icons = icons

    def run(self) -> int:
        bun = find_bun()
        if bun is None:
            print(
                "Error: Bun is required for the VeriForge OpenTUI frontend. Install Bun 1.4 or newer.",
                file=sys.stderr,
            )
            return 1

        frontend_dir = Path(__file__).resolve().parents[1] / "frontend" / "opentui"
        command = [
            bun,
            "run",
            "src/main.tsx",
            "--",
            "--bridge",
            "--python",
            sys.executable,
            "--cwd",
            str(self.cwd),
            "--theme",
            self.theme,
            "--icons",
            self.icons,
        ]
        if self.first_task:
            command.extend(["--first-task", self.first_task])
        if self.no_alt_screen:
            command.append("--no-alt-screen")

        env = os.environ.copy()
        repo_root = str(Path(__file__).resolve().parents[1])
        python_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = repo_root if not python_path else repo_root + os.pathsep + python_path
        env.setdefault("PYTHONUTF8", "1")
        env["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(command, cwd=frontend_dir, env=env, check=False)
        return completed.returncode
