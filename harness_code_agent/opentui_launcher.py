"""Process launcher for the Bun/OpenTUI frontend."""

from __future__ import annotations

import functools
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .workspace.state_paths import veriforge_home

_ASSETS = Path(__file__).resolve().parents[1] / "frontend" / "opentui"


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


def _installed(directory: Path) -> bool:
    packages = ["@opentui/core", "@opentui/react"]
    if os.name == "nt":
        packages.append("@deepseek-ai/dsh-sandbox-windows-acl")
    return all((directory / "node_modules" / name / "package.json").is_file() for name in packages)


def _asset_files(assets: Path) -> list[Path]:
    return [assets / name for name in ("package.json", "bun.lock", "tsconfig.json")] + sorted(
        path for folder in ("src", "patches") for path in (assets / folder).rglob("*")
        if path.is_file() and not path.name.endswith((".test.ts", ".test.tsx"))
    )


@functools.lru_cache(maxsize=1)
def _asset_hash(assets: Path) -> str:
    digest = hashlib.sha256()
    for path in _asset_files(assets):
        digest.update(path.relative_to(assets).as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:16]


def _frontend_cache() -> Path:
    return veriforge_home() / "opentui" / _asset_hash(_ASSETS)


def frontend_directory() -> Path:
    # Editable development can reuse the already installed frontend.
    return _ASSETS if _installed(_ASSETS) else _frontend_cache()


def setup_tui() -> int:
    bun = find_bun()
    if bun is None:
        print("Error: Install Bun 1.4 or newer, then run veriforge --setup-tui.", file=sys.stderr)
        return 1
    directory = frontend_directory()
    if not _installed(directory):
        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".setup-", dir=directory.parent) as staging:
            source = Path(staging) / "opentui"
            for path in _asset_files(_ASSETS):
                target = source / path.relative_to(_ASSETS)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            result = subprocess.run([bun, "install", "--frozen-lockfile", "--production"], cwd=source, check=False)
            if result.returncode:
                return result.returncode
            try:
                source.rename(directory)
            except FileExistsError:
                if not _installed(directory):
                    raise
    print(f"OpenTUI ready: {directory}")
    return 0


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

        frontend_dir = frontend_directory()
        if not _installed(frontend_dir):
            print("Error: Run veriforge --setup-tui to install OpenTUI dependencies.", file=sys.stderr)
            return 1
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
