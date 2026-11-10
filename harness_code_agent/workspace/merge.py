"""Three-way text merge used by change proposals and worktree integration."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def git_merge_text(current: str, base: str, worker: str) -> tuple[str, bool]:
    with tempfile.TemporaryDirectory(prefix="hca-merge-") as temp_dir:
        root = Path(temp_dir)
        current_path = root / "current"
        base_path = root / "base"
        worker_path = root / "worker"
        current_path.write_text(current, encoding="utf-8")
        base_path.write_text(base, encoding="utf-8")
        worker_path.write_text(worker, encoding="utf-8")
        env = dict(os.environ)
        env.update({"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never"})
        completed = subprocess.run(
            ["git", "merge-file", "-p", "--diff3", str(current_path), str(base_path), str(worker_path)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=30,
            check=False,
        )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(f"git merge-file failed: {completed.stderr.strip()}")
    return completed.stdout, completed.returncode == 1
