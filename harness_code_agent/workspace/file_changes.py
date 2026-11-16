"""Optional per-command workspace snapshots for diagnostics.

Only used when ``HARNESS_TRACE_FILE_CHANGES=1``.  It is deliberately a plain
manifest + diff, not a monitor: the Shell runner takes a snapshot before and
after one command and puts the change set in the tool result metadata under
``file_change_summary``.  That key is intentionally distinct from the
list-shaped ``file_changes`` field that feeds ``FileChangeEvent``.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import TypedDict

IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".harness",
    ".veriforge",
}
MAX_PATHS = 200


def enabled() -> bool:
    return os.environ.get("HARNESS_TRACE_FILE_CHANGES", "").strip().lower() in {
        "1", "true", "yes",
    }


class FileSnapshot(TypedDict):
    files: dict[str, tuple[int, int, str]]
    scan_ms: float


def snapshot(root: str | Path) -> FileSnapshot:
    """Hash every regular file under ``root`` except known generated trees."""
    root = Path(root)
    started = time.perf_counter()
    files: dict[str, tuple[int, int, str]] = {}
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in IGNORED_DIRS)
        for name in filenames:
            path = Path(directory) / name
            if path.is_symlink():
                continue
            try:
                stat = path.stat()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                continue
            files[path.relative_to(root).as_posix()] = (
                stat.st_size,
                stat.st_mtime_ns,
                digest,
            )
    return {"files": files, "scan_ms": round((time.perf_counter() - started) * 1000, 2)}


def diff(before: dict, after: dict) -> dict:
    old = before.get("files") or {}
    new = after.get("files") or {}
    created = sorted(set(new) - set(old))
    deleted = sorted(set(old) - set(new))
    modified = sorted(
        path for path in set(old) & set(new)
        if old[path][2] != new[path][2]
    )
    return {
        "created": created[:MAX_PATHS],
        "modified": modified[:MAX_PATHS],
        "deleted": deleted[:MAX_PATHS],
        "counts": {
            "created": len(created),
            "modified": len(modified),
            "deleted": len(deleted),
        },
        "truncated": any(len(paths) > MAX_PATHS for paths in (created, modified, deleted)),
        "before_scan_ms": before.get("scan_ms"),
        "after_scan_ms": after.get("scan_ms"),
    }
