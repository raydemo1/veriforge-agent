"""Content fingerprints for supported source files, independent of Git."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

_SOURCE_SUFFIXES = {
    ".py",
    ".go",
    ".rs",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mts",
    ".cts",
    ".mjs",
    ".cjs",
}
_EXCLUDE = {
    ".git",
    ".harness",
    ".harbor",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "ENV",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "target",
    "dist",
    "build",
    ".next",
    ".cache",
}
_CONFIG_NAMES = {
    "Cargo.toml",
    "pyproject.toml",
    "pyrightconfig.json",
    "package.json",
    "go.mod",
}


def is_source_location(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return not any(part in _EXCLUDE for part in relative.parts) and relative.parts[
        :2
    ] != ("eval", "results")


def is_source_path(root: Path, path: Path) -> bool:
    return is_source_location(root, path) and (
        path.suffix in _SOURCE_SUFFIXES
        or path.name in _CONFIG_NAMES
        or path.name.startswith("tsconfig")
    )


def iter_source_files(root: Path, subtree: Path | None = None):
    if not is_source_location(root, subtree or root) or not (
        subtree or root
    ).resolve().is_relative_to(root):
        return
    for directory, children, names in os.walk(subtree or root, followlinks=False):
        children[:] = [
            name
            for name in children
            if name not in _EXCLUDE
            and (Path(directory) / name).relative_to(root).parts != ("eval", "results")
            and not (Path(directory) / name).is_symlink()
        ]
        for name in names:
            path = Path(directory) / name
            if not is_source_path(root, path):
                continue
            try:
                if path.resolve().is_relative_to(root) and path.is_file():
                    yield path
            except OSError:
                continue


def source_fingerprints(root: Path) -> dict[str, str]:
    files = {}
    for path in iter_source_files(root):
        try:
            files[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        except OSError:
            continue
    return files
