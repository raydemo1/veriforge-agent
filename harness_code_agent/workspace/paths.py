"""Path rules shared by permissions, proposals, attachments and tools.

These helpers exist because the same three questions were answered by four
slightly different implementations across the codebase:

* is this declared path inside one of the roots a worker may write?
* is this real path inside the workspace?
* where do per-session observation artifacts live?

Kept in one module so the rules cannot drift apart. Everything here is pure
path arithmetic: no filesystem access, no symlink resolution unless the caller
asks for it.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def normalize_rel_path(value: str) -> str:
    """Normalize a declared workspace-relative path for comparison."""
    normalized = str(value or "").replace("\\", "/").removeprefix("./")
    return normalized.rstrip("/") or "."


def path_within(path: str, roots: Iterable[str]) -> bool:
    """Whether ``path`` is one of ``roots`` or below it, segment by segment.

    String-only on purpose: ``allowed_paths`` describes the shape of a path a
    worker declared, not a resolved location on disk, so the check must not
    depend on what happens to exist or be linked at the time.
    """
    roots = list(roots)
    if not roots:
        return False
    needle = normalize_rel_path(path).strip("/").split("/")
    for root in roots:
        base = normalize_rel_path(root).strip("/").split("/")
        if needle[: len(base)] == base:
            return True
    return False


def is_relative_to(path: Path, root: Path, *, resolve: bool = False) -> bool:
    """Whether ``path`` is ``root`` or below it."""
    if resolve:
        path, root = path.resolve(), root.resolve()
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def observation_dir_for(
    root: str | Path, session_id: str | None, agent_id: str | None = None
) -> Path:
    """Directory for observation files and streamed tool-output artifacts."""
    safe_session_id = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(session_id or "default")
    )
    safe_agent_id = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(agent_id or "main")
    )
    return Path(root) / ".harness" / "observations" / safe_session_id / safe_agent_id
