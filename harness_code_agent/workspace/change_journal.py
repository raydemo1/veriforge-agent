"""Thread-safe, ordered record of workspace mutations."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspaceChange:
    sequence: int
    path: Path
    operation: str
    snapshot_path: Path | None = None


class WorkspaceChangeJournal:
    """Append-only mutation history with stable cursors for turn-local queries."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._changes: list[WorkspaceChange] = []

    def record(
        self,
        path: str | Path,
        *,
        operation: str,
        snapshot_path: Path | None = None,
    ) -> WorkspaceChange:
        relative_path = Path(path)
        with self._lock:
            change = WorkspaceChange(
                sequence=len(self._changes) + 1,
                path=relative_path,
                operation=str(operation),
                snapshot_path=snapshot_path,
            )
            self._changes.append(change)
            return change

    def cursor(self) -> int:
        with self._lock:
            return len(self._changes)

    def changes_since(self, cursor: int = 0) -> tuple[WorkspaceChange, ...]:
        start = max(0, int(cursor))
        with self._lock:
            return tuple(self._changes[start:])

    def paths_since(self, cursor: int = 0) -> tuple[Path, ...]:
        return _unique_paths(change.path for change in self.changes_since(cursor))

    def changed_paths(self) -> tuple[Path, ...]:
        return self.paths_since(0)


def _unique_paths(paths) -> tuple[Path, ...]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return tuple(unique)
