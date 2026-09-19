from __future__ import annotations

import shutil
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from .. import config
from .change_journal import WorkspaceChangeJournal


class WorkspaceQuotaError(RuntimeError):
    """Raised when a write would exceed the workspace write/disk quota."""

    def __init__(
        self,
        reason: str,
        *,
        charged_bytes: int = 0,
        limit_bytes: int = 0,
        resource_kind: str = "workspace_quota",
    ):
        super().__init__(reason)
        self.reason = reason
        self.charged_bytes = charged_bytes
        self.limit_bytes = limit_bytes
        #: "workspace_quota" or "free_disk" — feeds the failure taxonomy.
        self.resource_kind = resource_kind


@dataclass
class WorkspaceWriteResult:
    path: Path
    snapshot_path: Path | None
    old_content: str | None = None


@dataclass
class WorkspacePatchResult:
    path: Path
    snapshot_path: Path | None
    replacements: int
    old_content: str = ""
    new_content: str = ""


class WorkspaceService:
    """Workspace path resolution and file snapshot service.

    Per-path locks keep each file transaction atomic without serializing
    unrelated files. Metadata bookkeeping has its own short critical section.
    """

    DEFAULT_PROTECTED_NAMES: ClassVar[set] = {".env", ".env.local", ".env.production"}

    def __init__(
        self,
        *,
        root: str | Path,
        snapshots_dir: str | Path | None = None,
        protected_names: set[str] | None = None,
    ):
        self.root = Path(root).resolve()
        self.snapshots_dir = Path(snapshots_dir).resolve() if snapshots_dir else self.root / ".harness" / "snapshots"
        self.protected_names = protected_names or self.DEFAULT_PROTECTED_NAMES
        self.change_journal = WorkspaceChangeJournal()
        self._metadata_lock = threading.RLock()
        self._path_locks_lock = threading.Lock()
        self._path_locks: dict[str, threading.RLock] = {}
        # Cumulative net-new bytes written through this service; the cheap
        # backstop against unbounded file generation. run_bash writes bypass
        # this ledger and are expected to be bounded by ulimit/docker caps.
        self._charged_bytes = 0

    def resolve(self, path: str | Path) -> Path:
        raw = Path(path)
        if raw.is_absolute():
            candidate = raw.resolve()
        else:
            candidate = (self.root / raw).resolve()
        if not self._is_relative_to(candidate, self.root):
            raise ValueError(f"Path escapes workspace: {path}")
        return candidate

    def read_text(self, path: str | Path) -> str:
        resolved = self.resolve(path)
        with self._path_lock(resolved):
            return resolved.read_text(encoding="utf-8", errors="replace")

    def write_text(self, path: str | Path, content: str) -> WorkspaceWriteResult:
        resolved = self.resolve(path)
        with self._path_lock(resolved):
            self._ensure_writable(resolved)
            old_content = resolved.read_text(encoding="utf-8", errors="replace") if resolved.exists() else None
            old_size = resolved.stat().st_size if resolved.exists() else 0
            new_size = len(content.encode("utf-8"))
            self._check_write_quota(new_size - old_size)
            snapshot_path = self._snapshot_unlocked(resolved) if resolved.exists() else None
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            self._charge_quota(new_size - old_size)
            rel = resolved.relative_to(self.root)
            self._record_changed(rel, operation="write_file", snapshot_path=snapshot_path)
            return WorkspaceWriteResult(path=resolved, snapshot_path=snapshot_path, old_content=old_content)

    def write_text_batch(self, changes: dict[str, str]) -> list[WorkspaceWriteResult]:
        """Apply a validated text change set, rolling every file back on failure."""
        if not changes:
            return []
        resolved_changes = {
            self.resolve(path): str(content)
            for path, content in changes.items()
        }
        ordered = sorted(resolved_changes, key=lambda path: str(path).casefold())
        with ExitStack() as stack:
            for path in ordered:
                stack.enter_context(self._path_lock(path))
            originals: dict[Path, tuple[bool, bytes | None]] = {}
            results: list[WorkspaceWriteResult] = []
            total_net_new = 0
            for path in ordered:
                self._ensure_writable(path)
                existed = path.exists()
                if existed and not path.is_file():
                    raise ValueError(f"Refusing to replace non-file path: {path.relative_to(self.root)}")
                old_bytes = path.read_bytes() if existed else None
                old_content = old_bytes.decode("utf-8", errors="replace") if old_bytes is not None else None
                originals[path] = (existed, old_bytes)
                new_bytes = resolved_changes[path].encode("utf-8")
                total_net_new += len(new_bytes) - (len(old_bytes) if old_bytes is not None else 0)
                snapshot_path = self._snapshot_unlocked(path) if existed else None
                results.append(WorkspaceWriteResult(path=path, snapshot_path=snapshot_path, old_content=old_content))
            self._check_write_quota(total_net_new)
            try:
                for path in ordered:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(resolved_changes[path], encoding="utf-8")
            except Exception:
                for path in reversed(ordered):
                    existed, old_bytes = originals[path]
                    if existed:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(old_bytes or b"")
                    elif path.exists():
                        path.unlink()
                raise
            self._charge_quota(total_net_new)
            for result in results:
                self._record_changed(
                    result.path.relative_to(self.root),
                    operation="write_file",
                    snapshot_path=result.snapshot_path,
                )
            return results

    def apply_text_patch(
        self,
        path: str | Path,
        *,
        search: str,
        replace: str,
    ) -> WorkspacePatchResult:
        resolved = self.resolve(path)
        with self._path_lock(resolved):
            if not search:
                raise ValueError("Patch search text must not be empty")
            self._ensure_writable(resolved)
            if not resolved.exists() or not resolved.is_file():
                raise FileNotFoundError(f"File not found: {path}")
            original = resolved.read_text(encoding="utf-8", errors="replace")
            count = original.count(search)
            if count != 1:
                raise ValueError(f"Patch search text must match exactly once; found {count}")
            snapshot_path = self._snapshot_unlocked(resolved)
            updated = original.replace(search, replace, 1)
            net_new_bytes = len(updated.encode("utf-8")) - resolved.stat().st_size
            self._check_write_quota(net_new_bytes)
            resolved.write_text(updated, encoding="utf-8")
            self._charge_quota(net_new_bytes)
            rel = resolved.relative_to(self.root)
            self._record_changed(rel, operation="apply_patch", snapshot_path=snapshot_path)
            return WorkspacePatchResult(
                path=resolved,
                snapshot_path=snapshot_path,
                replacements=1,
                old_content=original,
                new_content=updated,
            )

    def rollback_latest_snapshot(self, path: str | Path) -> WorkspaceWriteResult:
        resolved = self.resolve(path)
        with self._path_lock(resolved):
            self._ensure_writable(resolved)
            rel = resolved.relative_to(self.root)
            snapshot_dir = self.snapshots_dir / rel.parent
            pattern = f"{resolved.name}.*.bak"
            snapshots = sorted(
                snapshot_dir.glob(pattern),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            if not snapshots:
                raise FileNotFoundError(f"No snapshot found for: {rel}")
            old_content = resolved.read_text(encoding="utf-8", errors="replace") if resolved.exists() else None
            rollback_snapshot = self._snapshot_unlocked(resolved) if resolved.exists() else None
            resolved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snapshots[0], resolved)
            self._record_changed(rel, operation="rollback", snapshot_path=rollback_snapshot)
            return WorkspaceWriteResult(path=resolved, snapshot_path=rollback_snapshot, old_content=old_content)

    def snapshot(self, path: str | Path) -> Path | None:
        resolved = self.resolve(path)
        with self._path_lock(resolved):
            return self._snapshot_unlocked(resolved)

    def _snapshot_unlocked(self, resolved: Path) -> Path | None:
        if not resolved.exists() or not resolved.is_file():
            return None
        rel = resolved.relative_to(self.root)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        snapshot_path = self.snapshots_dir / rel.parent / f"{resolved.name}.{stamp}.bak"
        suffix = 1
        while snapshot_path.exists():
            snapshot_path = self.snapshots_dir / rel.parent / f"{resolved.name}.{stamp}.{suffix}.bak"
            suffix += 1
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved, snapshot_path)
        return snapshot_path

    def _path_lock(self, resolved: Path) -> threading.RLock:
        key = str(resolved).casefold()
        with self._path_locks_lock:
            return self._path_locks.setdefault(key, threading.RLock())

    @property
    def changed_files(self) -> list[Path]:
        """Compatibility snapshot; callers cannot mutate journal state through it."""

        return list(self.change_journal.changed_paths())

    def _record_changed(
        self,
        rel: Path,
        *,
        operation: str,
        snapshot_path: Path | None,
    ) -> None:
        with self._metadata_lock:
            self.change_journal.record(
                rel,
                operation=operation,
                snapshot_path=snapshot_path,
            )

    _SAFE_ENV_SUFFIXES: ClassVar[set] = {'.example', '.template', '.sample', '.default', '.dist'}

    def _ensure_writable(self, path: Path) -> None:
        rel = path.relative_to(self.root)
        parts = rel.parts
        if ".git" in parts:
            raise ValueError(f"Refusing to write inside .git: {rel}")
        if self._is_protected_name(path.name):
            raise ValueError(f"Refusing to write protected file: {rel}")

    def _check_write_quota(self, net_new_bytes: int) -> None:
        """Reject a write that would cross the free-disk floor or write quota."""
        net_new_bytes = max(0, int(net_new_bytes))
        if net_new_bytes <= 0:
            return
        min_free = int(getattr(config, "MIN_FREE_DISK_BYTES", 0) or 0)
        if min_free > 0:
            free = shutil.disk_usage(self.root).free
            if free < min_free + net_new_bytes:
                raise WorkspaceQuotaError(
                    f"workspace disk floor reached: only {free} bytes free "
                    f"(write needs {net_new_bytes}, floor is {min_free})",
                    charged_bytes=self._charged_bytes,
                    limit_bytes=min_free,
                    resource_kind="free_disk",
                )
        quota = int(getattr(config, "WORKSPACE_WRITE_QUOTA_BYTES", 0) or 0)
        if quota > 0 and self._charged_bytes + net_new_bytes > quota:
            raise WorkspaceQuotaError(
                f"workspace write quota exceeded: {self._charged_bytes} bytes already written, "
                f"quota is {quota} bytes (this write needs {net_new_bytes})",
                charged_bytes=self._charged_bytes,
                limit_bytes=quota,
            )

    def _charge_quota(self, net_new_bytes: int) -> None:
        with self._metadata_lock:
            self._charged_bytes += max(0, int(net_new_bytes))

    def _is_protected_name(self, name: str) -> bool:
        if name in self.protected_names:
            return True
        for p in self.protected_names:
            if name.startswith(p + '.') and not any(name.endswith(s) for s in self._SAFE_ENV_SUFFIXES):
                return True
        return False

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False
