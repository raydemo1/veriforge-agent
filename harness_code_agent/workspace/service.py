from __future__ import annotations

import shutil
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from .. import config
from ..support.cancellation import current_token
from .access import WorkspaceAccessCoordinator, WorkspaceConflict, fingerprint
from .change_journal import WorkspaceChangeJournal
from .paths import is_relative_to
from .state_paths import ensure_workspace_state

_UNOBSERVED = object()


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

    def __init__(
        self,
        *,
        root: str | Path,
        snapshots_dir: str | Path | None = None,
        session_id: str | None = None,
        state_root: str | Path | None = None,
    ):
        self.root = Path(root).resolve()
        self.session_id = session_id
        self.turn_id = ""
        self.state_root = (
            Path(state_root).expanduser().resolve()
            if state_root is not None
            else ensure_workspace_state(self.root)
        )
        self.access = WorkspaceAccessCoordinator(self.root, state_root=self.state_root)
        self._versions = {}
        self.snapshots_dir = (
            Path(snapshots_dir).resolve()
            if snapshots_dir
            else self.state_root / "recovery" / "snapshots"
        )
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
            data = resolved.read_bytes()
            self._versions[str(resolved).casefold()] = fingerprint(data)
            return data.decode("utf-8", errors="replace")

    def write_text(self, path: str | Path, content: str) -> WorkspaceWriteResult:
        resolved = self.resolve(path)
        with self._path_lock(resolved, write=True):
            if resolved.exists() and str(resolved).casefold() not in self._versions:
                raise WorkspaceConflict(f"覆盖已有文件前必须先读取当前版本：{path}")
            self._ensure_writable(resolved)
            old_content = resolved.read_text(encoding="utf-8", errors="replace") if resolved.exists() else None
            old_size = resolved.stat().st_size if resolved.exists() else 0
            new_size = len(content.encode("utf-8"))
            self._check_write_quota(new_size - old_size)
            snapshot_path = self._snapshot_unlocked(resolved) if resolved.exists() else None
            resolved.parent.mkdir(parents=True, exist_ok=True)
            from ..runtime.recovery import atomic_bytes
            atomic_bytes(resolved, content.encode("utf-8"))
            self._charge_quota(new_size - old_size)
            rel = resolved.relative_to(self.root)
            self._record_changed(rel, operation="write_file", snapshot_path=snapshot_path)
            return WorkspaceWriteResult(path=resolved, snapshot_path=snapshot_path, old_content=old_content)

    def write_text_batch(self, changes: dict[str, str], *, expected=None) -> list[WorkspaceWriteResult]:
        """Apply a validated text change set, rolling every file back on failure."""
        return self.write_batch(
            {path: str(content).encode("utf-8") for path, content in changes.items()},
            expected=expected,
        )

    def write_batch(
        self, changes: dict[str, bytes | None], *, expected=None
    ) -> list[WorkspaceWriteResult]:
        """Apply bytes or deletions atomically, rolling every path back on failure.

        A ``None`` value deletes the file. Binary content is written as-is;
        text callers use :meth:`write_text_batch`.
        """
        if not changes:
            return []
        resolved_changes = {self.resolve(path): value for path, value in changes.items()}
        ordered = sorted(resolved_changes, key=lambda path: str(path).casefold())
        versions = {self.resolve(path): data for path, data in expected.items()} if expected is not None else {}
        with ExitStack() as stack:
            for path in ordered:
                stack.enter_context(self._path_lock(path, write=True, expected=versions.get(path, _UNOBSERVED)))
            originals: dict[Path, tuple[bool, bytes | None]] = {}
            results: list[WorkspaceWriteResult] = []
            total_net_new = 0
            for path in ordered:
                self._ensure_writable(path)
                existed = path.exists()
                if existed and not path.is_file():
                    raise ValueError(f"Refusing to replace non-file path: {path.relative_to(self.root)}")
                old_bytes = path.read_bytes() if existed else None
                new_bytes = resolved_changes[path]
                if new_bytes is None and not existed:
                    raise FileNotFoundError(f"File not found: {path.relative_to(self.root)}")
                old_content = old_bytes.decode("utf-8", errors="replace") if old_bytes is not None else None
                originals[path] = (existed, old_bytes)
                total_net_new += (len(new_bytes) if new_bytes is not None else 0) - (
                    len(old_bytes) if old_bytes is not None else 0
                )
                snapshot_path = self._snapshot_unlocked(path) if existed else None
                results.append(WorkspaceWriteResult(path=path, snapshot_path=snapshot_path, old_content=old_content))
            self._check_write_quota(total_net_new)
            try:
                from ..runtime.recovery import atomic_bytes
                for path in ordered:
                    new_bytes = resolved_changes[path]
                    if new_bytes is None:
                        path.unlink()
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_bytes(path, new_bytes)
            except Exception:
                for path in reversed(ordered):
                    existed, old_bytes = originals[path]
                    if existed:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        atomic_bytes(path, old_bytes or b"")
                    elif path.exists():
                        path.unlink()
                raise
            self._charge_quota(total_net_new)
            for result in results:
                self._record_changed(
                    result.path.relative_to(self.root),
                    operation="write_file" if resolved_changes[result.path] is not None else "delete_file",
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
        with self._path_lock(resolved, write=True):
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
            from ..runtime.recovery import atomic_bytes
            atomic_bytes(resolved, updated.encode("utf-8"))
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
        with self._path_lock(resolved, write=True):
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
        with self._path_lock(resolved, write=True):
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

    @contextmanager
    def _path_lock(self, resolved: Path, *, write=False, expected=_UNOBSERVED):
        key = str(resolved).casefold()
        with self._path_locks_lock:
            lock = self._path_locks.setdefault(key, threading.RLock())
        claims = [self.access.write_claim(resolved)] if write else []
        with self.access.acquire(claims, current_token.get()), lock:
            before = resolved.read_bytes() if resolved.is_file() else None
            version = self._versions.get(key, _UNOBSERVED) if expected is _UNOBSERVED else fingerprint(expected)
            if write and version is not _UNOBSERVED and version != fingerprint(before):
                raise WorkspaceConflict(f"文件版本已变化，请重新读取并重新计算修改：{resolved.relative_to(self.root)}")
            completed = False
            try:
                yield
                completed = True
            finally:
                after = resolved.read_bytes() if resolved.is_file() else None
                if write and before != after:
                    self.access.record(self.session_id, self.turn_id, resolved, before, after)
                if write and completed:
                    self._versions[key] = fingerprint(after)

    @property
    def changed_files(self) -> list[Path]:
        """Return the changed paths without exposing the mutable journal."""

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

    def _ensure_writable(self, path: Path) -> None:
        # Workspace boundary only: .git, .env and other project files are
        # ordinary workspace content. Snapshot and conflict checks still apply.
        path.relative_to(self.root)

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

    _is_relative_to = staticmethod(is_relative_to)
