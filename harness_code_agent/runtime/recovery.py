from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .middleware.base import AgentMiddleware

EXCLUDED_NAMES = frozenset({
    ".git", ".harness", "node_modules", ".venv", "venv", "__pycache__",
    ".pytest_cache", ".ruff_cache", ".mypy_cache", ".cache", "build", "dist",
})


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".recovery-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: dict) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))


class RecoveryStore:
    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()
        self.root = self.workspace / ".harness" / "recovery"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._depth = 0

    @contextmanager
    def exclusive(self):
        with self._lock:
            handle = None
            if self._depth == 0:
                handle = (self.root / "workspace.lock").open("a+b")
                if handle.tell() == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                try:
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    handle.close()
                    raise RuntimeError("另一个会话正在操作此工作区，请等待它结束") from exc
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
                if handle is not None:
                    if os.name == "nt":
                        import msvcrt
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle, fcntl.LOCK_UN)
                    handle.close()

    def _put_blob(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        path = self.root / "blobs" / digest
        if not path.exists():
            atomic_bytes(path, data)
        return digest

    def blob(self, digest: str) -> bytes:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("恢复数据的摘要无效")
        data = (self.root / "blobs" / digest).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("恢复数据校验失败")
        return data

    def capture(self) -> str:
        manifest = {}

        def visit(directory: Path):
            for path in sorted(directory.iterdir()):
                if path.name in EXCLUDED_NAMES:
                    continue
                rel = path.relative_to(self.workspace).as_posix()
                info = path.lstat()
                if getattr(path, "is_junction", lambda: False)():
                    raise ValueError(f"无法为目录联接创建恢复快照：{rel}")
                mode = stat.S_IMODE(info.st_mode)
                if stat.S_ISLNK(info.st_mode):
                    manifest[rel] = {"kind": "link", "target": os.readlink(path), "directory": bool(getattr(info, "st_file_attributes", 0) & 0x10)}
                elif stat.S_ISDIR(info.st_mode):
                    manifest[rel] = {"kind": "directory", "mode": mode}
                    visit(path)
                elif stat.S_ISREG(info.st_mode):
                    if info.st_nlink > 1:
                        raise ValueError(f"无法为硬链接创建恢复快照：{rel}")
                    data = path.read_bytes()
                    after = path.lstat()
                    if (info.st_size, info.st_mtime_ns, info.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                        raise RuntimeError(f"快照期间文件发生变化，请重试：{rel}")
                    manifest[rel] = {"kind": "file", "hash": self._put_blob(data), "mode": mode}
                else:
                    raise ValueError(f"无法为特殊文件创建恢复快照：{rel}")

        visit(self.workspace)
        digest = self._put_blob(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        return digest

    def manifest(self, snapshot: str) -> dict:
        result = json.loads(self.blob(snapshot))
        if not isinstance(result, dict):
            raise TypeError("文件快照无效")
        for name, entry in result.items():
            self._path(name)
            if entry.get("kind") not in {"file", "directory", "link"}:
                raise ValueError(f"文件快照条目无效：{name}")
        return result

    def _path(self, name: str) -> Path:
        parts = PurePosixPath(name).parts
        if not parts or "\\" in name or ":" in name or PurePosixPath(name).is_absolute() or any(p in {"..", "."} or p in EXCLUDED_NAMES for p in parts):
            raise ValueError(f"恢复路径无效：{name}")
        return self.workspace.joinpath(*parts)

    def read_point(self, point_id: str) -> dict:
        if not point_id.startswith("rp_") or any(c not in "0123456789abcdef" for c in point_id[3:]) or len(point_id) != 35:
            raise ValueError("回撤点无效")
        return json.loads((self.root / "points" / f"{point_id}.json").read_text(encoding="utf-8"))

    def publish(self, *, session_id: str, turn: int, journal_sequence: int, event_count: int, before: str | None, snapshot: str) -> dict:
        point = {
            "id": "rp_" + uuid.uuid4().hex, "session_id": session_id, "turn": turn,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "journal_sequence": journal_sequence, "event_count": event_count,
            "before_snapshot": before, "snapshot": snapshot,
        }
        atomic_json(self.root / "points" / f"{point['id']}.json", point)
        return point

    def changed(self, current: str, target: str) -> list[str]:
        left, right = self.manifest(current), self.manifest(target)
        return sorted(name for name in left.keys() | right.keys() if left.get(name) != right.get(name))

    def restore(self, current: str, target: str) -> None:
        left, right = self.manifest(current), self.manifest(target)
        changed = self.changed(current, target)
        # Validate every blob before the first workspace mutation.
        for name in changed:
            entry = right.get(name, {})
            if entry.get("kind") == "file":
                self.blob(entry["hash"])
        directories = set()
        for name in changed:
            parts = PurePosixPath(name).parts
            for length in range(1, len(parts) + 1):
                parent = "/".join(parts[:length])
                if any(manifest.get(parent, {}).get("kind") == "directory" for manifest in (left, right)):
                    directories.add(parent)
        ordered_directories = sorted(directories, key=lambda n: (n.count("/"), n))
        try:
            for name in ordered_directories:
                entry = left.get(name, {})
                if entry.get("kind") == "directory":
                    self._path(name).chmod(entry["mode"] | stat.S_IRWXU)
            self._restore_entries(left, right, changed)
        finally:
            for name in reversed(ordered_directories):
                path = self._path(name)
                if path.is_symlink() or not path.is_dir():
                    continue
                entry = right.get(name, {})
                if entry.get("kind") != "directory":
                    entry = left.get(name, {})
                path.chmod(entry["mode"])

    def _restore_entries(self, left: dict, right: dict, changed: list[str]) -> None:
        for name in sorted(changed, key=lambda n: (n.count("/"), n), reverse=True):
            previous, desired = left.get(name), right.get(name)
            if previous is None or (desired is not None and previous["kind"] == desired["kind"]):
                continue
            path = self._path(name)
            if previous["kind"] == "directory":
                path.rmdir()
            else:
                path.unlink()
        for name in sorted(changed, key=lambda n: (n.count("/"), n)):
            entry = right.get(name)
            if entry is None:
                continue
            path = self._path(name)
            for parent in path.parents:
                if parent == self.workspace:
                    break
                if parent.is_symlink() or getattr(parent, "is_junction", lambda: False)():
                    raise ValueError(f"恢复路径经过链接：{name}")
            if entry["kind"] == "directory":
                path.mkdir(exist_ok=True)
                path.chmod(entry["mode"] | stat.S_IRWXU)
            elif entry["kind"] == "link":
                if path.is_symlink():
                    path.unlink()
                path.symlink_to(entry["target"], target_is_directory=entry["directory"])
            else:
                atomic_bytes(path, self.blob(entry["hash"]))
            if entry["kind"] == "file":
                path.chmod(entry["mode"])

    def recover_interrupted(self) -> None:
        with self.exclusive():
            for path in sorted((self.root / "transactions").glob("*.json")):
                transaction = json.loads(path.read_text(encoding="utf-8"))
                if transaction["status"] != "prepared":
                    continue
                self.restore(self.capture(), transaction["before"])
                transaction["status"] = "rolled_back"
                atomic_json(path, transaction)
                target_session = transaction.get("target_session")
                if target_session and Path(target_session).name == target_session:
                    metadata_path = self.workspace / ".harness" / "sessions" / target_session / "session.json"
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    metadata["status"] = "discarded"
                    atomic_json(metadata_path, metadata)


class RecoveryService:
    def __init__(self, workspace: str | Path):
        self.store = RecoveryStore(workspace)
        self.store.recover_interrupted()
        self.before: str | None = None
        self.observed: str | None = None
        self.origin: dict | None = None
        self._start_location: dict = {}

    def begin_turn(self, **location) -> None:
        self.before = None
        self._start_location = location

    def protect(self) -> None:
        if self.before is None:
            self.before = self.store.capture()
            if self._start_location.get("turn") == 0 and self.origin is None:
                self.origin = self.store.publish(before=None, snapshot=self.before, **self._start_location)

    def finish_turn(self, **location) -> dict:
        if location["turn"] == 1 and self.origin is None and self.before is None:
            self.protect()
            snapshot = self.before
        else:
            snapshot = self.store.capture()
        session_root = self.store.workspace / ".harness" / "sessions" / location["session_id"]
        for name in ("journal.jsonl", "events.jsonl"):
            with (session_root / name).open("ab") as handle:
                handle.flush()
                os.fsync(handle.fileno())
        point = self.store.publish(before=self.before, snapshot=snapshot, **location)
        self.observed = snapshot
        return point


class RecoveryMiddleware(AgentMiddleware):
    def __init__(self, recovery: RecoveryService, tool_context, tool_registry):
        self.recovery = recovery
        self.context = tool_context
        self.registry = tool_registry

    def on_tool_allowed(self, tool_name, tool_args, messages, runtime_state=None, agent_name=None):
        if agent_name != "main_agent":
            return
        effect = self.registry.effect_for(tool_name, tool_args, self.context)
        if effect.kind in {"interaction", "control"}:
            return
        if any(
            claim.access == "write" and claim.domain in {"workspace", "global"}
            for claim in effect.resources
        ):
            self.recovery.protect()
