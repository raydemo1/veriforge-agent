from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


EXCLUDED_NAMES = frozenset({
    ".git", ".harness", "node_modules", ".venv", "venv", "__pycache__",
    ".pytest_cache", ".ruff_cache", ".mypy_cache", ".cache", "build", "dist",
})


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".recovery-", dir=path.parent)
    temporary = Path(name)
    try:
        if path.exists():
            os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
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

    def capture(self, paths=None) -> str:
        manifest = {}

        def visit(path):
            rel = path.relative_to(self.workspace).as_posix()
            if any(part in EXCLUDED_NAMES for part in Path(rel).parts):
                return
            info = path.lstat()
            if getattr(path, "is_junction", lambda: False)():
                raise ValueError(f"无法为目录联接创建恢复快照：{rel}")
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                manifest[rel] = {"kind": "link", "target": os.readlink(path), "directory": bool(getattr(info, "st_file_attributes", 0) & 0x10)}
            elif stat.S_ISDIR(info.st_mode):
                manifest[rel] = {"kind": "directory", "mode": mode}
                for child in sorted(path.iterdir()):
                    visit(child)
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

        for path in paths if paths is not None else [self.workspace]:
            path = Path(path)
            path.relative_to(self.workspace)
            if path == self.workspace:
                for child in sorted(path.iterdir()):
                    visit(child)
            elif path.exists() or path.is_symlink():
                visit(path)
        return self._put_blob(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"))

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



class RecoveryService:
    def __init__(self, workspace: str | Path):
        from ..workspace.access import WorkspaceAccessCoordinator
        self.store = RecoveryStore(workspace)
        self.access = WorkspaceAccessCoordinator(workspace)
        self.origin = None

    def begin_turn(self, **location):
        if location.get("turn") == 0 and self.origin is None:
            self.origin = self._publish(**location)

    def _publish(self, **location):
        point = dict(location, id="rp_" + uuid.uuid4().hex,
                     created_at=datetime.now(timezone.utc).isoformat(), change_cursor=self.access.cursor())
        atomic_json(self.store.root / "points" / f"{point['id']}.json", point)
        return point

    def finish_turn(self, **location):
        return self._publish(**location)
