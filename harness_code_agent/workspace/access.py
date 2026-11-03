"""Project-wide write leases and versioned, session-owned file transactions."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import psutil

from ..runtime.execution_planner import ResourceClaim, claims_conflict

_owner = ContextVar("workspace_write_owner", default=None)


class WorkspaceConflict(RuntimeError):
    pass


def fingerprint(data: bytes | None) -> str | None:
    return hashlib.sha256(data).hexdigest() if data is not None else None


class WriteLease:
    def __init__(self, coordinator, key):
        self.coordinator = coordinator
        self.key = key
        self._lock = threading.Lock()
        self._released = False
        self.finalizers = []

    def release(self):
        with self._lock:
            if self._released:
                return
            try:
                for finalize in self.finalizers:
                    finalize()
            finally:
                self.finalizers.clear()
                with self.coordinator.connect() as db:
                    db.execute("DELETE FROM leases WHERE id=?", (self.key,))
                self._released = True


class WorkspaceAccessCoordinator:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.directory = self.root / ".harness" / "workspace"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "access.sqlite3"
        self.pid = os.getpid()
        self.born = psutil.Process(self.pid).create_time()
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS leases (
                    id TEXT PRIMARY KEY, owner TEXT, pid INTEGER, born REAL, claims TEXT);
                CREATE TABLE IF NOT EXISTS runtimes (session TEXT PRIMARY KEY, pid INTEGER, born REAL);
                CREATE TABLE IF NOT EXISTS changes (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT, turn TEXT,
                    path TEXT, before BLOB, after BLOB, reverted INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY, payload TEXT, status TEXT);
            """)
            pending = db.execute("SELECT 1 FROM transactions LIMIT 1").fetchone()
        if pending:
            self.recover_interrupted()

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=10)
        try:
            db.execute("PRAGMA busy_timeout=10000")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _alive(pid, born):
        try:
            return abs(psutil.Process(pid).create_time() - born) < 0.01
        except psutil.NoSuchProcess:
            return False

    @contextmanager
    def acquire(self, claims, cancellation_token=None):
        claims = tuple(claim for claim in claims if claim.access == "write")
        owner = _owner.get() or uuid.uuid4().hex
        key = uuid.uuid4().hex
        lease = WriteLease(self, key)
        while claims:
            if cancellation_token is not None:
                cancellation_token.check()
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                blocked = False
                for lease_id, other_owner, pid, born, raw in db.execute("SELECT * FROM leases").fetchall():
                    if not self._alive(pid, born):
                        db.execute("DELETE FROM leases WHERE id=?", (lease_id,))
                        continue
                    if other_owner != owner and any(
                        claims_conflict(wanted, ResourceClaim(**active))
                        for wanted in claims for active in json.loads(raw)
                    ):
                        blocked = True
                if not blocked:
                    db.execute("INSERT INTO leases VALUES (?,?,?,?,?)", (
                        key, owner, self.pid, self.born,
                        json.dumps([dict(domain=c.domain, key=c.key, scope=c.scope, access=c.access) for c in claims]),
                    ))
                    break
            if cancellation_token is not None:
                cancellation_token.wait(0.05)
            else:
                time.sleep(0.05)
        binding = _owner.set(owner)
        try:
            if cancellation_token is not None:
                cancellation_token.check()
            yield lease
        finally:
            _owner.reset(binding)
            lease.release()

    def reserve_runtime(self, session_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT pid,born FROM runtimes WHERE session=?", (session_id,)).fetchone()
            if existing and self._alive(*existing):
                raise RuntimeError("此会话已在另一个工作台运行")
            db.execute("INSERT OR REPLACE INTO runtimes VALUES (?,?,?)", (session_id, self.pid, self.born))

    def release_runtime(self, session_id):
        with self.connect() as db:
            db.execute("DELETE FROM runtimes WHERE session=? AND pid=? AND born=?", (session_id, self.pid, self.born))

    def write_claim(self, path: Path | None = None):
        path = (path or self.root).resolve()
        return ResourceClaim("workspace", os.path.normcase(str(path)),
                             "subtree" if path == self.root else "exact", "write")

    def cursor(self) -> int:
        with self.connect() as db:
            return db.execute("SELECT COALESCE(MAX(sequence),0) FROM changes").fetchone()[0]

    def record(self, session, turn, path, before, after):
        if before == after:
            return
        with self.connect() as db:
            db.execute("INSERT INTO changes(session,turn,path,before,after) VALUES (?,?,?,?,?)",
                       (session, turn, Path(path).relative_to(self.root).as_posix(), before, after))

    def limit_recovery(self, session, turn, reason):
        with self.connect() as db:
            db.execute("INSERT INTO changes(session,turn,path,before,after) VALUES (?,?,NULL,?,NULL)",
                       (session, turn, reason.encode("utf-8")))

    def check_rewind(self, session, cursor):
        with self.connect() as db:
            row = db.execute("SELECT before FROM changes WHERE session=? AND sequence>? AND path IS NULL AND reverted=0 LIMIT 1",
                             (session, cursor)).fetchone()
        if row:
            raise WorkspaceConflict("无法安全回撤：目标点之后包含无法完整追踪的操作（" + row[0].decode("utf-8") + "）。现有文件已保留。")

    def record_external(self, session, turn, path, before, after, cursor):
        if before == after:
            return
        name = Path(path).relative_to(self.root).as_posix()
        with self.connect() as db:
            db.execute("DELETE FROM changes WHERE session=? AND turn=? AND path=? AND sequence>?", (session, turn, name, cursor))
            db.execute("INSERT INTO changes(session,turn,path,before,after) VALUES (?,?,?,?,?)", (session, turn, name, before, after))

    def owned_changes(self, session, after=0):
        with self.connect() as db:
            return db.execute("SELECT sequence,path,before,after FROM changes WHERE session=? AND sequence>? AND reverted=0 ORDER BY sequence",
                              (session, after)).fetchall()

    @contextmanager
    def rewind_transaction(self, session, cursor, cancellation_token=None, *, participants=()):
        with self.acquire([self.write_claim()], cancellation_token):
            self.check_rewind(session, cursor)
            changes = self.owned_changes(session, cursor)
            grouped = {}
            for seq, name, before, after in changes:
                if name not in grouped:
                    grouped[name] = [before, after, seq]
                else:
                    if grouped[name][1] != before:
                        raise WorkspaceConflict(f"文件包含无法归属本会话的中间修改：{name}")
                    grouped[name][1] = after
                    grouped[name][2] = seq
            with self.connect() as db:
                for name, (before, after, last_seq) in grouped.items():
                    path = self.root / name
                    current = path.read_bytes() if path.exists() else None
                    intervening = db.execute("SELECT 1 FROM changes WHERE path=? AND sequence>? AND session<>? AND reverted=0 LIMIT 1",
                                             (name, last_seq, session)).fetchone()
                    if current != after or intervening:
                        raise WorkspaceConflict(f"其他会话或外部程序已修改文件，无法安全回撤：{name}")
            with self.transaction({name: before for name, (before, _, _) in grouped.items()},
                                  participants=participants, reverted=[row[0] for row in changes]):
                yield

    def rewind(self, session, cursor, cancellation_token=None):
        with self.rewind_transaction(session, cursor, cancellation_token):
            pass

    @contextmanager
    def transaction(self, changes: dict[str, bytes | None], *, participants=(), reverted=(), change_records=()):
        from ..runtime.recovery import atomic_bytes
        originals = {}
        for name in changes:
            path = (self.root / name).resolve()
            path.relative_to(self.root)
            if ".git" in path.relative_to(self.root).parts or ".harness" in path.relative_to(self.root).parts:
                raise ValueError(f"不允许修改内部路径：{name}")
            originals[name] = path.read_bytes() if path.exists() else None
        key = uuid.uuid4().hex
        saved = {str(Path(path).resolve()): Path(path).read_bytes() if Path(path).exists() else None
                 for path in participants}
        payload = json.dumps({"files": {name: {"before": before.hex() if before is not None else None,
                                    "after": changes[name].hex() if changes[name] is not None else None}
                              for name, before in originals.items()},
                              "participants": {name: data.hex() if data is not None else None for name, data in saved.items()}})
        with self.connect() as db:
            db.execute("INSERT INTO transactions VALUES (?,?,?)", (key, payload, "prepared"))
        try:
            for name, data in changes.items():
                path = self.root / name
                if data is None:
                    path.unlink(missing_ok=True)
                else:
                    atomic_bytes(path, data)
            yield
            with self.connect() as db:
                db.executemany("INSERT INTO changes(session,turn,path,before,after) VALUES (?,?,?,?,?)", change_records)
                db.executemany("UPDATE changes SET reverted=1 WHERE sequence=?", [(seq,) for seq in reverted])
                db.execute("DELETE FROM transactions WHERE id=?", (key,))
        except BaseException:
            for name, data in originals.items():
                if data is None:
                    (self.root / name).unlink(missing_ok=True)
                else:
                    atomic_bytes(self.root / name, data)
            for name, data in saved.items():
                if data is None:
                    Path(name).unlink(missing_ok=True)
                else:
                    atomic_bytes(Path(name), data)
            with self.connect() as db:
                db.execute("DELETE FROM transactions WHERE id=?", (key,))
            raise

    def apply(self, changes: dict[str, bytes | None]):
        with self.transaction(changes):
            pass

    def recover_interrupted(self):
        from ..runtime.recovery import atomic_bytes
        with self.acquire([self.write_claim()]):
            with self.connect() as db:
                pending = db.execute("SELECT id,payload FROM transactions WHERE status='prepared'").fetchall()
            for key, payload in pending:
                transaction = json.loads(payload)
                files = transaction["files"]
                for name, values in files.items():
                    path = (self.root / name).resolve()
                    path.relative_to(self.root)
                    current = path.read_bytes() if path.is_file() else None
                    before = bytes.fromhex(values["before"]) if values["before"] is not None else None
                    after = bytes.fromhex(values["after"]) if values["after"] is not None else None
                    if current not in (before, after):
                        raise WorkspaceConflict(f"未完成事务遇到后续修改，已保留现场：{name}")
                for name, values in files.items():
                    if values["before"] is None:
                        (self.root / name).unlink(missing_ok=True)
                    else:
                        atomic_bytes(self.root / name, bytes.fromhex(values["before"]))
                for name, data in transaction["participants"].items():
                    path = Path(name)
                    if data is None:
                        path.unlink(missing_ok=True)
                    else:
                        atomic_bytes(path, bytes.fromhex(data))
                with self.connect() as db:
                    db.execute("DELETE FROM transactions WHERE id=?", (key,))
