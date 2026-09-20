from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

MemoryScope = Literal["project", "user"]
MemoryStatus = Literal["active", "review_required", "superseded"]
GIT_IDENTITY_TIMEOUT_SECONDS = 0.5
_PROCESS_LOCK = threading.RLock()
_HEADER = "<!-- veriforge-memory\n"
_FOOTER = "\n-->\n"


@dataclass
class MemoryDocument:
    id: str
    scope: MemoryScope
    topic: str
    status: MemoryStatus = "active"
    applicability: str = ""
    source_sessions: list[str] = field(default_factory=list)
    source_paths: list[str] = field(default_factory=list)
    evidence_fingerprints: dict[str, str] = field(default_factory=dict)
    supersedes: str | None = None
    superseded_by: str | None = None
    version: int = 1
    updated_at: str = ""
    body: str = ""

    @classmethod
    def from_dict(cls, data: dict, *, body: str) -> MemoryDocument:
        names = {item.name for item in cls.__dataclass_fields__.values()}
        values = {key: value for key, value in data.items() if key in names and key != "body"}
        return cls(**values, body=body)

    def metadata(self) -> dict:
        data = asdict(self)
        data.pop("body")
        return data


@dataclass(frozen=True)
class MemoryWriteCommand:
    topic: str
    body: str
    scope: MemoryScope = "project"
    applicability: str = ""
    source_sessions: list[str] = field(default_factory=list)
    source_paths: list[str] = field(default_factory=list)
    memory_id: str | None = None
    expected_version: int | None = None
    supersedes: str | None = None


def default_memory_root(workspace: str | Path, *, scope: MemoryScope = "project") -> Path:
    override = os.environ.get("HARNESS_MEMORY_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        return root if scope == "project" else root.parent / "user-memory"
    base = Path.home() / ".harness"
    if scope == "user":
        return base / "memory"
    return base / "projects" / resolve_repo_key(Path(workspace)) / "memory-v2"


def resolve_repo_key(workspace: Path) -> str:
    workspace = workspace.resolve()
    repo_name = workspace.name
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"], cwd=workspace,
            capture_output=True, text=True, check=True,
            timeout=GIT_IDENTITY_TIMEOUT_SECONDS, stdin=subprocess.DEVNULL,
            env=_noninteractive_git_env(), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        common = Path(proc.stdout.strip())
        common = common if common.is_absolute() else (workspace / common).resolve()
        basis = str(common)
        if common.name.lower() == ".git":
            repo_name = common.parent.name
    except (OSError, subprocess.SubprocessError):
        basis = str(workspace)
    digest = hashlib.sha256(basis.encode()).hexdigest()[:12]
    name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in repo_name)
    return f"{name or 'workspace'}-{digest}"


class MemoryStore:
    """Markdown is authoritative; SQLite is a rebuildable index and work queue."""

    def __init__(self, root: str | Path, *, workspace: str | Path, scope: MemoryScope = "project"):
        self.root = Path(root).expanduser().resolve()
        self.workspace = Path(workspace).resolve()
        self.scope = scope
        self.entries_dir = self.root / "entries"
        self.index_path = self.root / "MEMORY.md"
        self.db_path = self.root / "memory.db"
        self._lock_path = self.root / ".memory.lock"

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with _PROCESS_LOCK:
            while True:
                try:
                    fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                    break
                except FileExistsError:
                    if time.time() - self._lock_path.stat().st_mtime > 300:
                        self._lock_path.unlink(missing_ok=True)
                        continue
                    time.sleep(0.05)
            try:
                yield
            finally:
                self._lock_path.unlink(missing_ok=True)

    def ensure_initialized(self) -> None:
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                  id TEXT PRIMARY KEY, path TEXT NOT NULL, scope TEXT NOT NULL,
                  topic TEXT NOT NULL, status TEXT NOT NULL, version INTEGER NOT NULL,
                  updated_at TEXT NOT NULL, mtime_ns INTEGER NOT NULL, search_text TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS extraction_jobs (
                  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                  boundary INTEGER NOT NULL, journal_path TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                  available_at REAL NOT NULL DEFAULT 0, lease_until REAL, error TEXT,
                  UNIQUE(session_id, boundary));
                CREATE TABLE IF NOT EXISTS suppressions (key TEXT PRIMARY KEY, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS usage (
                  memory_id TEXT PRIMARY KEY, use_count INTEGER NOT NULL DEFAULT 0, last_used_at TEXT);
            """)
        if not self.index_path.exists():
            self.rebuild_navigation()

    @contextlib.contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.root.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.db_path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def list_documents(self, *, include_superseded: bool = False) -> list[MemoryDocument]:
        self.ensure_initialized()
        self.sync_index()
        docs = []
        for path in sorted(self.entries_dir.glob("*.md")):
            try:
                doc = self._read_path(path)
            except (OSError, ValueError, json.JSONDecodeError, TypeError):
                continue
            if include_superseded or doc.status != "superseded":
                docs.append(doc)
        return docs

    def read(self, memory_id: str) -> MemoryDocument:
        path = self._entry_path(memory_id)
        if not path.exists():
            raise KeyError(f"Unknown memory: {memory_id}")
        return self._read_path(path)

    def write(self, command: MemoryWriteCommand) -> MemoryDocument:
        if not command.topic.strip() or not command.body.strip():
            raise ValueError("topic and body are required")
        if command.scope != self.scope:
            raise ValueError(f"store scope is {self.scope}, not {command.scope}")
        self.ensure_initialized()
        with self.lock():
            self.sync_index()
            current = self.read(command.memory_id) if command.memory_id else None
            superseded = self.read(command.supersedes) if command.supersedes else None
            if current:
                if command.expected_version is None:
                    raise ValueError("expected_version is required when updating a memory")
                self._check_version(current, command.expected_version)
            if superseded:
                if current:
                    raise ValueError("memory_id and supersedes cannot be used together")
                if command.expected_version is None:
                    raise ValueError("expected_version is required when superseding a memory")
                self._check_version(superseded, command.expected_version)
                if superseded.status == "superseded":
                    raise ValueError(f"memory is already superseded by {superseded.superseded_by}")
            memory_id = current.id if current else command.memory_id or "mem_" + uuid.uuid4().hex[:16]
            doc = MemoryDocument(
                id=memory_id, scope=self.scope, topic=command.topic.strip(),
                status=current.status if current else "active",
                applicability=command.applicability.strip(),
                source_sessions=_clean_list(command.source_sessions),
                source_paths=_clean_list(command.source_paths),
                evidence_fingerprints=fingerprint_paths(self.workspace, command.source_paths),
                supersedes=command.supersedes or (current.supersedes if current else None),
                superseded_by=current.superseded_by if current else None,
                version=current.version + 1 if current else 1,
                updated_at=_utc_now(), body=command.body.strip(),
            )
            self._atomic_write(self._entry_path(doc.id), _render(doc))
            if superseded:
                superseded.status, superseded.superseded_by = "superseded", doc.id
                superseded.version += 1
                superseded.updated_at = doc.updated_at
                self._atomic_write(self._entry_path(superseded.id), _render(superseded))
            self.sync_index(force=True)
            self.rebuild_navigation()
            return doc

    def set_status(self, memory_id: str, status: MemoryStatus, *, expected_version: int) -> MemoryDocument:
        if status not in {"active", "review_required", "superseded"}:
            raise ValueError(f"invalid memory status: {status}")
        with self.lock():
            self.sync_index()
            doc = self.read(memory_id)
            self._check_version(doc, expected_version)
            doc.status, doc.version, doc.updated_at = status, doc.version + 1, _utc_now()
            self._atomic_write(self._entry_path(doc.id), _render(doc))
            self.sync_index(force=True)
            self.rebuild_navigation()
            return doc

    def validate(self, memory_id: str, *, expected_version: int) -> MemoryDocument:
        with self.lock():
            self.sync_index()
            doc = self.read(memory_id)
            self._check_version(doc, expected_version)
            doc.evidence_fingerprints = fingerprint_paths(self.workspace, doc.source_paths)
            doc.status, doc.version, doc.updated_at = "active", doc.version + 1, _utc_now()
            self._atomic_write(self._entry_path(doc.id), _render(doc))
            self.sync_index(force=True)
            self.rebuild_navigation()
            return doc

    def forget(self, memory_id: str, *, expected_version: int) -> None:
        with self.lock():
            self.sync_index()
            doc = self.read(memory_id)
            self._check_version(doc, expected_version)
            lineage = self._linked_documents(doc)
            keys = [f"id:{item.id}" for item in lineage]
            keys.extend(
                "session:" + hashlib.sha256(session.encode()).hexdigest()
                for item in lineage
                for session in item.source_sessions
            )
            for item in lineage:
                self._entry_path(item.id).unlink(missing_ok=True)
            with self.connect() as db:
                db.executemany("DELETE FROM documents WHERE id=?", [(item.id,) for item in lineage])
                db.executemany("DELETE FROM usage WHERE memory_id=?", [(item.id,) for item in lineage])
                db.executemany("INSERT OR IGNORE INTO suppressions VALUES (?,?)", [(x, _utc_now()) for x in keys])
            self.rebuild_navigation()

    def _linked_documents(self, root: MemoryDocument) -> list[MemoryDocument]:
        docs = {}
        for path in self.entries_dir.glob("*.md"):
            with contextlib.suppress(OSError, ValueError, json.JSONDecodeError, TypeError):
                item = self._read_path(path)
                docs[item.id] = item
        selected = {root.id}
        changed = True
        while changed:
            changed = False
            for item in docs.values():
                if item.id in selected or item.supersedes in selected or item.superseded_by in selected:
                    before = len(selected)
                    selected.add(item.id)
                    if item.supersedes:
                        selected.add(item.supersedes)
                    if item.superseded_by:
                        selected.add(item.superseded_by)
                    changed = changed or len(selected) != before
        return [docs[item] for item in selected if item in docs]

    def refresh_applicability(self, doc: MemoryDocument) -> MemoryDocument:
        current = fingerprint_paths(self.workspace, doc.source_paths)
        if doc.status == "active" and doc.evidence_fingerprints and current != doc.evidence_fingerprints:
            return self.set_status(doc.id, "review_required", expected_version=doc.version)
        return doc

    def enqueue_extraction(self, session_id: str, boundary: int, journal_path: str | Path) -> None:
        self.ensure_initialized()
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO extraction_jobs(session_id,boundary,journal_path,available_at) VALUES(?,?,?,?)",
                (session_id, boundary, str(Path(journal_path).resolve()), time.time()),
            )

    def sync_index(self, *, force: bool = False) -> None:
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            known = {row["id"]: row for row in db.execute("SELECT * FROM documents")}
            seen = set()
            for path in self.entries_dir.glob("*.md"):
                doc = self._read_path(path)
                seen.add(doc.id)
                mtime = path.stat().st_mtime_ns
                if not force and doc.id in known and known[doc.id]["mtime_ns"] == mtime:
                    continue
                if not force and doc.id in known and doc.version <= int(known[doc.id]["version"]):
                    doc.version = int(known[doc.id]["version"]) + 1
                    doc.updated_at = _utc_now()
                    self._atomic_write(path, _render(doc))
                    mtime = path.stat().st_mtime_ns
                db.execute(
                    "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                    "path=excluded.path,scope=excluded.scope,topic=excluded.topic,status=excluded.status,"
                    "version=excluded.version,updated_at=excluded.updated_at,mtime_ns=excluded.mtime_ns,search_text=excluded.search_text",
                    (doc.id, str(path), doc.scope, doc.topic, doc.status, doc.version, doc.updated_at, mtime, _search_text(doc)),
                )
            db.executemany("DELETE FROM documents WHERE id=?", [(item,) for item in set(known) - seen])

    def indexed_documents(self) -> list[sqlite3.Row]:
        self.ensure_initialized()
        self.sync_index()
        with self.connect() as db:
            return list(db.execute("SELECT * FROM documents WHERE status IN ('active','review_required')"))

    def note_usage(self, memory_ids: list[str]) -> None:
        with self.connect() as db:
            db.executemany(
                "INSERT INTO usage VALUES(?,1,?) ON CONFLICT(memory_id) DO UPDATE SET "
                "use_count=use_count+1,last_used_at=excluded.last_used_at",
                [(item, _utc_now()) for item in memory_ids],
            )

    def rebuild_index(self) -> None:
        self.ensure_initialized()
        self.sync_index(force=True)
        self.rebuild_navigation()

    def rebuild_navigation(self) -> None:
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        docs = []
        for path in sorted(self.entries_dir.glob("*.md")):
            with contextlib.suppress(OSError, ValueError, json.JSONDecodeError, TypeError):
                docs.append(self._read_path(path))
        lines = ["# Long-term memory", "", "Generated navigation. Entry files are authoritative.", ""]
        lines.extend(f"- [{d.topic}](entries/{d.id}.md) — {d.status}; v{d.version}" for d in docs)
        if not docs:
            lines.append("_No memories yet._")
        self._atomic_write(self.index_path, "\n".join(lines) + "\n")

    def _entry_path(self, memory_id: str) -> Path:
        if not memory_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in memory_id):
            raise ValueError("invalid memory id")
        return self.entries_dir / f"{memory_id}.md"

    @staticmethod
    def _check_version(doc: MemoryDocument, expected: int) -> None:
        if doc.version != expected:
            raise ValueError(f"memory version changed: expected {expected}, current {doc.version}")

    @staticmethod
    def _read_path(path: Path) -> MemoryDocument:
        text = path.read_text(encoding="utf-8")
        if not text.startswith(_HEADER) or _FOOTER not in text:
            raise ValueError(f"invalid memory document: {path.name}")
        raw, body = text[len(_HEADER):].split(_FOOTER, 1)
        body = body.strip()
        if body.startswith("# "):
            body = body.split("\n", 1)[1].strip() if "\n" in body else ""
        return MemoryDocument.from_dict(json.loads(raw), body=body)

    @staticmethod
    def _atomic_write(target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
            os.replace(temp, target)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp)


def fingerprint_paths(workspace: Path, paths: list[str]) -> dict[str, str]:
    result = {}
    root = workspace.resolve()
    for raw in _clean_list(paths):
        path = Path(raw) if Path(raw).is_absolute() else root / raw
        try:
            path = path.resolve()
            key = path.relative_to(root).as_posix()
        except (OSError, ValueError):
            continue
        if not path.is_file():
            result[key] = "missing"
            continue
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(65536), b""):
                    digest.update(chunk)
            result[key] = digest.hexdigest()
        except OSError:
            result[key] = "unreadable"
    return result


def _render(doc: MemoryDocument) -> str:
    return f"{_HEADER}{json.dumps(doc.metadata(), ensure_ascii=False, indent=2, sort_keys=True)}{_FOOTER}# {doc.topic}\n\n{doc.body}\n"


def _search_text(doc: MemoryDocument) -> str:
    return " ".join([doc.topic, doc.body, doc.applicability, *doc.source_paths])


def _clean_list(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _noninteractive_git_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    return env


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
