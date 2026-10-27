from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import sqlite3
import tempfile
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .extraction_schema import EXTRACTION_SCHEMA
from .search_index import search_terms
from .vector_schema import VECTOR_SCHEMA, load_vector_extension

MemoryKind = Literal["preference", "project_fact", "convention", "decision", "lesson", "workflow"]
KINDS = frozenset({"preference", "project_fact", "convention", "decision", "lesson", "workflow"})
STATUSES = frozenset({"active", "review_required", "expired"})


@dataclass(frozen=True)
class EvidenceReference:
    session_id: str
    sequence: int
    kind: Literal["user_input", "verification", "manual"]
    content_hash: str
    path: str = ""
    fingerprint: str = ""
    role: Literal["support", "validity"] = "support"
    start_offset: int = 0
    end_offset: int = 0
    source_namespace: str = ""
    observed_at: float = 0.0


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    namespace: str
    kind: str
    topic_id: str
    topic: str
    fact_key: str
    body: str
    applicability: str
    status: str
    version: int
    updated_at: str
    evidence: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class WriteResult:
    record: MemoryRecord
    action: Literal["created", "updated", "unchanged", "conflict"]


class MemoryDatabase:
    def __init__(self, root: str | Path, namespace: str):
        if namespace != "user" and not namespace.startswith("project:"):
            raise ValueError("namespace must be user or project:<repo_key>")
        if namespace == "project:":
            raise ValueError("project identity is required")
        self.root = Path(root).expanduser().resolve()
        self.path = self.root / "memory.db"
        self.namespace = namespace
        self.export_root = self.root / "views" / hashlib.sha256(namespace.encode()).hexdigest()[:16]
        self.initialize()

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with contextlib.closing(sqlite3.connect(self.path, timeout=5)) as db:
            load_vector_extension(db)
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS state (
                    id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL);
                INSERT OR IGNORE INTO state VALUES(1,0);
                CREATE TABLE IF NOT EXISTS topics (
                    id TEXT PRIMARY KEY, namespace TEXT NOT NULL, title TEXT NOT NULL,
                    normalized_title TEXT NOT NULL, aliases TEXT NOT NULL DEFAULT '[]',
                    UNIQUE(namespace,normalized_title));
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY, namespace TEXT NOT NULL,
                    topic_id TEXT NOT NULL REFERENCES topics(id), fact_key TEXT NOT NULL,
                    applicability TEXT NOT NULL, kind TEXT NOT NULL, body TEXT NOT NULL,
                    status TEXT NOT NULL, version INTEGER NOT NULL, updated_at TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    UNIQUE(namespace,topic_id,fact_key,applicability));
                CREATE TABLE IF NOT EXISTS revisions (
                    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                    version INTEGER NOT NULL, snapshot TEXT NOT NULL,
                    PRIMARY KEY(memory_id,version));
                CREATE TABLE IF NOT EXISTS conflicts (
                    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                    proposals TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS suppressions (
                    namespace TEXT NOT NULL, key_hash TEXT NOT NULL,
                    PRIMARY KEY(namespace,key_hash));
                CREATE TABLE IF NOT EXISTS export_outbox (
                    namespace TEXT NOT NULL, topic_id TEXT NOT NULL,
                    PRIMARY KEY(namespace,topic_id));
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY, namespace TEXT NOT NULL, memory_id TEXT NOT NULL,
                    action TEXT NOT NULL, version INTEGER NOT NULL, timestamp TEXT NOT NULL);
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_search USING fts5(memory_id UNINDEXED, terms);
                CREATE INDEX IF NOT EXISTS memories_namespace ON memories(namespace,status);
            """)
            db.executescript(EXTRACTION_SCHEMA)
            db.executescript(VECTOR_SCHEMA)

    @contextlib.contextmanager
    def connect(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        load_vector_extension(db)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        try:
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @property
    def revision(self) -> int:
        with self.connect() as db:
            return db.execute("SELECT revision FROM state WHERE id=1").fetchone()[0]

    def _read(self, db, memory_id: str) -> MemoryRecord:
        row = db.execute(
            "SELECT m.*,t.title AS topic FROM memories m JOIN topics t ON t.id=m.topic_id "
            "WHERE m.id=? AND m.namespace=?", (memory_id, self.namespace),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown memory: {memory_id}")
        data = dict(row)
        data["evidence"] = json.loads(data["evidence"])
        return MemoryRecord(**data)

    def read(self, memory_id: str) -> MemoryRecord:
        with self.connect() as db:
            return self._read(db, memory_id)

    def records(self, *, include_inactive: bool = False, topic_id: str | None = None) -> list[MemoryRecord]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT id FROM memories WHERE namespace=? "
                "AND (? OR (status='active' AND NOT EXISTS (SELECT 1 FROM memory_holds h WHERE h.memory_id=memories.id))) AND (? IS NULL OR topic_id=?) ORDER BY id",
                (self.namespace, include_inactive, topic_id, topic_id),
            ).fetchall()
            return [self._read(db, row["id"]) for row in rows]

    def history(self, memory_id: str) -> list[dict]:
        with self.connect() as db:
            self._read(db, memory_id)
            return [json.loads(row[0]) for row in db.execute(
                "SELECT snapshot FROM revisions WHERE memory_id=? ORDER BY version", (memory_id,),
            )]

    def _topic(self, db, title: str) -> str:
        normalized = " ".join(title.split()).casefold()
        existing = _find_topic(db, self.namespace, title)
        if existing is not None:
            return existing
        topic_id = "topic_" + uuid.uuid4().hex
        db.execute("INSERT INTO topics(id,namespace,title,normalized_title) VALUES(?,?,?,?)",
                   (topic_id, self.namespace, title.strip(), normalized))
        return topic_id

    def _persist(self, db, record: MemoryRecord, action: str) -> None:
        db.execute(
            "INSERT INTO revisions VALUES(?,?,?)",
            (record.id, record.version, json.dumps(asdict(record), ensure_ascii=False)),
        )
        db.execute("UPDATE state SET revision=revision+1 WHERE id=1")
        db.execute("INSERT INTO audit(namespace,memory_id,action,version,timestamp) VALUES(?,?,?,?,?)",
                   (self.namespace, record.id, action, record.version, record.updated_at))
        self._queue_export(db, record.topic_id)
        db.execute("DELETE FROM memory_search WHERE memory_id=?", (record.id,))
        aliases = db.execute("SELECT aliases FROM topics WHERE id=?", (record.topic_id,)).fetchone()[0]
        db.execute("INSERT INTO memory_search VALUES(?,?)", (record.id, search_terms(
            f"{record.topic} {record.fact_key} {record.body} {record.applicability} {aliases}")))
        self._invalidate_vector(db, record.id)

    def scoped(self, namespace: str) -> MemoryDatabase:
        if namespace != "user" and (not namespace.startswith("project:") or namespace == "project:"):
            raise ValueError("invalid namespace")
        store = copy.copy(self)
        store.namespace = namespace
        store.export_root = self.root / "views" / hashlib.sha256(namespace.encode()).hexdigest()[:16]
        return store

    def _invalidate_vector(self, db, memory_id: str) -> None:
        db.execute("DELETE FROM memory_vectors WHERE rowid IN (SELECT id FROM vector_entries WHERE memory_id=?)",
                   (memory_id,))
        db.execute("DELETE FROM vector_entries WHERE memory_id=?", (memory_id,))
        db.execute("INSERT INTO vector_outbox(memory_id) VALUES(?) ON CONFLICT(memory_id) DO UPDATE SET "
                   "failures=0,available_at=0,error_type=NULL", (memory_id,))

    def _queue_export(self, db, topic_id: str) -> None:
        db.execute("INSERT OR IGNORE INTO export_outbox VALUES(?,?)", (self.namespace, topic_id))

    def write(
        self, *, topic: str, fact_key: str, kind: MemoryKind, body: str,
        evidence: list[EvidenceReference], applicability: str = "",
        memory_id: str | None = None, expected_version: int | None = None,
        restore_forgotten: bool = False,
    ) -> WriteResult:
        with self.connect(write=True) as db:
            return self._write(db, topic=topic, fact_key=fact_key, kind=kind, body=body,
                               evidence=evidence, applicability=applicability,
                               memory_id=memory_id, expected_version=expected_version, restore_forgotten=restore_forgotten)

    def _write(
        self, db, *, topic: str, fact_key: str, kind: MemoryKind, body: str,
        evidence: list[EvidenceReference], applicability: str = "",
        memory_id: str | None = None, expected_version: int | None = None,
        restore_forgotten: bool = False,
    ) -> WriteResult:
        if kind not in KINDS or not all(isinstance(v, str) and v.strip() for v in (topic, fact_key, body)):
            raise ValueError("kind, topic, fact_key and body must be valid")
        refs = _evidence(evidence)
        if self.namespace == "user" and (kind != "preference" or applicability.strip().startswith("path:")
                                         or any(ref["kind"] == "verification" or ref.get("role") == "validity" for ref in refs)):
            raise ValueError("User memory requires personal preferences independent of project files or verification")
        topic, fact_key, body, applicability = (v.strip() for v in (topic, fact_key, body, applicability))
        topic_id = self._topic(db, topic)
        topic = db.execute("SELECT title FROM topics WHERE id=?", (topic_id,)).fetchone()[0]
        key = _hash(self.namespace, " ".join(topic.split()).casefold(), fact_key, applicability)
        if restore_forgotten:
            db.execute("DELETE FROM suppressions WHERE namespace=? AND key_hash=?", (self.namespace, key))
        if db.execute("SELECT 1 FROM suppressions WHERE namespace=? AND key_hash=?",
                      (self.namespace, key)).fetchone():
            raise ValueError("This fact was forgotten; explicit restore is required")
        for ref in refs:
            source_key = _hash(ref["session_id"], str(ref["sequence"]), ref["content_hash"])
            if db.execute("SELECT 1 FROM suppressions WHERE namespace=? AND key_hash=?",
                          (self.namespace, source_key)).fetchone():
                raise ValueError("This source was forgotten")
        if memory_id:
            current = self._read(db, memory_id)
            _check_version(current, expected_version)
            if (current.topic_id, current.fact_key, current.applicability) != (topic_id, fact_key, applicability):
                raise ValueError("Updates cannot change fact identity")
        else:
            row = db.execute(
                "SELECT id FROM memories WHERE namespace=? AND topic_id=? AND fact_key=? AND applicability=?",
                (self.namespace, topic_id, fact_key, applicability),
            ).fetchone()
            current = self._read(db, row[0]) if row else None
            if expected_version is not None:
                raise ValueError("memory_id is required with expected_version")
        if current and current.body == body and current.kind == kind:
            replaced_paths = {ref["path"] for ref in refs if ref.get("role") == "validity"}
            retained = [ref for ref in current.evidence if ref.get("role") != "validity" or ref["path"] not in replaced_paths]
            merged = _merge_evidence(retained, refs)
            if merged == current.evidence and (not memory_id or current.status == "active"):
                return WriteResult(current, "unchanged")
            refs = merged
            status = "active" if memory_id else current.status
        elif current and not memory_id:
            proposal = {"body": body, "kind": kind, "evidence": refs}
            previous = db.execute("SELECT proposals FROM conflicts WHERE memory_id=?", (current.id,)).fetchone()
            proposals = json.loads(previous[0]) if previous else []
            if proposal in proposals:
                return WriteResult(current, "conflict")
            proposals.append(proposal)
            db.execute("INSERT INTO conflicts VALUES(?,?) ON CONFLICT(memory_id) "
                       "DO UPDATE SET proposals=excluded.proposals", (current.id, json.dumps(proposals)))
            db.execute("UPDATE memories SET status='review_required',version=version+1,updated_at=? WHERE id=?",
                       (_now(), current.id))
            record = self._read(db, current.id)
            self._persist(db, record, "conflict")
            return WriteResult(record, "conflict")
        else:
            status = "active"
        record_id = current.id if current else "mem_" + uuid.uuid4().hex
        version = current.version + 1 if current else 1
        db.execute(
            "INSERT INTO memories VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
            "kind=excluded.kind,body=excluded.body,status=excluded.status,version=excluded.version,"
            "updated_at=excluded.updated_at,evidence=excluded.evidence",
            (record_id, self.namespace, topic_id, fact_key, applicability, kind, body, status,
             version, _now(), json.dumps(refs, ensure_ascii=False)),
        )
        if memory_id:
            db.execute("DELETE FROM conflicts WHERE memory_id=?", (memory_id,))
        record = self._read(db, record_id)
        action = "updated" if current else "created"
        self._persist(db, record, action)
        if memory_id:
            self._retire_older_holds(db, record, refs)
        return WriteResult(record, action)

    def _retire_older_holds(self, db, record, evidence):
        confirmed_at = max((ref.get("observed_at", 0) for ref in evidence if ref.get("role") == "support"), default=0)
        if not confirmed_at:
            return
        rows = db.execute("SELECT c.id,c.payload FROM memory_candidates c JOIN memory_holds h ON h.candidate_id=c.id "
                          "WHERE h.memory_id=?", (record.id,)).fetchall()
        for row in rows:
            refs = json.loads(row["payload"])["evidence"]
            observed = max((ref.get("observed_at", 0) for ref in refs), default=0)
            if 0 < observed < confirmed_at:
                db.execute("INSERT OR IGNORE INTO memory_decisions VALUES(?,?,?,?,?,?)",
                           (row["id"], record.id, "history", record.version - 1, record.version, _now()))
                db.execute("UPDATE memory_candidates SET status='processed' WHERE id=?", (row["id"],))
                db.execute("DELETE FROM memory_holds WHERE candidate_id=?", (row["id"],))

    def proposals(self, memory_id: str) -> list[dict]:
        with self.connect() as db:
            self._read(db, memory_id)
            row = db.execute("SELECT proposals FROM conflicts WHERE memory_id=?", (memory_id,)).fetchone()
            return json.loads(row[0]) if row else []

    def resolve(self, memory_id: str, expected_version: int, *, body: str,
                evidence: list[EvidenceReference]) -> WriteResult:
        current = self.read(memory_id)
        _check_version(current, expected_version)
        known = current.evidence + [ref for proposal in self.proposals(memory_id) for ref in proposal["evidence"]]
        if not any(ref.kind in {"user_input", "verification", "manual"}
                   and asdict(ref) not in known for ref in evidence):
            raise ValueError("Resolving a memory requires new confirmation or verification evidence")
        return self.write(topic=current.topic, fact_key=current.fact_key, kind=current.kind,
                          body=body, evidence=evidence, applicability=current.applicability,
                          memory_id=memory_id, expected_version=expected_version)

    def restore_revision(self, memory_id: str, version: int, *, expected_version: int) -> MemoryRecord:
        with self.connect(write=True) as db:
            current = self._read(db, memory_id)
            _check_version(current, expected_version)
            row = db.execute("SELECT snapshot FROM revisions WHERE memory_id=? AND version=?",
                             (memory_id, version)).fetchone()
            if row is None:
                raise KeyError("Unknown memory revision")
            value = json.loads(row[0])
            if value["status"] != "active":
                raise ValueError("Restore requires a previously confirmed revision")
            db.execute("UPDATE memories SET body=?,kind=?,status=?,evidence=?,version=version+1,updated_at=? WHERE id=?",
                       (value["body"], value["kind"], value["status"], json.dumps(value["evidence"]), _now(), memory_id))
            db.execute("DELETE FROM conflicts WHERE memory_id=?", (memory_id,))
            # Keep pending changes visible as uncertainty after a rollback.
            restored = self._read(db, memory_id)
            self._persist(db, restored, "restored")
            return restored

    def forget(self, memory_id: str, expected_version: int) -> None:
        with self.connect(write=True) as db:
            current = self._read(db, memory_id)
            _check_version(current, expected_version)
            keys = {_hash(self.namespace, " ".join(current.topic.split()).casefold(),
                          current.fact_key, current.applicability)}
            references = []
            for row in db.execute("SELECT snapshot FROM revisions WHERE memory_id=?", (memory_id,)):
                references.extend(json.loads(row[0])["evidence"])
            for proposal in self.proposals_from_connection(db, memory_id):
                references.extend(proposal["evidence"])
            keys.update(_hash(ref["session_id"], str(ref["sequence"]), ref["content_hash"]) for ref in references)
            db.executemany("INSERT OR IGNORE INTO suppressions VALUES(?,?)", [(self.namespace, key) for key in keys])
            linked = [row[0] for row in db.execute("SELECT candidate_id FROM memory_decisions WHERE memory_id=?", (memory_id,))]
            for candidate_id in linked:
                row = db.execute("SELECT payload FROM memory_candidates WHERE id=?", (candidate_id,)).fetchone()
                if row:
                    references.extend(json.loads(row[0])["evidence"])
                db.execute("DELETE FROM memory_candidates WHERE id=?", (candidate_id,))
            keys.update(_hash(ref["session_id"], str(ref["sequence"]), ref["content_hash"]) for ref in references)
            db.executemany("INSERT OR IGNORE INTO suppressions VALUES(?,?)", [(self.namespace, key) for key in keys])
            self._forget_buffered(db, keys, references)
            self._invalidate_vector(db, memory_id)
            db.execute("DELETE FROM memory_search WHERE memory_id=?", (memory_id,))
            db.execute("DELETE FROM memories WHERE id=? AND namespace=?", (memory_id, self.namespace))
            db.execute("DELETE FROM topics WHERE id=? AND NOT EXISTS "
                       "(SELECT 1 FROM memories WHERE topic_id=?)", (current.topic_id, current.topic_id))
            db.execute("UPDATE state SET revision=revision+1 WHERE id=1")
            db.execute("INSERT INTO audit(namespace,memory_id,action,version,timestamp) VALUES(?,?,?,?,?)",
                       (self.namespace, memory_id, "forgotten", expected_version, _now()))
            self._queue_export(db, current.topic_id)

    def _forget_buffered(self, db, keys: set[str], references: list[dict]) -> None:
        for row in db.execute("SELECT id,payload FROM memory_candidates WHERE target_namespace=?",
                              (self.namespace,)).fetchall():
            value = json.loads(row["payload"])
            source_keys = {_hash(ref["session_id"], str(ref["sequence"]), ref["content_hash"])
                           for ref in value["evidence"]}
            fact_hash = _hash(self.namespace, " ".join(value["topic"].split()).casefold(),
                              value["fact_key"], value.get("applicability", "").strip())
            if fact_hash in keys or source_keys & keys:
                references.extend(value["evidence"])
                db.execute("DELETE FROM memory_candidates WHERE id=?", (row["id"],))
        for ref in references:
            db.execute("DELETE FROM memory_episodes WHERE namespace=? AND session_id=? AND sequence=?",
                       (ref.get("source_namespace") or self.namespace, ref["session_id"], ref["sequence"]))

    @staticmethod
    def proposals_from_connection(db, memory_id: str) -> list[dict]:
        row = db.execute("SELECT proposals FROM conflicts WHERE memory_id=?", (memory_id,)).fetchone()
        return json.loads(row[0]) if row else []

    def flush_exports(self) -> None:
        with self.connect(write=True) as db:
            pending = db.execute("SELECT topic_id FROM export_outbox WHERE namespace=?", (self.namespace,)).fetchall()
            if not pending:
                return
            self.export_root.mkdir(parents=True, exist_ok=True)
            for row in pending:
                topic_id = row[0]
                records = [self._read(db, item[0]) for item in db.execute(
                    "SELECT id FROM memories WHERE namespace=? AND topic_id=? AND status='active' AND NOT EXISTS (SELECT 1 FROM memory_holds h WHERE h.memory_id=memories.id) ORDER BY fact_key",
                    (self.namespace, topic_id),
                ).fetchall()]
                path = self.export_root / f"{topic_id}.md"
                if records:
                    lines = [f"# {records[0].topic}", "", "Generated view. Edit through memory commands.", ""]
                    for record in records:
                        lines.extend([f"## {record.fact_key}",
                                      f"{record.id} · v{record.version} · {record.kind} · {record.status}",
                                      "", record.body, "", "Sources:"])
                        lines.extend(f"- {ref['session_id']}:{ref['sequence']} ({ref['kind']})" for ref in record.evidence)
                        lines.append("")
                    _atomic_write(path, "\n".join(lines))
                else:
                    path.unlink(missing_ok=True)
            topics = db.execute(
                "SELECT DISTINCT t.id,t.title FROM topics t JOIN memories m ON m.topic_id=t.id "
                "WHERE m.namespace=? AND m.status='active' AND NOT EXISTS (SELECT 1 FROM memory_holds h WHERE h.memory_id=m.id) ORDER BY t.title", (self.namespace,),
            ).fetchall()
            lines = ["# Memory topics", "", "Generated view. SQLite is authoritative.", ""]
            lines.extend(f"- [{row['title']}]({row['id']}.md)" for row in topics)
            _atomic_write(self.export_root / "MEMORY.md", "\n".join(lines) + "\n")
            db.execute("DELETE FROM export_outbox WHERE namespace=?", (self.namespace,))


def _find_topic(db, namespace: str, title: str) -> str | None:
    normalized = " ".join(title.split()).casefold()
    row = db.execute("SELECT id FROM topics WHERE namespace=? AND normalized_title=?",
                     (namespace, normalized)).fetchone()
    if row:
        return row[0]
    topic_id = None
    for alias in db.execute("SELECT t.id,a.value FROM topics t,json_each(t.aliases) a WHERE t.namespace=?", (namespace,)):
        if " ".join(str(alias["value"]).split()).casefold() == normalized:
            if topic_id is not None and topic_id != alias["id"]:
                raise ValueError("Topic alias is ambiguous; use a canonical topic title")
            topic_id = alias["id"]
    return topic_id


def _evidence(values: list[EvidenceReference]) -> list[dict]:
    if not values:
        raise ValueError("Memory evidence is required")
    result = []
    for ref in values:
        if (not ref.session_id or isinstance(ref.sequence, bool) or not isinstance(ref.sequence, int)
                or ref.sequence < 1 or ref.kind not in {"user_input", "verification", "manual"}
                or ref.role not in {"support", "validity"} or len(ref.content_hash) != 64
                or any(ch not in "0123456789abcdef" for ch in ref.content_hash)):
            raise ValueError("Invalid memory evidence reference")
        if type(ref.observed_at) not in {int, float} or not math.isfinite(ref.observed_at) or ref.observed_at < 0:
            raise ValueError("Invalid evidence observation time")
        if ref.role == "validity" and (not ref.path or not ref.fingerprint):
            raise ValueError("Validity evidence requires a path and fingerprint")
        if (type(ref.start_offset) is not int or type(ref.end_offset) is not int
                or ref.start_offset < 0 or ref.end_offset < ref.start_offset):
            raise ValueError("Invalid evidence character range")
        result = _merge_evidence(result, [asdict(ref)])
    return result


def _merge_evidence(first: list[dict], second: list[dict]) -> list[dict]:
    result = list(first)
    for ref in second:
        if ref not in result:
            result.append(ref)
    return result


def _check_version(record: MemoryRecord, expected: int | None) -> None:
    if isinstance(expected, bool) or not isinstance(expected, int) or record.version != expected:
        raise ValueError(f"memory version changed: expected {expected}, current {record.version}")


def _hash(*values: str) -> str:
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".memory-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
