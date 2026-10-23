from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time

import httpx

from .database import MemoryDatabase, _hash
from .security import contains_secret

MODEL = "BAAI/bge-m3"
DIMENSIONS = 1024


class SiliconFlowEmbeddings:
    def __init__(self):
        self.base_url = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
        self.contract = _hash(self.base_url, MODEL, str(DIMENSIONS), "memory-text-v1")

    def embed(self, texts: list[str]) -> tuple[list[list[float]], dict]:
        key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
        if not key:
            raise ValueError("SILICONFLOW_API_KEY is required for memory embeddings")
        if not 1 <= len(texts) <= 32 or any(not text.strip() or contains_secret(text) for text in texts):
            raise ValueError("Invalid embedding input")
        with httpx.Client(timeout=httpx.Timeout(30, connect=10)) as client:
            response = client.post(self.base_url + "/embeddings", headers={"Authorization": "Bearer " + key},
                                   json={"model": MODEL, "input": texts, "encoding_format": "float"})
            response.raise_for_status()
            body = response.json()
            rows = body.get("data", [])
            if (len(rows) != len(texts) or {r.get("index") for r in rows} != set(range(len(texts)))
                    or body.get("model") != MODEL):
                raise ValueError("Embedding response contract mismatch")
            vectors = [_normalize(r["embedding"]) for r in sorted(rows, key=lambda r: r["index"])]
            return vectors, {"usage": body.get("usage"), "trace_id": response.headers.get("x-siliconcloud-trace-id")}


class EmbeddingBudgetExceeded(RuntimeError):
    pass


class BudgetedEmbeddings:
    def __init__(self, limit: int):
        self.client = SiliconFlowEmbeddings()
        self.contract = self.client.contract
        self.limit = limit
        self.used = 0

    @property
    def remaining(self):
        return max(0, self.limit - self.used)

    def embed(self, texts):
        if self.remaining == 0:
            raise EmbeddingBudgetExceeded("Embedding request budget exhausted")
        self.used += 1
        return self.client.embed(texts)


def _normalize(vector):
    if (not isinstance(vector, list) or len(vector) != DIMENSIONS
            or any(type(v) not in {int, float} or not math.isfinite(v) for v in vector)):
        raise ValueError("Invalid embedding dimensions or values")
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 0 or not math.isfinite(norm):
        raise ValueError("Invalid embedding norm")
    return [v / norm for v in vector]


def _blob(vector):
    return struct.pack(f"{DIMENSIONS}f", *_normalize(vector))


def _text(record):
    return f"{record.topic}\n{record.fact_key}\n{record.applicability}\n{record.body}"


class VectorIndex:
    def __init__(self, database: MemoryDatabase, *, embeddings=None):
        self.database = database
        self.embeddings = embeddings or SiliconFlowEmbeddings()
        self._query_cache = {}

    def _embed(self, texts, purpose):
        with self.database.connect(write=True) as db:
            call_id = db.execute(
                "INSERT INTO embedding_calls(namespace,purpose,contract,input_hash,input_count,error_type) "
                "VALUES(?,?,?,?,?,'in_flight')", (self.database.namespace, purpose, self.embeddings.contract,
                                               _hash(*texts), len(texts)),
            ).lastrowid
        started, metadata, error = time.monotonic(), {}, None
        try:
            vectors, metadata = self.embeddings.embed(texts)
            if len(vectors) != len(texts):
                raise ValueError("Embedding count mismatch")
            return [_blob(vector) for vector in vectors]
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            with self.database.connect(write=True) as db:
                db.execute("UPDATE embedding_calls SET usage=?,trace_id=?,elapsed_ms=?,error_type=? WHERE id=?",
                           (json.dumps(metadata.get("usage")), metadata.get("trace_id"),
                            round((time.monotonic() - started) * 1000), error, call_id))

    def sync(self, *, max_requests: int = 4, should_continue=None) -> dict:
        if not 0 <= max_requests <= 32:
            raise ValueError("Invalid embedding request budget")
        calls, indexed, obsolete = 0, 0, 0
        for _ in range(max_requests):
            if should_continue is not None and not should_continue():
                break
            with self.database.connect() as db:
                # Rebuilds after contract changes are explicit pending work, never mixed with old vectors.
                rows = db.execute(
                    "SELECT m.id FROM memories m LEFT JOIN vector_entries v ON v.memory_id=m.id "
                    "LEFT JOIN vector_outbox o ON o.memory_id=m.id WHERE m.namespace=? AND m.status<>'expired' "
                    "AND (v.memory_id IS NULL OR v.version<>m.version OR v.contract<>?) "
                    "AND (o.failures IS NULL OR (o.failures<3 AND o.available_at<=?)) ORDER BY m.id LIMIT 16",
                    (self.database.namespace, self.embeddings.contract, time.time()),
                ).fetchall()
                records = [self.database._read(db, row[0]) for row in rows]
            if not records:
                break
            # The provider documents stricter limits for batched inputs; long facts use single requests.
            if any(len(_text(record).encode("utf-8")) > 512 for record in records):
                records = records[:1]
            try:
                calls += 1
                vectors = self._embed([_text(record) for record in records], "index")
            except EmbeddingBudgetExceeded:
                return {"calls": calls - 1, "indexed": indexed, "obsolete": obsolete, "budget_exhausted": True}
            except Exception as exc:  # noqa: BLE001 - index errors never roll back authoritative facts
                with self.database.connect(write=True) as db:
                    for record in records:
                        current = db.execute("SELECT version FROM memories WHERE id=?", (record.id,)).fetchone()
                        if current is not None and current[0] == record.version:
                            db.execute("INSERT INTO vector_outbox VALUES(?,1,?,?) ON CONFLICT(memory_id) DO UPDATE SET "
                                       "failures=failures+1,available_at=excluded.available_at,error_type=excluded.error_type",
                                       (record.id, time.time() + 300, type(exc).__name__))
                return {"calls": calls, "indexed": indexed, "obsolete": obsolete, "error_type": type(exc).__name__}
            with self.database.connect(write=True) as db:
                for record, vector in zip(records, vectors):
                    current = db.execute("SELECT version FROM memories WHERE id=?", (record.id,)).fetchone()
                    if current is None or current[0] != record.version:
                        obsolete += 1
                        continue
                    self.database._invalidate_vector(db, record.id)
                    row_id = db.execute("INSERT INTO vector_entries(memory_id,version,contract,input_hash) VALUES(?,?,?,?)",
                                        (record.id, record.version, self.embeddings.contract, _hash(_text(record)))).lastrowid
                    db.execute("INSERT INTO memory_vectors(rowid,embedding,namespace) VALUES(?,?,?)",
                               (row_id, vector, record.namespace))
                    db.execute("DELETE FROM vector_outbox WHERE memory_id=?", (record.id,))
                    indexed += 1
        return {"calls": calls, "indexed": indexed, "obsolete": obsolete}

    def search(self, text: str, *, limit: int = 30) -> list[tuple[str, float]]:
        if not 1 <= limit <= 200:
            raise ValueError("Invalid vector candidate count")
        with self.database.connect() as db:
            if not db.execute("SELECT 1 FROM memories WHERE namespace=? AND status<>'expired' LIMIT 1",
                              (self.database.namespace,)).fetchone():
                return []
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest not in self._query_cache:
            vector = self._embed([text], "query")[0]
            if len(self._query_cache) >= 128:
                self._query_cache.clear()
            self._query_cache[digest] = vector
        with self.database.connect() as db:
            rows = db.execute(
                "SELECT v.memory_id,n.distance FROM memory_vectors n JOIN vector_entries v ON v.id=n.rowid "
                "JOIN memories m ON m.id=v.memory_id WHERE n.embedding MATCH ? AND k=? AND n.namespace=? "
                "AND v.contract=? AND v.version=m.version ORDER BY n.distance",
                (self._query_cache[digest], limit, self.database.namespace, self.embeddings.contract),
            ).fetchall()
            return [(row[0], 1 - row[1]) for row in rows]

    def pending(self) -> int:
        with self.database.connect() as db:
            return db.execute(
                "SELECT COUNT(*) FROM memories m LEFT JOIN vector_entries v ON v.memory_id=m.id "
                "WHERE m.namespace=? AND m.status<>'expired' AND (v.memory_id IS NULL OR v.version<>m.version OR v.contract<>?)",
                (self.database.namespace, self.embeddings.contract),
            ).fetchone()[0]
