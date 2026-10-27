"""Explicit, offline memory robustness checks; excluded from the default pytest suite."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness_code_agent.memory.database import EvidenceReference, MemoryDatabase
from harness_code_agent.memory.extraction import ExtractionQueue
from harness_code_agent.memory.reconciliation import (
    ConsolidationQueue,
    CurrentMemoryView,
)
from harness_code_agent.memory.vectors import VectorIndex
from harness_code_agent.sessions.journal import SessionJournal


def response(value):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
        content=json.dumps(value, ensure_ascii=False)))], usage=None)


class LocalVectors:
    contract = "offline-fixture-not-a-semantic-model"

    def embed(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * 1024
            for char in text:
                vector[int(hashlib.sha256(char.encode()).hexdigest()[:8], 16) % 1024] += 1
            vectors.append(vector)
        return vectors, {}


def factory(database):
    return VectorIndex(database, embeddings=LocalVectors())


def seed(root):
    db = MemoryDatabase(root / "memory", "project:fixture")
    extraction = ExtractionQueue(db)
    journal = SessionJournal(root / "journal.jsonl")
    text = "已经确认：项目数据库使用 SQLite。"
    entry = journal.append_user_input(text, turn_id="old", memory_generation=extraction.generation)
    extraction.enqueue("session", journal.path)
    with db.connect(write=True) as sql:
        sql.execute("UPDATE extraction_tasks SET status='done'")
        sql.execute("INSERT INTO extraction_checkpoints VALUES(?,?,?,0)",
                    (db.namespace, "session", journal.path.stat().st_size))
    ref = EvidenceReference("session", entry.sequence, "user_input", hashlib.sha256(text.encode()).hexdigest(),
                            end_offset=len(text), source_namespace=db.namespace)
    old = db.write(topic="数据库", fact_key="database.engine", kind="project_fact", body="项目使用 SQLite。",
                   evidence=[ref]).record
    assert factory(db).sync()["indexed"] == 1
    return db, extraction, journal, old


def candidate(db, extraction, journal, text="迁移已经完成，后续统一使用 PostgreSQL。", *, key="database.engine",
              body="项目使用 PostgreSQL。", applicability=""):
    journal.append_user_input(text, turn_id="new", memory_generation=extraction.generation)
    extraction.enqueue("session", journal.path)
    item = {"topic": "数据库", "fact_key": key, "kind": "project_fact", "body": body,
            "scope": "project", "quotes": [text], "applicability": applicability}
    outcome = extraction.process(extraction.claim(), max_calls=1,
                                 invoke=lambda **kw: response({"complete": True, "episode": "", "candidates": [item]}))
    assert outcome["state"] == "done", outcome


def decide(action, target=None):
    def invoke(**kwargs):
        payload = json.loads(kwargs["messages"][-1]["content"])
        return response({"decisions": [{"candidate_id": c["id"], "action": action,
                                        "target_id": target.id if target else None,
                                        "topic": target.topic if target else c["fact"]["topic"],
                                        "fact_key": target.fact_key if target else c["fact"]["fact_key"],
                                        "applicability": target.applicability if target else c["fact"].get("applicability", ""),
                                        "body": c["fact"]["body"], "reason": "fixture decision"} for c in payload["candidates"]]})
    return invoke


def retry_now(db):
    with db.connect(write=True) as sql:
        sql.execute("UPDATE consolidation_tasks SET available_at=0")


def checks(root):
    passed = []
    db, extraction, journal, old = seed(root / "update")
    candidate(db, extraction, journal)
    view = CurrentMemoryView(db, vectors=factory(db))
    assert view.read(old.id)["status"] == "pending_update" and db.records() == []
    db.flush_exports()
    assert not (db.export_root / f"{old.topic_id}.md").exists()
    passed.append("pending update excludes definite old value")
    worker = ConsolidationQueue(db)
    for attempt in range(3):
        retry_now(db)
        result = worker.process(worker.claim(), invoke=lambda **kw: (_ for _ in ()).throw(TimeoutError()))
        assert result["state"] == "failed_attempt"
    assert worker.claim() is None and view.read(old.id)["status"] == "pending_update"
    passed.append("three failed calls stop, uncertainty remains")
    with db.connect(write=True) as sql:
        sql.execute("UPDATE consolidation_tasks SET status='pending',failures=0,available_at=0")
    with patch.object(MemoryDatabase, "_queue_export", side_effect=OSError("injected disk failure")):
        assert worker.process(worker.claim(), invoke=decide("update", old))["state"] == "failed_attempt"
    assert db.read(old.id).version == 1 and len(db.history(old.id)) == 1
    assert extraction.candidates()[0]["status"] == "pending"
    passed.append("atomic rollback includes fact, history, candidate and decision")
    retry_now(db)
    result = worker.process(worker.claim(), invoke=decide("update", old))
    assert result["state"] == "done", result
    assert db.records()[0].body == "项目使用 PostgreSQL。" and len(db.history(old.id)) == 2
    assert worker.claim() is None
    with db.connect() as sql:
        assert sql.execute("SELECT COUNT(*) FROM vector_entries").fetchone()[0] == 0
        assert sql.execute("SELECT COUNT(*) FROM memory_decisions").fetchone()[0] == 1
    passed.append("confirmed update retains raw evidence and invalidates old vector")
    assert factory(db).sync()["indexed"] == 1
    hit = view.search("数据库", max_tokens=512)
    assert hit["items"][0]["body"] == "项目使用 PostgreSQL。"
    assert "SQLite" not in json.dumps(hit, ensure_ascii=False)
    db.flush_exports()
    topic = db.export_root / f"{old.topic_id}.md"
    assert "PostgreSQL" in topic.read_text(encoding="utf-8") and "SQLite" not in topic.read_text(encoding="utf-8")
    passed.append("current search and Markdown exclude superseded values")
    restored = db.restore_revision(old.id, 1, expected_version=2)
    assert restored.version == 3 and restored.body == "项目使用 SQLite。"
    assert len(db.history(old.id)) == 3
    passed.append("incorrect update can be undone without erasing history")

    db, extraction, journal, old = seed(root / "stale")
    candidate(db, extraction, journal)
    worker = ConsolidationQueue(db)
    def concurrent(**kwargs):
        db.write(topic=old.topic, fact_key=old.fact_key, kind=old.kind, body="最新人工确认值",
                 evidence=[EvidenceReference("manual", 1, "manual", "f" * 64)],
                 memory_id=old.id, expected_version=1)
        return decide("update", old)(**kwargs)
    assert worker.process(worker.claim(), invoke=concurrent)["state"] == "obsolete"
    assert db.read(old.id).body == "最新人工确认值" and extraction.candidates()[0]["status"] == "pending"
    passed.append("in-flight stale decision cannot overwrite a concurrent correction")

    db, extraction, journal, old = seed(root / "review")
    candidate(db, extraction, journal, "数据库似乎也有 PostgreSQL，是否迁移不确定。")
    worker = ConsolidationQueue(db)
    assert worker.process(worker.claim(), invoke=decide("review", old))["state"] == "done"
    page = CurrentMemoryView(db, vectors=factory(db)).search("数据库")
    assert page["items"][0]["status"] == "review_required" and "body" not in page["items"][0]
    assert db.proposals(old.id)[0]["body"] == "项目使用 PostgreSQL。"
    passed.append("unresolved conflict returns warning instead of an asserted value")

    db, extraction, journal, old = seed(root / "scope")
    candidate(db, extraction, journal, applicability="tools/")
    worker = ConsolidationQueue(db)
    assert worker.process(worker.claim(), invoke=decide("add"))["state"] == "done"
    assert len(db.records()) == 2
    passed.append("different applicability coexists without replacing project-wide fact")

    db, extraction, journal, old = seed(root / "forged")
    candidate(db, extraction, journal)
    other = db.scoped("project:other")
    foreign = other.write(topic="数据库", fact_key=old.fact_key, kind=old.kind, body="FOREIGN",
                          evidence=[EvidenceReference("manual", 1, "manual", "f" * 64)]).record
    factory(other).sync()
    worker = ConsolidationQueue(db)
    assert worker.process(worker.claim(), invoke=decide("update", foreign))["state"] == "failed_attempt"
    assert other.read(foreign.id).body == "FOREIGN" and db.read(old.id).version == 1
    assert not any(i["id"] == foreign.id for i in CurrentMemoryView(db, vectors=factory(db)).search("数据库")["items"])
    passed.append("foreign-namespace target and retrieval are blocked")

    db, extraction, journal, old = seed(root / "tamper")
    candidate(db, extraction, journal)
    journal.path.write_text(journal.path.read_text(encoding="utf-8").replace("迁移已经完成", "内容已经篡改"), encoding="utf-8")
    worker = ConsolidationQueue(db)
    result = worker.process(worker.claim(), invoke=lambda **kw: (_ for _ in ()).throw(AssertionError("no model call")))
    assert result["calls"] == 0 and result["state"] == "failed_attempt"
    passed.append("changed journal evidence rejected before model call")

    db, _, _, old = seed(root / "vector_failure")
    db.write(topic=old.topic, fact_key=old.fact_key, kind=old.kind, body="NEW",
             evidence=old.evidence and [EvidenceReference(**old.evidence[0])], memory_id=old.id, expected_version=1)
    broken = LocalVectors()
    broken.embed = lambda texts: (_ for _ in ()).throw(TimeoutError())
    assert VectorIndex(db, embeddings=broken).sync()["error_type"] == "TimeoutError"
    assert db.read(old.id).body == "NEW" and factory(db).search("SQLite") == []
    with db.connect(write=True) as sql:
        sql.execute("UPDATE vector_outbox SET available_at=0")
    assert factory(db).sync()["indexed"] == 1
    passed.append("failed embedding preserves fact and never serves superseded vector")
    db.write(topic=old.topic, fact_key=old.fact_key, kind=old.kind, body="NEXT",
             evidence=[EvidenceReference(**old.evidence[0])], memory_id=old.id, expected_version=2)
    delayed = LocalVectors()
    def forget_during_embedding(texts):
        db.forget(old.id, 3)
        return LocalVectors().embed(texts)
    delayed.embed = forget_during_embedding
    assert VectorIndex(db, embeddings=delayed).sync()["obsolete"] == 1
    assert db.records() == [] and factory(db).search("NEXT") == []
    passed.append("forget during embedding cannot resurrect removed memory")

    db, extraction, journal, old = seed(root / "confirmation")
    candidate(db, extraction, journal)
    text = "最新人工确认：仍使用 SQLite。"
    entry = journal.append_user_input(text, turn_id="confirm", memory_generation=extraction.generation)
    fresh = EvidenceReference("session", entry.sequence, "user_input", hashlib.sha256(text.encode()).hexdigest(),
                              end_offset=len(text), source_namespace=db.namespace, observed_at=entry.timestamp)
    confirmed = db.write(topic=old.topic, fact_key=old.fact_key, kind=old.kind, body=old.body,
                         evidence=[fresh], memory_id=old.id, expected_version=1).record
    worker = ConsolidationQueue(db)
    outcome = worker.process(worker.claim(), invoke=lambda **kw: (_ for _ in ()).throw(AssertionError("no model call")))
    assert outcome["state"] == "done" and outcome["calls"] == 0
    assert db.records()[0].id == confirmed.id
    passed.append("fresh confirmation retires older pending changes atomically")
    db.forget(confirmed.id, confirmed.version)
    new_ref = EvidenceReference("new-confirmation", 1, "manual", "c" * 64)
    args = {"topic": old.topic, "fact_key": old.fact_key, "kind": old.kind, "body": "RESTORED",
            "evidence": [new_ref]}
    with patch.object(db, "_queue_export", side_effect=OSError("restore failure")):
        try:
            db.write(**args, restore_forgotten=True)
        except OSError:
            pass
        else:
            raise AssertionError("restore must fail atomically")
    try:
        db.write(**args)
    except ValueError:
        pass
    else:
        raise AssertionError("suppression was cleared before failed restore committed")
    assert db.write(**args, restore_forgotten=True).record.body == "RESTORED"
    passed.append("explicit re-remember is possible and failed restores preserve suppression")

    db, extraction, journal, old = seed(root / "late_job")
    candidate(db, extraction, journal, key="database.engine.alias")
    text = "更晚确认：使用 CURRENT_FINAL。"
    entry = journal.append_user_input(text, turn_id="late-confirm", memory_generation=extraction.generation)
    fresh = EvidenceReference("session", entry.sequence, "user_input", hashlib.sha256(text.encode()).hexdigest(),
                              end_offset=len(text), source_namespace=db.namespace, observed_at=entry.timestamp)
    latest = db.write(topic=old.topic, fact_key=old.fact_key, kind=old.kind, body="CURRENT_FINAL",
                      evidence=[fresh], memory_id=old.id, expected_version=1).record
    factory(db).sync()
    worker = ConsolidationQueue(db)
    result = worker.process(worker.claim(), invoke=decide("update", latest))
    assert result["actions"] == ["history"] and db.read(old.id).body == "CURRENT_FINAL"
    passed.append("a delayed older correction cannot overwrite a later confirmed value")

    db = MemoryDatabase(root / "scale", "project:scale")
    ref = EvidenceReference("manual", 1, "manual", "f" * 64)
    with db.connect(write=True) as sql:
        for number in range(2000):
            db._write(sql, topic="构建规范", fact_key=f"build.{number}", kind="convention",
                      body=f"构建入口 BUILD_{number:04d}", evidence=[ref])
    index = factory(db)
    while index.pending():
        result = index.sync(max_requests=32)
        assert not result.get("error_type"), result
    view = CurrentMemoryView(db, vectors=index)
    hit = view.search("BUILD_1977", max_tokens=512)
    assert hit["items"][0]["body"] == "构建入口 BUILD_1977"
    assert len(hit["items"]) <= 10
    assert view.search("构建规范", max_tokens=512)["has_more"]
    passed.append("2000 facts use indexed bounded retrieval and pagination")
    return passed


def main():
    with tempfile.TemporaryDirectory(prefix="memory-consolidation-") as temporary:
        passed = checks(Path(temporary))
    for label in passed:
        print("PASS:", label)
    print(f"{len(passed)} standalone checks passed; no network calls; fixture vectors do not measure semantic quality.")


if __name__ == "__main__":
    main()
