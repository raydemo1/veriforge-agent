import hashlib
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from harness_code_agent.memory.database import EvidenceReference, MemoryDatabase


def evidence(text="confirmed", sequence=1):
    return EvidenceReference("session-1", sequence, "user_input", hashlib.sha256(text.encode()).hexdigest())


@pytest.fixture
def database(tmp_path):
    return MemoryDatabase(tmp_path / "memory", "project:sample")


def save(database, **overrides):
    values = {"topic": "发布规范", "fact_key": "release.build", "kind": "convention", "body": "BUILD_V1",
              "evidence": [evidence()]}
    values.update(overrides)
    return database.write(**values)


def test_records_are_sqlite_authoritative_with_topic_exports(database):
    result = save(database)
    database.flush_exports()
    exported = database.export_root / f"{result.record.topic_id}.md"
    assert "BUILD_V1" in exported.read_text(encoding="utf-8")
    exported.write_text("untrusted external edit", encoding="utf-8")
    assert database.read(result.record.id).body == "BUILD_V1"
    assert database.revision == 1


def test_correction_preserves_identity_and_prior_revision(database):
    old = save(database).record
    updated = save(database, memory_id=old.id, expected_version=1, body="BUILD_V2", evidence=[evidence("new", 2)]).record
    assert updated.id == old.id
    assert updated.version == 2
    assert [value["body"] for value in database.history(old.id)] == ["BUILD_V1", "BUILD_V2"]
    assert len(database.records()) == 1
    with pytest.raises(ValueError, match="version changed"):
        save(database, memory_id=old.id, expected_version=1, body="BUILD_STALE")
    assert database.revision == 2


def test_duplicate_merges_evidence_without_adding_a_second_fact(database):
    first = save(database)
    assert save(database).action == "unchanged"
    merged = save(database, evidence=[evidence("other", 2)])
    assert merged.action == "updated"
    assert merged.record.id == first.record.id
    assert len(merged.record.evidence) == 2
    assert len(database.records()) == 1


def test_ambiguous_conflict_is_not_published_as_a_current_fact(database):
    old = save(database).record
    result = save(database, body="BUILD_OTHER", evidence=[evidence("other", 2)])
    assert result.action == "conflict"
    assert result.record.status == "review_required"
    assert database.records() == []
    assert database.proposals(old.id)[0]["body"] == "BUILD_OTHER"
    resolved = database.resolve(old.id, 2, body="BUILD_V1", evidence=[evidence("confirmed original", 3)])
    assert resolved.record.status == "active"
    assert database.proposals(old.id) == []
    assert database.records()[0].body == "BUILD_V1"


def test_resolving_without_new_evidence_is_rejected(database):
    old = save(database).record
    with pytest.raises(ValueError, match="new confirmation"):
        database.resolve(old.id, 1, body="BUILD_V2", evidence=[evidence()])


def test_project_and_user_scope_are_isolated_in_one_database(database):
    current = save(database).record
    other = MemoryDatabase(database.root, "project:other")
    user = MemoryDatabase(database.root, "user")
    for store in (other, user):
        assert store.records() == []
        with pytest.raises(KeyError):
            store.read(current.id)
        with pytest.raises(KeyError):
            store.forget(current.id, 1)
        with pytest.raises(KeyError):
            store.history(current.id)
    assert save(user, body="pnpm", kind="preference").record.namespace == "user"
    assert database.records()[0].body == "BUILD_V1"


def test_partial_update_preserves_other_facts_in_topic(database):
    build = save(database).record
    gate = save(database, fact_key="release.gate", body="GATE_1").record
    save(database, memory_id=build.id, expected_version=1, body="BUILD_V2")
    assert database.read(gate.id).body == "GATE_1"
    assert len({record.topic_id for record in database.records()}) == 1


def test_failure_before_commit_rolls_back_fact_revision_and_outbox(database):
    old = save(database).record
    database.flush_exports()
    with (patch.object(database, "_queue_export", side_effect=OSError("injected failure")),
          pytest.raises(OSError)):
        save(database, memory_id=old.id, expected_version=1, body="BUILD_V2")
    assert database.read(old.id).body == "BUILD_V1"
    assert len(database.history(old.id)) == 1
    assert database.revision == 1
    with database.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM export_outbox").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM audit").fetchone()[0] == 1


def test_new_fact_failure_rolls_back_topic_as_well(database):
    with patch.object(database, "_queue_export", side_effect=OSError("failure")), pytest.raises(OSError):
        save(database)
    assert database.records() == []
    with database.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM topics").fetchone()[0] == 0


def test_concurrent_writers_cannot_overwrite_same_version(database):
    old = save(database).record
    barrier = threading.Barrier(2)

    def update(body):
        store = MemoryDatabase(database.root, database.namespace)
        barrier.wait(timeout=5)
        try:
            save(store, memory_id=old.id, expected_version=1, body=body)
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(update, ["BUILD_A", "BUILD_B"])) == [False, True]
    assert len(database.history(old.id)) == 2
    assert database.read(old.id).version == 2


def test_export_failure_keeps_authoritative_fact_and_retry(database):
    result = save(database)
    with (patch("harness_code_agent.memory.database._atomic_write", side_effect=OSError("disk full")),
          pytest.raises(OSError)):
        database.flush_exports()
    assert database.read(result.record.id).body == "BUILD_V1"
    with database.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM export_outbox").fetchone()[0] == 1
    database.flush_exports()
    assert (database.export_root / "MEMORY.md").is_file()


def test_forget_removes_history_conflicts_and_exports_and_blocks_reextraction(database):
    old = save(database).record
    save(database, body="BUILD_V2", evidence=[evidence("changed", 2)])
    database.flush_exports()
    database.forget(old.id, 2)
    database.flush_exports()
    assert not (database.export_root / f"{old.topic_id}.md").exists()
    for method in (database.read, database.history, database.proposals):
        with pytest.raises(KeyError):
            method(old.id)
    with database.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM topics").fetchone()[0] == 0
        assert "BUILD" not in json.dumps([tuple(row) for row in db.execute("SELECT * FROM audit")])
    with pytest.raises(ValueError, match="forgotten"):
        save(database)
    with pytest.raises(ValueError, match="forgotten"):
        save(database, evidence=[evidence("new source", 3)])
    with pytest.raises(ValueError, match="source was forgotten"):
        save(database, topic="renamed topic", fact_key="different.key")


def test_forget_failure_rolls_back_suppressions_and_deletion(database):
    old = save(database).record
    with patch.object(database, "_queue_export", side_effect=OSError("failure")), pytest.raises(OSError):
        database.forget(old.id, 1)
    assert database.read(old.id).body == "BUILD_V1"
    with database.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM suppressions").fetchone()[0] == 0


def test_process_crash_during_transaction_preserves_committed_revision(database):
    old = save(database).record
    script = """import os, sqlite3, sys
db = sqlite3.connect(sys.argv[1])
db.execute('BEGIN IMMEDIATE')
db.execute('UPDATE memories SET body=? WHERE id=?', ('UNCOMMITTED', sys.argv[2]))
db.execute('UPDATE state SET revision=99 WHERE id=1')
os._exit(7)
"""
    process = subprocess.run([sys.executable, "-c", script, str(database.path), old.id],
                             capture_output=True, timeout=10, check=False)
    assert process.returncode == 7
    reopened = MemoryDatabase(database.root, database.namespace)
    assert reopened.read(old.id).body == "BUILD_V1"
    assert reopened.revision == 1
    assert len(reopened.history(old.id)) == 1


def test_mutating_returned_snapshot_cannot_change_database(database):
    saved = save(database).record
    saved.evidence[0]["session_id"] = "mutated"
    history = database.history(saved.id)
    history[0]["body"] = "mutated"
    history[0]["evidence"].clear()
    assert database.read(saved.id).evidence[0]["session_id"] == "session-1"
    assert database.history(saved.id)[0]["body"] == "BUILD_V1"


@pytest.mark.parametrize("overrides", [
    {"kind": "anything"}, {"body": ""}, {"fact_key": " "}, {"evidence": []},
    {"evidence": [EvidenceReference("s", True, "user_input", "0" * 64)]},
])
def test_invalid_records_cannot_be_committed(database, overrides):
    with pytest.raises(ValueError):
        save(database, **overrides)
    assert database.revision == 0
