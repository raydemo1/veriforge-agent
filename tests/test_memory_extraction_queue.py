import hashlib
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from harness_code_agent import config
from harness_code_agent.memory.database import EvidenceReference, MemoryDatabase
from harness_code_agent.memory.extraction import ExtractionQueue
from harness_code_agent.memory.security import contains_secret, redact_secrets
from harness_code_agent.sessions.journal import SessionJournal


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_EXTRACTION_IDLE_SECONDS", 0)
    database = MemoryDatabase(tmp_path / "memory", "project:sample")
    queue = ExtractionQueue(database)
    journal = SessionJournal(tmp_path / "journal.jsonl")
    return database, queue, journal


def user(queue, journal, text="长期约定：构建使用 BUILD_V1。"):
    return journal.append_user_input(text, turn_id="turn-1", memory_generation=queue.generation)


def candidate(quote="构建使用 BUILD_V1", **overrides):
    return {"topic": "构建规范", "fact_key": "build.command", "kind": "convention", "scope": "project",
            "body": "构建使用 BUILD_V1", "quotes": [quote], **overrides}


def reply(candidates=None, *, episode="", complete=True, finish_reason="stop", raw=None):
    content = raw if raw is not None else json.dumps({
        "complete": complete, "candidates": candidates or [], "episode": episode,
    }, ensure_ascii=False)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason,
                                                   message=SimpleNamespace(content=content))], usage=None)


def enqueue(queue, journal):
    task_id = queue.enqueue("session-1", journal.path, idle_seconds=0)
    return task_id, queue.claim()


def rows(database, table):
    with database.connect() as db:
        return [dict(row) for row in db.execute(f"SELECT * FROM {table} ORDER BY rowid")]


def test_cancel_resume_requeues_conflict_without_stranding_hold(runtime):
    from harness_code_agent.memory.reconciliation import (
        ConsolidationQueue,
        CurrentMemoryView,
    )

    database, queue, journal = runtime
    old = database.write(topic="构建规范", fact_key="build.command", kind="convention", body="构建使用 BUILD_V0",
                         evidence=[EvidenceReference("manual", 1, "manual", "a" * 64)]).record
    user(queue, journal)
    _, task = enqueue(queue, journal)
    assert queue.process(task, invoke=lambda **kw: reply([candidate()]))["state"] == "done"
    consolidation = ConsolidationQueue(database)
    stale = consolidation.claim()
    view = CurrentMemoryView(database)
    assert view.read(old.id)["status"] == "pending_update"
    with database.connect(write=True) as db:
        db.execute("CREATE TRIGGER fail_cancel BEFORE UPDATE OF status ON memory_candidates "
                   "WHEN NEW.status='cancelled' BEGIN SELECT RAISE(ABORT,'cancel failed'); END")
    with pytest.raises(sqlite3.IntegrityError, match="cancel failed"):
        queue.cancel("session-1")
    assert view.read(old.id)["status"] == "pending_update"
    assert rows(database, "consolidation_tasks")[0]["status"] == "running"
    with database.connect(write=True) as db:
        db.execute("DROP TRIGGER fail_cancel")
    queue.cancel("session-1")
    assert view.read(old.id)["body"] == old.body
    assert queue.candidates()[0]["status"] == "cancelled"
    queue.resume("session-1")
    assert view.read(old.id)["status"] == "pending_update"
    resumed = consolidation.claim()
    assert resumed["id"] == stale["id"] and resumed["lease_token"] != stale["lease_token"]
    assert consolidation.process(stale, invoke=Mock())["state"] == "obsolete"
    revision, candidates, records = consolidation._batch(resumed)
    item = candidates[0]
    consolidation._commit(resumed, revision, candidates, [{"candidate_id": item["id"], "action": "update",
                          "target_id": old.id, "topic": old.topic, "fact_key": old.fact_key,
                          "applicability": "", "body": "构建使用 BUILD_V1", "reason": "confirmed change"}], records)
    assert view.read(old.id)["body"] == "构建使用 BUILD_V1"


def test_generation_disabled_sources_are_not_extracted_after_resume(runtime):
    _, queue, journal = runtime
    journal.append_user_input("请保存 DISABLED_SOURCE", turn_id="off", memory_generation=queue.generation,
                              auto_extract=False)
    user(queue, journal)
    _, task = enqueue(queue, journal)
    invoke = Mock(return_value=reply([candidate()]))
    assert queue.process(task, invoke=invoke)["state"] == "done"
    assert invoke.call_count == 1
    assert "DISABLED_SOURCE" not in json.dumps(invoke.call_args.kwargs)


def ready(database):
    with database.connect(write=True) as db:
        db.execute("UPDATE extraction_tasks SET available_at=0 WHERE status='pending'")


def test_raw_sources_are_distinct_from_recalled_and_model_messages(runtime):
    database, queue, journal = runtime
    original = "请长期记住：构建使用 BUILD_V1。"
    source = user(queue, journal, original)
    journal.append_message({"role": "user", "content": "RECALLED_MEMORY\nUser turn:\n" + original})
    journal.append_message({"role": "assistant", "content": "UNVERIFIED_CLAIM", "reasoning_content": "HIDDEN"})
    journal.append("context_epoch", {"base_messages": [{"role": "user", "content": "EPOCH_DUPLICATE"}]})
    _, task = enqueue(queue, journal)
    invoke = Mock(return_value=reply([candidate()], episode="用户确认了构建约定。"))
    assert queue.process(task, invoke=invoke) == {"state": "done", "calls": 1}
    payload = json.loads(invoke.call_args.kwargs["messages"][-1]["content"])
    assert payload["source"] == original
    assert not any(text in json.dumps(payload) for text in ("RECALLED_MEMORY", "UNVERIFIED_CLAIM", "HIDDEN", "EPOCH_DUPLICATE"))
    stored = queue.candidates()[0]
    evidence = stored["payload"]["evidence"][0]
    assert evidence["sequence"] == source.sequence
    assert original[evidence["start_offset"]:evidence["end_offset"]] == "构建使用 BUILD_V1"
    assert evidence["content_hash"] == hashlib.sha256("构建使用 BUILD_V1".encode()).hexdigest()
    assert database.records() == []
    assert len(rows(database, "consolidation_tasks")) == 1
    assert rows(database, "consolidation_tasks")[0]["status"] == "pending"
    assert queue.checkpoint("session-1") == (journal.path.stat().st_size, 0)


def test_valid_sources_can_be_replayed_from_journal_without_becoming_model_messages(runtime):
    _, queue, journal = runtime
    user(queue, journal)
    journal.append_message({"role": "user", "content": "real projected request"})
    reloaded = SessionJournal(journal.path)
    assert reloaded.read()[0].kind == "user_input"
    messages = reloaded.recovery_messages("system")
    assert [message["content"] for message in messages] == ["system", "real projected request"]


def test_more_than_four_candidates_are_retained(runtime):
    _, queue, journal = runtime
    user(queue, journal, "；".join(f"长期约定 FIELD_{i}=VALUE_{i}" for i in range(7)))
    _, task = enqueue(queue, journal)
    values = [candidate(f"FIELD_{i}=VALUE_{i}", body=f"FIELD_{i}=VALUE_{i}", fact_key=f"field.{i}") for i in range(7)]
    assert queue.process(task, invoke=Mock(return_value=reply(values)))["state"] == "done"
    assert len(queue.candidates()) == 7


def test_identical_candidates_in_one_response_do_not_fail_the_batch(runtime):
    _, queue, journal = runtime
    user(queue, journal)
    _, task = enqueue(queue, journal)
    assert queue.process(task, invoke=Mock(return_value=reply([candidate(), candidate()])))["state"] == "done"
    assert len(queue.candidates()) == 1


def test_candidates_checkpoint_episode_and_next_stage_commit_together(runtime):
    database, queue, journal = runtime
    user(queue, journal)
    _, task = enqueue(queue, journal)
    with patch.object(queue, "_enqueue_consolidation", side_effect=OSError("disk failure")):
        assert queue.process(task, invoke=Mock(return_value=reply([candidate()], episode="historical summary")))["state"] == "failed_attempt"
    assert queue.candidates() == []
    assert rows(database, "memory_episodes") == []
    assert rows(database, "consolidation_tasks") == []
    assert queue.checkpoint("session-1") == (0, 0)
    ready(database)
    restarted = ExtractionQueue(MemoryDatabase(database.root, database.namespace))
    assert restarted.process(restarted.claim(), invoke=Mock(return_value=reply([candidate()])))["state"] == "done"
    assert len(queue.candidates()) == 1
    assert len(rows(database, "memory_model_calls")) == 2


def test_incremental_extraction_starts_at_successful_checkpoint(runtime):
    _, queue, journal = runtime
    user(queue, journal)
    _, first = enqueue(queue, journal)
    queue.process(first, invoke=Mock(return_value=reply([candidate()])))
    next_source = user(queue, journal, "长期约定：测试使用 SUITE_V2。")
    _, second = enqueue(queue, journal)
    invoke = Mock(return_value=reply([candidate("测试使用 SUITE_V2", body="测试使用 SUITE_V2", fact_key="test.suite")]))
    assert queue.process(second, invoke=invoke)["state"] == "done"
    payload = json.loads(invoke.call_args.kwargs["messages"][-1]["content"])
    assert payload["sequence"] == next_source.sequence
    assert "BUILD_V1" not in payload["source"]
    assert len(queue.candidates()) == 2


def test_long_source_resumes_inside_message_without_truncation(runtime):
    database, _, journal = runtime
    queue = ExtractionQueue(database, chunk_tokens=40)
    text = "EARLY_FACT\n" + "这是临时工作描述。" * 50 + "\nLATE_FACT"
    user(queue, journal, text)
    queue.enqueue("session-1", journal.path, idle_seconds=0)
    seen = []

    def invoke(**kwargs):
        payload = json.loads(kwargs["messages"][-1]["content"])
        seen.append(payload["source"])
        return reply()

    first = queue.process(queue.claim(), invoke=invoke, max_calls=1)
    assert first == {"state": "pending", "calls": 1}
    assert queue.checkpoint("session-1")[0] == 0
    assert 0 < queue.checkpoint("session-1")[1] < len(text)
    for _ in range(30):
        ready(database)
        queue = ExtractionQueue(MemoryDatabase(database.root, database.namespace), chunk_tokens=40)
        result = queue.process(queue.claim(), invoke=invoke, max_calls=3)
        if result["state"] == "done":
            break
    assert result["state"] == "done"
    assert "".join(seen) == text
    assert queue.checkpoint("session-1") == (journal.path.stat().st_size, 0)


def test_snapshot_does_not_include_later_user_input(runtime):
    _, queue, journal = runtime
    user(queue, journal)
    _, task = enqueue(queue, journal)
    user(queue, journal, "LATER_ACTIVE_INPUT")
    invoke = Mock(return_value=reply([candidate()]))
    assert queue.process(task, invoke=invoke)["state"] == "done"
    assert queue.checkpoint("session-1") == (task["boundary"], 0)
    assert "LATER_ACTIVE_INPUT" not in json.dumps(invoke.call_args.kwargs)


def test_old_generation_and_ordinary_messages_never_trigger_extraction(runtime):
    _, queue, journal = runtime
    journal.append_user_input("LEGACY", turn_id="t", memory_generation="old-generation")
    journal.append_message({"role": "user", "content": "OLD_MIXED_PROMPT"})
    _, task = enqueue(queue, journal)
    invoke = Mock()
    assert queue.process(task, invoke=invoke) == {"state": "done", "calls": 0}
    invoke.assert_not_called()


@pytest.mark.parametrize("raw", ["[]", "{}", "not json", '{"complete":true,"candidates":[],"episode":"","unexpected":1}'])
def test_invalid_model_output_never_consumes_source(runtime, raw):
    database, queue, journal = runtime
    user(queue, journal)
    queue.enqueue("session-1", journal.path, idle_seconds=0)
    for _ in range(3):
        ready(database)
        assert queue.process(queue.claim(), invoke=Mock(return_value=reply(raw=raw)))["state"] == "failed_attempt"
    assert rows(database, "extraction_tasks")[0]["status"] == "failed"
    assert len(rows(database, "memory_model_calls")) == 3
    assert queue.checkpoint("session-1") == (0, 0)
    assert queue.candidates() == []
    assert queue.claim() is None


@pytest.mark.parametrize("complete,finish", [(False, "stop"), (True, "length"), (True, None)])
def test_incomplete_output_retries_with_smaller_chunk(runtime, complete, finish):
    database, queue, journal = runtime
    user(queue, journal)
    _, task = enqueue(queue, journal)
    assert queue.process(task, invoke=Mock(return_value=reply(complete=complete, finish_reason=finish)))["state"] == "failed_attempt"
    assert queue.checkpoint("session-1") == (0, 0)
    assert rows(database, "extraction_tasks")[0]["chunk_tokens"] == task["chunk_tokens"] // 2


@pytest.mark.parametrize("value", [
    candidate("not in the journal"), candidate(scope="user"),
    candidate(body="password: SYNTHETIC_SECRET"), candidate(body="unsupported", kind="permission"),
])
def test_unverifiable_or_disallowed_candidates_are_not_committed(runtime, value):
    _, queue, journal = runtime
    user(queue, journal)
    _, task = enqueue(queue, journal)
    assert queue.process(task, invoke=Mock(return_value=reply([value])))["state"] == "failed_attempt"
    assert queue.candidates() == []
    assert queue.checkpoint("session-1") == (0, 0)


def test_cancel_during_request_prevents_commit(runtime):
    database, queue, journal = runtime
    user(queue, journal)
    _, task = enqueue(queue, journal)

    def invoke(**kwargs):
        assert rows(database, "memory_model_calls")[0]["error_type"] == "in_flight"
        queue.cancel("session-1")
        return reply([candidate()])

    assert queue.process(task, invoke=invoke) == {"state": "obsolete", "calls": 1}
    assert queue.candidates() == []
    assert queue.checkpoint("session-1") == (0, 0)
    assert rows(database, "extraction_tasks")[0]["status"] == "cancelled"


def test_activity_and_lease_reclaim_fence_old_worker(runtime):
    database, queue, journal = runtime
    user(queue, journal)
    _, stale = enqueue(queue, journal)
    with database.connect(write=True) as db:
        db.execute("UPDATE extraction_tasks SET lease_until=0")
    current = queue.claim()
    assert current["lease_token"] != stale["lease_token"]
    invoke = Mock(return_value=reply([candidate()]))
    assert queue.process(stale, invoke=invoke) == {"state": "obsolete", "calls": 0}
    invoke.assert_not_called()
    queue.set_active("session-1")
    assert queue.process(current, invoke=invoke) == {"state": "obsolete", "calls": 0}
    assert queue.claim() is None
    queue.finish_activity("session-1")
    assert queue.process(queue.claim(), invoke=invoke)["state"] == "done"


def test_new_snapshot_supersedes_inflight_task(runtime):
    _, queue, journal = runtime
    user(queue, journal)
    _, old = enqueue(queue, journal)

    def invoke(**kwargs):
        user(queue, journal, "NEW_SOURCE")
        queue.enqueue("session-1", journal.path, idle_seconds=0)
        return reply([candidate()])

    assert queue.process(old, invoke=invoke)["state"] == "obsolete"
    assert queue.candidates() == []
    assert queue.checkpoint("session-1") == (0, 0)


def test_forget_during_request_suppresses_candidate_and_historical_summary(runtime):
    database, queue, journal = runtime
    entry = user(queue, journal)
    quote = "构建使用 BUILD_V1"
    memory = database.write(topic="构建规范", fact_key="build.command", kind="convention", body=quote,
                            evidence=[EvidenceReference("session-1", entry.sequence, "user_input",
                                                        hashlib.sha256(quote.encode()).hexdigest())]).record
    _, task = enqueue(queue, journal)

    def invoke(**kwargs):
        database.forget(memory.id, memory.version)
        return reply([candidate()], episode="这次用户确认构建使用 BUILD_V1。")

    assert queue.process(task, invoke=invoke)["state"] == "done"
    assert queue.candidates() == []
    assert rows(database, "memory_episodes") == []
    assert rows(database, "consolidation_tasks") == []


def test_budget_and_context_limits_cannot_be_bypassed(runtime, monkeypatch):
    database, queue, journal = runtime
    user(queue, journal)
    _, task = enqueue(queue, journal)
    invoke = Mock()
    assert queue.process(task, invoke=invoke, max_tokens=1) == {"state": "pending", "calls": 0}
    invoke.assert_not_called()
    monkeypatch.setattr(config, "CONTEXT_WINDOW_TOKENS", 1)
    ready(database)
    assert queue.process(queue.claim(), invoke=invoke)["state"] == "failed_attempt"
    invoke.assert_not_called()
    assert queue.checkpoint("session-1") == (0, 0)


def test_corrupt_snapshot_fails_without_model_request(runtime):
    _, queue, journal = runtime
    user(queue, journal)
    _, task = enqueue(queue, journal)
    journal.path.write_bytes(journal.path.read_bytes().replace(b"BUILD_V1", b"BUILD_V2"))
    invoke = Mock()
    assert queue.process(task, invoke=invoke)["state"] == "failed_attempt"
    invoke.assert_not_called()
    assert queue.checkpoint("session-1") == (0, 0)


def test_partial_record_snapshot_is_rejected(runtime):
    _, queue, journal = runtime
    user(queue, journal)
    with pytest.raises(ValueError, match="complete record"):
        queue.enqueue("session-1", journal.path, boundary=journal.path.stat().st_size - 1)


def test_failure_telemetry_never_records_provider_message_or_prompt(runtime):
    database, queue, journal = runtime
    user(queue, journal)
    _, task = enqueue(queue, journal)
    result = queue.process(task, invoke=Mock(side_effect=RuntimeError("password: SENSITIVE_PROVIDER_MESSAGE")))
    assert result["error_type"] == "RuntimeError"
    telemetry = rows(database, "memory_model_calls")
    assert telemetry[0]["error_type"] == "RuntimeError"
    assert "SENSITIVE" not in json.dumps(telemetry)
    assert "BUILD_V1" not in json.dumps(telemetry)


@pytest.mark.parametrize("text", [
    "api_key=SYNTHETIC_SECRET", '{"api_key":"SYNTHETIC_SECRET"}', "password: 'synthetic secret with spaces'",
    "sk-" + "a" * 30, "-----BEGIN PRIVATE KEY-----\nSYNTHETIC\n-----END PRIVATE KEY-----",
])
def test_secrets_are_redacted_from_extraction_inputs(text):
    assert contains_secret(text)
    assert redact_secrets(text) != text
    assert "SYNTHETIC" not in redact_secrets(text)


def test_secret_redaction_spans_multiple_source_chunks(runtime):
    database, _, journal = runtime
    queue = ExtractionQueue(database, chunk_tokens=30)
    user(queue, journal, "-----BEGIN PRIVATE KEY-----\n" + "SYNTHETIC_PRIVATE_MATERIAL " * 50
         + "\n-----END PRIVATE KEY-----")
    queue.enqueue("session-1", journal.path, idle_seconds=0)

    def invoke(**kwargs):
        value = json.loads(kwargs["messages"][-1]["content"])
        assert "PRIVATE_MATERIAL" not in value["source"] + value["context_only"]
        assert "SYNTHETIC" not in value["source"] + value["context_only"]
        return reply()

    for _ in range(30):
        ready(database)
        result = queue.process(queue.claim(), invoke=invoke)
        if result["state"] == "done":
            break
    assert result["state"] == "done"


def test_structured_verification_cannot_create_user_preference(runtime):
    _, queue, journal = runtime
    journal.append_verification(summary="构建使用 BUILD_V1", outcome="passed", tool_call_id="call-1",
                                memory_generation=queue.generation)
    _, task = enqueue(queue, journal)
    assert queue.process(task, invoke=Mock(return_value=reply([candidate(scope="user", kind="preference")])))["state"] == "failed_attempt"


def test_namespace_bound_queue_cannot_read_other_project_candidates(runtime):
    database, queue, journal = runtime
    user(queue, journal)
    _, task = enqueue(queue, journal)
    queue.process(task, invoke=Mock(return_value=reply([candidate()])))
    other = ExtractionQueue(MemoryDatabase(database.root, "project:other"))
    assert other.candidates() == []
    assert other.claim() is None
    assert other.generation == queue.generation


def test_idle_deadline_is_persisted_and_new_snapshot_coalesces_pending_work(runtime):
    database, queue, journal = runtime
    user(queue, journal)
    with patch("harness_code_agent.memory.extraction.time.time", return_value=1000):
        first = queue.enqueue("session-1", journal.path, idle_seconds=60)
    user(queue, journal, "新一轮真实输入。")
    with patch("harness_code_agent.memory.extraction.time.time", return_value=1030):
        latest = queue.enqueue("session-1", journal.path, idle_seconds=60)
    assert first != latest
    assert [row["status"] for row in rows(database, "extraction_tasks")] == ["superseded", "pending"]
    reloaded = ExtractionQueue(MemoryDatabase(database.root, database.namespace))
    with patch("harness_code_agent.memory.extraction.time.time", return_value=1089):
        assert reloaded.claim() is None
    with patch("harness_code_agent.memory.extraction.time.time", return_value=1090):
        assert reloaded.claim()["id"] == latest


def test_forget_removes_already_buffered_candidates_and_episode(runtime):
    database, queue, journal = runtime
    user(queue, journal)
    _, task = enqueue(queue, journal)
    queue.process(task, invoke=Mock(return_value=reply([candidate()], episode="确认构建规范。")))
    stored = queue.candidates()[0]["payload"]
    memory = database.write(topic=stored["topic"], fact_key=stored["fact_key"], kind=stored["kind"],
                            body=stored["body"], evidence=[EvidenceReference(**ref) for ref in stored["evidence"]]).record
    database.forget(memory.id, memory.version)
    assert queue.candidates() == []
    assert rows(database, "memory_episodes") == []


def test_repeated_process_crashes_stop_after_three_expired_leases(runtime):
    database, queue, journal = runtime
    user(queue, journal)
    queue.enqueue("session-1", journal.path, idle_seconds=0)
    for _ in range(3):
        assert queue.claim() is not None
        with database.connect(write=True) as db:
            db.execute("UPDATE extraction_tasks SET lease_until=0")
    assert queue.claim() is None
    assert rows(database, "extraction_tasks")[0]["status"] == "failed"


def test_interrupt_preserves_checkpoint_and_records_attempt(runtime):
    database, queue, journal = runtime
    user(queue, journal)
    _, task = enqueue(queue, journal)
    with pytest.raises(KeyboardInterrupt):
        queue.process(task, invoke=Mock(side_effect=KeyboardInterrupt()))
    assert queue.checkpoint("session-1") == (0, 0)
    assert rows(database, "memory_model_calls")[0]["error_type"] == "KeyboardInterrupt"
