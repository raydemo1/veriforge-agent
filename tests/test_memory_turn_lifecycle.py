import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from openai.types.chat import ChatCompletion

from harness_code_agent.core.interactive import InteractiveSession
from harness_code_agent.memory import (
    MemoryService,
    background,
    extraction,
    reconciliation,
)
from harness_code_agent.memory.database import EvidenceReference
from harness_code_agent.memory.reconciliation import ConsolidationQueue
from harness_code_agent.runtime.approvals import NoApprovalProvider
from harness_code_agent.sessions.journal import SessionJournal


def reply(value):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
        content=json.dumps(value, ensure_ascii=False)))], usage=None)


def test_turn_completion_enqueues_immediately_and_survives_store_reload(tmp_path):
    service = MemoryService(tmp_path)
    journal = SessionJournal(tmp_path / "journal.jsonl")
    queue = service.extraction
    queue.register("session", journal.path)
    journal.append_user_input("构建使用 BUILD_NOW。", turn_id="turn", memory_generation=queue.generation)
    queue.set_active("session")
    task_id = queue.enqueue("session", journal.path)
    assert queue.claim() is None
    queue.finish_activity("session")
    task = MemoryService(tmp_path).extraction.claim()
    assert task is not None and task["id"] == task_id


def test_background_extraction_starts_after_reply_without_blocking(tmp_path):
    service = MemoryService(tmp_path)
    journal = SessionJournal(tmp_path / "journal.jsonl")
    started, release = threading.Event(), threading.Event()
    key = background._key(service)
    def extract(**kwargs):
        started.set()
        assert release.wait(5)
        return reply({"complete": True, "candidates": [], "episode": ""})
    with patch.object(extraction, "_call_model", side_effect=extract):
        background.open_memory_session(tmp_path, "session", journal.path)
        try:
            background.begin_memory_turn(tmp_path, "session")
            journal.append_user_input("确认新的构建约定。", turn_id="turn",
                                      memory_generation=service.extraction.generation)
            background.finish_memory_turn(tmp_path, "session", journal.path)
            assert started.wait(2), "extraction should start without an idle timer"
            assert not release.is_set()
        finally:
            release.set()
            deadline = time.monotonic() + 5
            while key in background._running and time.monotonic() < deadline:
                time.sleep(.01)
            background.close_memory_session(tmp_path, "session")
        assert key not in background._running
        assert key not in background._timers
        assert service.extraction.checkpoint("session") == (journal.path.stat().st_size, 0)


@pytest.mark.parametrize("mode", ["direct", "zero-budget", "broken-embeddings"])
def test_consolidation_updates_facts_while_vectors_are_unavailable(tmp_path, mode):
    service = MemoryService(tmp_path)
    store = service.stores["project"]
    old = store.write(topic="构建", fact_key="build.command", kind="convention", body="使用 BUILD_OLD。",
                      evidence=[EvidenceReference("manual", 1, "manual", "a" * 64)]).record
    journal = SessionJournal(tmp_path / "journal.jsonl")
    text = "构建约定更新为 BUILD_NEW。"
    journal.append_user_input(text, turn_id="turn", memory_generation=service.extraction.generation)
    service.extraction.enqueue("session", journal.path)
    candidate = {"topic": "构建", "fact_key": "build.command", "kind": "convention", "body": "使用 BUILD_NEW。",
                 "scope": "project", "quotes": [text]}
    service.extraction.process(service.extraction.claim(), invoke=lambda **kw: reply(
        {"complete": True, "candidates": [candidate], "episode": ""}))
    def consolidate(**kwargs):
        payload = json.loads(kwargs["messages"][-1]["content"])
        assert any(record["id"] == old.id for record in payload["existing"])
        return reply({"decisions": [{"candidate_id": payload["candidates"][0]["id"], "action": "update",
                      "target_id": old.id, "topic": old.topic, "fact_key": old.fact_key, "applicability": "",
                      "body": "使用 BUILD_NEW。", "reason": "用户明确更新"}]})
    with patch.object(reconciliation.VectorIndex, "search", side_effect=AssertionError("no embedding request")):
        if mode == "direct":
            worker = ConsolidationQueue(store)
            assert worker.process(worker.claim(), invoke=consolidate)["state"] == "done"
        else:
            with (patch.object(reconciliation, "_call_model", side_effect=consolidate),
                  patch("harness_code_agent.memory.vectors.SiliconFlowEmbeddings.embed", side_effect=RuntimeError("offline"))):
                result = background.drain_memory_jobs(service, max_model_calls=1,
                                                      max_embedding_requests=0 if mode == "zero-budget" else 8)
            assert result["states"] == ["done"], result
    assert store.read(old.id).body == "使用 BUILD_NEW。"
    assert len(store.history(old.id)) == 2
    assert service.vectors["project"].pending() == 1


def test_user_turns_do_not_search_or_inject_memory_bodies(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_MEMORY_GENERATION_DISABLED", "1")
    (tmp_path / ".harness").mkdir()
    (tmp_path / ".harness/mcp.json").write_text('{"servers":{}}', encoding="utf-8")
    service = MemoryService(tmp_path)
    service.stores["project"].write(topic="构建入口", fact_key="build.command", kind="project_fact",
                                     body="UNREQUESTED_MEMORY_BODY",
                                     evidence=[EvidenceReference("manual", 1, "manual", "a" * 64)])
    completion = ChatCompletion.model_validate({"id": "fixture", "object": "chat.completion", "created": 0,
        "model": "fixture", "choices": [{"index": 0, "finish_reason": "stop",
                                         "message": {"role": "assistant", "content": "done"}}]})
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=completion))),
                             close=lambda: None)
    with patch("harness_code_agent.agent.conversation.get_client", return_value=client):
        session = InteractiveSession(cwd=tmp_path, profile_name="general", profile_explicit=True,
                                     approval_provider=NoApprovalProvider(), enable_turn_summary=False)
        try:
            with patch.object(session.memory_service, "search", side_effect=AssertionError("automatic recall")):
                session.submit("需求" * 2500)
                previous = list(session.conversation.messages)
                session.submit("继续")
            assert session.conversation.messages[:len(previous)] == previous
            messages = json.dumps(session.conversation.messages, ensure_ascii=False)
            assert "[HARNESS_MEMORY_INDEX]" in messages and "构建入口" in messages
            assert "UNREQUESTED_MEMORY_BODY" not in messages
            assert "[HARNESS_MEMORY_REFERENCE]" not in messages
            names = {schema["function"]["name"] for schema in session.agent.tool_schemas}
            assert {"memory_read", "memory_search"} <= names
        finally:
            session.close()


def test_closed_session_does_not_commit_a_late_extraction_result(tmp_path):
    service = MemoryService(tmp_path)
    journal = SessionJournal(tmp_path / "journal.jsonl")
    text = "构建使用 BUILD_LATE。"
    journal.append_user_input(text, turn_id="turn", memory_generation=service.extraction.generation)
    service.extraction.enqueue("session", journal.path)
    eligible = True
    def extract(**kwargs):
        nonlocal eligible
        eligible = False
        return reply({"complete": True, "episode": "", "candidates": [{"topic": "构建",
                      "fact_key": "build.command", "kind": "convention", "body": text,
                      "scope": "project", "quotes": [text]}]})
    result = service.extraction.process(service.extraction.claim(), invoke=extract, should_continue=lambda: eligible)
    assert result == {"state": "pending", "calls": 1}
    assert service.extraction.candidates() == []
    assert service.extraction.checkpoint("session") == (0, 0)
    assert service.extraction.claim() is not None


def test_closed_session_does_not_commit_a_late_consolidation_result(tmp_path):
    service = MemoryService(tmp_path)
    journal = SessionJournal(tmp_path / "journal.jsonl")
    text = "构建使用 BUILD_LATE。"
    journal.append_user_input(text, turn_id="turn", memory_generation=service.extraction.generation)
    service.extraction.enqueue("session", journal.path)
    service.extraction.process(service.extraction.claim(), invoke=lambda **kw: reply({"complete": True,
        "episode": "", "candidates": [{"topic": "构建", "fact_key": "build.command", "kind": "convention",
        "body": text, "scope": "project", "quotes": [text]}]}))
    eligible = True
    def consolidate(**kwargs):
        nonlocal eligible
        eligible = False
        payload = json.loads(kwargs["messages"][-1]["content"])
        return reply({"decisions": [{"candidate_id": payload["candidates"][0]["id"], "action": "add",
                      "target_id": None, "topic": "构建", "fact_key": "build.command", "applicability": "",
                      "body": text, "reason": "确认新约定"}]})
    queue = ConsolidationQueue(service.stores["project"])
    result = queue.process(queue.claim(), invoke=consolidate, should_continue=lambda: eligible)
    assert result == {"state": "pending", "calls": 1}
    assert service.stores["project"].records() == []
    assert service.extraction.candidates()[0]["status"] == "pending"
