import hashlib
import json
from types import SimpleNamespace

import pytest

from harness_code_agent.memory import (
    EvidenceReference,
    MemoryService,
    MemoryWriteCommand,
)
from harness_code_agent.memory.vectors import DIMENSIONS, SiliconFlowEmbeddings
from harness_code_agent.runtime.builtins.memory_tools import (
    memory_forget,
    memory_read,
    memory_validate,
    memory_write,
)
from harness_code_agent.sessions.journal import SessionJournal
from harness_code_agent.support.tokens import count_text_tokens


def test_file_evidence_expires_and_confirmation_replaces_old_fingerprint(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "config.py"
    source.write_text("value=1", encoding="utf-8")
    service = MemoryService(workspace)
    ref = EvidenceReference("manual", 1, "manual", "a" * 64)
    record = service.write(MemoryWriteCommand(topic="配置", fact_key="config.value", kind="project_fact",
                           body="value=1", evidence=[ref], source_paths=["config.py"]))
    source.write_text("value=2", encoding="utf-8")
    expired = service.read(record.id)
    assert expired["status"] == "expired" and "body" not in expired
    confirmed = service.validate(record.id, expired["version"], [EvidenceReference("manual", 2, "manual", "b" * 64)])
    assert service.read(confirmed.id)["status"] == "active"
    service.restore(record.id, 1, confirmed.version)
    assert service.read(record.id)["status"] == "expired"


def test_tool_requires_original_user_quote_and_preserves_version_history(tmp_path):
    service = MemoryService(tmp_path)
    journal = SessionJournal(tmp_path / "journal.jsonl")
    text = "请记住：构建使用 BUILD_A。"
    source = journal.append_user_input(text, turn_id="t1", memory_generation=service.extraction.generation)
    context = SimpleNamespace(workspace=SimpleNamespace(root=tmp_path), session_id="s1", memory_service=service,
                              memory_journal_path=str(journal.path), memory_user_sequence=source.sequence)
    service.extraction.register("s1", journal.path)
    args = {"topic": "构建", "fact_key": "build.command", "kind": "convention",
            "body": "构建使用 BUILD_A。", "tool_context": context}
    assert memory_write(**args, source_quote="model-invented quote").status == "failed"
    assert memory_write(**args, source_quote=text).status == "success"
    record = service.stores["project"].records()[0]
    assert memory_read(record.id, tool_context=context).status == "success"
    context.memory_use_enabled = False
    assert memory_read(record.id, tool_context=context).status == "failed"
    assert service.stores["project"].history(record.id)[0]["body"] == "构建使用 BUILD_A。"
    assert record.evidence[0]["content_hash"] == hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def unavailable_embeddings(monkeypatch):
    def embed(self, texts):
        raise OSError("offline")
    monkeypatch.setattr(SiliconFlowEmbeddings, "embed", embed)


def test_expired_matches_do_not_crowd_out_active_lexical_result(tmp_path, unavailable_embeddings):
    service = MemoryService(tmp_path)
    store = service.stores["project"]
    ref = EvidenceReference("manual", 1, "manual", "a" * 64)
    with store.connect(write=True) as db:
        for index in range(201):
            record = store._write(db, topic="build", fact_key=str(index), kind="convention",
                                  body="needle", evidence=[ref]).record
            if index < 200:
                db.execute("UPDATE memories SET status='expired' WHERE id=?", (record.id,))
        # Make the valid item rank after every expired match, independent of random IDs.
        db.execute("UPDATE memory_search SET terms='needle'")
        db.execute("UPDATE memory_search SET terms=? WHERE memory_id=?", ("needle " + "extra " * 100, record.id))
    page = service.views["project"].search("needle")
    assert [item["id"] for item in page["items"]] == [record.id]
    assert page["semantic_status"] == "OSError"


@pytest.mark.parametrize("scope", ["project", "both"])
def test_small_search_page_advances_and_stays_within_budget(tmp_path, scope, unavailable_embeddings):
    service = MemoryService(tmp_path)
    ref = EvidenceReference("manual", 1, "manual", "a" * 64)
    for index in range(2):
        service.write(MemoryWriteCommand(topic="build", fact_key=str(index), kind="convention",
                      body="needle " * 250, evidence=[ref]))
    seen, offset = set(), 0
    while True:
        page = service.search("needle", scope=scope, max_tokens=512, offset=offset)
        assert page["items"], page
        assert count_text_tokens(json.dumps(page, ensure_ascii=False)) <= 512
        seen.update(item["id"] for item in page["items"])
        project_page = next(p for p in page["pages"] if p["scope"] == "project")
        assert project_page["next_offset"] > offset
        if not project_page["has_more"]:
            break
        offset = project_page["next_offset"]
    assert len(seen) == 2


@pytest.mark.parametrize("offline", [False, True])
def test_both_memory_scopes_share_one_embedding_request(tmp_path, monkeypatch, offline):
    requests = []

    def embed(self, texts):
        requests.append(texts)
        vector = [1.0, *([0.0] * (DIMENSIONS - 1))]
        return [vector for _ in texts], {"usage": {"total_tokens": 1}}

    monkeypatch.setattr(SiliconFlowEmbeddings, "embed", embed)
    service = MemoryService(tmp_path)
    ref = EvidenceReference("manual", 1, "manual", "a" * 64)
    for store in service.stores.values():
        with store.connect(write=True) as db:
            store._write(db, topic="build", fact_key="command", kind="preference", body="needle", evidence=[ref])
    for vectors in service.vectors.values():
        assert vectors.sync()["indexed"] == 1
    requests.clear()

    if offline:
        def fail(self, texts):
            requests.append(texts)
            raise OSError("offline")
        monkeypatch.setattr(SiliconFlowEmbeddings, "embed", fail)
    result = service.search("needle")
    assert requests == [["needle"]]
    assert {item["scope"] for item in result["items"]} == {"project", "user"}
    assert {page["semantic_status"] for page in result["pages"]} == {"OSError" if offline else "ready"}
    with service.stores["project"].connect() as db:
        assert db.execute("SELECT COUNT(*) FROM embedding_calls WHERE purpose='query'").fetchone()[0] == 1


def test_memory_service_enabled_is_session_state_not_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_MEMORY_DISABLED", "1")
    service = MemoryService(tmp_path)

    assert service.enabled is False
    assert service.search("anything").get("disabled") is True

    service.enabled = True
    assert service.search("anything").get("disabled") is not True

    service.enabled = False
    assert service.search("anything").get("disabled") is True


def test_memory_tools_are_blocked_by_the_master_switch(tmp_path):
    context = SimpleNamespace(
        workspace=SimpleNamespace(root=tmp_path),
        memory_use_enabled=False,
    )

    result = memory_read(tool_context=context)

    assert result.status == "failed"
    assert "memory is disabled" in result.error


def test_master_switch_pauses_and_resumes_memory_and_extraction(tmp_path):
    from unittest.mock import patch

    from harness_code_agent.core.interactive import InteractiveSession

    calls = []
    service = SimpleNamespace(
        enabled=True,
        extraction=SimpleNamespace(
            resume=lambda session_id: calls.append(("resume", session_id)),
            cancel=lambda session_id: calls.append(("cancel", session_id)),
        ),
    )
    session = object.__new__(InteractiveSession)
    session.memory_use_enabled = True
    session.memory_auto_extract_enabled = True
    session.memory_service = service
    session.session = SimpleNamespace(id="session-1", journal_path=tmp_path / "journal.jsonl")
    session.tool_context = SimpleNamespace(
        memory_use_enabled=True, memory_auto_extract_enabled=True
    )
    session.conversation = SimpleNamespace(
        refresh_memory_index=lambda: calls.append(("refresh",))
    )
    session.cwd = tmp_path

    with patch("harness_code_agent.core.interactive.open_memory_session") as open_session:
        message = session.set_memory_enabled(False)

        assert service.enabled is False
        assert session.memory_use_enabled is False
        assert session.memory_auto_extract_enabled is False
        assert session.tool_context.memory_use_enabled is False
        assert session.tool_context.memory_auto_extract_enabled is False
        assert ("cancel", "session-1") in calls
        assert "仅作用于当前会话" in message
        assert open_session.call_count == 0

        session.set_memory_enabled(True)

        assert service.enabled is True
        assert session.memory_use_enabled is True
        assert session.memory_auto_extract_enabled is True
        assert session.tool_context.memory_use_enabled is True
        assert session.tool_context.memory_auto_extract_enabled is True
        assert ("resume", "session-1") in calls
        assert ("refresh",) in calls
        assert open_session.call_count == 1


def test_explicit_memory_operations_work_with_auto_generation_disabled(tmp_path):
    service = MemoryService(tmp_path)
    journal = SessionJournal(tmp_path / "journal.jsonl")
    text = "请记住构建使用 BUILD_A，确认并在测试后忘记这条记忆。"
    source = journal.append_user_input(text, turn_id="t1", memory_generation=service.extraction.generation)
    context = SimpleNamespace(workspace=SimpleNamespace(root=tmp_path), session_id="s1", memory_service=service,
                              memory_auto_extract_enabled=False, memory_journal_path=str(journal.path),
                              memory_user_sequence=source.sequence)
    assert memory_write(topic="构建", fact_key="build.command", kind="convention", body="BUILD_A",
                        source_quote=text, tool_context=context).status == "success"
    record = service.stores["project"].records()[0]
    confirmation = journal.append_user_input(text, turn_id="t2", memory_generation=service.extraction.generation,
                                              auto_extract=False)
    context.memory_user_sequence = confirmation.sequence
    validated = memory_validate(record.id, record.version, text, tool_context=context)
    assert validated.status == "success", validated.output
    version = json.loads(validated.output)["version"]
    assert memory_forget(record.id, version, "invented", tool_context=context).status == "failed"
    assert memory_forget(record.id, version, text, tool_context=context).status == "success"
