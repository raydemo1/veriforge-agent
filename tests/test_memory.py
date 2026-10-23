import hashlib
import json
from types import SimpleNamespace

import pytest

from harness_code_agent.agent.context import count_text_tokens
from harness_code_agent.memory import (
    EvidenceReference,
    MemoryService,
    MemoryWriteCommand,
)
from harness_code_agent.memory.reconciliation import CurrentMemoryView
from harness_code_agent.runtime.builtins.memory_tools import (
    memory_forget,
    memory_read,
    memory_validate,
    memory_write,
)
from harness_code_agent.sessions.journal import SessionJournal


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


class UnavailableVectors:
    def __init__(self, database):
        pass

    def search(self, query, *, limit):
        raise OSError("offline")

    def pending(self):
        return 0


def test_expired_matches_do_not_crowd_out_active_lexical_result(tmp_path):
    service = MemoryService(tmp_path, vector_factory=UnavailableVectors)
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
    page = CurrentMemoryView(store, vectors=UnavailableVectors(store)).search("needle")
    assert [item["id"] for item in page["items"]] == [record.id]
    assert page["semantic_status"] == "OSError"


@pytest.mark.parametrize("scope", ["project", "both"])
def test_small_search_page_advances_and_stays_within_budget(tmp_path, scope):
    service = MemoryService(tmp_path, vector_factory=UnavailableVectors)
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
