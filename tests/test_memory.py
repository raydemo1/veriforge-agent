import hashlib
from types import SimpleNamespace

from harness_code_agent.memory import (
    EvidenceReference,
    MemoryService,
    MemoryWriteCommand,
)
from harness_code_agent.runtime.builtins.memory_tools import memory_read, memory_write
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
