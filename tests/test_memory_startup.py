import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from harness_code_agent.agent.context import count_text_tokens
from harness_code_agent.agent.conversation import Agent
from harness_code_agent.memory import MemoryService
from harness_code_agent.memory.database import EvidenceReference
from harness_code_agent.runtime.permissions import PermissionPolicy
from harness_code_agent.runtime.tool_context import ToolContext
from harness_code_agent.sessions.events import EventBus
from harness_code_agent.sessions.journal import SessionJournal
from harness_code_agent.workspace.service import WorkspaceService


def save(service, key, body, *, scope="project", kind="convention", topic="构建与测试", applicability=""):
    return service.stores[scope].write(topic=topic, fact_key=key, kind=kind, body=body,
        applicability=applicability, evidence=[EvidenceReference("manual", 1, "manual", "a" * 64)]).record


def test_startup_separates_defaults_and_topic_navigation(tmp_path):
    service = MemoryService(tmp_path)
    user = save(service, "reply.language", "PERSONAL_DEFAULT", scope="user", kind="preference", topic="交流")
    save(service, "research.sources", "CONDITIONAL_PERSONAL_BODY", scope="user", kind="preference",
         topic="研究偏好", applicability="research")
    project = save(service, "test.policy", "PROJECT_DEFAULT")
    fact = save(service, "test.entry", "ON_DEMAND_FACT", kind="project_fact")
    save(service, "test.steps", "ON_DEMAND_WORKFLOW", kind="workflow")
    save(service, "docs.policy", "PATH_SPECIFIC_BODY", topic="文档", applicability="path:docs/")
    snapshot = service.startup_block()
    assert "## Personal defaults" in snapshot and "## Project defaults" in snapshot
    assert "## Memory topics" in snapshot
    assert "PERSONAL_DEFAULT" in snapshot and "PROJECT_DEFAULT" in snapshot
    assert user.id in snapshot and project.id in snapshot
    assert all(body not in snapshot for body in ("ON_DEMAND_FACT", "ON_DEMAND_WORKFLOW",
                                                "PATH_SPECIFIC_BODY", "CONDITIONAL_PERSONAL_BODY"))
    assert fact.topic_id in snapshot and "test.entry" in snapshot and "path:docs/" in snapshot
    assert count_text_tokens(snapshot) <= 1200


def test_startup_budgets_are_independent_and_skip_oversized_facts(tmp_path):
    service = MemoryService(tmp_path)
    save(service, "a.huge", "UNSPLIT_LONG_BODY " * 1000, scope="user", kind="preference", topic="偏好")
    save(service, "z.short", "SMALL_PERSONAL_DEFAULT", scope="user", kind="preference", topic="偏好")
    for index in range(20):
        save(service, f"project.{index:02}", "PROJECT_POLICY " * 30)
    snapshot = service.startup_block()
    assert "SMALL_PERSONAL_DEFAULT" in snapshot
    assert "UNSPLIT_LONG_BODY" not in snapshot
    assert "PROJECT_POLICY" in snapshot
    assert count_text_tokens(snapshot) <= 1200
    personal, rest = snapshot.split("## Project defaults", 1)
    project, topics = rest.split("## Memory topics", 1)
    assert count_text_tokens(personal.split("## Personal defaults", 1)[1]) <= 300
    assert count_text_tokens("## Project defaults" + project) <= 500
    assert count_text_tokens("## Memory topics" + topics) <= 400


def test_project_override_shadows_personal_default_without_changing_storage(tmp_path):
    service = MemoryService(tmp_path)
    user = save(service, "package.manager", "PERSONAL_PNPM", scope="user", kind="preference", topic="工具习惯")
    save(service, "package.manager", "PROJECT_NPM", topic="构建")
    snapshot = service.startup_block()
    assert "PROJECT_NPM" in snapshot and "PERSONAL_PNPM" not in snapshot
    assert service.stores["user"].read(user.id).body == "PERSONAL_PNPM"


def test_conditional_project_rule_does_not_override_global_default(tmp_path):
    service = MemoryService(tmp_path)
    save(service, "package.manager", "PERSONAL_PNPM", scope="user", kind="preference")
    save(service, "package.manager", "DOCS_NPM_ONLY", applicability="path:docs/")
    snapshot = service.startup_block()
    assert "PERSONAL_PNPM" in snapshot and "DOCS_NPM_ONLY" not in snapshot


def test_inactive_facts_are_not_startup_topics_or_defaults(tmp_path):
    service = MemoryService(tmp_path)
    expired = save(service, "old", "EXPIRED_BODY", topic="EXPIRED_TOPIC")
    pending = save(service, "pending", "UNCERTAIN_BODY", topic="UNCERTAIN_TOPIC")
    with service.stores["project"].connect(write=True) as db:
        db.execute("UPDATE memories SET status='expired' WHERE id=?", (expired.id,))
        db.execute("UPDATE memories SET status='review_required' WHERE id=?", (pending.id,))
    assert service.topics()["topics"] == []
    assert service.startup_block() == ""


def test_pending_project_update_is_not_replaced_by_personal_default(tmp_path):
    service = MemoryService(tmp_path)
    save(service, "package.manager", "PERSONAL_DEFAULT", scope="user", kind="preference")
    record = save(service, "package.manager", "OLD_PROJECT_POLICY")
    store = service.stores["project"]
    journal = SessionJournal(tmp_path / "journal.jsonl")
    journal.append_user_input("更改项目的包管理器。", turn_id="turn",
                              memory_generation=service.extraction.generation)
    task_id = service.extraction.enqueue("session", journal.path)
    with store.connect(write=True) as db:
        db.execute("INSERT INTO memory_candidates(id,owner_namespace,target_namespace,session_id,task_id,"
                   "sequence,start_offset,end_offset,payload) VALUES('pending',?,?, 'session',?,1,0,1,'{}')",
                   (store.namespace, store.namespace, task_id))
        db.execute("INSERT INTO memory_holds VALUES(?, 'pending')", (record.id,))
    snapshot = service.startup_block()
    assert "PERSONAL_DEFAULT" not in snapshot and "OLD_PROJECT_POLICY" not in snapshot
    assert service.topics()["topics"] == []
    assert service.topic(record.topic_id)["items"] == []


def test_topic_navigation_reserves_room_for_both_scopes(tmp_path):
    service = MemoryService(tmp_path)
    for index in range(20):
        save(service, f"entry.{index}", "FACT_BODY", kind="project_fact", topic=f"Project {index:02}")
    user = save(service, "research.citations", "CONDITIONAL_PREF", scope="user", kind="preference",
                topic="研究引用", applicability="research")
    snapshot = service.startup_block()
    assert f"user:{user.topic_id}" in snapshot and "project:topic_" in snapshot
    assert "FACT_BODY" not in snapshot and "CONDITIONAL_PREF" not in snapshot
    assert count_text_tokens(snapshot) <= 1200


def test_topic_is_independent_of_kind_and_isolated_by_scope(tmp_path):
    service = MemoryService(tmp_path)
    first = save(service, "test.policy", "规范")
    second = save(service, "test.steps", "流程", kind="workflow")
    user = save(service, "test.preference", "个人习惯", scope="user", kind="preference")
    assert first.topic_id == second.topic_id and user.topic_id != first.topic_id
    topics = service.topics()["topics"]
    assert len(topics) == 1 and topics[0]["count"] == 2
    assert {item["kind"] for item in service.topic(first.topic_id)["items"]} == {"convention", "workflow"}


def test_topic_alias_reuses_canonical_domain_across_kinds(tmp_path):
    service = MemoryService(tmp_path)
    first = save(service, "test.policy", "规范")
    with service.stores["project"].connect(write=True) as db:
        db.execute("UPDATE topics SET aliases=? WHERE id=?", (json.dumps(["测试命令", "Test Commands"]), first.topic_id))
    second = save(service, "test.steps", "流程", kind="workflow", topic="test commands")
    assert second.topic_id == first.topic_id and second.topic == first.topic


def test_topic_alias_cannot_bypass_forgotten_fact_suppression(tmp_path):
    service = MemoryService(tmp_path)
    first = save(service, "test.policy", "规范")
    save(service, "test.steps", "保留主题", kind="workflow")
    store = service.stores["project"]
    with store.connect(write=True) as db:
        db.execute("UPDATE topics SET aliases=? WHERE id=?", (json.dumps(["测试命令"]), first.topic_id))
    store.forget(first.id, first.version)
    with pytest.raises(ValueError, match="forgotten"):
        store.write(topic="测试命令", fact_key=first.fact_key, kind=first.kind, body="重新归类的旧事实",
                    evidence=[EvidenceReference("new-source", 2, "manual", "b" * 64)])


def test_pending_candidate_using_topic_alias_holds_the_same_fact(tmp_path):
    service = MemoryService(tmp_path)
    record = save(service, "test.policy", "OLD_POLICY")
    with service.stores["project"].connect(write=True) as db:
        db.execute("UPDATE topics SET aliases=? WHERE id=?", (json.dumps(["测试命令"]), record.topic_id))
    journal = SessionJournal(tmp_path / "journal.jsonl")
    journal.append_user_input("测试约定改为 NEW_POLICY。", turn_id="turn",
                              memory_generation=service.extraction.generation)
    service.extraction.enqueue("session", journal.path)
    candidate = {"topic": "测试命令", "fact_key": record.fact_key, "kind": record.kind,
                 "body": "NEW_POLICY", "scope": "project", "quotes": ["测试约定改为 NEW_POLICY。"]}
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
        content=json.dumps({"complete": True, "candidates": [candidate], "episode": ""})))], usage=None)
    assert service.extraction.process(service.extraction.claim(), invoke=lambda **kw: response)["state"] == "done"
    assert service.read(record.id)["status"] == "pending_update"
    assert "OLD_POLICY" not in service.startup_block()


@pytest.mark.parametrize("kind,applicability,evidence", [
    ("project_fact", "", EvidenceReference("manual", 1, "manual", "a" * 64)),
    ("preference", "path:docs", EvidenceReference("manual", 1, "manual", "a" * 64)),
    ("preference", "", EvidenceReference("check", 1, "verification", "a" * 64)),
])
def test_user_scope_cannot_store_project_facts_or_evidence(tmp_path, kind, applicability, evidence):
    service = MemoryService(tmp_path)
    with pytest.raises(ValueError):
        service.stores["user"].write(topic="偏好", fact_key="setting", kind=kind, body="value",
                                      applicability=applicability, evidence=[evidence])


def test_profile_and_compaction_reuse_snapshot_resume_refreshes_once(tmp_path):
    service = MemoryService(tmp_path)
    record = save(service, "test.policy", "ORIGINAL_POLICY")
    context = ToolContext(workspace=WorkspaceService(root=tmp_path),
        permission_policy=PermissionPolicy(mode="danger-full-access"), event_bus=EventBus(),
        session_id="snapshot", memory_service=service)
    with (patch("harness_code_agent.agent.conversation.get_client", return_value=SimpleNamespace(close=lambda: None)),
          patch.object(service, "startup_block", wraps=service.startup_block) as load):
        conversation = Agent("test", "stable", tool_context=context).start_conversation()
        try:
            initial = conversation.messages[0]["content"]
            assert initial.count("[HARNESS_MEMORY_INDEX]") == 1 and "ORIGINAL_POLICY" in initial
            service.stores["project"].write(topic=record.topic, fact_key=record.fact_key, kind=record.kind,
                body="UPDATED_POLICY", memory_id=record.id, expected_version=record.version,
                evidence=[EvidenceReference("manual", 2, "manual", "b" * 64)])
            conversation.rebind_agent(Agent("review", "review rules", tool_context=context))
            context.blocked_tool_names.add("write_file")
            conversation._ensure_request_contract()
            assert conversation._request_contract()["system"] == conversation.messages[0]["content"]
            compacted = conversation._refreshed_context(conversation.messages, phase="manual")
            assert "ORIGINAL_POLICY" in compacted[0]["content"] and "UPDATED_POLICY" not in compacted[0]["content"]
            assert load.call_count == 1
            conversation._replace_messages(compacted, reason="resume")
            assert load.call_count == 2
            current = conversation.messages[0]["content"]
            assert current.count("[HARNESS_MEMORY_INDEX]") == 1
            assert "UPDATED_POLICY" in current and "ORIGINAL_POLICY" not in current
            assert not any(message["role"] == "user" for message in conversation.messages)
            initial_entry = next(entry for entry in conversation.journal.read() if entry.kind == "message")
            assert "ORIGINAL_POLICY" in initial_entry.payload["message"]["content"]
        finally:
            conversation.close()


def test_different_projects_share_only_personal_defaults(tmp_path):
    first, second = MemoryService(tmp_path / "first"), MemoryService(tmp_path / "second")
    save(first, "reply.language", "SHARED_PERSONAL", scope="user", kind="preference")
    save(first, "build.policy", "FIRST_PROJECT")
    save(second, "build.policy", "SECOND_PROJECT")
    snapshot = second.startup_block()
    assert "SHARED_PERSONAL" in snapshot and "SECOND_PROJECT" in snapshot
    assert "FIRST_PROJECT" not in snapshot


def test_worker_uses_parent_project_memory_once(tmp_path):
    from harness_code_agent.agent.coordinator import AgentCoordinator, AgentRecord
    from harness_code_agent.runtime.builtins.memory_tools import memory_read
    from harness_code_agent.runtime.builtins.registry import BUILTIN_TOOL_REGISTRY

    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "source.py").write_text("value = 1\n", encoding="utf-8")
    service = MemoryService(workspace)
    record = save(service, "build.policy", "PARENT_PROJECT_MEMORY")
    context = ToolContext(workspace=WorkspaceService(root=workspace),
        permission_policy=PermissionPolicy(mode="danger-full-access"), event_bus=EventBus(),
        session_id="parent", memory_service=service, tool_registry=BUILTIN_TOOL_REGISTRY.copy())
    with patch("harness_code_agent.agent.conversation.get_client", return_value=SimpleNamespace(close=lambda: None)):
        parent = Agent("main", "rules with [HARNESS_MEMORY_INDEX] as a literal; KEEP_THIS_RULE",
                       tool_context=context).start_conversation()
        coordinator = AgentCoordinator(context, parent_messages=lambda: parent.messages,
                                       parent_system_prompt=lambda: parent.agent.system_prompt)
        worker = AgentRecord(id="memory_worker", name="worker", role="worker", task="Inspect source.py",
                             expected_output="", allowed_paths=["source.py"], fork_turns="all", model_intensity=None)
        coordinator._records[worker.id] = worker
        coordinator._names[worker.name] = worker.id
        try:
            coordinator._ensure_conversation(worker)
            child_context = worker.conversation.agent.tool_context
            assert child_context.workspace.root != workspace
            assert child_context.memory_service is service
            content = worker.conversation.messages[0]["content"]
            assert content.count("[HARNESS_MEMORY_INDEX]\nHistorical references") == 1
            assert "KEEP_THIS_RULE" in content
            assert "PARENT_PROJECT_MEMORY" in content
            assert "PARENT_PROJECT_MEMORY" in memory_read(record.id, tool_context=child_context).output
        finally:
            coordinator.close()
            parent.close()


def test_cache_diagnostics_track_the_loaded_snapshot(tmp_path):
    service = MemoryService(tmp_path)
    save(service, "test.policy", "PROJECT_POLICY")
    bus = EventBus()
    context = ToolContext(workspace=WorkspaceService(root=tmp_path),
        permission_policy=PermissionPolicy(mode="danger-full-access"), event_bus=bus,
        session_id="cache-snapshot", memory_service=service)
    with patch("harness_code_agent.agent.conversation.get_client", return_value=SimpleNamespace(close=lambda: None)):
        conversation = Agent("test", "stable", tool_context=context).start_conversation()
        try:
            from harness_code_agent.agent.utils import _short_hash
            emitted = []
            with patch.object(bus, "emit_event", side_effect=emitted.append):
                conversation.record_llm_usage({"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}, None)
            assert emitted[-1].payload["cache_diagnostics"]["memory_index_hash"] == _short_hash(conversation._memory_index)
        finally:
            conversation.close()
