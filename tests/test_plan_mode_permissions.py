from __future__ import annotations

import pytest

from harness_code_agent.core.interactive import InteractiveSession, TurnResult
from harness_code_agent.runtime.permissions import PermissionPolicy


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_MEMORY_DISABLED", "1")
    monkeypatch.setenv("HARNESS_MEMORY_GENERATION_DISABLED", "1")
    monkeypatch.setenv("HARNESS_MENTION_MODE", "off")
    monkeypatch.setenv("HARNESS_PERMISSION_MODE", PermissionPolicy.WORKSPACE_WRITE)
    live = InteractiveSession(
        cwd=tmp_path,
        enable_turn_summary=False,
        output_sink=lambda _text: None,
    )
    yield live
    live.close()


def test_plan_mode_denies_writes_and_restores_after_execute(session, tmp_path, monkeypatch):
    root = str(tmp_path)
    assert session.tool_context.permission_policy.mode == PermissionPolicy.WORKSPACE_WRITE

    session.start_plan()
    policy = session.tool_context.permission_policy
    assert policy.mode == PermissionPolicy.READ_ONLY
    assert policy.decide_tool_call(
        "read_file", {"path": "a.py"}, tool_permission="read", workspace_root=root
    ).action == "allow"
    assert policy.decide_tool_call(
        "write_file", {"path": "a.py", "content": "x"}, tool_permission="edit", workspace_root=root
    ).action == "deny"
    assert policy.decide_tool_call(
        "run_bash", {"command": "echo hi > a.txt"}, tool_permission="shell", workspace_root=root
    ).action == "deny"
    assert policy.decide_tool_call(
        "mcp__docs__delete", {}, tool_permission="dangerous", workspace_root=root
    ).action == "deny"

    session.pending_plan_markdown = "# Plan\n\n1. Do the thing."
    monkeypatch.setattr(session, "_run_turn", lambda *args, **kwargs: TurnResult(text="done"))
    session.execute_pending_plan()

    assert session.plan_mode is False
    assert session.tool_context.permission_policy.mode == PermissionPolicy.WORKSPACE_WRITE


def test_restored_pending_plan_resumes_plan_mode(session, tmp_path):
    session.plan_mode = False
    session._restore_pending_plan([
        {"type": "plan_ready", "payload": {"plan_markdown": "# Rev 2", "plan_revision": 2}},
    ])

    assert session.pending_plan_markdown == "# Rev 2"
    assert session.pending_plan_revision == 2
    assert session.plan_mode is True
    assert session.tool_context.permission_policy.mode == PermissionPolicy.READ_ONLY


def test_planning_turn_tells_the_model_not_to_write(session, monkeypatch):
    captured: dict[str, str | None] = {}

    def fake_run(user_prompt, cancellation_token=None, turn_instruction=None, attachments=()):
        captured["instruction"] = turn_instruction
        return TurnResult(text="# Plan")

    session.start_plan()
    monkeypatch.setattr(session, "_run_turn", fake_run)
    session.submit("plan the parser migration")

    instruction = captured.get("instruction") or ""
    assert "planning turn" in instruction
    assert "Do not modify the workspace" in instruction


def test_delegated_worker_cannot_write_under_read_only_parent(tmp_path):
    parent = PermissionPolicy(mode=PermissionPolicy.READ_ONLY)
    worker = PermissionPolicy.delegated_from(parent, "worker", allowed_paths=["src"])

    assert worker.mode == PermissionPolicy.READ_ONLY
    assert worker.decide_tool_call(
        "write_file",
        {"path": "src/a.py", "content": "x"},
        tool_permission="edit",
        workspace_root=str(tmp_path),
    ).action == "deny"

    open_parent = PermissionPolicy(mode=PermissionPolicy.WORKSPACE_WRITE)
    assert PermissionPolicy.delegated_from(open_parent, "worker").mode == PermissionPolicy.EVAL
