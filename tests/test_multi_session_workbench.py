from __future__ import annotations

import json
import subprocess
import sys
import threading
from unittest.mock import patch

import pytest

from harness_code_agent import config
from harness_code_agent.agent.cancellation import CancellationToken, CancelledError
from harness_code_agent.workspace.access import WorkspaceAccessCoordinator, WorkspaceConflict
from harness_code_agent.workspace.service import WorkspaceService
from harness_code_agent.workspace.shell_jobs import ShellJob
from harness_code_agent.workspace.worktrees import WorktreeService


def test_profiles_models_and_histories_are_session_owned(runtime_factory):
    create, clients = runtime_factory
    first, second = create(), create()
    second_model = config.resolve_model_profile(config.MODEL_INTENSITY, second.model_selection)
    first.apply_model_override(model="deepseek-flash", reasoning_effort="max")
    first.switch_profile("review")
    assert second.profile.name() == "general"
    assert config.resolve_model_profile(config.MODEL_INTENSITY, second.model_selection) == second_model
    assert first.submit("分析 A").text == "任务完成"
    assert second.submit("分析 B").text == "任务完成"
    assert "分析 B" not in str(first.conversation.messages)
    assert "分析 A" not in str(second.conversation.messages)


def test_resume_preserves_id_and_fork_uses_stable_boundary(runtime_factory):
    create, _ = runtime_factory
    original = create()
    original.submit("已经完成")
    session_id = original.session.id
    original.conversation.add_user_turn("尚未结算")
    fork = original.session_store.fork(session_id)
    assert fork.id != session_id
    branch = create(session_id=fork.id)
    assert "已经完成" in str(branch.conversation.messages)
    assert "尚未结算" not in str(branch.conversation.messages)
    original.close()
    resumed = create(session_id=session_id)
    assert resumed.session.id == session_id
    assert "已经完成" in str(resumed.conversation.messages)
    assert "尚未结算" not in str(resumed.conversation.messages)


def test_file_versions_prevent_stale_writes(tmp_path):
    (tmp_path / "shared.py").write_text("old", encoding="utf-8")
    first = WorkspaceService(root=tmp_path, session_id="A")
    second = WorkspaceService(root=tmp_path, session_id="B")
    assert first.read_text("shared.py") == second.read_text("shared.py") == "old"
    first.write_text("shared.py", "new A")
    with pytest.raises(WorkspaceConflict):
        second.write_text("shared.py", "stale B")
    assert (tmp_path / "shared.py").read_text(encoding="utf-8") == "new A"
    second.read_text("shared.py")
    second.write_text("shared.py", "coordinated B")
    assert second.access.owned_changes("B")[0][2:] == (b"new A", b"coordinated B")


def test_process_lease_blocks_only_conflicting_writes(tmp_path):
    access = WorkspaceAccessCoordinator(tmp_path)
    script = """
import sys
from pathlib import Path
from harness_code_agent.workspace.access import WorkspaceAccessCoordinator
a=WorkspaceAccessCoordinator(Path(sys.argv[1]))
with a.acquire([a.write_claim(Path(sys.argv[1]) / 'a.txt')]):
 print('locked', flush=True)
 input()
"""
    process = subprocess.Popen([sys.executable, "-u", "-c", script, str(tmp_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "locked"
        with access.acquire([access.write_claim(tmp_path / "b.txt")]):
            pass
        token = CancellationToken()
        timer = threading.Timer(0.1, token.cancel)
        timer.start()
        with pytest.raises(CancelledError), access.acquire([access.write_claim(tmp_path / "a.txt")], token):
            pass
        timer.join()
    finally:
        process.communicate("exit\n", timeout=5)
    with access.acquire([access.write_claim(tmp_path / "a.txt")]):
        pass


def test_background_job_holds_project_lease_until_real_exit(tmp_path):
    first, second = WorkspaceAccessCoordinator(tmp_path), WorkspaceAccessCoordinator(tmp_path)
    job = ShellJob("job", "server", None, None)
    with first.acquire([first.write_claim()]) as lease:
        lease.transfer_to(job)
    token = CancellationToken()
    token.cancel()
    with pytest.raises(CancelledError), second.acquire([second.write_claim(tmp_path / "a")], token):
        pass
    job.mark("exited", exit_code=0)
    with second.acquire([second.write_claim(tmp_path / "a")]):
        pass


def test_rewind_preserves_another_sessions_changes(runtime_factory, tmp_path):
    create, _ = runtime_factory
    first, second = create(), create()
    first.submit("保存稳定边界")
    point = max(first.recovery_points().values(), key=lambda item: item["turn"])
    first.tool_context.workspace.write_text("a", "A")
    second.tool_context.workspace.write_text("b", "B")
    old_id = first.session.id
    old_conversation = first.conversation
    assert first.rewind_to_point(point["id"])
    assert first.session.id == old_id
    assert first.conversation is not old_conversation
    assert not (tmp_path / "a").exists()
    assert (tmp_path / "b").read_text(encoding="utf-8") == "B"
    first.tool_context.workspace.write_text("a", "A2")
    second.tool_context.workspace.read_text("a")
    second.tool_context.workspace.write_text("a", "B2")
    with pytest.raises(WorkspaceConflict):
        first.rewind_to_point(point["id"])
    assert (tmp_path / "a").read_text(encoding="utf-8") == "B2"


def test_worktree_inherits_dirty_local_and_applies_only_task_delta(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    git("init")
    (tmp_path / "tracked").write_text("base", encoding="utf-8")
    git("add", "tracked")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "base")
    (tmp_path / "tracked").write_text("dirty", encoding="utf-8")
    (tmp_path / "untracked").write_text("kept", encoding="utf-8")
    service = WorktreeService(tmp_path)
    tree = service.create("isolated")
    assert (tree / "tracked").read_text(encoding="utf-8") == "dirty"
    assert (tree / "untracked").read_text(encoding="utf-8") == "kept"
    (tree / "task").write_text("new task", encoding="utf-8")
    (tmp_path / "tracked").write_text("later Local", encoding="utf-8")
    assert service.apply("isolated") == ["task"]
    assert (tmp_path / "tracked").read_text(encoding="utf-8") == "later Local"
    assert (tmp_path / "task").read_text(encoding="utf-8") == "new task"


def test_bridge_parallel_turns_switch_cancel_and_independent_approval(runtime_factory, tmp_path):
    from harness_code_agent.tui_bridge import BridgeServer
    from harness_code_agent.runtime.approvals import ApprovalRequest
    condition = threading.Condition()
    messages = []
    class Workbench(BridgeServer):
        def _write(self, message):
            with condition:
                messages.append(message)
                condition.notify_all()
    def wait(predicate):
        with condition:
            assert condition.wait_for(lambda: predicate(messages), timeout=10), json.dumps([row for row in messages if row.get("event", {}).get("type") == "progress"], ensure_ascii=False, indent=2)
    server = Workbench(cwd=tmp_path, profile_name="general", profile_explicit=True)
    try:
        first_id = server.registry.active_id
        second_id = server.registry.create()
        for session_id in (first_id, second_id):
            wait(lambda rows, session_id=session_id: any(row.get("event", {}).get("sessionId") == session_id
                 and row["event"]["type"] == "progress" and row["event"].get("status") == "ready" for row in rows))
        first = server.registry.get(first_id).host._require_session()
        client = first.conversation.client
        client.gate, client.started = threading.Event(), threading.Event()
        server._handle_request({"id": "submit-a", "method": "submit", "sessionId": first_id, "params": {"text": "等待 A", "attachmentIds": []}})
        assert client.started.wait(5)
        server._handle_request({"id": "submit-b", "method": "submit", "sessionId": second_id, "params": {"text": "完成 B", "attachmentIds": []}})
        wait(lambda rows: any(row.get("event", {}).get("sessionId") == second_id and row["event"]["type"] == "turn_state"
                             and row["event"].get("state") == "idle" for row in rows))
        assert not client.gate.is_set()
        assert first._turn_thread is not None
        server._handle_request({"id": "cancel-a", "method": "cancel", "sessionId": first_id})
        wait(lambda rows: any(row.get("event", {}).get("sessionId") == first_id and row["event"]["type"] == "turn_state"
                             and row["event"].get("state") == "idle" for row in rows))
        client.gate.set()
        answer = []
        approval = threading.Thread(target=lambda: answer.append(first.approval_provider.request(
            ApprovalRequest("custom", {}, "受控请求", "验证会话归属"))))
        approval.start()
        wait(lambda rows: any(row.get("event", {}).get("sessionId") == first_id and row["event"]["type"] == "interaction" for row in rows))
        interaction = next(row["event"] for row in reversed(messages) if row.get("event", {}).get("type") == "interaction")
        assert server.registry.active_id == second_id
        server._handle_request({"id": "wrong", "method": "resolve_interaction", "sessionId": second_id,
                                "params": {"id": interaction["id"], "result": {"decision": "approve"}}})
        assert not answer
        server._handle_request({"id": "right", "method": "resolve_interaction", "sessionId": first_id,
                                "params": {"id": interaction["id"], "result": {"decision": "approve"}}})
        approval.join(2)
        assert answer[0].approved
        assert "完成 B" not in str(first.conversation.messages)
    finally:
        if "client" in locals():
            client.gate.set()
        server.close()


def test_rewind_failure_keeps_files_journal_and_runtime(runtime_factory, tmp_path):
    create, _ = runtime_factory
    runtime = create()
    runtime.submit("稳定边界")
    point = max(runtime.recovery_points().values(), key=lambda item: item["turn"])
    runtime.tool_context.workspace.write_text("a", "later")
    previous = runtime.conversation
    journal = runtime.session.journal_path.read_bytes()
    metadata = runtime.session.metadata_path.read_bytes()
    from harness_code_agent.runtime.recovery import atomic_json
    def fail(path, value):
        if path == runtime.session.metadata_path and value.get("checkpoint", {}).get("id") == point["id"]:
            raise OSError("metadata unavailable")
        return atomic_json(path, value)
    with patch("harness_code_agent.core.interactive.atomic_json", side_effect=fail), pytest.raises(OSError):
        runtime.rewind_to_point(point["id"])
    assert runtime.conversation is previous
    assert not previous._closed
    assert (tmp_path / "a").read_text(encoding="utf-8") == "later"
    assert runtime.session.journal_path.read_bytes() == journal
    assert runtime.session.metadata_path.read_bytes() == metadata
