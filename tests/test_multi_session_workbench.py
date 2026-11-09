from __future__ import annotations

import json
import subprocess
import sys
import threading
from unittest.mock import patch

import pytest

from harness_code_agent import config
from harness_code_agent.agent.cancellation import CancellationToken, CancelledError
from harness_code_agent.workspace.access import (
    WorkspaceAccessCoordinator,
    WorkspaceConflict,
)
from harness_code_agent.workspace.service import WorkspaceService
from harness_code_agent.workspace.worktrees import WorktreeService

pytestmark = pytest.mark.session


def test_models_and_histories_are_session_owned(runtime_factory):
    create, _clients = runtime_factory
    first, second = create(), create()
    second_model = config.resolve_model_profile(config.MODEL_INTENSITY, second.model_selection)
    first.apply_model_override(model="deepseek-flash", reasoning_effort="max")
    first.set_permission_mode("read-only")
    assert second.permission_mode == "workspace-write"
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
    for error in (CancelledError("停止"), RuntimeError("执行失败")):
        with pytest.raises(type(error)), original._turn_scope():
            original.conversation.add_user_turn("尚未结算")
            raise error
    fork = original.session_store.fork(session_id)
    assert fork.id != session_id
    branch = create(session_id=fork.id)
    assert "已经完成" in str(branch.conversation.messages)
    assert "尚未结算" not in str(branch.conversation.messages)
    original.close()
    resumed = create(session_id=session_id)
    assert resumed.session.id == session_id
    assert "已经完成" in str(resumed.conversation.messages)
    assert "尚未结算" in str(resumed.conversation.messages)


def test_registry_creation_failure_preserves_active_session_and_rejects_worktree_forks(tmp_path):
    from types import SimpleNamespace

    from harness_code_agent.core.session_runtime_registry import SessionRuntimeRegistry

    events = []
    registry = SessionRuntimeRegistry(tmp_path, lambda _: SimpleNamespace(), events.append)
    first = registry.create(environment="worktree")
    for isolated in (False, True):
        with pytest.raises(ValueError, match="Worktree"):
            registry.fork(first, worktree=isolated)
    with (
        patch.object(registry, "factory", side_effect=RuntimeError("初始化失败")),
        pytest.raises(RuntimeError, match="初始化失败"),
    ):
        registry.create()
    assert registry.active_id == first
    assert events[-1]["activeSessionId"] == first
    assert [record["id"] for record in events[-1]["sessions"]] == [first]
    registry.access.release_runtime(first)


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

    from harness_code_agent.workspace.worktrees import (
        WorktreeMergeConflict,
        prepare_merge_review,
    )
    (tree / "tracked").write_text("agent candidate", encoding="utf-8")
    with pytest.raises(WorktreeMergeConflict) as caught:
        service.apply("isolated")
    files = caught.value.files
    _, approved = prepare_merge_review(tree / ".harness/review", files, files)
    assert (tmp_path / "tracked").read_text(encoding="utf-8") == "later Local"
    (tmp_path / "tracked").write_text("newer Local", encoding="utf-8")
    with pytest.raises(WorkspaceConflict):
        service.apply("isolated", approved=approved)
    assert (tmp_path / "tracked").read_text(encoding="utf-8") == "newer Local"
    (tmp_path / "tracked").write_text("later Local", encoding="utf-8")
    assert service.apply("isolated", approved=approved) == ["tracked"]
    assert (tmp_path / "tracked").read_text(encoding="utf-8") == "agent candidate"


def test_bridge_parallel_turns_switch_cancel_and_independent_approval(runtime_factory, tmp_path):
    from harness_code_agent.runtime.approvals import ApprovalRequest
    from harness_code_agent.tui_bridge import BridgeServer
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
    server = Workbench(cwd=tmp_path)
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


def test_write_leases_are_shared_within_an_owner_and_waits_are_reported(tmp_path):
    from contextlib import contextmanager

    access = WorkspaceAccessCoordinator(tmp_path)
    claim = access.write_claim(tmp_path / "dist")
    background = access.lease([claim], owner="session-A")
    try:
        with access.acquire([access.write_claim(tmp_path / "dist")], owner="session-A"):
            pass
        reports = []

        @contextmanager
        def report():
            reports.append("waiting")
            try:
                yield
            finally:
                reports.append("done")

        token = CancellationToken()
        timer = threading.Timer(0.2, token.cancel)
        timer.start()
        with pytest.raises(CancelledError), access.acquire([claim], token, owner="session-B", on_blocked=report):
            pass
        timer.join()
        assert reports == ["waiting", "done"]
    finally:
        background.release()
    with access.acquire([claim], owner="session-B"):
        pass


def test_refused_suspension_leaves_runtime_usable(runtime_factory):
    create, _ = runtime_factory
    runtime = create()
    runtime.submit("第一轮")
    coordinator = runtime.tool_context.agent_coordinator
    with (
        patch.object(coordinator.changes, "has_pending_changes", return_value=True),
        pytest.raises(RuntimeError, match="代码提案"),
    ):
        runtime.close(suspend=True)
    assert not runtime._closing and not coordinator._closed
    assert runtime.submit("第二轮").text == "任务完成"


def test_suspension_keeps_finished_agents_and_shell_jobs(runtime_factory):
    from harness_code_agent.workspace.shell_jobs import ShellJob, ShellOutputBuffer

    create, _ = runtime_factory
    runtime = create()
    runtime.submit("准备")
    session_id = runtime.session.id
    coordinator = runtime.tool_context.agent_coordinator
    coordinator.spawn(name="scout", role="explorer", task="inspect")
    assert not coordinator.wait(["scout"], timeout_seconds=10)["timed_out"]
    output = ShellOutputBuffer()
    output.append("构建完成\n")
    job = ShellJob(job_id="shell-job-done", command="build", pid=None, process=None, output=output)
    job.mark("exited", exit_code=0)
    runtime.conversation.runtime_state.shell_job_manager._store(job)

    runtime.close(suspend=True)
    assert runtime._suspended_state_path.exists()
    resumed = create(session_id=session_id)
    assert not resumed._suspended_state_path.exists()
    manager = resumed.conversation.runtime_state.shell_job_manager
    assert manager.get("shell-job-done").exit_code == 0
    assert manager.read_output("shell-job-done") == "构建完成\n"
    agents = resumed.tool_context.agent_coordinator
    [agent] = agents.list()["agents"]
    assert (agent["name"], agent["status"], agent["summary"]) == ("scout", "completed", "任务完成")
    agents.followup("scout", "继续检查")
    assert not agents.wait(["scout"], timeout_seconds=10)["timed_out"]
    messages = str(agents._record("scout").conversation.messages)
    assert "inspect" in messages and "继续检查" in messages


def test_worktree_fork_is_refused_while_a_turn_runs(tmp_path):
    from types import SimpleNamespace

    from harness_code_agent.tui_bridge import SessionBridge

    bridge = SessionBridge.__new__(SessionBridge)
    bridge._session = SimpleNamespace(_execution_condition=threading.Condition(threading.RLock()), _turn_thread=1)
    with pytest.raises(ValueError, match="回合仍在执行"), bridge.turns_paused():
        pass
    bridge._session._turn_thread = None
    with bridge.turns_paused():
        assert bridge._session._execution_condition._is_owned()
