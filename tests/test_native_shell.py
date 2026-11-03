from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

import pytest

from harness_code_agent.agent.cancellation import CancellationToken, CancelledError
from harness_code_agent.runtime.builtins.shell import run_bash
from harness_code_agent.workspace.access import WorkspaceAccessCoordinator
from harness_code_agent.workspace.native_shell import validate_native_shell, write_claims
from harness_code_agent.workspace.shell_jobs import ShellJobManager


pytestmark = pytest.mark.integration


@pytest.fixture
def native_manager(tmp_path, require_integration_tool):
    require_integration_tool(os.name == "nt", "Native sandbox requires Windows")
    try:
        validate_native_shell()
    except RuntimeError as exc:
        require_integration_tool(False, str(exc))
    manager = ShellJobManager(tmp_path)
    yield manager
    manager.close()


def test_native_directory_grants_do_not_leak_between_calls(native_manager, tmp_path):
    runtime = SimpleNamespace(shell_job_manager=native_manager)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    result = run_bash("Set-Content a/first.txt '中文'; Set-Content b/keep.txt 'keep'",
                      write_paths=["a", "b"], runtime_state=runtime)
    assert result.return_code == 0
    for script in ("Set-Content b/forbidden.txt 'bad'", "Remove-Item b/keep.txt"):
        result = run_bash(script, write_paths=["a"], runtime_state=runtime)
        assert result.return_code != 0
    assert not (tmp_path / "b" / "forbidden.txt").exists()
    assert (tmp_path / "b" / "keep.txt").read_text(encoding="utf-8").strip() == "keep"
    result = run_bash("Set-Content a/readonly.txt 'bad'", runtime_state=runtime)
    assert result.return_code != 0
    assert not (tmp_path / "a" / "readonly.txt").exists()


def test_background_native_job_holds_only_its_directory_until_exit(native_manager, tmp_path):
    (tmp_path / "out").mkdir()
    job = native_manager.start("Set-Content out/ready 'ready'; Start-Sleep -Seconds 30", write_paths=["out"])
    deadline = time.monotonic() + 10
    while not (tmp_path / "out" / "ready").exists() and time.monotonic() < deadline:
        time.sleep(.02)
    assert (tmp_path / "out" / "ready").exists(), job.output_tail
    access = WorkspaceAccessCoordinator(tmp_path)
    with access.acquire([access.write_claim(tmp_path / "unrelated.txt")]):
        (tmp_path / "unrelated.txt").write_text("another session", encoding="utf-8")
    token = CancellationToken()
    timer = threading.Timer(.1, token.cancel)
    timer.start()
    try:
        with pytest.raises(CancelledError), access.acquire(write_claims(tmp_path, ["out"]), token):
            pass
    finally:
        timer.join()
    native_manager.stop(job.job_id, grace_seconds=.2)
    assert job._finished.wait(5)
    with access.acquire(write_claims(tmp_path, ["out"])):
        pass


def test_native_rejects_whole_local_background_without_creating_worktree(native_manager, tmp_path):
    job = native_manager.start("Set-Content unwanted.txt 'bad'", write_paths=["."])
    assert job.status == "failed"
    assert not (tmp_path / "unwanted.txt").exists()
    assert not (tmp_path / ".harness" / "worktrees").exists()


def test_cancelled_native_output_keeps_full_unicode_artifact(native_manager, tmp_path):
    token = CancellationToken()
    errors = []

    def cancel_when_ready():
        deadline = time.monotonic() + 10
        while not (tmp_path / "out" / "ready").exists() and time.monotonic() < deadline:
            time.sleep(.02)
        if not (tmp_path / "out" / "ready").exists():
            errors.append("process did not become ready")
        token.cancel()

    thread = threading.Thread(target=cancel_when_ready)
    thread.start()
    try:
        with pytest.raises(CancelledError) as caught:
            run_bash("Write-Host -NoNewline ('begin-' + ('甲' * 10000) + '-end'); "
                     "Set-Content out/ready ready; Start-Sleep -Seconds 30",
                     write_paths=["out"], runtime_state=SimpleNamespace(shell_job_manager=native_manager),
                     cancellation_token=token)
    finally:
        thread.join(15)
    assert not errors
    result = caught.value.tool_result
    assert result.metadata["output_spilled"]
    from pathlib import Path
    content = Path(result.metadata["artifact_path"]).read_text(encoding="utf-8")
    assert "begin-" + "甲" * 10000 + "-end" in content
    assert result.metadata["cancelled"]
