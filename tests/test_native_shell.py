from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness_code_agent.runtime.builtins.shell import run_bash
from harness_code_agent.support.cancellation import CancellationToken, CancelledError
from harness_code_agent.workspace.access import WorkspaceAccessCoordinator
from harness_code_agent.workspace.native_shell import (
    validate_native_shell,
    write_claims,
)
from harness_code_agent.workspace.shell_jobs import ShellJobManager

pytestmark = pytest.mark.integration


@pytest.fixture
def tmp_path():
    # Python 3.14 implements mkdir(mode=0o700) with OWNER RIGHTS on Windows.
    # Pytest uses that mode for its temp tree, but restricted tokens cannot
    # use owner rights. Model a normal workspace with inherited user ACLs;
    # do not alter pytest's shared directories or sandbox write permissions.
    root = Path(tempfile.gettempdir()) / f"veriforge native '中文' {uuid.uuid4().hex}"
    root.mkdir()
    try:
        yield root
    finally:
        shutil.rmtree(root)


def test_wrapped_powershell_keeps_object_output_and_exit_code(
    tmp_path, require_integration_tool
):
    require_integration_tool(os.name == "nt", "Windows only")
    from harness_code_agent.workspace.native_shell import _powershell_script
    from harness_code_agent.workspace.shell_environment import shell_executable

    (tmp_path / "app").mkdir()
    (tmp_path / "README.md").write_text("readme\n", encoding="utf-8")

    def run(command: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                shell_executable(), "-NoLogo", "-NoProfile", "-NonInteractive",
                "-Command", _powershell_script(tmp_path, command),
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    listing = run(f'Get-ChildItem -Force | Select-Object Mode,Name; & "{sys.executable}" -c "import sys; sys.exit(3)"')
    assert listing.returncode == 3
    assert "app" in listing.stdout
    assert "README.md" in listing.stdout


def test_native_pipeline_and_private_temp(native_manager, tmp_path):
    runtime = SimpleNamespace(shell_job_manager=native_manager)
    result = run_bash(
        f'& "{sys.executable}" -c "print(1)" | Select-Object -First 1; '
        'Write-Output $env:TEMP; Set-Content -Path "$env:TEMP\\probe.txt" -Value ok; '
        'Get-Content "$env:TEMP\\probe.txt"',
        runtime_state=runtime,
    )
    assert result.return_code == 0, result.output
    lines = [line.strip() for line in result.output.splitlines() if line.strip()]
    assert lines[0] == "1"
    temp_path = next(line for line in lines if "veriforge-shell-" in line)
    assert str(tmp_path).lower() not in temp_path.lower()
    assert "ok" in result.output


def test_write_outside_the_workspace_is_denied(native_manager, tmp_path):
    runtime = SimpleNamespace(shell_job_manager=native_manager)
    outside = tmp_path.parent / f"veriforge-outside-{uuid.uuid4().hex}"
    outside.mkdir()
    try:
        target = outside / "escape.txt"
        result = run_bash(f"Set-Content -LiteralPath '{target}' -Value bad", runtime_state=runtime)
        assert result.return_code != 0
        assert not target.exists()
    finally:
        shutil.rmtree(outside, ignore_errors=True)


def test_scoped_extra_directory_grant_stays_sandboxed(native_manager, tmp_path):
    from harness_code_agent.runtime.permissions import PermissionPolicy

    runtime = SimpleNamespace(shell_job_manager=native_manager)
    outside = tmp_path.with_name(f"veriforge-extra-{uuid.uuid4().hex}")
    outside.mkdir()
    try:
        granted = outside / "granted"
        blocked = outside / "blocked"
        granted.mkdir()
        blocked.mkdir()
        target = granted / "probe.txt"
        forbidden = blocked / "probe.txt"
        command = (
            f"Set-Content -LiteralPath '{target}' -Value ok; "
            f"[System.IO.File]::WriteAllText('{forbidden}', 'bad')"
        )
        args = {
            "command": command,
            "sandbox_permissions": "with_additional_permissions",
            "additional_permissions": {"file_system": {"write": [str(granted)]}},
        }
        decision = PermissionPolicy("llm-auto").decide_tool_call(
            "bash", args, workspace_root=str(tmp_path)
        )
        assert decision.requires_approval
        result = run_bash(command, runtime_state=runtime, permission_decision=decision)
        assert result.return_code != 0, result.output
        assert target.read_text(encoding="utf-8").strip() == "ok"
        assert not forbidden.exists()
    finally:
        shutil.rmtree(outside)


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


def test_native_directory_grants_are_isolated_and_reused(native_manager, tmp_path):
    from harness_code_agent.workspace.shell_environment import shell_executable

    runtime = SimpleNamespace(shell_job_manager=native_manager)
    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()

    def explicit_aces():
        literal = str(tmp_path / "a").replace("'", "''")
        script = f"@((Get-Acl -LiteralPath '{literal}').Access | Where-Object {{ -not $_.IsInherited }}).Count"
        result = subprocess.run([shell_executable(), "-NoProfile", "-Command", script],
                                capture_output=True, encoding="utf-8", errors="replace", check=True)
        return int(result.stdout.strip())

    result = run_bash("Set-Content a/first.txt '中文'; Set-Content b/keep.txt 'keep'",
                      write_paths=["a", "b"], runtime_state=runtime)
    assert result.return_code == 0, result.output
    baseline = explicit_aces()
    result = run_bash("Set-Content a/first.txt '续写'", write_paths=["a", "c"], runtime_state=runtime)
    assert result.return_code == 0, result.output
    assert explicit_aces() == baseline
    for script in ("Set-Content b/forbidden.txt 'bad'", "Remove-Item b/keep.txt"):
        result = run_bash(script, write_paths=["a", "c"], runtime_state=runtime)
        assert result.return_code != 0
    assert not (tmp_path / "b" / "forbidden.txt").exists()
    assert (tmp_path / "b" / "keep.txt").read_text(encoding="utf-8").strip() == "keep"


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


def test_background_workspace_write_does_not_hold_the_whole_project_lease(native_manager, tmp_path):
    job = native_manager.start("Set-Content bg.txt 'ok'; Start-Sleep -Seconds 30")
    deadline = time.monotonic() + 10
    while not (tmp_path / "bg.txt").exists() and time.monotonic() < deadline:
        time.sleep(.02)
    assert (tmp_path / "bg.txt").exists(), job.output_tail

    # The job may write the whole workspace, but must not hold the whole-project
    # lease while it runs: another session can still take it and write.
    access = WorkspaceAccessCoordinator(tmp_path)
    with access.acquire([access.write_claim()]):
        (tmp_path / "other-session.txt").write_text("other", encoding="utf-8")
    assert (tmp_path / "other-session.txt").exists()

    # An explicit whole-workspace declaration is accepted too, without holding
    # a whole-project lease for the job's lifetime.
    explicit = native_manager.start("Set-Content explicit-bg.txt 'ok'", write_paths=["."])
    deadline = time.monotonic() + 10
    while not (tmp_path / "explicit-bg.txt").exists() and time.monotonic() < deadline:
        time.sleep(.02)
    assert (tmp_path / "explicit-bg.txt").exists(), explicit.output_tail
    with access.acquire([access.write_claim()]):
        pass
    native_manager.stop(explicit.job_id, grace_seconds=.2)
    assert explicit._finished.wait(5)

    native_manager.stop(job.job_id, grace_seconds=.2)
    assert job._finished.wait(5)
    with access.acquire([access.write_claim()]):
        pass


def test_workspace_write_reaches_git_and_harness_directories(native_manager, tmp_path):
    runtime = SimpleNamespace(shell_job_manager=native_manager)
    script = (
        "New-Item -ItemType Directory .git, .harness -Force | Out-Null; "
        "Set-Content .git/agent-note.txt 'git'; "
        "Set-Content .harness/agent-note.txt 'state'"
    )
    result = run_bash(script, runtime_state=runtime)
    assert result.return_code == 0, result.output
    assert (tmp_path / ".git" / "agent-note.txt").read_text(encoding="utf-8").strip() == "git"
    assert (tmp_path / ".harness" / "agent-note.txt").read_text(encoding="utf-8").strip() == "state"


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
