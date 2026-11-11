import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import psutil
import pytest


class ShellJobManagerTests(unittest.TestCase):

    def test_natural_exit_records_status_and_exit_code(self):
        from harness_code_agent.workspace.shell_jobs import ShellJobManager

        command = _python_command("import sys; print('done', flush=True); sys.exit(7)")

        with tempfile.TemporaryDirectory() as tmp:
            manager = ShellJobManager(Path(tmp))
            try:
                job = manager.start(command)
                self._wait_for_status(manager, job.job_id, "exited")
                summary = manager.get(job.job_id)

                self.assertEqual(summary.status, "exited")
                self.assertEqual(summary.exit_code, 7)
                self.assertIn("done", manager.read_output(job.job_id, max_chars=1000))
            finally:
                manager.close()

    def test_output_buffer_keeps_all_content_and_supports_offsets(self):
        from harness_code_agent.workspace.shell_jobs import ShellOutputBuffer

        with tempfile.TemporaryDirectory() as tmp:
            buffer = ShellOutputBuffer(Path(tmp))
            buffer.append("abcdef")
            buffer.append("ghijkl")
            self.assertEqual(buffer.read(0, 100), "abcdefghijkl")
            self.assertEqual(buffer.read(4, 4), "efgh")
            self.assertEqual(buffer.read_tail(4), "ijkl")
            self.assertIsNone(buffer.path)

    def test_invalid_job_id_is_reported(self):
        from harness_code_agent.workspace.shell_jobs import (
            ShellJobManager,
            ShellJobNotFound,
        )

        with tempfile.TemporaryDirectory() as tmp:
            manager = ShellJobManager(Path(tmp))
            try:
                with self.assertRaises(ShellJobNotFound):
                    manager.read_output("missing")
                with self.assertRaises(ShellJobNotFound):
                    manager.stop("missing")
            finally:
                manager.close()

    def test_stop_uses_psutil_process_tree(self):
        from harness_code_agent.workspace.shell_jobs import ShellJob, ShellJobManager

        calls = []

        class FakeChild:
            pid = 124

            def create_time(self):
                return 1.0

            def terminate(self):
                calls.append("child-terminate")

            def kill(self):
                calls.append("child-kill")

        class FakeParent:
            pid = 123

            def create_time(self):
                return 1.0

            def children(self, recursive=False):
                calls.append(("children", recursive))
                return [FakeChild()]

            def terminate(self):
                calls.append("parent-terminate")

            def kill(self):
                calls.append("parent-kill")

        class FakePsutil:
            NoSuchProcess = RuntimeError
            TimeoutExpired = TimeoutError

            @staticmethod
            def Process(pid):
                calls.append(("process", pid))
                return FakeParent()

            @staticmethod
            def wait_procs(processes, timeout):
                calls.append(("wait", len(processes), timeout))
                return ([], processes)

        with tempfile.TemporaryDirectory() as tmp:
            manager = ShellJobManager(Path(tmp))
            job = ShellJob(job_id="shell-job-test", command="sleep", pid=123, process=SimpleNamespace(poll=lambda: None))
            with patch("harness_code_agent.workspace.shell_jobs.psutil", FakePsutil):
                manager._terminate_process_tree(job, grace_seconds=0.01)

        self.assertIn(("process", 123), calls)
        self.assertIn(("children", True), calls)
        self.assertIn("child-terminate", calls)
        self.assertIn("parent-terminate", calls)
        self.assertIn("child-kill", calls)
        self.assertIn("parent-kill", calls)

    def test_monitor_waits_for_reader_threads_before_marking_exited(self):
        import threading

        from harness_code_agent.workspace.shell_jobs import ShellJob, ShellJobManager

        class FakeProcess:
            def poll(self):
                return 0

            def wait(self):
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            manager = ShellJobManager(Path(tmp))
            job = ShellJob(job_id="shell-job-test", command="exit", pid=123, process=FakeProcess())

            def late_reader():
                time.sleep(0.05)
                job.output.append("late output\n")

            reader = threading.Thread(target=late_reader)
            job._reader_threads = [reader]
            reader.start()

            manager._monitor(job)

        self.assertEqual(job.status, "exited")
        self.assertIn("late output", job.output_tail)

    def _wait_for_status(self, manager, job_id, status):
        # A job is launched through Bun + PowerShell on Windows, so startup
        # alone costs a few seconds on a loaded machine.  The failure message
        # carries the observed state so a slow machine is distinguishable from
        # a job that never reached a terminal state.
        deadline = time.time() + 20
        while time.time() < deadline:
            job = manager.get(job_id)
            if job.status == status:
                return job
            time.sleep(0.05)
        job = manager.get(job_id)
        self.fail(
            f"Timed out waiting for status {status!r}; "
            f"observed status={job.status!r} pid={job.pid} exit_code={job.exit_code} "
            f"launcher_exited={job.launcher_exited} drained={job.drained} "
            f"warnings={job.warnings} last_output={job.output_tail[-500:]!r}"
        )


def _python_command(script: str) -> str:
    if os.name == "nt":
        return f'& "{sys.executable}" -c "{script}"'
    return f'"{sys.executable}" -c "{script}"'


if __name__ == "__main__":
    unittest.main()


def test_stop_keeps_stopping_until_exit_is_confirmed(tmp_path):
    import threading

    from harness_code_agent.workspace.shell_jobs import ShellJob, ShellJobManager
    released = threading.Event()
    process = SimpleNamespace(poll=lambda: -9 if released.is_set() else None, wait=lambda: -9)
    manager = ShellJobManager(tmp_path)
    job = ShellJob("pending-stop", "controlled", None, process)
    manager._store(job)
    monitor = threading.Thread(target=manager._monitor, args=(job,), daemon=True)
    monitor.start()
    try:
        with patch.object(job._finished, "wait", return_value=False):
            assert manager.stop(job.job_id, grace_seconds=0).status == "stopping"
        assert job.ended_at is None
        assert manager.running_jobs() == [job]
    finally:
        released.set()
        monitor.join(2)
        manager.close()
    assert job.status == "stopped"
    assert job._finished.is_set()


# ---------------------------------------------------------------------------
# Lifecycle: the launched command exiting, the process tree draining and the
# write lease being released are three separate facts.
# ---------------------------------------------------------------------------


class _ExitedLauncher:
    """A launched process that has already exited."""

    def __init__(self, exit_code: int = 0):
        self.pid = 4242
        self._exit_code = exit_code

    def poll(self):
        return self._exit_code

    def wait(self):
        return self._exit_code


class _ManagedProcess:
    """A tracked descendant; ``stubborn`` survivors ignore termination."""

    def __init__(self, pid: int = 555, *, stubborn: bool = False):
        self.pid = pid
        self.stubborn = stubborn
        self.alive = True
        self.terminated = False

    def create_time(self):
        return 1.0

    def children(self, recursive=False):
        return []

    def is_running(self):
        return self.alive

    def status(self):
        return "running" if self.alive else "zombie"

    def terminate(self):
        self.terminated = True
        if not self.stubborn:
            self.alive = False

    def kill(self):
        self.terminate()


class _Lease:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


class _FlakyLease:
    """Fails the first release attempt, like a lease held by a stuck controller."""

    def __init__(self):
        self.released = False
        self.fail = True

    def release(self):
        if self.fail:
            raise RuntimeError("lease busy")
        self.released = True


class _FlakyTempDirectory:
    def __init__(self):
        self.cleaned = False
        self.fail = True

    def cleanup(self):
        if self.fail:
            raise OSError("directory still in use")
        self.cleaned = True


class _UnreadableProcess:
    """Reports as gone, but its children cannot be read (AccessDenied)."""

    pid = 999

    def create_time(self):
        return 1.0

    def children(self, recursive=False):
        raise psutil.AccessDenied(pid=self.pid)

    def is_running(self):
        return False

    def status(self):
        return "zombie"

    def terminate(self):
        pass

    def kill(self):
        pass


class _LiveLauncherProcess(_ManagedProcess):
    """A launcher that is still running; ``readable`` toggles AccessDenied."""

    def __init__(self, pid=4242, *, readable=True):
        super().__init__(pid=pid)
        self.readable = readable

    def poll(self):
        return None if self.alive else 0

    def wait(self):
        return 0

    def children(self, recursive=False):
        if not self.readable:
            raise psutil.AccessDenied(pid=self.pid)
        return []


def _job_with_survivor(tmp_path, *, stubborn: bool, background: bool = False):
    from harness_code_agent.workspace.shell_jobs import ShellJob, ShellJobManager

    manager = ShellJobManager(tmp_path)
    launcher = _ExitedLauncher()
    job = ShellJob("survivor", "cmd", launcher.pid, launcher, background=background)
    survivor = _ManagedProcess(stubborn=stubborn)
    job._processes[(survivor.pid, 1.0)] = survivor
    lease = _Lease()
    job._write_lease = lease
    manager._store(job)
    return manager, job, survivor, lease


def test_foreground_survivor_is_terminated_after_the_grace_period(tmp_path, monkeypatch):
    from harness_code_agent import config

    monkeypatch.setattr(config, "SHELL_JOB_EXIT_DRAIN_SECONDS", 0.05)
    manager, job, survivor, lease = _job_with_survivor(tmp_path, stubborn=False)
    try:
        manager._monitor(job)
    finally:
        manager.close()

    assert job.status == "exited"
    assert survivor.terminated
    assert any("terminated 1 process" in warning for warning in job.warnings)
    assert job.resources_released and lease.released


def test_surviving_process_keeps_the_write_lease(tmp_path, monkeypatch):
    from harness_code_agent import config

    monkeypatch.setattr(config, "SHELL_JOB_EXIT_DRAIN_SECONDS", 0.05)
    monkeypatch.setattr(config, "SHELL_JOB_TERMINATION_CONFIRM_SECONDS", 0.05)
    manager, job, survivor, lease = _job_with_survivor(tmp_path, stubborn=True)
    try:
        manager._monitor(job)

        assert job.status == "exited"
        assert job.drained is False
        assert job.resources_released is False
        assert lease.released is False
        assert any("write lease retained" in warning for warning in job.warnings)
    finally:
        survivor.alive = False
    assert job.release_resources() is True
    manager.close()


def test_monitor_failure_reaches_a_terminal_state(tmp_path):
    from harness_code_agent.workspace.shell_jobs import ShellJob, ShellJobManager

    manager = ShellJobManager(tmp_path)
    job = ShellJob("boom", "cmd", None, _ExitedLauncher())
    lease = _Lease()
    job._write_lease = lease
    manager._store(job)

    def explode(_job):
        raise RuntimeError("psutil exploded")

    with patch.object(manager, "_wait_for_exit", explode):
        manager._monitor(job)
    manager.close()

    assert job.status == "failed"
    assert job._finished.is_set()
    assert "psutil exploded" in (job.error or "")
    assert lease.released


def test_reader_that_never_returns_does_not_block_the_job(tmp_path, monkeypatch):
    from harness_code_agent import config
    from harness_code_agent.workspace.shell_jobs import ShellJob, ShellJobManager

    monkeypatch.setattr(config, "SHELL_JOB_READER_JOIN_SECONDS", 0.05)
    manager = ShellJobManager(tmp_path)
    job = ShellJob("stuck-reader", "cmd", None, _ExitedLauncher())
    manager._store(job)
    release = threading.Event()
    reader = threading.Thread(target=release.wait, name="stuck-reader")
    job._reader_threads = [reader]
    reader.start()
    try:
        manager._monitor(job)
        assert job.status == "exited"
        # The process tree is gone, but the job is not drained: a stuck reader
        # must not be reported as a complete drain.
        assert job.processes_drained is True
        assert job.readers_drained is False
        assert job.drained is False
        assert any("output readers did not finish" in warning for warning in job.warnings)
    finally:
        release.set()
        reader.join(2)
    manager.close()


def test_stop_retains_resources_when_a_process_survives(tmp_path, monkeypatch):
    from harness_code_agent import config

    monkeypatch.setattr(config, "SHELL_JOB_EXIT_DRAIN_SECONDS", 0.05)
    monkeypatch.setattr(config, "SHELL_JOB_TERMINATION_CONFIRM_SECONDS", 0.05)
    manager, job, survivor, lease = _job_with_survivor(tmp_path, stubborn=True)

    manager._monitor(job)

    assert job.status == "exited"
    assert survivor.alive is True
    assert job.resources_released is False
    assert lease.released is False
    with pytest.raises(RuntimeError):
        manager.close()


def test_stop_cleans_up_a_terminal_job_that_still_holds_the_lease(tmp_path, monkeypatch):
    """A terminal job with survivors must stay stoppable through the tool API."""
    from harness_code_agent import config

    monkeypatch.setattr(config, "SHELL_JOB_EXIT_DRAIN_SECONDS", 0.05)
    monkeypatch.setattr(config, "SHELL_JOB_TERMINATION_CONFIRM_SECONDS", 0.05)
    manager, job, survivor, lease = _job_with_survivor(tmp_path, stubborn=True)
    manager._monitor(job)
    assert job.status == "exited"
    assert manager.retained_jobs() == [job]

    survivor.stubborn = False
    manager.stop(job.job_id)

    assert survivor.alive is False
    assert job.resources_released is True
    assert lease.released is True
    assert manager.retained_jobs() == []
    manager.close()


def test_terminal_job_with_retained_resources_cannot_be_exported(tmp_path, monkeypatch):
    from harness_code_agent import config

    monkeypatch.setattr(config, "SHELL_JOB_EXIT_DRAIN_SECONDS", 0.05)
    monkeypatch.setattr(config, "SHELL_JOB_TERMINATION_CONFIRM_SECONDS", 0.05)
    manager, job, survivor, _lease = _job_with_survivor(tmp_path, stubborn=True)
    try:
        manager._monitor(job)
        with pytest.raises(RuntimeError):
            manager.export_jobs()
    finally:
        survivor.alive = False


def test_failed_lease_release_keeps_the_handle_for_a_retry(tmp_path):
    from harness_code_agent.workspace.shell_jobs import ShellJob, ShellJobManager

    manager = ShellJobManager(tmp_path)
    job = ShellJob("retry-lease", "cmd", None, _ExitedLauncher())
    lease = _FlakyLease()
    job._write_lease = lease
    manager._store(job)

    job.mark("exited")

    assert job.resources_released is False
    assert job._write_lease is lease
    assert lease.released is False
    assert any("write lease release failed" in warning for warning in job.warnings)

    lease.fail = False
    assert job.release_resources() is True
    assert lease.released is True
    assert job._write_lease is None
    manager.close()


def test_failed_temp_cleanup_keeps_the_handle_for_a_retry(tmp_path):
    from harness_code_agent.workspace.shell_jobs import ShellJob, ShellJobManager

    manager = ShellJobManager(tmp_path)
    job = ShellJob("retry-temp", "cmd", None, _ExitedLauncher())
    temporary = _FlakyTempDirectory()
    job._temporary_directory = temporary
    manager._store(job)

    job.mark("exited")

    assert job.resources_released is False
    assert job._temporary_directory is temporary
    assert any("private temp directory retained" in warning for warning in job.warnings)

    temporary.fail = False
    assert job.release_resources() is True
    assert temporary.cleaned is True
    assert job._temporary_directory is None
    manager.close()


def test_unreadable_descendants_keep_the_lease_after_the_launcher_exits(tmp_path):
    """An empty tracked set after the launcher exits is not proof of an empty tree."""
    from harness_code_agent.workspace.shell_jobs import ShellJob, ShellJobManager

    manager = ShellJobManager(tmp_path)
    job = ShellJob("unreadable", "cmd", 4242, _ExitedLauncher())
    unreadable = _UnreadableProcess()
    job._processes[(unreadable.pid, 1.0)] = unreadable
    lease = _Lease()
    job._write_lease = lease
    manager._store(job)

    manager._track_process_tree(job)
    assert job.tracking_failed is True

    # The dead descendant is pruned, so the next pass has nothing left to read.
    # That pass must not be mistaken for proof that every descendant is gone.
    manager._prune_dead(job)
    manager._track_process_tree(job)
    assert job._processes == {}
    assert job.tracking_failed is True
    assert job.release_resources() is False
    assert lease.released is False
    assert any("could not be confirmed" in warning for warning in job.warnings)


def test_live_launcher_pass_clears_a_tracking_failure(tmp_path):
    """Re-reading the whole tree from a live launcher is real evidence."""
    from harness_code_agent.workspace.shell_jobs import ShellJob, ShellJobManager

    manager = ShellJobManager(tmp_path)
    launcher = _LiveLauncherProcess(readable=False)
    job = ShellJob("live", "cmd", launcher.pid, launcher)
    lease = _Lease()
    job._write_lease = lease
    manager._store(job)

    with patch("harness_code_agent.workspace.shell_jobs.psutil.Process", return_value=launcher):
        manager._track_process_tree(job)
        assert job.tracking_failed is True

        launcher.readable = True
        manager._track_process_tree(job)
        assert job.tracking_failed is False

    launcher.alive = False
    assert job.release_resources() is True
    assert lease.released is True
    job.mark("exited")
    manager.close()


def test_terminal_job_with_retained_resources_blocks_suspension(tmp_path, monkeypatch):
    """Auto-suspension must wait for a lease a finished job still holds."""
    from harness_code_agent import config
    from harness_code_agent.core.interactive import InteractiveSession

    monkeypatch.setattr(config, "SHELL_JOB_EXIT_DRAIN_SECONDS", 0.05)
    monkeypatch.setattr(config, "SHELL_JOB_TERMINATION_CONFIRM_SECONDS", 0.05)
    manager, job, survivor, _lease = _job_with_survivor(tmp_path, stubborn=True)
    try:
        manager._monitor(job)
        assert manager.running_jobs() == []
        assert manager.retained_jobs() == [job]

        session = SimpleNamespace(
            tool_context=SimpleNamespace(
                agent_coordinator=SimpleNamespace(
                    changes=SimpleNamespace(has_pending_changes=lambda: False),
                    has_active_agents=lambda: False,
                    _condition=threading.Condition(),
                    _records={},
                ),
                tool_tasks=SimpleNamespace(pending_count=0),
            ),
            _turn_thread=None,
            conversation=SimpleNamespace(
                runtime_state=SimpleNamespace(shell_job_manager=manager),
            ),
        )
        assert InteractiveSession._suspend_blocker(session) is not None
    finally:
        survivor.alive = False
