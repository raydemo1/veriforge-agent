import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


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

    def _wait_for_output(self, manager, job_id, needle):
        deadline = time.time() + 5
        while time.time() < deadline:
            output = manager.read_output(job_id, max_chars=12000)
            if needle in output:
                return output
            time.sleep(0.05)
        self.fail(f"Timed out waiting for output containing {needle!r}")

    def _wait_for_status(self, manager, job_id, status):
        deadline = time.time() + 5
        while time.time() < deadline:
            job = manager.get(job_id)
            if job.status == status:
                return job
            time.sleep(0.05)
        self.fail(f"Timed out waiting for status {status!r}")


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
