import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


class ShellJobManagerTests(unittest.TestCase):
    def test_wsl_background_job_uses_wsl_bash_without_fallback(self):
        if os.name != "nt":
            self.skipTest("Windows-only shell selection")
        from harness_code_agent import config
        from harness_code_agent.workspace.shell_jobs import ShellJobManager

        with tempfile.TemporaryDirectory() as tmp:
            manager = ShellJobManager(Path(tmp))
            with (
                patch.object(config, "WINDOWS_SHELL", "wsl"),
                patch("harness_code_agent.workspace.shell_jobs.windows_shell_path", return_value="C:/Windows/System32/wsl.exe"),
                patch("harness_code_agent.workspace.shell_jobs.subprocess.Popen") as popen,
            ):
                manager._start_windows_process("python -m http.server")

        args = popen.call_args.args[0]
        self.assertEqual(args[:4], ["C:/Windows/System32/wsl.exe", "--cd", str(Path(tmp).resolve()), "--exec"])
        self.assertEqual(args[4:6], ["bash", "-lc"])
        self.assertEqual(args[6], "python -m http.server")

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

    def test_ring_buffer_keeps_recent_tail(self):
        from harness_code_agent.workspace.shell_jobs import RingBuffer

        buffer = RingBuffer(max_chars=10)
        buffer.append("abcdef")
        buffer.append("ghijkl")

        self.assertEqual(buffer.read_tail(100), "cdefghijkl")
        self.assertEqual(buffer.read_tail(4), "ijkl")

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
            def terminate(self):
                calls.append("child-terminate")

            def kill(self):
                calls.append("child-kill")

        class FakeParent:
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
            job = ShellJob(job_id="shell-job-test", command="sleep", pid=123, process=None)
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
