import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from harness_code_agent.agent.cancellation import CancellationToken, CancelledError
from harness_code_agent.runtime.builtins.browser import browser_test


class _FakeJobManager:
    def __init__(self):
        self.started = threading.Event()
        self.stopped = []
        self.job = SimpleNamespace(
            job_id="shell-job-browser",
            pid=123,
            status="running",
            error=None,
        )

    def start(self, command, **kwargs):
        self.command = command
        self.started.set()
        return self.job

    def get(self, job_id):
        return self.job

    def stop(self, job_id):
        self.stopped.append(job_id)
        self.job.status = "stopped"
        return self.job

    def read_output(self, job_id, max_chars):
        return ""


class BrowserRuntimeTests(unittest.TestCase):
    def test_cancel_during_dev_server_start_stops_the_owned_job(self):
        manager = _FakeJobManager()
        runtime_state = SimpleNamespace(shell_job_manager=manager, browser_job_id=None)
        token = CancellationToken()
        errors = []

        def invoke():
            try:
                browser_test(
                    "http://127.0.0.1:5173",
                    screenshot=False,
                    start_command="npm run dev",
                    startup_wait=5,
                    runtime_state=runtime_state,
                    cancellation_token=token,
                )
            except Exception as exc:  # noqa: BLE001 - thread records the result for assertion
                errors.append(exc)

        with patch("harness_code_agent.runtime.builtins.browser.HAS_PLAYWRIGHT", True):
            worker = threading.Thread(target=invoke)
            worker.start()
            self.assertTrue(manager.started.wait(1))
            token.cancel()
            worker.join(1)

        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], CancelledError)
        self.assertEqual(manager.stopped, ["shell-job-browser"])


if __name__ == "__main__":
    unittest.main()
