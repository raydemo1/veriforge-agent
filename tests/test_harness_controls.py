"""Tests for the runtime control knobs: retries, quotas, subagent fan-out."""
from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from harness_code_agent import config
from harness_code_agent.agent import llm_channel
from harness_code_agent.agent.cancellation import CancelledError, CancellationToken
from harness_code_agent.agent.coordinator import AgentCoordinator
from harness_code_agent.agent.tool_executor import ToolExecutor
from harness_code_agent.runtime.builtins.filesystem import write_file
from harness_code_agent.runtime.builtins.registry import BUILTIN_TOOL_REGISTRY
from harness_code_agent.runtime.builtins.shell import run_bash
from harness_code_agent.runtime.execution_planner import _CONCURRENCY_LIMITS
from harness_code_agent.runtime.permissions import PermissionPolicy
from harness_code_agent.runtime.tool_context import ToolContext
from harness_code_agent.sessions.events import EventBus
from harness_code_agent.workspace.service import WorkspaceQuotaError, WorkspaceService


class LlmRetryHelperTests(unittest.TestCase):
    def test_retries_retryable_errors_then_succeeds(self):
        calls = 0

        def action():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError("429 rate limit")

        with patch.object(llm_channel.time, "sleep"):
            result = llm_channel._call_with_retry(
                action, operation="test", attempts=4
            )

        self.assertIsNone(result)
        self.assertEqual(calls, 3)

    def test_non_retryable_error_propagates_immediately(self):
        calls = 0

        def action():
            nonlocal calls
            calls += 1
            raise ValueError("bad request")

        with patch.object(llm_channel.time, "sleep") as slept:
            with self.assertRaises(ValueError):
                llm_channel._call_with_retry(action, operation="test", attempts=3)

        self.assertEqual(calls, 1)
        slept.assert_not_called()

    def test_cancelled_token_aborts_retry_loop(self):
        token = CancellationToken()
        token.cancel()

        def action():
            raise AssertionError("action must not run after cancellation")

        with self.assertRaises(llm_channel.CancelledError):
            llm_channel._call_with_retry(
                action, operation="test", attempts=3, cancellation_token=token
            )

    def test_retryable_classification_matrix(self):
        retryable = SimpleNamespace(status_code=429)
        server_error = SimpleNamespace(status_code=503)
        connection = type("APIConnectionError", (Exception,), {})("boom")
        self.assertTrue(llm_channel._is_retryable_llm_error(retryable))
        self.assertTrue(llm_channel._is_retryable_llm_error(server_error))
        self.assertTrue(llm_channel._is_retryable_llm_error(connection))
        self.assertFalse(llm_channel._is_retryable_llm_error(SimpleNamespace(status_code=400)))
        self.assertFalse(llm_channel._is_retryable_llm_error(ValueError("nope")))


class RunBashTimeoutClampTests(unittest.TestCase):
    """run_bash owns its own timeout; the clamp is observable via metadata."""

    def _run_with_timeout(self, timeout):
        calls = []

        def fake_run(command, timeout=300, artifact_dir=None):
            calls.append(timeout)
            return SimpleNamespace(
                stdout="ok",
                stderr="",
                exit_code=0,
                timed_out=False,
                output_spilled=False,
                output_bytes=2,
            )

        fake_session = SimpleNamespace(
            run=fake_run, close=lambda: None, interrupt=lambda: None
        )
        with patch(
            "harness_code_agent.workspace.shell_session.PersistentShellSession",
            return_value=fake_session,
        ):
            result = run_bash("echo ok", timeout=timeout)
        return result, calls[0]

    def test_run_bash_timeout_is_clamped_to_maximum(self):
        with patch.object(config, "TOOL_MAX_TIMEOUT_SECONDS", 1800):
            result, effective = self._run_with_timeout(999999)
        self.assertEqual(effective, 1800)
        self.assertEqual(result.metadata["requested_timeout"], 999999)
        self.assertEqual(result.metadata["effective_timeout"], 1800)

    def test_run_bash_timeout_is_clamped_to_floor(self):
        result, effective = self._run_with_timeout(0)
        self.assertEqual(effective, 1)
        self.assertEqual(result.metadata["requested_timeout"], 0)
        self.assertEqual(result.metadata["effective_timeout"], 1)

    def test_invalid_timeout_falls_back_to_shell_default(self):
        with patch.object(config, "SHELL_DEFAULT_TIMEOUT_SECONDS", 300):
            result, effective = self._run_with_timeout("soon")
        self.assertEqual(effective, 300)
        self.assertEqual(result.metadata["requested_timeout"], "soon")
        self.assertEqual(result.metadata["effective_timeout"], 300)

    def test_timeout_result_carries_requested_and_effective_timeout(self):
        def fake_run(command, timeout=300, artifact_dir=None):
            return SimpleNamespace(
                stdout="", stderr="", exit_code=130, timed_out=True, output_spilled=False
            )

        fake_session = SimpleNamespace(
            run=fake_run, close=lambda: None, interrupt=lambda: None
        )
        with patch(
            "harness_code_agent.workspace.shell_session.PersistentShellSession",
            return_value=fake_session,
        ):
            result = run_bash("slow", timeout=999999)
        self.assertEqual(result.status, "failed")
        self.assertTrue(result.metadata["timed_out"])
        self.assertEqual(result.metadata["requested_timeout"], 999999)
        self.assertEqual(result.metadata["effective_timeout"], 1800)

    def test_no_global_tool_deadline_exists(self):
        # The global per-tool deadline was removed: timeout is a backend
        # capability (run_bash, HTTP, MCP...), not a ToolExecutor concept.
        self.assertFalse(hasattr(ToolExecutor, "_deadline_seconds"))
        self.assertFalse(hasattr(config, "TOOL_DEFAULT_TIMEOUT_SECONDS"))


class WorkspaceQuotaTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp(prefix="hca-quota-test-"))
        self.workspace = WorkspaceService(root=self.temp)
        # The 100MB free-disk floor would make these tests machine-dependent;
        # quota behaviour is exercised with the floor effectively disabled.
        self._floor = patch.object(config, "MIN_FREE_DISK_BYTES", 0)
        self._floor.start()

    def tearDown(self):
        self._floor.stop()
        shutil.rmtree(self.temp, ignore_errors=True)

    def test_write_quota_blocks_after_cumulative_limit(self):
        with patch.object(config, "WORKSPACE_WRITE_QUOTA_BYTES", 20):
            self.workspace.write_text("a.txt", "x" * 15)
            with self.assertRaises(WorkspaceQuotaError):
                self.workspace.write_text("b.txt", "y" * 50)

    def test_overwriting_with_smaller_content_is_not_quota_charged(self):
        with patch.object(config, "WORKSPACE_WRITE_QUOTA_BYTES", 10):
            self.workspace.write_text("a.txt", "x" * 10)
            # Shrinking an existing file adds no net bytes; must stay allowed.
            result = self.workspace.write_text("a.txt", "x")
            self.assertEqual(result.old_content, "x" * 10)

    def test_batch_write_is_checked_as_one_unit(self):
        with patch.object(config, "WORKSPACE_WRITE_QUOTA_BYTES", 10):
            with self.assertRaises(WorkspaceQuotaError):
                self.workspace.write_text_batch({"a.txt": "x" * 8, "b.txt": "y" * 8})
            # Nothing should have been written after the rejection.
            self.assertFalse((self.temp / "a.txt").exists())
            self.assertFalse((self.temp / "b.txt").exists())

    def test_free_disk_floor_blocks_positive_writes(self):
        usage = SimpleNamespace(total=1000, used=995, free=5)
        with patch.object(config, "MIN_FREE_DISK_BYTES", 100), patch(
            "harness_code_agent.workspace.service.shutil.disk_usage", return_value=usage
        ):
            with self.assertRaises(WorkspaceQuotaError):
                self.workspace.write_text("a.txt", "x" * 10)

    def test_write_file_tool_returns_structured_quota_result(self):
        context = ToolContext(
            workspace=self.workspace,
            permission_policy=PermissionPolicy(mode="danger-full-access"),
            event_bus=EventBus(),
            tool_registry=BUILTIN_TOOL_REGISTRY,
        )
        with patch.object(config, "WORKSPACE_WRITE_QUOTA_BYTES", 5):
            result = write_file("big.txt", "z" * 500, tool_context=context)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metadata["status_source"], "resource")
        self.assertEqual(result.metadata["resource_kind"], "workspace_quota")
        self.assertFalse((self.temp / "big.txt").exists())


# --- Subagent fan-out controls ---------------------------------------------


class _ControlledConversation:
    def __init__(self):
        self.messages = [{"role": "user", "content": "task"}]
        self.started = threading.Event()
        self.release = threading.Event()

    def run_until_idle(self, cancellation_token=None):
        self.started.set()
        remove = cancellation_token.add_callback(self.release.set)
        self.release.wait(5)
        remove()
        cancellation_token.check()
        return "done"

    def queue_message(self, message):
        pass

    def has_queued_messages(self):
        return False

    def add_user_turn(self, task):
        pass

    def close(self):
        self.release.set()


class _ControlledAgent:
    def __init__(self, **kwargs):
        self.name = kwargs["name"]

    def start_conversation(self, task):
        return _ControlledConversation()


class SubagentControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp(prefix="hca-fanout-test-"))
        self.context = ToolContext(
            workspace=WorkspaceService(root=self.temp),
            permission_policy=PermissionPolicy(mode="danger-full-access"),
            event_bus=EventBus(),
            tool_registry=BUILTIN_TOOL_REGISTRY,
        )
        self.coordinator = AgentCoordinator(self.context, max_concurrent=3)
        self.context.agent_coordinator = self.coordinator

    def tearDown(self):
        self.coordinator.close()
        shutil.rmtree(self.temp, ignore_errors=True)

    def test_concurrency_limits_follow_config(self):
        self.assertEqual(
            _CONCURRENCY_LIMITS["subagent"], max(1, int(config.MAX_CONCURRENT_AGENTS))
        )

    def test_role_registries_cannot_spawn_agents(self):
        # Structural depth cap: neither read-only roles nor worker may spawn.
        for role in ("explorer", "reviewer", "verifier", "test_designer", "worker"):
            names = {spec.name for spec in self.coordinator._role_registry(role).specs()}
            self.assertNotIn("spawn_agent", names, msg=f"role {role} can spawn agents")

    def test_open_agent_limit_is_enforced(self):
        from harness_code_agent.agent.coordinator import SubagentCapacityError

        with patch("harness_code_agent.agent.conversation.Agent", _ControlledAgent), patch(
            "harness_code_agent.agent.coordinator.MAX_OPEN_AGENTS", 1
        ):
            self.coordinator.spawn(name="only", role="explorer", task="one")
            with self.assertRaises(SubagentCapacityError):
                self.coordinator.spawn(name="second", role="explorer", task="two")

    def test_spawn_is_rejected_when_parent_turn_is_cancelled(self):
        token = CancellationToken()
        token.cancel()
        with self.assertRaisesRegex(ValueError, "cancelled"):
            self.coordinator.spawn(
                name="late", role="explorer", task="x", parent_cancellation_token=token
            )

    def test_parent_cancellation_interrupts_running_subagent(self):
        with patch("harness_code_agent.agent.conversation.Agent", _ControlledAgent):
            parent_token = CancellationToken()
            spawned = self.coordinator.spawn(
                name="child",
                role="explorer",
                task="inspect",
                parent_cancellation_token=parent_token,
            )
            deadline = threading.Event()
            deadline.wait(0.5)
            self.assertEqual(
                self.coordinator.list()["agents"][0]["status"], "running"
            )

            parent_token.cancel()
            terminal = self.coordinator.wait([spawned["agent_id"]], timeout_seconds=5)
            self.assertFalse(terminal["timed_out"])
            self.assertEqual(terminal["agents"][0]["status"], "interrupted")


# --- Cancellation token tree -----------------------------------------------


class CancellationTokenTreeTests(unittest.TestCase):
    def test_parent_cancel_propagates_to_child_and_grandchild(self):
        root = CancellationToken()
        child = root.create_child()
        grandchild = CancellationToken(parent=child)
        root.cancel()
        self.assertTrue(child.is_cancelled)
        self.assertTrue(grandchild.is_cancelled)

    def test_child_cancel_does_not_propagate_upward(self):
        root = CancellationToken()
        child = root.create_child()
        child.cancel()
        self.assertTrue(child.is_cancelled)
        self.assertFalse(root.is_cancelled)

    def test_child_is_born_cancelled(self):
        root = CancellationToken()
        root.cancel()
        child = CancellationToken(parent=root)
        self.assertTrue(child.is_cancelled)

    def test_close_detaches_child_from_parent(self):
        root = CancellationToken()
        child = root.create_child()
        self.assertEqual(len(root._callbacks), 1)
        child.close()
        self.assertEqual(len(root._callbacks), 0)
        root.cancel()
        self.assertFalse(child.is_cancelled)

    def test_close_is_idempotent_and_safe_for_root_token(self):
        root = CancellationToken()
        root.close()
        root.close()
        child = root.create_child()
        child.close()
        child.close()
        root.cancel()
        self.assertFalse(child.is_cancelled)

    def test_new_children_after_close_are_independent_of_old_one(self):
        root = CancellationToken()
        old = root.create_child()
        old.close()
        new = root.create_child()
        root.cancel()
        self.assertFalse(old.is_cancelled)
        self.assertTrue(new.is_cancelled)

    def test_wait_returns_true_on_cancel_and_false_on_timeout(self):
        token = CancellationToken()
        self.assertFalse(token.wait(0.02))
        token.cancel()
        self.assertTrue(token.wait(1.0))

    def test_wait_unblocks_immediately_when_cancelled_from_another_thread(self):
        import threading
        import time as _time

        token = CancellationToken()

        def cancel_later():
            _time.sleep(0.1)
            token.cancel()

        threading.Thread(target=cancel_later, daemon=True).start()
        start = _time.monotonic()
        self.assertTrue(token.wait(30.0))
        self.assertLess(_time.monotonic() - start, 1.0)


# --- Retry-After handling --------------------------------------------------


class RetryAfterTests(unittest.TestCase):
    def _exc(self, header=None):
        headers = {"retry-after": header} if header is not None else None
        return SimpleNamespace(status_code=429, response=SimpleNamespace(headers=headers))

    def test_delta_seconds_header(self):
        with patch.object(llm_channel.random, "uniform", return_value=0.0):
            self.assertEqual(llm_channel._retry_delay(0, self._exc("3")), 3.0)

    def test_missing_header_uses_exponential_backoff(self):
        no_header = SimpleNamespace(
            status_code=500, response=SimpleNamespace(headers=None)
        )
        with patch.object(llm_channel.random, "uniform", return_value=0.0):
            self.assertEqual(llm_channel._retry_delay(0, no_header), 2.0)

    def test_header_is_capped(self):
        self.assertEqual(
            llm_channel._retry_after_seconds(self._exc("9999")),
            llm_channel._RETRY_AFTER_CAP_SECONDS,
        )

    def test_malformed_header_falls_back_to_backoff(self):
        self.assertIsNone(llm_channel._retry_after_seconds(self._exc("soon")))
        self.assertIsNone(
            llm_channel._retry_after_seconds(SimpleNamespace(status_code=503))
        )

    def test_retry_loop_honours_retry_after(self):
        class RateLimit(Exception):
            def __init__(self):
                super().__init__("rate limited")
                self.status_code = 429
                self.response = SimpleNamespace(headers={"retry-after": "0"})

        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RateLimit()
            return "ok"

        with patch.object(llm_channel.time, "sleep") as slept:
            self.assertEqual(
                llm_channel._call_with_retry(flaky, operation="t", attempts=3), "ok"
            )
        self.assertEqual(calls["n"], 2)
        slept.assert_called_once()

    def test_cancellation_during_backoff_raises_immediately(self):
        import threading
        import time as _time

        class RateLimited(Exception):
            def __init__(self):
                super().__init__("429")
                self.status_code = 429
                self.response = SimpleNamespace(headers={"retry-after": "30"})

        token = CancellationToken()
        calls = {"n": 0}

        def always_429():
            calls["n"] += 1
            raise RateLimited()

        threading.Thread(
            target=lambda: (_time.sleep(0.15), token.cancel()), daemon=True
        ).start()
        start = _time.monotonic()
        with self.assertRaises(CancelledError):
            llm_channel._call_with_retry(
                always_429,
                operation="t",
                attempts=5,
                cancellation_token=token,
            )
        # Must surface well inside the 30s Retry-After window.
        self.assertLess(_time.monotonic() - start, 2.0)
        self.assertEqual(calls["n"], 1)


if __name__ == "__main__":
    unittest.main()
