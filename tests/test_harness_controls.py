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
from harness_code_agent.agent.cancellation import CancellationToken
from harness_code_agent.agent.coordinator import AgentCoordinator
from harness_code_agent.agent.tool_executor import ToolExecutor
from harness_code_agent.runtime import tools
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


class ToolDeadlineTests(unittest.TestCase):
    def _deadline(self, name: str, args: dict | None = None):
        prepared = SimpleNamespace(name=name, args=args or {})
        return ToolExecutor._deadline_seconds(None, prepared)

    def test_run_bash_timeout_is_clamped_to_maximum(self):
        with patch.object(config, "TOOL_MAX_TIMEOUT_SECONDS", 1800.0):
            self.assertEqual(self._deadline("run_bash", {"timeout": 999999}), 1800.0)

    def test_run_bash_timeout_is_clamped_to_floor(self):
        self.assertEqual(self._deadline("run_bash", {"timeout": 0}), 1.0)

    def test_invalid_timeout_falls_back_to_default(self):
        with patch.object(config, "TOOL_DEFAULT_TIMEOUT_SECONDS", 300.0):
            self.assertEqual(self._deadline("run_bash", {"timeout": "soon"}), 300.0)

    def test_other_tools_get_default_deadline(self):
        with patch.object(config, "TOOL_DEFAULT_TIMEOUT_SECONDS", 42.0):
            self.assertEqual(self._deadline("read_file"), 42.0)


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
            tool_registry=tools.BUILTIN_TOOL_REGISTRY,
        )
        with patch.object(config, "WORKSPACE_WRITE_QUOTA_BYTES", 5):
            result = tools.write_file("big.txt", "z" * 500, tool_context=context)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metadata["status_source"], "quota")
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
            tool_registry=tools.BUILTIN_TOOL_REGISTRY,
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
        with patch("harness_code_agent.agent.conversation.Agent", _ControlledAgent), patch(
            "harness_code_agent.agent.coordinator.MAX_OPEN_AGENTS", 1
        ):
            self.coordinator.spawn(name="only", role="explorer", task="one")
            with self.assertRaisesRegex(ValueError, "at most 1"):
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


if __name__ == "__main__":
    unittest.main()
