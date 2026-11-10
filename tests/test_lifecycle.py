from __future__ import annotations

import unittest

from harness_code_agent.runtime.lifecycle import LifecycleScope


class LifecycleScopeTests(unittest.TestCase):
    def test_closes_higher_order_and_newer_resources_first(self):
        closed: list[str] = []
        lifecycle = LifecycleScope()
        lifecycle.register("mcp-old", lambda: closed.append("mcp-old"), order=10)
        lifecycle.register("mcp-new", lambda: closed.append("mcp-new"), order=10)
        lifecycle.register("conversation", lambda: closed.append("conversation"), order=40)

        self.assertEqual(lifecycle.close(), [])
        self.assertEqual(closed, ["conversation", "mcp-new", "mcp-old"])

    def test_cleanup_continues_after_one_resource_fails(self):
        closed: list[str] = []
        lifecycle = LifecycleScope()
        lifecycle.register("last", lambda: closed.append("last"), order=10)

        def fail() -> None:
            raise RuntimeError("broken close")

        lifecycle.register("broken", fail, order=20)
        lifecycle.register("first", lambda: closed.append("first"), order=30)

        errors = lifecycle.close()

        self.assertEqual(closed, ["first", "last"])
        self.assertEqual([(error.name, error.error) for error in errors], [
            ("broken", "RuntimeError: broken close"),
        ])

    def test_close_is_idempotent_and_rejects_late_registration(self):
        closed: list[str] = []
        lifecycle = LifecycleScope()
        lifecycle.register("resource", lambda: closed.append("resource"), order=10)

        lifecycle.close()
        self.assertEqual(lifecycle.close(), [])
        self.assertEqual(closed, ["resource"])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            lifecycle.register("late", lambda: None, order=10)


class ExecutionBoundaryTests(unittest.TestCase):
    def test_cancelled_resource_wait_never_enters_handler(self):
        import threading

        from harness_code_agent.support.cancellation import (
            CancellationToken,
            CancelledError,
        )

        from harness_code_agent.runtime.execution_planner import (
            ResourceClaim,
            ResourceCoordinator,
        )
        coordinator = ResourceCoordinator()
        claim = ResourceClaim("workspace", "file", access="write")
        token = CancellationToken()
        waiting, exited = threading.Event(), threading.Event()
        entered = []
        def worker():
            waiting.set()
            try:
                with coordinator.acquire([claim], token):
                    entered.append(True)
            except CancelledError:
                exited.set()
        with coordinator.acquire([claim]):
            thread = threading.Thread(target=worker)
            thread.start()
            self.assertTrue(waiting.wait(1))
            token.cancel()
            self.assertTrue(exited.wait(1))
        thread.join(1)
        self.assertFalse(entered)

    def test_unknown_effect_conflicts_with_workspace_write(self):
        from harness_code_agent.runtime.execution_planner import (
            CallEffect,
            ResourceClaim,
            effects_conflict,
        )
        self.assertTrue(effects_conflict(CallEffect.global_exclusive(),
                        CallEffect((ResourceClaim("workspace", "file", access="write"),))))

    def test_supervisor_keeps_running_future_until_real_completion(self):
        from concurrent.futures import Future

        from harness_code_agent.support.cancellation import CancellationToken

        from harness_code_agent.runtime.task_supervisor import ToolTaskSupervisor
        supervisor, future, token = ToolTaskSupervisor(), Future(), CancellationToken()
        future.set_running_or_notify_cancel()
        supervisor.track(future, turn_id="old", cancellation_token=token)
        supervisor.cancel(turn_id="other")
        self.assertFalse(token.is_cancelled)
        self.assertFalse(supervisor.close(timeout=0))
        self.assertTrue(token.is_cancelled)
        self.assertEqual(supervisor.pending_count, 1)
        future.set_result("late")
        self.assertEqual(supervisor.pending_count, 0)

    def test_approval_wait_cancels_only_own_execution(self):
        import tempfile
        import threading
        from pathlib import Path

        from harness_code_agent.support.cancellation import (
            CancellationToken,
            CancelledError,
            cancellation_scope,
        )

        from harness_code_agent.tui_bridge import BridgeInteractionProvider
        opened, exited = threading.Event(), threading.Event()
        token = CancellationToken()
        with tempfile.TemporaryDirectory() as directory:
            provider = BridgeInteractionProvider(lambda event: opened.set(), project_root=Path(directory))
            def worker():
                try:
                    with cancellation_scope(token):
                        provider._open("approval", {})
                except CancelledError:
                    exited.set()
            thread = threading.Thread(target=worker)
            thread.start()
            self.assertTrue(opened.wait(1))
            token.cancel()
            self.assertTrue(exited.wait(1))
            thread.join(1)
            self.assertEqual(provider._pending, {})

    def test_cancellation_runs_remaining_callbacks_after_failure(self):
        from harness_code_agent.support.cancellation import CancellationToken
        token = CancellationToken()
        calls = []
        token.add_callback(lambda: 1 / 0)
        token.add_callback(lambda: calls.append(True))
        token.cancel()
        self.assertEqual(calls, [True])


    def test_cancelled_startup_wait_retains_session_future(self):
        from concurrent.futures import Future

        from harness_code_agent.support.cancellation import (
            CancellationToken,
            CancelledError,
            cancellation_scope,
        )

        from harness_code_agent.core.interactive import InteractiveSession
        future, token = Future(), CancellationToken()
        future.set_running_or_notify_cancel()
        token.cancel()
        session = InteractiveSession.__new__(InteractiveSession)
        with cancellation_scope(token), self.assertRaises(CancelledError):
            session._wait_session_future(future)
        self.assertFalse(future.cancelled())
        future.set_result("initialized")
        self.assertEqual(future.result(), "initialized")

    def test_session_close_retains_resources_while_foreground_tool_is_alive(self):
        import threading
        from types import SimpleNamespace
        from unittest.mock import Mock

        from harness_code_agent.core.interactive import InteractiveSession
        session = InteractiveSession.__new__(InteractiveSession)
        session._execution_condition = threading.Condition(threading.RLock())
        session._close_lock = threading.Lock()
        session._closed = session._closing = False
        session._turn_thread = session._turn_token = None
        tasks = Mock()
        tasks.wait_idle.return_value = False
        session.tool_context = SimpleNamespace(tool_tasks=tasks, agent_coordinator=Mock())
        session.lifecycle = Mock()
        with self.assertRaisesRegex(RuntimeError, "资源未释放"):
            session.close()
        session.lifecycle.close.assert_not_called()
        self.assertTrue(session._closing)
        self.assertFalse(session._closed)

    def test_runtime_mutation_is_rejected_for_lingering_foreground_work(self):
        import threading
        from types import SimpleNamespace

        from harness_code_agent.core.interactive import InteractiveSession
        session = InteractiveSession.__new__(InteractiveSession)
        session._execution_condition = threading.Condition(threading.RLock())
        session._closed = session._closing = False
        session._turn_thread = None
        session.tool_context = SimpleNamespace(tool_tasks=SimpleNamespace(pending_count=1))
        with self.assertRaisesRegex(RuntimeError, "前台工具尚未退出"):
            session.set_permission_mode("danger-full-access")


if __name__ == "__main__":
    unittest.main()
