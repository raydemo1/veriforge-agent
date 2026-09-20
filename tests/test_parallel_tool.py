import asyncio
import os
import shutil
import sys
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


def _install_fake_openai_module() -> None:
    openai = types.ModuleType("openai")

    class OpenAI:
        def __init__(self, *args, **kwargs):
            pass

    openai.OpenAI = OpenAI
    sys.modules["openai"] = openai


_install_fake_openai_module()

from harness_code_agent.agent.cancellation import CancellationToken
from harness_code_agent.runtime.builtins.registry import (
    BUILTIN_TOOL_REGISTRY,
    TOOL_SCHEMAS,
)
from harness_code_agent.runtime.execution_planner import (
    CallEffect,
    ExecutionPlanner,
    ResourceClaim,
    ResourceCoordinator,
)
from harness_code_agent.runtime.mcp import McpClientManager, McpToolBinding
from harness_code_agent.runtime.tool_registry import ToolRegistry
from harness_code_agent.runtime.permissions import PermissionPolicy
from harness_code_agent.runtime.tool_context import ToolContext
from harness_code_agent.sessions.events import EventBus
from harness_code_agent.workspace.service import WorkspaceService


class ParallelToolTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(os.getcwd(), "workspace", "test-parallel-tools")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True)
        (self.root / "sample.txt").write_text("needle\n", encoding="utf-8")
        self.context = ToolContext(
            workspace=WorkspaceService(root=self.root),
            permission_policy=PermissionPolicy(mode="danger-full-access"),
            event_bus=EventBus(),
        )

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_planner_allows_independent_files_to_share_a_wave(self):
        root = self.context.workspace.root
        effects = [
            CallEffect((ResourceClaim("workspace", str(root / "a.txt").casefold(), "exact", "write"),)),
            CallEffect((ResourceClaim("workspace", str(root / "b.txt").casefold(), "exact", "write"),)),
        ]
        planner = ExecutionPlanner(enumerate(effects))

        self.assertEqual(planner.ready({0, 1}, set()), [0, 1])

    def test_planner_serializes_same_file_read_write(self):
        key = str(self.context.workspace.root / "sample.txt").casefold()
        planner = ExecutionPlanner([
            (0, CallEffect((ResourceClaim("workspace", key, "exact", "read"),))),
            (1, CallEffect((ResourceClaim("workspace", key, "exact", "write"),))),
        ])

        self.assertEqual(planner.ready({0, 1}, set()), [0])
        self.assertEqual(planner.ready({1}, {0}), [1])

    def test_network_read_can_overlap_local_write(self):
        network = BUILTIN_TOOL_REGISTRY.effect_for("web_search", {"query": "x"}, self.context)
        local_write = BUILTIN_TOOL_REGISTRY.effect_for("write_file", {"path": "a.txt"}, self.context)
        planner = ExecutionPlanner([(0, network), (1, local_write)])

        self.assertEqual(planner.ready({0, 1}, set()), [0, 1])

    def test_shell_inspections_parallel_but_verifications_serialize(self):
        inspections = [
            BUILTIN_TOOL_REGISTRY.effect_for("run_bash", {"command": command}, self.context)
            for command in ("git status --short", "rg needle .")
        ]
        verifications = [
            BUILTIN_TOOL_REGISTRY.effect_for("run_bash", {"command": command}, self.context)
            for command in ("pytest -q", "bun run check")
        ]

        inspect_plan = ExecutionPlanner(enumerate(inspections))
        verify_plan = ExecutionPlanner(enumerate(verifications))
        self.assertEqual(inspect_plan.ready({0, 1}, set()), [0, 1])
        self.assertEqual(verify_plan.ready({0, 1}, set()), [0])

    def test_mcp_read_only_hint_and_server_write_effects(self):
        manager = McpClientManager(workspace=self.root)
        manager.tool_bindings = [
            McpToolBinding("mcp_read_a", "docs", "a", "", {}, "network_read", {"readOnlyHint": True}),
            McpToolBinding("mcp_read_b", "docs", "b", "", {}, "network_read", {"readOnlyHint": True}),
            McpToolBinding("mcp_write_a", "state", "a", "", {}, "dangerous", {}),
            McpToolBinding("mcp_write_b", "state", "b", "", {}, "dangerous", {}),
        ]
        registry = ToolRegistry()
        manager.register_tools(registry)

        read_plan = ExecutionPlanner([
            (0, registry.effect_for("mcp_read_a", {}, self.context)),
            (1, registry.effect_for("mcp_read_b", {}, self.context)),
        ])
        write_plan = ExecutionPlanner([
            (0, registry.effect_for("mcp_write_a", {}, self.context)),
            (1, registry.effect_for("mcp_write_b", {}, self.context)),
        ])

        self.assertEqual(read_plan.ready({0, 1}, set()), [0, 1])
        self.assertEqual(write_plan.ready({0, 1}, set()), [0])

    def test_mcp_handler_keeps_runtime_context_out_of_server_arguments(self):
        manager = McpClientManager(workspace=self.root)
        binding = McpToolBinding("mcp_read", "docs", "read", "", {}, "network_read", {})
        captured = {}

        def call_tool(name, arguments, cancellation_token=None):
            captured.update(
                name=name,
                arguments=arguments,
                cancellation_token=cancellation_token,
            )
            return "ok"

        manager.call_tool = call_tool
        token = CancellationToken()

        manager._handler_for(binding)(
            query="needle",
            runtime_state=object(),
            agent_name="reader",
            tool_context=self.context,
            cancellation_token=token,
        )

        self.assertEqual(captured["name"], "mcp_read")
        self.assertEqual(captured["arguments"], {"query": "needle"})
        self.assertIs(captured["cancellation_token"], token)

    def test_mcp_loop_cancels_one_call_and_remains_usable(self):
        from harness_code_agent.agent.cancellation import CancelledError
        from harness_code_agent.runtime.mcp import _AsyncLoopThread

        loop_thread = _AsyncLoopThread()
        token = CancellationToken()
        started = threading.Event()

        async def blocked():
            started.set()
            await asyncio.Event().wait()

        errors = []

        def invoke():
            try:
                loop_thread.run(blocked(), timeout=5, cancellation_token=token)
            except Exception as exc:  # noqa: BLE001 - thread records the result for assertion
                errors.append(exc)

        worker = threading.Thread(target=invoke)
        worker.start()
        self.assertTrue(started.wait(1))
        token.cancel()
        worker.join(1)
        try:
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], CancelledError)

            async def healthy():
                return "ok"

            self.assertEqual(loop_thread.run(healthy(), timeout=1), "ok")
        finally:
            loop_thread.close()

    def test_mcp_worker_cancel_propagates_to_session_call_tool(self):
        from types import SimpleNamespace

        import mcp
        from mcp.client import stdio as mcp_stdio

        from harness_code_agent.runtime.mcp import McpServerConfig

        class FakeStdioClient:
            async def __aenter__(self):
                return (object(), object())

            async def __aexit__(self, *args):
                return False

        class FakeClientSession:
            # Events are installed per scenario (they must be created on the
            # running loop); instances reach them through the class.
            started: asyncio.Event | None = None
            cancelled: asyncio.Event | None = None

            def __init__(self, read_stream=None, write_stream=None):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def initialize(self):
                return None

            async def list_tools(self):
                tool = SimpleNamespace(
                    name="slow_tool",
                    description="",
                    inputSchema={"type": "object", "properties": {}},
                    annotations=None,
                )
                return SimpleNamespace(tools=[tool])

            async def call_tool(self, name, arguments):
                type(self).started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    type(self).cancelled.set()

        async def scenario():
            started = asyncio.Event()
            cancelled = asyncio.Event()
            FakeClientSession.started = started
            FakeClientSession.cancelled = cancelled

            manager = McpClientManager(workspace=self.root)
            server = McpServerConfig(
                name="fake", transport="stdio", command="fake-cmd", args=[]
            )
            request_queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            ready = loop.create_future()
            worker_task = asyncio.create_task(
                manager._connection_worker(server, request_queue, ready)
            )
            bindings = await ready

            response = loop.create_future()
            await request_queue.put(("call", (bindings[0], {}), response))
            await started.wait()
            # Exactly what _call_tool does when its awaiter is cancelled.
            response.cancel()
            # The worker must propagate the response cancellation into the
            # task blocked inside session.call_tool().
            await asyncio.wait_for(cancelled.wait(), timeout=2)

            # Close must return promptly rather than wait on the remote call.
            close_response = loop.create_future()
            await request_queue.put(("close", None, close_response))
            await asyncio.wait_for(close_response, timeout=2)
            await asyncio.wait_for(worker_task, timeout=2)

        with (
            patch.object(mcp, "ClientSession", FakeClientSession),
            patch.object(
                mcp_stdio,
                "stdio_client",
                lambda params: FakeStdioClient(),
            ),
        ):
            asyncio.run(scenario())

    def test_coordinator_allows_different_file_writes_to_overlap(self):
        coordinator = ResourceCoordinator()
        both_entered = threading.Event()
        release = threading.Event()
        entered = []

        def worker(key):
            claim = ResourceClaim("workspace", key, "exact", "write")
            with coordinator.acquire((claim,)):
                entered.append(key)
                if len(entered) == 2:
                    both_entered.set()
                release.wait(2)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(worker, key) for key in ("a", "b")]
            self.assertTrue(both_entered.wait(1))
            release.set()
            for future in futures:
                future.result()

    def test_coordinator_serializes_subtree_read_and_child_write(self):
        coordinator = ResourceCoordinator()
        reader_entered = threading.Event()
        release_reader = threading.Event()
        writer_entered = threading.Event()
        subtree_key = BUILTIN_TOOL_REGISTRY.effect_for(
            "list_files", {"directory": "src"}, self.context
        ).resources[0].key
        child_key = BUILTIN_TOOL_REGISTRY.effect_for(
            "write_file", {"path": "src/a.py"}, self.context
        ).resources[0].key

        def reader():
            with coordinator.acquire((ResourceClaim("workspace", subtree_key, "subtree", "read"),)):
                reader_entered.set()
                release_reader.wait(2)

        def writer():
            reader_entered.wait(1)
            with coordinator.acquire((ResourceClaim("workspace", child_key, "exact", "write"),)):
                writer_entered.set()

        with ThreadPoolExecutor(max_workers=2) as executor:
            read_future = executor.submit(reader)
            write_future = executor.submit(writer)
            self.assertTrue(reader_entered.wait(1))
            self.assertFalse(writer_entered.wait(0.1))
            release_reader.set()
            self.assertTrue(writer_entered.wait(1))
            read_future.result()
            write_future.result()

    def test_legacy_parallel_tools_are_removed(self):
        schema_names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}

        self.assertNotIn("parallel_commands", schema_names)
        self.assertNotIn("parallel_agents", schema_names)
        self.assertNotIn("parallel", schema_names)
        self.assertIn("spawn_agent", schema_names)


if __name__ == "__main__":
    unittest.main()
