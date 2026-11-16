import os
import shutil
import subprocess
import tempfile
import threading
import types
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest

from harness_code_agent.agent.coordinator import AgentCoordinator, AgentRecord
from harness_code_agent.runtime.builtins.registry import (
    BUILTIN_TOOL_REGISTRY,
    TOOL_SCHEMAS,
)
from harness_code_agent.runtime.permissions import PermissionPolicy
from harness_code_agent.runtime.tool_context import ToolContext
from harness_code_agent.sessions.events import EventBus
from harness_code_agent.workspace.access import WorkspaceConflict
from harness_code_agent.workspace.paths import path_within
from harness_code_agent.workspace.service import WorkspaceService


class AgentPathOwnershipTests(unittest.TestCase):
    def test_only_workspace_relative_owned_paths_are_allowed(self):
        self.assertTrue(path_within("src/app.py", ["src"]))
        self.assertTrue(path_within("docs", ["docs"]))
        self.assertFalse(path_within("docs2/x.md", ["docs"]))
        with self.assertRaises(ValueError):
            from harness_code_agent.agent.coordinator import _normalize_rel

            _normalize_rel("src/../../secret")

    def test_dot_root_owns_the_whole_workspace(self):
        self.assertTrue(path_within("src/app.py", ["."]))
        self.assertTrue(path_within(".", ["."]))
        self.assertFalse(path_within(".", ["src"]))
        self.assertFalse(path_within("src2/app.py", ["src"]))


def _make_directory_link(link: Path, target: Path) -> bool:
    try:
        link.symlink_to(target, target_is_directory=True)
        return True
    except OSError:
        if os.name != "nt":
            return False
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


class WorkerScopePathTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="hca-worker-scope-"))
        (self.root / "src").mkdir()
        (self.root / "docs").mkdir()
        (self.root / "other.txt").write_text("keep", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _decide(self, path):
        policy = PermissionPolicy.for_role("worker", allowed_paths=["src"])
        return policy.decide_tool_call(
            "write_file", {"path": path}, tool_permission="edit", workspace_root=str(self.root)
        )

    def test_inside_scope_is_allowed(self):
        self.assertTrue(self._decide("src/nested/new.txt").allowed)

    def test_dotdot_escape_is_denied(self):
        decision = self._decide("src/../other.txt")
        self.assertFalse(decision.allowed)
        self.assertIn("resolves outside allowed_paths", decision.reason)

    def test_link_escape_is_denied(self):
        link = self.root / "src" / "linked"
        if not _make_directory_link(link, self.root / "docs"):
            self.skipTest("cannot create a directory symlink or junction")
        decision = self._decide("src/linked/new.txt")
        self.assertFalse(decision.allowed)
        self.assertIn("resolves outside allowed_paths", decision.reason)

    def test_file_scope_allows_the_file_but_shell_requires_directories(self):
        (self.root / "src" / "app.py").write_text("x", encoding="utf-8")
        policy = PermissionPolicy.for_role("worker", allowed_paths=["src/app.py"])
        allowed = policy.decide_tool_call(
            "write_file", {"path": "src/app.py"}, tool_permission="edit", workspace_root=str(self.root)
        )
        denied = policy.decide_tool_call(
            "write_file", {"path": "src/other.py"}, tool_permission="edit", workspace_root=str(self.root)
        )
        self.assertTrue(allowed.allowed)
        self.assertFalse(denied.allowed)

        from harness_code_agent.runtime.builtins.shell import _effective_write_paths
        from harness_code_agent.workspace.native_shell import write_directories

        context = types.SimpleNamespace(permission_policy=policy)
        self.assertEqual(_effective_write_paths([], context), ["src/app.py"])
        # Shell grants are directory-based: a file-only scope fails loudly
        # instead of silently widening to the parent directory.
        with self.assertRaisesRegex(ValueError, "directories, not files"):
            write_directories(self.root, ["src/app.py"])


class SharedWorkspaceConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="hca-shared-workspace-"))
        (self.root / "frontend").mkdir()
        (self.root / "backend").mkdir()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_distinct_files_written_by_two_services_survive(self):
        main = WorkspaceService(root=self.root)
        worker = WorkspaceService(root=self.root)
        threads = [
            threading.Thread(target=main.write_text, args=("frontend/app.ts", "frontend")),
            threading.Thread(target=worker.write_text, args=("backend/app.py", "backend")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        self.assertEqual((self.root / "frontend" / "app.ts").read_text(encoding="utf-8"), "frontend")
        self.assertEqual((self.root / "backend" / "app.py").read_text(encoding="utf-8"), "backend")

    def test_worker_write_after_main_change_conflicts_instead_of_overwriting(self):
        path = self.root / "frontend" / "app.ts"
        path.write_text("original", encoding="utf-8")
        main = WorkspaceService(root=self.root)
        worker = WorkspaceService(root=self.root)
        worker.read_text("frontend/app.ts")
        main.read_text("frontend/app.ts")
        main.write_text("frontend/app.ts", "main change")
        with self.assertRaises(WorkspaceConflict):
            worker.write_text("frontend/app.ts", "worker change")
        self.assertEqual(path.read_text(encoding="utf-8"), "main change")


class AgentToolSurfaceTests(unittest.TestCase):
    def test_registry_exposes_lifecycle_tools_only(self):
        names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
        for name in ("spawn_agent", "send_agent_message", "wait_agents", "close_agent"):
            self.assertIn(name, names)
        self.assertNotIn("read_agent_changes", names)
        self.assertNotIn("apply_agent_changes", names)

    def test_coordination_tools_are_deferred_until_search_tools_mounts_them(self):
        from harness_code_agent.runtime.tool_registry import tool_schemas_for_agent

        def names_of(**kwargs):
            return {
                schema["function"]["name"]
                for schema in tool_schemas_for_agent(registry=BUILTIN_TOOL_REGISTRY, **kwargs)
            }

        core = names_of()
        deferred = names_of(disclosure={"deferred"})
        for name in ("spawn_agent", "wait_agents", "close_agent"):
            self.assertNotIn(name, core, msg=f"{name} must not be mounted by default")
            self.assertIn(name, deferred, msg=f"{name} must stay discoverable")
        child_view = {
            schema["function"]["name"]
            for schema in tool_schemas_for_agent(
                registry=BUILTIN_TOOL_REGISTRY, capabilities={"worker_agent"}
            )
        }
        self.assertIn("send_parent_message", child_view)


class ToolDiscoveryTests(unittest.TestCase):
    def test_workflow_group_expands_independently_of_ranking(self):
        from harness_code_agent.runtime.builtins.discovery import (
            TOOL_GROUPS,
            _expand_groups,
        )
        from harness_code_agent.runtime.tool_registry import ToolSpec

        group = next(iter(TOOL_GROUPS))
        catalog = {
            name: ToolSpec(
                name=name,
                schema={
                    "type": "function",
                    "function": {"name": name, "parameters": {"type": "object"}},
                },
                handler=lambda **_: None,
                permission="read",
                effect_resolver=lambda _args, _context: None,
            )
            for name in group
        }

        mounted = _expand_groups(["spawn_agent"], catalog)

        self.assertEqual(mounted[0], "spawn_agent")
        self.assertEqual(set(mounted), set(group))

        partial = {name: catalog[name] for name in ("spawn_agent", "wait_agents")}
        self.assertEqual(_expand_groups(["spawn_agent"], partial), ["spawn_agent", "wait_agents"])

    def test_search_without_server_does_not_connect_mcp_servers(self):
        from harness_code_agent.runtime.builtins.discovery import search_tools

        calls = []

        def discover(server_name=None):
            calls.append(server_name)
            return {}

        with tempfile.TemporaryDirectory() as tmp:
            context = self._context(Path(tmp), discover=discover)
            result = search_tools("spawn agent", limit=1, tool_context=context)

        self.assertEqual(result.status, "success")
        self.assertIn("spawn_agent", result.metadata["revealed_tool_names"])
        self.assertEqual(calls, [])

    def test_requested_server_failure_is_reported_with_cached_tools(self):
        from harness_code_agent.runtime.builtins.discovery import search_tools

        registry = BUILTIN_TOOL_REGISTRY.copy()
        registry.register(
            {
                "type": "function",
                "function": {
                    "name": "mcp__docs__search",
                    "description": "Search product documentation",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            },
            lambda **_: None,
            permission="read",
            disclosure="deferred",
            origin_server="docs",
            origin_tool="search",
        )
        failure = "TimeoutError: did not answer"

        def discover(server_name=None):
            return {"docs": failure}

        with tempfile.TemporaryDirectory() as tmp:
            context = self._context(Path(tmp), discover=discover, registry=registry)
            result = search_tools("product documentation", server="docs", tool_context=context)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.metadata["failed_mcp_servers"], {"docs": failure})
        self.assertIn("Unavailable MCP server(s)", result.output)

    def test_requested_server_without_cached_tools_fails_clearly(self):
        from harness_code_agent.runtime.builtins.discovery import search_tools

        def discover(server_name=None):
            raise TimeoutError("did not answer")

        with tempfile.TemporaryDirectory() as tmp:
            context = self._context(Path(tmp), discover=discover)
            result = search_tools("documents", server="docs", tool_context=context)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error, "mcp_unavailable")
        self.assertEqual(
            result.metadata["failed_mcp_servers"],
            {"docs": "TimeoutError: did not answer"},
        )

    def _context(self, root, *, discover, registry=None):
        return ToolContext(
            workspace=WorkspaceService(root=root),
            permission_policy=PermissionPolicy(mode=PermissionPolicy.DANGER_FULL_ACCESS),
            event_bus=EventBus(),
            tool_registry=registry or BUILTIN_TOOL_REGISTRY,
            discover_mcp_tools=discover,
        )


class _ControlledConversation:
    instances: ClassVar[dict] = {}

    def __init__(self, name: str):
        self.name = name
        self.messages = [{"role": "user", "content": "task"}]
        self.started = threading.Event()
        self.release = threading.Event()
        self.message_received = threading.Event()
        self.queued = []
        self.consumed = []
        self.__class__.instances[name] = self

    def run_until_idle(self, cancellation_token=None):
        remove = cancellation_token.add_callback(self.release.set)
        self.started.set()
        self.release.wait(2)
        remove()
        cancellation_token.check()
        self.consumed.extend(self.queued)
        self.queued.clear()
        return "done"

    def queue_message(self, message):
        self.queued.append(message)
        self.message_received.set()

    def has_queued_messages(self):
        return bool(self.queued)

    def add_user_turn(self, task):
        self.messages.append({"role": "user", "content": task})

    def close(self):
        self.release.set()


class _ControlledAgent:
    contexts: ClassVar[dict] = {}

    def __init__(self, **kwargs):
        self.name = kwargs["name"]
        self.__class__.contexts[self.name] = kwargs["tool_context"]

    def start_conversation(self, task):
        return _ControlledConversation(self.name)


class _RunFailingConversation:
    def __init__(self, name):
        self.name = name
        self.messages = [{"role": "user", "content": "task"}]

    def run_until_idle(self, cancellation_token=None):
        raise RuntimeError("run exploded")

    def has_queued_messages(self):
        return False

    def add_user_turn(self, task):
        self.messages.append({"role": "user", "content": task})

    def close(self):
        pass


class _RunFailingAgent:
    def __init__(self, **kwargs):
        self.name = kwargs["name"]

    def start_conversation(self, task):
        return _RunFailingConversation(self.name)


class _SetupFailingAgent:
    def __init__(self, **kwargs):
        self.name = kwargs["name"]

    def start_conversation(self, task):
        raise ValueError("setup exploded")


class AgentCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp(prefix="hca-coordinator-test-"))
        self.context = ToolContext(
            workspace=WorkspaceService(root=self.temp),
            permission_policy=PermissionPolicy(mode="danger-full-access"),
            event_bus=EventBus(),
            tool_registry=BUILTIN_TOOL_REGISTRY,
        )
        self.coordinator = AgentCoordinator(self.context)
        self.context.agent_coordinator = self.coordinator
        _ControlledConversation.instances.clear()
        _ControlledAgent.contexts.clear()

    def tearDown(self):
        self.coordinator.close()
        shutil.rmtree(self.temp, ignore_errors=True)

    def test_registered_agent_survives_requesting_turn_cancellation(self):
        from harness_code_agent.support.cancellation import CancellationToken
        token = CancellationToken()
        with patch("harness_code_agent.agent.conversation.Agent", _ControlledAgent):
            spawned = self.coordinator.spawn(name="independent", role="explorer", task="inspect",
                                              parent_cancellation_token=token)
            conversation = self._wait_conversation("independent")
            self.assertTrue(conversation.started.wait(1))
            token.cancel()
            self.assertFalse(conversation.release.wait(0.1))
            self.assertEqual(self.coordinator.list()["agents"][0]["agent_id"], spawned["agent_id"])
            conversation.release.set()
            self.assertEqual(self._wait_terminal(spawned["agent_id"])["status"], "completed")

    def test_overlapping_worker_scopes_are_rejected(self):
        with patch("harness_code_agent.agent.conversation.Agent", _ControlledAgent):
            first = self.coordinator.spawn(
                name="worker-one", role="worker", task="one", allowed_paths=["src"]
            )
            with self.assertRaisesRegex(ValueError, "overlap"):
                self.coordinator.spawn(
                    name="worker-two", role="worker", task="two", allowed_paths=["src/app.py"]
                )
            self._wait_conversation("worker-one").release.set()
            self.assertEqual(self._wait_terminal(first["agent_id"])["status"], "completed")

    def test_followup_rechecks_worker_scope_overlap(self):
        with patch("harness_code_agent.agent.conversation.Agent", _ControlledAgent):
            finished = self.coordinator.spawn(
                name="former", role="worker", task="one", allowed_paths=["src"]
            )
            conversation = self._wait_conversation("former")
            self.assertTrue(conversation.started.wait(1))
            conversation.release.set()
            self.assertEqual(self._wait_terminal(finished["agent_id"])["status"], "completed")

            active = self.coordinator.spawn(
                name="current", role="worker", task="two", allowed_paths=["src"]
            )
            self._wait_conversation("current")
            with self.assertRaisesRegex(ValueError, "overlap"):
                self.coordinator.followup(finished["agent_id"], "continue")

            state = next(
                item for item in self.coordinator.list()["agents"]
                if item["agent_id"] == finished["agent_id"]
            )
            self.assertEqual(state["status"], "completed")
            _ControlledConversation.instances["current"].release.set()
            self.assertEqual(self._wait_terminal(active["agent_id"])["status"], "completed")

    def test_disjoint_worker_scopes_share_the_workspace_and_run_together(self):
        (self.temp / "frontend").mkdir()
        (self.temp / "backend").mkdir()
        with patch("harness_code_agent.agent.conversation.Agent", _ControlledAgent):
            first = self.coordinator.spawn(
                name="frontend", role="worker", task="ui", allowed_paths=["frontend"]
            )
            second = self.coordinator.spawn(
                name="backend", role="worker", task="api", allowed_paths=["backend"]
            )
            first_conversation = self._wait_conversation("frontend")
            second_conversation = self._wait_conversation("backend")
            self.assertTrue(first_conversation.started.wait(1))
            self.assertTrue(second_conversation.started.wait(1))

            states = {item["name"]: item["status"] for item in self.coordinator.list()["agents"]}
            self.assertEqual(states, {"frontend": "running", "backend": "running"})
            frontend = _ControlledAgent.contexts["frontend"]
            backend = _ControlledAgent.contexts["backend"]
            self.assertEqual(frontend.workspace.root, self.temp.resolve())
            self.assertEqual(backend.workspace.root, self.temp.resolve())
            self.assertIsNot(frontend.workspace, self.context.workspace)
            self.assertIsNot(frontend.workspace, backend.workspace)
            self.assertEqual(frontend.permission_policy.allowed_paths, ("frontend",))

            first_conversation.release.set()
            second_conversation.release.set()
            self.assertEqual(self._wait_terminal(first["agent_id"])["status"], "completed")
            self.assertEqual(self._wait_terminal(second["agent_id"])["status"], "completed")

    def test_message_reaches_running_agent_without_waiting_for_turn_end(self):
        with patch("harness_code_agent.agent.conversation.Agent", _ControlledAgent):
            spawned = self.coordinator.spawn(name="explore", role="explorer", task="inspect")
            conversation = self._wait_conversation("explore")
            self.assertTrue(conversation.started.wait(1))

            state = self.coordinator.send(spawned["agent_id"], "also inspect cancellation")

            self.assertEqual(state["status"], "running")
            self.assertTrue(conversation.message_received.is_set())
            self.assertFalse(conversation.release.is_set())
            conversation.release.set()
            completed = self._wait_terminal(spawned["agent_id"])
            self.assertEqual(completed["status"], "completed")
            self.assertIn("also inspect cancellation", conversation.consumed)

    def test_interrupt_is_per_agent_and_shared_coordinator_is_reused(self):
        with patch("harness_code_agent.agent.conversation.Agent", _ControlledAgent):
            first = self.coordinator.spawn(name="first", role="explorer", task="one")
            second = self.coordinator.spawn(name="second", role="explorer", task="two")
            first_conversation = self._wait_conversation("first")
            second_conversation = self._wait_conversation("second")
            self.assertTrue(first_conversation.started.wait(1))
            self.assertTrue(second_conversation.started.wait(1))

            self.coordinator.interrupt(first["agent_id"])
            interrupted = self._wait_terminal(first["agent_id"])

            self.assertEqual(interrupted["status"], "interrupted")
            self.assertEqual(self.coordinator.list()["agents"][1]["status"], "running")
            self.assertIs(_ControlledAgent.contexts["first"].resource_coordinator, self.context.resource_coordinator)
            second_conversation.release.set()
            self.assertEqual(self._wait_terminal(second["agent_id"])["status"], "completed")

    def test_run_exception_is_caught_at_single_thread_boundary(self):
        with patch("harness_code_agent.agent.conversation.Agent", _RunFailingAgent):
            spawned = self.coordinator.spawn(name="boom", role="explorer", task="inspect")
            terminal = self._wait_terminal(spawned["agent_id"])
        self.assertEqual(terminal["status"], "failed")
        self.assertEqual(terminal["error"], "RuntimeError: run exploded")

    def test_setup_exception_is_caught_before_agent_ran(self):
        with patch("harness_code_agent.agent.conversation.Agent", _SetupFailingAgent):
            spawned = self.coordinator.spawn(name="setupboom", role="explorer", task="inspect")
            terminal = self._wait_terminal(spawned["agent_id"])
        self.assertEqual(terminal["status"], "failed")
        self.assertEqual(terminal["error"], "ValueError: setup exploded")

    def test_spawn_submission_failure_does_not_keep_the_name_or_slot(self):
        with (
            patch.object(
                self.coordinator, "_submit_record", side_effect=RuntimeError("executor closed")
            ),
            self.assertRaisesRegex(RuntimeError, "executor closed"),
        ):
            self.coordinator.spawn(name="rejected", role="explorer", task="inspect")
        self.assertEqual(self.coordinator.list()["agents"], [])

        # The name and capacity are free again.
        with patch("harness_code_agent.agent.conversation.Agent", _ControlledAgent):
            spawned = self.coordinator.spawn(name="rejected", role="explorer", task="inspect")
            self._wait_conversation("rejected").release.set()
            self.assertEqual(self._wait_terminal(spawned["agent_id"])["status"], "completed")

    def test_followup_submission_failure_restores_the_terminal_state(self):
        with patch("harness_code_agent.agent.conversation.Agent", _RunFailingAgent):
            spawned = self.coordinator.spawn(name="failed", role="explorer", task="inspect")
            self._wait_terminal(spawned["agent_id"])
        record = self.coordinator._records[spawned["agent_id"]]
        messages_before = len(record.conversation.messages)
        future_before = record.future

        with (
            patch.object(
                self.coordinator, "_submit_record", side_effect=RuntimeError("executor closed")
            ),
            self.assertRaisesRegex(RuntimeError, "executor closed"),
        ):
            self.coordinator.followup(spawned["agent_id"], "try again")

        state = next(
            item for item in self.coordinator.list()["agents"]
            if item["agent_id"] == spawned["agent_id"]
        )
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["error"], "RuntimeError: run exploded")
        self.assertIs(record.future, future_before)
        self.assertEqual(len(record.conversation.messages), messages_before)

    def test_child_event_bus_stamps_the_stable_agent_id(self):
        with patch("harness_code_agent.agent.conversation.Agent", _ControlledAgent):
            spawned = self.coordinator.spawn(name="attributed", role="explorer", task="inspect")
            self._wait_conversation("attributed")
            context = _ControlledAgent.contexts["attributed"]
            event = context.event_bus.emit("tool_call", payload={"tool": "bash", "args": {}})
            self.assertEqual(event.agent, "attributed")
            self.assertEqual(event.payload["agent_id"], spawned["agent_id"])
            # Same shared stream as the main agent; only identity is scoped.
            self.assertIn(event, self.context.event_bus.events)
            self.assertIs(context.event_bus.events, self.context.event_bus.events)
            self._wait_conversation("attributed").release.set()
            self.assertEqual(self._wait_terminal(spawned["agent_id"])["status"], "completed")

    def test_wait_ignores_unrelated_agent_state_changes(self):
        selected = AgentRecord(
            id="agent_selected",
            name="selected",
            role="explorer",
            task="selected task",
            expected_output="",
            allowed_paths=[],
            fork_turns="none",
            model_intensity=None,
            state="running",
        )
        unrelated = AgentRecord(
            id="agent_unrelated",
            name="unrelated",
            role="explorer",
            task="unrelated task",
            expected_output="",
            allowed_paths=[],
            fork_turns="none",
            model_intensity=None,
            state="running",
        )
        with self.coordinator._condition:
            self.coordinator._records = {
                selected.id: selected,
                unrelated.id: unrelated,
            }
            self.coordinator._names = {
                selected.name: selected.id,
                unrelated.name: unrelated.id,
            }

        def unrelated_change(predicate, timeout=None):
            unrelated.state = "completed"
            self.coordinator._changed_locked()
            return predicate()

        with patch.object(self.coordinator._condition, "wait_for", side_effect=unrelated_change):
            payload = self.coordinator.wait([selected.id], timeout_seconds=1)

        self.assertTrue(payload["timed_out"])
        self.assertEqual(payload["agents"][0]["status"], "running")

    def test_agent_runtime_uses_its_tool_context_workspace_for_background_jobs(self):
        from harness_code_agent.agent.conversation import Agent

        isolated = self.temp / "isolated-worker"
        isolated.mkdir()
        context = ToolContext(
            workspace=WorkspaceService(root=isolated),
            permission_policy=PermissionPolicy(mode="danger-full-access"),
            event_bus=EventBus(),
        )
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=None),
            close=lambda: None,
        )
        with patch("harness_code_agent.agent.conversation.get_client", return_value=fake_client):
            conversation = Agent("worker", "system", use_tools=False, tool_context=context).start_conversation()
        try:
            self.assertEqual(
                Path(conversation.runtime_state.shell_job_manager.workspace),
                isolated.resolve(),
            )
        finally:
            conversation.close()

    def test_role_registries_are_derived_from_tool_capabilities(self):
        explorer = self.coordinator._role_registry("explorer")
        worker = self.coordinator._role_registry("worker")

        self.assertIsNotNone(explorer.get("read_file"))
        self.assertIsNone(explorer.get("write_file"))
        self.assertIsNone(explorer.get("spawn_agent"))
        self.assertIsNotNone(explorer.get("search_tools"))
        self.assertIsNotNone(worker.get("read_file"))
        self.assertIsNotNone(worker.get("write_file"))
        self.assertIsNone(worker.get("spawn_agent"))

    def test_child_registries_hold_upward_channel_only(self):
        # Fixed Main <-> Child topology: children may report upward but hold
        # no downward or sibling addressing, and cannot spawn agents.
        explorer = self.coordinator._role_registry("explorer")
        worker = self.coordinator._role_registry("worker")

        self.assertIsNotNone(explorer.get("send_parent_message"))
        self.assertIsNotNone(worker.get("send_parent_message"))
        self.assertIsNone(explorer.get("send_agent_message"))
        self.assertIsNone(worker.get("send_agent_message"))

    def test_send_to_parent_queues_tagged_message_for_main(self):
        queued = []
        self.coordinator._parent_message_sink = lambda text, tag: queued.append((text, tag))
        with patch("harness_code_agent.agent.conversation.Agent", _ControlledAgent):
            spawned = self.coordinator.spawn(name="reporter", role="explorer", task="inspect")
            self._wait_conversation("reporter")

        snapshot = self.coordinator.send_to_parent(
            spawned["agent_id"], "evidence contradicts the current plan"
        )

        self.assertEqual(
            queued,
            [("evidence contradicts the current plan", "SUBAGENT MESSAGE from reporter")],
        )
        self.assertEqual(snapshot["status"], "running")

    def test_send_to_parent_without_sink_raises(self):
        with patch("harness_code_agent.agent.conversation.Agent", _ControlledAgent):
            spawned = self.coordinator.spawn(name="adrift", role="explorer", task="inspect")
            self._wait_conversation("adrift")

        with self.assertRaisesRegex(ValueError, "no parent agent is available"):
            self.coordinator.send_to_parent(spawned["agent_id"], "hello")

    def test_send_to_parent_rejects_empty_message(self):
        with patch("harness_code_agent.agent.conversation.Agent", _ControlledAgent):
            spawned = self.coordinator.spawn(name="quiet", role="explorer", task="inspect")
            self._wait_conversation("quiet")

        with self.assertRaisesRegex(ValueError, "message is required"):
            self.coordinator.send_to_parent(spawned["agent_id"], "   ")

    def test_send_parent_message_tool_blocks_without_child_identity(self):
        from harness_code_agent.runtime.builtins.agents import send_parent_message

        result = send_parent_message("hello", tool_context=self.context)

        self.assertEqual(result.status, "failed")
        self.assertIn("only available to spawned agents", result.output)

    def test_send_parent_message_tool_routes_through_real_child_context(self):
        from harness_code_agent.runtime.builtins.agents import send_parent_message

        with patch("harness_code_agent.agent.conversation.Agent", _ControlledAgent):
            spawned = self.coordinator.spawn(name="routed", role="explorer", task="inspect")
            self._wait_conversation("routed")

        # Use the production child context, not the main context: it must
        # carry both the child identity and a reference to the coordinator.
        child_context = _ControlledAgent.contexts["routed"]
        self.assertEqual(child_context.agent_id, spawned["agent_id"])
        self.assertIs(child_context.agent_coordinator, self.coordinator)

        reported = []
        self.coordinator._parent_message_sink = lambda text, tag: reported.append((text, tag))

        result = send_parent_message("plan must change", tool_context=child_context)

        self.assertEqual(result.status, "success")
        self.assertEqual(
            reported,
            [("plan must change", "SUBAGENT MESSAGE from routed")],
        )

    def _wait_conversation(self, name):
        for _ in range(1000):
            conversation = _ControlledConversation.instances.get(name)
            if conversation is not None:
                return conversation
            threading.Event().wait(0.001)
        self.fail(f"conversation did not start: {name}")

    def _wait_terminal(self, agent_id):
        for _ in range(1000):
            state = self.coordinator.list()["agents"]
            selected = next(item for item in state if item["agent_id"] == agent_id)
            if selected["status"] not in {"queued", "running"}:
                return selected
            threading.Event().wait(0.001)
        self.fail(f"agent did not finish: {agent_id}")


if __name__ == "__main__":
    unittest.main()


def test_agent_role_contract_is_single_sourced():
    """Coordinator, permission policy and tool schema read one role definition."""
    from harness_code_agent.agent.coordinator import AGENT_ROLES as coordinator_roles
    from harness_code_agent.runtime.agent_roles import (
        AGENT_ROLE_DEFINITIONS,
        AGENT_ROLE_NAMES,
        AGENT_ROLES,
        READ_ONLY_ROLE_NAMES,
        WRITING_ROLE_NAMES,
        role_definition,
    )
    from harness_code_agent.runtime.builtins.schemas import CORE_TOOL_SCHEMAS

    schema = next(
        entry["function"] for entry in CORE_TOOL_SCHEMAS if entry["function"]["name"] == "spawn_agent"
    )
    assert schema["parameters"]["properties"]["role"]["enum"] == list(AGENT_ROLES)
    assert coordinator_roles == AGENT_ROLE_NAMES

    assert READ_ONLY_ROLE_NAMES | WRITING_ROLE_NAMES == AGENT_ROLE_NAMES
    assert not READ_ONLY_ROLE_NAMES & WRITING_ROLE_NAMES
    for definition in AGENT_ROLE_DEFINITIONS:
        assert definition.instruction.strip(), definition.name
        assert role_definition(definition.name) is definition

    for name in AGENT_ROLES:
        policy = PermissionPolicy.for_role(name)
        assert policy.mode == (
            PermissionPolicy.READ_ONLY if name in READ_ONLY_ROLE_NAMES else PermissionPolicy.EVAL
        )

    with pytest.raises(ValueError):
        PermissionPolicy.for_role("does-not-exist")
    with pytest.raises(ValueError):
        role_definition("does-not-exist")
