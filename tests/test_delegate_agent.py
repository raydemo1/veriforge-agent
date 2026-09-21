import shutil
import tempfile
import threading
import types
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from harness_code_agent.agent.change_proposal import ChangeProposalStore
from harness_code_agent.agent.coordinator import (
    AgentCoordinator,
    AgentRecord,
    _path_allowed,
)
from harness_code_agent.runtime.builtins.registry import (
    BUILTIN_TOOL_REGISTRY,
    TOOL_SCHEMAS,
)
from harness_code_agent.runtime.permissions import PermissionPolicy
from harness_code_agent.runtime.tool_context import ToolContext
from harness_code_agent.sessions.events import EventBus
from harness_code_agent.workspace.service import WorkspaceService


class AgentPathOwnershipTests(unittest.TestCase):
    def test_only_workspace_relative_owned_paths_are_allowed(self):
        self.assertTrue(_path_allowed("src/app.py", ["src"]))
        self.assertTrue(_path_allowed("docs", ["docs"]))
        self.assertFalse(_path_allowed("docs2/x.md", ["docs"]))
        with self.assertRaises(ValueError):
            from harness_code_agent.agent.coordinator import _normalize_rel

            _normalize_rel("src/../../secret")


class ChangeProposalTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp(prefix="hca-proposal-test-"))
        self.root = self.temp / "workspace"
        self.root.mkdir()
        self.workspace = WorkspaceService(root=self.root)
        self.store = ChangeProposalStore()

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.temp, ignore_errors=True)

    def _proposal(self, base: str, worker: str, *, path: str = "app.py"):
        (self.root / path).parent.mkdir(parents=True, exist_ok=True)
        (self.root / path).write_text(base, encoding="utf-8")
        sandbox = self.store.create_sandbox("agent_worker", self.root)
        (sandbox.workspace / path).write_text(worker, encoding="utf-8")
        return self.store.finalize("agent_worker", [path])

    def test_non_overlapping_same_file_changes_three_way_merge(self):
        proposal = self._proposal("one\ntwo\nthree\n", "ONE\ntwo\nthree\n")
        (self.root / "app.py").write_text("one\ntwo\nTHREE\n", encoding="utf-8")

        result = self.store.apply(proposal.id, self.workspace)

        self.assertEqual(result["status"], "applied")
        self.assertEqual((self.root / "app.py").read_text(encoding="utf-8"), "ONE\ntwo\nTHREE\n")

    def test_true_conflict_writes_nothing_until_parent_resolves(self):
        proposal = self._proposal("value = 1\n", "value = 2\n")
        (self.root / "app.py").write_text("value = 3\n", encoding="utf-8")

        result = self.store.apply(proposal.id, self.workspace)

        self.assertEqual(result["status"], "conflict")
        self.assertEqual((self.root / "app.py").read_text(encoding="utf-8"), "value = 3\n")
        self.assertNotIn("<<<<<<<", (self.root / "app.py").read_text(encoding="utf-8"))
        conflict = self.store.read_conflicts(result["conflict_id"])
        self.assertIn("value = 2", conflict["content"])

        resolved = self.store.resolve(
            result["conflict_id"],
            {"app.py": "value = 4\n"},
            self.workspace,
        )
        self.assertEqual(resolved["status"], "applied")
        self.assertEqual((self.root / "app.py").read_text(encoding="utf-8"), "value = 4\n")

    def test_conflict_resolution_rejects_stale_main_workspace(self):
        proposal = self._proposal("value = 1\n", "value = 2\n")
        (self.root / "app.py").write_text("value = 3\n", encoding="utf-8")
        conflict = self.store.apply(proposal.id, self.workspace)
        (self.root / "app.py").write_text("value = 5\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "workspace changed"):
            self.store.resolve(conflict["conflict_id"], {"app.py": "value = 4\n"}, self.workspace)
        self.assertEqual((self.root / "app.py").read_text(encoding="utf-8"), "value = 5\n")

    def test_large_proposal_is_paged_but_artifact_remains_complete(self):
        proposal = self._proposal("", "x" * 40_000 + "\n")

        first = self.store.read_changes(proposal.id, limit=10_000)
        second = self.store.read_changes(proposal.id, offset=first["next_offset"], limit=50_000)

        self.assertGreater(first["total_chars"], 30_000)
        self.assertIsNotNone(first["next_offset"])
        self.assertIsNone(second["next_offset"])
        self.assertTrue(self.store.sandbox_for("agent_worker").workspace.exists())

    def test_registry_exposes_lifecycle_tools_and_removes_legacy_batch_tools(self):
        names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
        self.assertIn("spawn_agent", names)
        self.assertIn("send_agent_message", names)
        self.assertIn("apply_agent_changes", names)
        self.assertIn("resolve_agent_conflicts", names)
        self.assertNotIn("delegate_agent", names)
        self.assertNotIn("parallel_agents", names)
        self.assertNotIn("parallel_commands", names)


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
        self.coordinator = AgentCoordinator(self.context, max_concurrent=3)
        self.context.agent_coordinator = self.coordinator
        _ControlledConversation.instances.clear()
        _ControlledAgent.contexts.clear()

    def tearDown(self):
        self.coordinator.close()
        shutil.rmtree(self.temp, ignore_errors=True)

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

    def test_close_agent_cannot_be_revived_by_concurrent_followup(self):
        closed = threading.Event()
        cleanup_started = threading.Event()
        release_cleanup = threading.Event()
        conversation = types.SimpleNamespace(
            close=lambda: closed.set(),
            add_user_turn=lambda _task: self.fail("closed agent was revived"),
        )
        record = AgentRecord(
            id="agent_done",
            name="done",
            role="explorer",
            task="task",
            expected_output="",
            allowed_paths=[],
            fork_turns="none",
            model_intensity=None,
            state="completed",
            conversation=conversation,
        )
        with self.coordinator._condition:
            self.coordinator._records[record.id] = record
            self.coordinator._names[record.name] = record.id

        def block_cleanup(_agent_id, *, discard_changes=False):
            cleanup_started.set()
            release_cleanup.wait(timeout=1)

        errors: list[Exception] = []

        def close_record() -> None:
            try:
                self.coordinator.close_agent(record.id)
            except Exception as exc:  # noqa: BLE001 - surfaced by the assertion below
                errors.append(exc)

        with patch.object(self.coordinator.changes, "close_agent", side_effect=block_cleanup):
            worker = threading.Thread(target=close_record)
            worker.start()
            self.assertTrue(cleanup_started.wait(timeout=1))
            with self.assertRaises((KeyError, ValueError)):
                self.coordinator.followup(record.id, "revive")
            release_cleanup.set()
            worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(closed.is_set())

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
        self.assertIsNone(explorer.get("tool_search"))
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

    def test_send_parent_message_tool_routes_through_coordinator(self):
        from harness_code_agent.runtime.builtins.agents import send_parent_message

        with patch("harness_code_agent.agent.conversation.Agent", _ControlledAgent):
            spawned = self.coordinator.spawn(name="routed", role="explorer", task="inspect")
            self._wait_conversation("routed")
        self.context.agent_id = spawned["agent_id"]
        self.coordinator._parent_message_sink = lambda text, tag: None

        result = send_parent_message("plan must change", tool_context=self.context)

        self.assertEqual(result.status, "success")

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
