from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from harness_code_agent.core.interactive import InteractiveSession, TurnResult
from harness_code_agent.runtime.builtins.registry import BUILTIN_TOOL_REGISTRY
from harness_code_agent.runtime.questions import QuestionResult
from harness_code_agent.runtime.recovery import (
    RecoveryMiddleware,
    RecoveryStore,
    atomic_json,
)
from harness_code_agent.tui_bridge import BridgeServer


class RecoveryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.store = RecoveryStore(self.root)

    def test_exact_bytes_create_delete_rename_and_directory_transitions(self):
        (self.root / "original.bin").write_bytes(b"\xff\x00\r\n")
        (self.root / "empty").mkdir()
        original = self.store.capture()
        self.assertEqual(original, self.store.capture())
        (self.root / "original.bin").rename(self.root / "renamed.bin")
        (self.root / "created.txt").write_bytes(b"new")
        (self.root / "empty").rmdir()
        (self.root / "empty").write_bytes(b"file")
        current = self.store.capture()
        self.store.restore(current, original)
        self.assertEqual((self.root / "original.bin").read_bytes(), b"\xff\x00\r\n")
        self.assertFalse((self.root / "renamed.bin").exists())
        self.assertFalse((self.root / "created.txt").exists())
        self.assertTrue((self.root / "empty").is_dir())
        self.assertEqual(self.store.capture(), original)

    def test_git_and_dependencies_are_untouched(self):
        for name in (".git", "node_modules"):
            (self.root / name).mkdir()
            (self.root / name / "state").write_bytes(b"before")
        snapshot = self.store.capture()
        for name in (".git", "node_modules"):
            (self.root / name / "state").write_bytes(b"after")
        (self.root / "new").write_bytes(b"new")
        self.store.restore(self.store.capture(), snapshot)
        for name in (".git", "node_modules"):
            self.assertEqual((self.root / name / "state").read_bytes(), b"after")

    def test_corrupt_blob_fails_before_any_file_changes(self):
        (self.root / "a").write_bytes(b"original")
        target = self.store.capture()
        digest = self.store.manifest(target)["a"]["hash"]
        (self.root / "a").write_bytes(b"current")
        current = self.store.capture()
        (self.store.root / "blobs" / digest).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "校验失败"):
            self.store.restore(current, target)
        self.assertEqual((self.root / "a").read_bytes(), b"current")

    def test_crash_recovery_restores_partial_transaction(self):
        (self.root / "a").write_bytes(b"before")
        before = self.store.capture()
        atomic_json(self.store.root / "transactions" / "test.json", {"status": "prepared", "before": before})
        (self.root / "a").write_bytes(b"half restored")
        (self.root / "partial").write_bytes(b"created")
        self.store.recover_interrupted()
        self.assertEqual(self.store.capture(), before)
        self.store.recover_interrupted()
        self.assertFalse((self.root / "partial").exists())

    def test_other_session_cannot_capture_while_turn_owns_workspace(self):
        other = RecoveryStore(self.root)
        with self.store.exclusive(), self.assertRaisesRegex(RuntimeError, "另一个会话"), other.exclusive():
            pass

    def test_readonly_directory_modes_follow_child_restoration(self):
        from harness_code_agent.runtime.recovery import atomic_bytes

        directory = self.root / "locked"
        directory.mkdir()
        child = directory / "child.txt"
        child.write_bytes(b"original")
        manifest = self.store.manifest(self.store.capture())
        manifest["locked"]["mode"] = 0o555
        target = self.store._put_blob(json.dumps(manifest, sort_keys=True).encode("utf-8"))
        real_chmod = Path.chmod
        for existing in (False, True):
            with self.subTest(existing=existing):
                directory.chmod(0o755)
                child.unlink(missing_ok=True)
                modes = {}
                if existing:
                    child.write_bytes(b"current")
                    left = self.store.manifest(self.store.capture())
                    left["locked"]["mode"] = 0o555
                    current = self.store._put_blob(json.dumps(left, sort_keys=True).encode("utf-8"))
                    modes[directory] = 0o555
                else:
                    directory.rmdir()
                    current = self.store.capture()

                def chmod(path, mode, *args, modes=modes, **kwargs):
                    modes[path] = mode
                    return real_chmod(path, mode, *args, **kwargs)

                def write(path, data, modes=modes):
                    if path == child and not modes.get(directory, 0o755) & 0o200:
                        raise PermissionError("parent directory is read-only")
                    return atomic_bytes(path, data)

                try:
                    with patch.object(Path, "chmod", chmod), patch("harness_code_agent.runtime.recovery.atomic_bytes", side_effect=write):
                        self.store.restore(current, target)
                    self.assertEqual(child.read_bytes(), b"original")
                    self.assertEqual(modes[directory], 0o555)
                finally:
                    if directory.exists():
                        directory.chmod(0o755)


class RewindTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"HARNESS_MEMORY_GENERATION_DISABLED": "1", "HARNESS_MEMORY_DISABLED": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.questions = []
        self.choice = "rewind"
        self.session = InteractiveSession(cwd=self.root, profile_name="coding-agent", profile_explicit=True, enable_turn_summary=False,
            question_provider=SimpleNamespace(ask=self._ask))
        self.addCleanup(self.session.close)
        external_tools = patch.object(self.session, "_ensure_mcp_tools_loaded")
        external_tools.start()
        self.addCleanup(external_tools.stop)

    def _ask(self, request):
        self.questions.append(request)
        return QuestionResult(value=self.choice)

    def turn(self, prompt: str, changes: dict[str, bytes | None] | None = None):
        def submit(content, **kwargs):
            self.session.conversation.add_user_turn(content)
            if changes:
                self.session.recovery.protect()
                for name, value in changes.items():
                    path = self.root / name
                    if value is None:
                        path.unlink()
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(value)
            self.session.conversation._append_message({"role": "assistant", "content": "done " + prompt})
            return "done " + prompt
        with patch.object(self.session.conversation, "submit", side_effect=submit):
            self.session._submit_to_current_agent(prompt)
        return max(self.session.recovery_points().values(), key=lambda p: p["turn"])

    def test_no_write_turns_share_snapshot_and_repeated_rewind_retains_correct_prefix(self):
        first = self.turn("one", {"a": b"one"})
        second = self.turn("two")
        third = self.turn("three", {"a": b"three", "new": b"new"})
        self.assertEqual(first["snapshot"], second["snapshot"])
        source = self.session.session
        self.assertTrue(self.session.rewind_to_point(second["id"]))
        self.assertEqual((self.root / "a").read_bytes(), b"one")
        self.assertFalse((self.root / "new").exists())
        self.assertEqual(self.session.turn_count, 2)
        self.assertNotEqual(self.session.session.id, source.id)
        self.assertIn("done three", source.journal_path.read_text(encoding="utf-8"))
        self.assertNotIn("done three", str(self.session.conversation.messages))
        self.assertNotIn(third["id"], self.session.recovery_points())
        fourth = self.turn("four", {"a": b"four"})
        fifth = self.turn("five", {"a": b"five"})
        self.assertTrue(self.session.rewind_to_point(fourth["id"]))
        self.assertEqual((self.root / "a").read_bytes(), b"four")
        self.assertIn("done two", str(self.session.conversation.messages))
        self.assertNotIn("done five", str(self.session.conversation.messages))
        self.assertNotIn(fifth["id"], self.session.recovery_points())

    def test_cancel_leaves_workspace_and_conversation_unchanged(self):
        first = self.turn("one", {"a": b"one"})
        self.turn("two", {"a": b"two"})
        source = self.session.session.id
        messages = list(self.session.conversation.messages)
        self.choice = "cancel"
        self.assertFalse(self.session.rewind_to_point(first["id"]))
        self.assertEqual((self.root / "a").read_bytes(), b"two")
        self.assertEqual(self.session.session.id, source)
        self.assertEqual(self.session.conversation.messages, messages)

    def test_manual_changes_are_listed_in_the_same_confirmation(self):
        first = self.turn("one", {"a": b"one"})
        self.turn("two", {"a": b"two"})
        (self.root / "a").write_bytes(b"manual")
        self.session.rewind_to_point(first["id"])
        self.assertIn("a", self.questions[-1].question)
        self.assertIn("将被覆盖", self.questions[-1].question)
        self.assertEqual(len(self.questions[-1].options), 2)

    def test_session_origin_can_undo_the_first_turn(self):
        (self.root / "a").write_bytes(b"user original")
        self.turn("one", {"a": b"one", "new": b"new"})
        origin = next(point for point in self.session.recovery_points().values() if point["turn"] == 0)
        self.session.rewind_to_point(origin["id"])
        self.assertEqual((self.root / "a").read_bytes(), b"user original")
        self.assertFalse((self.root / "new").exists())
        self.assertEqual(self.session.turn_count, 0)
        self.assertEqual([message["role"] for message in self.session.conversation.messages], ["system"])

    def test_partial_file_failure_restores_original_workspace(self):
        from harness_code_agent.runtime.recovery import atomic_bytes

        first = self.turn("one", {"a": b"one", "b": b"one"})
        self.turn("two", {"a": b"two", "b": b"two"})
        source = self.session.session.id
        failed = False
        def write(path, data):
            nonlocal failed
            if path == self.root / "b" and data == b"one" and not failed:
                failed = True
                raise OSError("file occupied")
            return atomic_bytes(path, data)
        with patch("harness_code_agent.runtime.recovery.atomic_bytes", side_effect=write), self.assertRaisesRegex(OSError, "file occupied"):
            self.session.rewind_to_point(first["id"])
        self.assertEqual((self.root / "a").read_bytes(), b"two")
        self.assertEqual((self.root / "b").read_bytes(), b"two")
        self.assertEqual(self.session.session.id, source)

    def test_plan_and_todo_state_return_to_the_target_turn(self):
        self.session.event_bus.emit("plan_ready", agent="main_agent", payload={"plan_markdown": "# original plan", "plan_revision": 1})
        self.session.event_bus.emit("tool_result", agent="main_agent", payload={"tool": "update_todo", "status": "success", "metadata": {"todo_state": {
            "revision": 1, "updated_at": "time", "next_seq": 3,
            "items": [{"id": "todo_3", "text": "original task", "status": "pending"}],
        }}})
        first = self.turn("one", {"a": b"one"})
        self.session.event_bus.emit("plan_ready", agent="main_agent", payload={"plan_markdown": "# future plan", "plan_revision": 2})
        self.turn("two", {"a": b"two"})
        self.session.rewind_to_point(first["id"])
        self.assertEqual(self.session.pending_plan_markdown, "# original plan")
        self.assertEqual(self.session.pending_plan_revision, 1)
        self.assertEqual(self.session.conversation.runtime_state.todo.items[0].text, "original task")
        self.assertEqual(self.session.conversation.runtime_state.todo.next_seq, 3)
        self.assertTrue((self.session.session.root / "todo" / "state.json").exists())

    def test_files_and_conversation_roll_back_when_activation_fails(self):
        first = self.turn("one", {"a": b"one"})
        self.turn("two", {"a": b"two"})
        source = self.session.session.id
        messages = list(self.session.conversation.messages)
        original = self.session._activate_session
        def activate(session):
            if session.id != source:
                original(session)
                raise OSError("activation failed")
            original(session)
        with patch.object(self.session, "_activate_session", side_effect=activate), self.assertRaisesRegex(OSError, "activation failed"):
            self.session.rewind_to_point(first["id"])
        self.assertEqual((self.root / "a").read_bytes(), b"two")
        self.assertEqual(self.session.session.id, source)
        self.assertEqual(self.session.conversation.messages, messages)

    def test_future_compaction_cannot_leak_into_rewound_conversation(self):
        first = self.turn("one", {"a": b"one"})
        self.turn("two", {"a": b"two"})
        journal = self.session.conversation.journal
        journal.append_compaction(summary="future secret", first_kept_sequence=journal.sequence + 1, phase="test")
        self.session.rewind_to_point(first["id"])
        self.assertNotIn("future secret", str(self.session.conversation.messages))
        self.assertIn("done one", str(self.session.conversation.messages))

    def test_first_allowed_shell_captures_original_bytes(self):
        (self.root / "a").write_bytes(b"original")
        self.session.recovery.begin_turn()
        middleware = RecoveryMiddleware(self.session.recovery, self.session.tool_context, BUILTIN_TOOL_REGISTRY)
        middleware.on_tool_allowed("run_bash", {"command": "python arbitrary_script.py"}, [], agent_name="main_agent")
        before = self.session.recovery.before
        (self.root / "a").write_bytes(b"changed")
        self.assertEqual(self.session.recovery.store.blob(self.session.recovery.store.manifest(before)["a"]["hash"]), b"original")
        middleware.on_tool_allowed("run_bash", {"command": "echo hi"}, [], agent_name="main_agent")
        self.assertEqual(self.session.recovery.before, before)

    def test_control_inspection_and_verification_do_not_capture(self):
        middleware = RecoveryMiddleware(self.session.recovery, self.session.tool_context, BUILTIN_TOOL_REGISTRY)
        calls = [
            ("update_todo", {}), ("ask_user", {}), ("tool_search", {}),
            ("list_shell_jobs", {}), ("read_shell_output", {}), ("stop_shell_job", {}),
            ("run_bash", {"command": "git status"}),
            ("run_bash", {"command": "rg function ."}),
            ("run_bash", {"command": "pytest"}),
            ("code_intelligence", {"operation": "definition", "path": "main.py", "line": 1, "column": 1}),
        ]
        with patch.object(self.session.recovery.store, "capture") as capture:
            for name, args in calls:
                with self.subTest(tool=name, args=args):
                    self.session.recovery.begin_turn()
                    middleware.on_tool_allowed(name, args, [], agent_name="main_agent")
                    capture.assert_not_called()

    def test_first_readonly_turn_captures_once_and_shares_origin_snapshot(self):
        with patch.object(self.session.recovery.store, "capture", wraps=self.session.recovery.store.capture) as capture:
            point = self.turn("read only")
        capture.assert_called_once()
        self.assertEqual(point["snapshot"], self.session.recovery.origin["snapshot"])

    def test_workspace_and_unknown_mutations_capture_once(self):
        middleware = RecoveryMiddleware(self.session.recovery, self.session.tool_context, BUILTIN_TOOL_REGISTRY)
        calls = [
            ("write_file", {"path": "a", "content": "changed"}),
            ("run_bash", {"command": "python arbitrary_script.py"}),
            ("run_bash", {"command": "echo changed > a"}),
            ("run_bash", {"command": "git checkout -- a"}),
            ("apply_agent_changes", {"proposal_id": "missing"}),
            ("resolve_agent_conflicts", {"conflict_id": "missing"}),
            ("browser_test", {"url": "http://localhost:5173", "start_command": "python arbitrary_script.py"}),
        ]
        for name, args in calls:
            with self.subTest(tool=name), patch.object(self.session.recovery.store, "capture", wraps=self.session.recovery.store.capture) as capture:
                self.session.recovery.begin_turn()
                middleware.on_tool_allowed(name, args, [], agent_name="main_agent")
                middleware.on_tool_allowed(name, args, [], agent_name="main_agent")
                capture.assert_called_once()

    def test_resume_unrelated_session_uses_its_origin_and_current_workspace(self):
        self.turn("A", {"a": b"A"})
        self.turn("A again")
        active = self.session
        other = InteractiveSession(cwd=self.root, profile_name="coding-agent", profile_explicit=True, enable_turn_summary=False,
            question_provider=SimpleNamespace(ask=self._ask))
        self.addCleanup(other.close)
        self.session = other
        try:
            with patch.object(other, "_ensure_mcp_tools_loaded"):
                self.turn("B", {"a": b"B"})
        finally:
            self.session = active
        origin = other.recovery.origin
        metadata = other.session_store.read_metadata(other.session.id)
        metadata["recovery_origin_point"] = active.recovery.origin["id"]
        atomic_json(other.session.metadata_path, metadata)
        active.resume_from_session(other.session.id)
        self.assertEqual(active.recovery.origin["id"], origin["id"])
        self.assertEqual(active.recovery.observed, active.recovery.store.capture())
        self.assertEqual(active.turn_count, 1)
        self.assertNotIn(metadata["recovery_origin_point"], active.recovery_points())
        self.turn("continue B")
        origins = [point for point in active.recovery_points().values() if point["turn"] == 0]
        self.assertEqual([point["id"] for point in origins], [origin["id"]])

    def test_resuming_empty_lineage_creates_its_own_origin(self):
        self.turn("A", {"a": b"A"})
        old_origin = self.session.recovery.origin["id"]
        other = self.session.session_store.create(profile="coding-agent", cwd=self.root, model="test", permission_mode="workspace-write")
        self.session.resume_from_session(other.id)
        self.assertIsNone(self.session.recovery.origin)
        self.assertEqual(self.session.turn_count, 0)
        self.turn("first B", {"a": b"B"})
        origins = [point for point in self.session.recovery_points().values() if point["turn"] == 0]
        self.assertEqual(len(origins), 1)
        self.assertNotEqual(origins[0]["id"], old_origin)
        self.assertTrue(self.session.rewind_to_point(origins[0]["id"]))
        self.assertEqual((self.root / "a").read_bytes(), b"A")
        self.assertEqual(self.session.turn_count, 0)

    def test_fork_keeps_origin_and_observes_manual_workspace_changes(self):
        self.turn("one", {"a": b"one"})
        origin = self.session.recovery.origin
        (self.root / "a").write_bytes(b"manual")
        self.session.fork_current_session()
        self.assertEqual(self.session.recovery.origin, origin)
        self.assertEqual(self.session.recovery.observed, self.session.recovery.store.capture())
        self.assertIsNone(self.session.recovery.before)

    def _plan_then_execution(self):
        self.session.switch_profile("plan")
        point = self.turn("prepare plan")
        self.session.pending_plan_markdown = None
        self.session.pending_plan_revision = 0
        self.session._switch_profile("coding-agent", reason="execute approved plan")
        self.turn("execute plan", {"a": b"executed"})
        return point

    def test_rewind_restores_plan_profile_and_continue_executes_pending_plan(self):
        point = self._plan_then_execution()
        self.session.rewind_to_point(point["id"])
        self.assertEqual(self.session.profile.name(), "plan")
        self.assertEqual(self.session.session_store.read_metadata(self.session.session.id)["profile"], "plan")
        self.assertEqual(self.session.pending_plan_markdown, "done prepare plan")
        self.assertEqual(self.session.conversation.messages[0]["content"], self.session.agent.full_system_prompt)
        with patch.object(self.session, "execute_pending_plan", return_value=TurnResult(text="executed")) as execute:
            self.session.submit("继续")
        execute.assert_called_once()

    def test_rewind_before_first_profile_switch_uses_initial_profile(self):
        point = self.turn("coding")
        self.session.switch_profile("plan")
        self.turn("future plan")
        self.session.rewind_to_point(point["id"])
        self.assertEqual(self.session.profile.name(), "coding-agent")
        self.assertIsNone(self.session.pending_plan_markdown)
        self.assertEqual(self.session.pending_plan_revision, 0)

    def test_failed_rewind_restores_profile_policy_and_recovery_state(self):
        point = self._plan_then_execution()
        old_agent = self.session.agent
        old_policy = self.session.tool_context.permission_policy
        old_observed = self.session.recovery.observed
        old_source = self.session.session.id
        with patch.object(self.session.session_store, "update_profile", side_effect=OSError("metadata write failed")), self.assertRaisesRegex(OSError, "metadata write failed"):
            self.session.rewind_to_point(point["id"])
        self.assertEqual(self.session.profile.name(), "coding-agent")
        self.assertIs(self.session.agent, old_agent)
        self.assertIs(self.session.conversation.agent, old_agent)
        self.assertIs(self.session.tool_context.permission_policy, old_policy)
        self.assertEqual(self.session.recovery.observed, old_observed)
        self.assertEqual(self.session.session.id, old_source)
        self.assertEqual((self.root / "a").read_bytes(), b"executed")

    def test_bridge_history_replays_recovery_buttons_and_work_state(self):
        first = self.turn("one", {"a": b"one"})
        self.turn("two", {"a": b"two"})
        self.session.rewind_to_point(first["id"])
        bridge = BridgeServer.__new__(BridgeServer)
        bridge.cwd = self.root
        bridge._session = self.session
        bridge._session_error = None
        bridge._assistant_group_id = None
        bridge.state = SimpleNamespace(snapshot=SimpleNamespace(profile="coding-agent", model="test", provider="test", permission_mode="workspace-write"))
        items = bridge._history_items(self.session.session.id)
        groups = [item for item in items if item.get("role") == "group"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["recoveryPointId"], first["id"])
        self.assertEqual(len({item["id"] for item in items}), len(items))
        self.assertNotIn("done two", str(items))
        self.assertEqual(bridge.state.snapshot.turn, 1)

    def test_rewinding_a_normal_fork_retains_parent_history(self):
        self.turn("one", {"a": b"one"})
        self.session.fork_current_session()
        second = self.turn("two", {"a": b"two"})
        self.turn("three", {"a": b"three"})
        self.session.rewind_to_point(second["id"])
        events = self.session.session_store.read_history_events(self.session.session.id)
        prompts = [event["payload"]["text"] for event in events if event["type"] == "user_input"]
        self.assertEqual(prompts, ["one", "two"])
        self.assertIn("done one", str(self.session.conversation.messages))
        self.assertNotIn("done three", str(self.session.conversation.messages))
        self.assertEqual((self.root / "a").read_bytes(), b"two")

    def test_rewind_requests_are_serialized_and_reject_duplicate_clicks(self):
        first = self.turn("one", {"a": b"one"})
        bridge = BridgeServer.__new__(BridgeServer)
        bridge._session = self.session
        bridge._session_error = None
        bridge._active_lock = threading.Lock()
        bridge._active_token = None
        bridge._rewind_pending = False
        bridge._stopping = threading.Event()
        bridge._tasks = queue.Queue()
        self.assertTrue(bridge._queue_rewind(first["id"])["queued"])
        with self.assertRaisesRegex(ValueError, "等待当前操作"):
            bridge._queue_rewind(first["id"])


if __name__ == "__main__":
    unittest.main()
