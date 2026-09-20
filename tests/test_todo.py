import json
import os
import shutil
import sys
import types
import unittest
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

from harness_code_agent import config
from harness_code_agent.agent.conversation import AgentRuntimeState
from harness_code_agent.agent.runtime_state import TodoList, normalize_todo_items
from harness_code_agent.runtime.tool_runner import execute_tool, execute_tool_result


def _items():
    return [
        {"text": "Inspect parser", "status": "completed"},
        {"text": "Fix validation", "status": "in_progress"},
        {"text": "Run tests", "status": "pending"},
    ]


class UpdateTodoToolTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = os.path.join(os.getcwd(), "workspace", "test-todo-tool")
        self.old_workspace = config.WORKSPACE
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        config.WORKSPACE = self.temp_dir

    def tearDown(self):
        config.WORKSPACE = self.old_workspace
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _state_path(self, session_id: str = "test-session") -> Path:
        return Path(self.temp_dir, ".harness", "sessions", session_id, "todo", "state.json")

    def test_create_todo_writes_only_session_state_json(self):
        state = AgentRuntimeState(session_id="test-session")
        result = execute_tool(
            "update_todo",
            {"items": _items()},
            runtime_state=state,
            agent_name="main_agent",
        )

        self.assertIn("Updated todo list", result)
        self.assertIsNotNone(state.todo)
        self.assertEqual(state.todo.revision, 1)
        self.assertEqual([item.status for item in state.todo.items],
                         ["completed", "in_progress", "pending"])
        state_path = self._state_path()
        self.assertTrue(state_path.exists())
        data = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(data["revision"], 1)
        self.assertEqual([item["text"] for item in data["items"]],
                         ["Inspect parser", "Fix validation", "Run tests"])
        self.assertFalse(Path(self.temp_dir, "global_plan", "current", "plan.md").exists())

    def test_metadata_exposes_todo_state(self):
        state = AgentRuntimeState(session_id="test-session")

        result = execute_tool_result(
            "update_todo",
            {"items": _items()},
            runtime_state=state,
            agent_name="main_agent",
        )

        self.assertEqual(result.status, "success")
        todo_state = result.metadata.get("todo_state")
        self.assertIsNotNone(todo_state)
        self.assertEqual(todo_state["items"][1]["status"], "in_progress")

    def test_replacement_increments_revision_and_assigns_stable_ids(self):
        state = AgentRuntimeState(session_id="test-session")

        first = execute_tool_result(
            "update_todo",
            {"items": [
                {"text": "one", "status": "pending"},
                {"text": "two", "status": "pending"},
            ]},
            runtime_state=state,
            agent_name="main_agent",
        )
        second = execute_tool_result(
            "update_todo",
            {"items": [
                {"id": "custom", "text": "one", "status": "completed"},
                {"text": "three", "status": "in_progress"},
            ]},
            runtime_state=state,
            agent_name="main_agent",
        )

        self.assertEqual(first.metadata["todo_state"]["items"][0]["id"], "todo_1")
        self.assertEqual(second.metadata["todo_state"]["revision"], 2)
        ids = [item["id"] for item in second.metadata["todo_state"]["items"]]
        self.assertEqual(ids, ["custom", "todo_3"])

    def test_empty_items_are_rejected(self):
        state = AgentRuntimeState(session_id="test-session")
        result = execute_tool_result(
            "update_todo",
            {"items": []},
            runtime_state=state,
            agent_name="main_agent",
        )

        self.assertEqual(result.status, "failed")
        self.assertIn("at least 1", result.output)
        self.assertFalse(self._state_path().exists())
        self.assertIsNone(state.todo)

    def test_unknown_status_is_rejected(self):
        state = AgentRuntimeState(session_id="test-session")
        result = execute_tool_result(
            "update_todo",
            {"items": [{"text": "x", "status": "doing"}]},
            runtime_state=state,
            agent_name="main_agent",
        )

        self.assertEqual(result.status, "failed")
        self.assertIn("status must be one of", result.output)
        self.assertFalse(self._state_path().exists())

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_todo_items(
                [
                    {"id": "same", "text": "a"},
                    {"id": "same", "text": "b"},
                ],
                next_seq=0,
            )

    def test_render_uses_status_markers(self):
        state = AgentRuntimeState(session_id="test-session")
        execute_tool(
            "update_todo",
            {"items": [
                {"text": "done", "status": "completed"},
                {"text": "active", "status": "in_progress"},
                {"text": "later", "status": "pending"},
                {"text": "dropped", "status": "cancelled"},
            ]},
            runtime_state=state,
            agent_name="main_agent",
        )

        rendered = state.todo.render()
        self.assertIn("[done] done", rendered)
        self.assertIn("[in progress] active", rendered)
        self.assertIn("[pending] later", rendered)
        self.assertIn("[cancelled] dropped", rendered)

    def test_atomic_replace_failure_keeps_previous_state_json(self):
        state = AgentRuntimeState(session_id="test-session")
        state_path = self._state_path()
        state_path.parent.mkdir(parents=True)
        state_path.write_text('{"revision": 1}\n', encoding="utf-8")

        with patch("harness_code_agent.runtime.builtins.todo.os.replace", side_effect=OSError("locked")):
            result = execute_tool(
                "update_todo",
                {"items": _items()},
                runtime_state=state,
                agent_name="main_agent",
            )

        self.assertIn("[error]", result)
        self.assertEqual(json.loads(state_path.read_text(encoding="utf-8")), {"revision": 1})
        self.assertEqual(list(state_path.parent.glob("state.json.tmp.*")), [])


class TodoListRenderTests(unittest.TestCase):
    def test_empty_list_renders_as_empty_string(self):
        self.assertEqual(TodoList().render(), "")


if __name__ == "__main__":
    unittest.main()
