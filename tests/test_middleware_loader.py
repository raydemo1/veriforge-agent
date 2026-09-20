"""Tests for the user middleware loader (~/.harness/middlewares.json)."""
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _install_fake_openai_module() -> None:
    if "openai" in sys.modules:
        return
    openai = types.ModuleType("openai")

    class OpenAI:
        def __init__(self, *args, **kwargs):
            pass

    openai.OpenAI = OpenAI
    sys.modules["openai"] = openai


_install_fake_openai_module()

from harness_code_agent.agent.conversation import Agent, AgentConversation
from harness_code_agent.runtime.middleware import AgentMiddleware
from harness_code_agent.runtime.middleware.loader import (
    MiddlewareConfigError,
    load_user_middlewares,
)
from harness_code_agent.runtime.permissions import PermissionPolicy
from harness_code_agent.runtime.tool_context import ToolContext
from harness_code_agent.runtime.tool_result import ToolResult
from harness_code_agent.sessions.events import EventBus
from harness_code_agent.workspace.service import WorkspaceService


_MIDDLEWARE_SOURCE = '''
from harness_code_agent.runtime.middleware import AgentMiddleware


class FirstMiddleware(AgentMiddleware):
    pass


class SecondMiddleware(AgentMiddleware):
    pass


class PlainClass:
    pass
'''


class UserMiddlewareLoaderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.module_path = self.root / "middlewares.py"
        self.module_path.write_text(_MIDDLEWARE_SOURCE, encoding="utf-8")
        self.config_path = self.root / "middlewares.json"

    def tearDown(self):
        self._tmp.cleanup()

    def _write_config(self, data) -> None:
        self.config_path.write_text(json.dumps(data), encoding="utf-8")

    def test_missing_config_returns_empty(self):
        self.assertEqual(load_user_middlewares(self.config_path), [])

    def test_loads_middlewares_in_array_order(self):
        self._write_config({"middlewares": [
            {"path": str(self.module_path), "class": "FirstMiddleware", "enabled": True},
            {"path": str(self.module_path), "class": "SecondMiddleware"},
        ]})

        middlewares = load_user_middlewares(self.config_path)

        self.assertEqual(
            [type(m).__name__ for m in middlewares],
            ["FirstMiddleware", "SecondMiddleware"],
        )

    def test_disabled_middleware_is_skipped(self):
        self._write_config({"middlewares": [
            {"path": str(self.module_path), "class": "FirstMiddleware", "enabled": False},
            {"path": str(self.module_path), "class": "SecondMiddleware", "enabled": True},
        ]})

        middlewares = load_user_middlewares(self.config_path)

        self.assertEqual([type(m).__name__ for m in middlewares], ["SecondMiddleware"])

    def test_missing_file_raises_clear_error(self):
        self._write_config({"middlewares": [
            {"path": str(self.root / "missing.py"), "class": "FirstMiddleware"},
        ]})

        with self.assertRaises(MiddlewareConfigError) as ctx:
            load_user_middlewares(self.config_path)
        self.assertIn("Failed to load middleware FirstMiddleware", str(ctx.exception))
        self.assertIn("file not found", str(ctx.exception))

    def test_missing_class_raises_clear_error(self):
        self._write_config({"middlewares": [
            {"path": str(self.module_path), "class": "NoSuchMiddleware"},
        ]})

        with self.assertRaises(MiddlewareConfigError) as ctx:
            load_user_middlewares(self.config_path)
        self.assertIn("class not found", str(ctx.exception))
        self.assertIn("NoSuchMiddleware", str(ctx.exception))

    def test_non_middleware_class_raises_clear_error(self):
        self._write_config({"middlewares": [
            {"path": str(self.module_path), "class": "PlainClass"},
        ]})

        with self.assertRaises(MiddlewareConfigError) as ctx:
            load_user_middlewares(self.config_path)
        self.assertIn("is not an AgentMiddleware subclass", str(ctx.exception))

    def test_invalid_json_raises_clear_error(self):
        self.config_path.write_text("{ not valid json", encoding="utf-8")

        with self.assertRaises(MiddlewareConfigError) as ctx:
            load_user_middlewares(self.config_path)
        self.assertIn("Failed to parse middleware config", str(ctx.exception))

    def test_entry_missing_path_or_class_raises(self):
        self._write_config({"middlewares": [{"enabled": True}]})

        with self.assertRaises(MiddlewareConfigError):
            load_user_middlewares(self.config_path)


class RecordingMiddleware(AgentMiddleware):
    """Records every lifecycle hook the agent invokes, in invocation order."""

    def __init__(self):
        self.calls: list[str] = []

    def _record(self, hook: str) -> None:
        self.calls.append(hook)

    def on_conversation_start(self, messages, runtime_state=None, agent_name=None):
        self._record("on_conversation_start")
        return []

    def begin_turn(self, task, messages, runtime_state=None, agent_name=None):
        self._record("begin_turn")

    def per_iteration(self, iteration, messages, runtime_state=None, agent_name=None):
        self._record("per_iteration")

    def before_tool(self, tool_name, tool_args, messages, runtime_state=None,
                    agent_name=None, permission_decision=None):
        self._record("before_tool")
        return None

    def on_tool_allowed(self, tool_name, tool_args, messages, runtime_state=None,
                        agent_name=None):
        self._record("on_tool_allowed")

    def post_tool(self, tool_name, tool_args, result, messages, runtime_state=None,
                  agent_name=None):
        self._record("post_tool")
        return None

    def on_tool_failure(self, failure, messages, runtime_state=None, agent_name=None):
        self._record("on_tool_failure")
        return None

    def on_context_compacted(self, messages, runtime_state=None, agent_name=None,
                             phase=None):
        self._record("on_context_compacted")
        return []

    def pre_exit(self, messages, runtime_state=None, agent_name=None):
        self._record("pre_exit")
        return None

    def on_conversation_close(self, messages, runtime_state=None, agent_name=None):
        self._record("on_conversation_close")


def _echo_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "echo",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            },
        },
    }


class _ScriptedCompletions:
    def __init__(self, responses):
        self._responses = responses
        self.calls = 0

    def create(self, **kwargs):
        response = self._responses[self.calls]
        self.calls += 1
        return response


class MiddlewareLifecycleContractTests(unittest.TestCase):
    def test_every_declared_hook_is_invoked_on_happy_path(self):
        from harness_code_agent.runtime.tool_registry import ToolRegistry

        registry = ToolRegistry()

        def echo():
            return ToolResult(tool="echo", status="success", output="ok")

        registry.register(_echo_schema(), echo, permission="read")

        recorder = RecordingMiddleware()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tool_context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
                permission_policy=PermissionPolicy(mode="danger-full-access"),
                event_bus=EventBus(),
                tool_registry=registry,
            )
            tool_response = SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[SimpleNamespace(
                            id="tc_1",
                            function=SimpleNamespace(name="echo", arguments="{}"),
                        )],
                    ),
                    finish_reason="tool_calls",
                )],
                usage=None,
            )
            stop_response = SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="done", tool_calls=None),
                    finish_reason="stop",
                )],
                usage=None,
            )
            fake_client = SimpleNamespace(
                chat=SimpleNamespace(completions=_ScriptedCompletions(
                    [tool_response, stop_response]
                ))
            )
            with patch(
                "harness_code_agent.agent.conversation.get_client",
                return_value=fake_client,
            ):
                agent = Agent(
                    "main_agent",
                    "system",
                    use_tools=True,
                    tool_schemas=registry.schemas(),
                    middlewares=[recorder],
                    tool_context=tool_context,
                )
                conversation = AgentConversation(agent)
                conversation.add_user_turn("task")
                conversation.run_until_idle()
                conversation.close()

        self.assertEqual(recorder.calls, [
            "on_conversation_start",
            "begin_turn",
            "per_iteration",
            "before_tool",
            "on_tool_allowed",
            "post_tool",
            "per_iteration",
            "pre_exit",
            "on_conversation_close",
        ])


class UserMiddlewareSessionCachingTests(unittest.TestCase):
    """User middlewares load once per session and survive profile switches."""

    def setUp(self):
        self.env_patch = patch.dict(os.environ, {
            "HARNESS_MEMORY_GENERATION_DISABLED": "1",
        })
        self.env_patch.start()
        self._tmp = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._tmp.name)

    def tearDown(self):
        if getattr(self, "interactive", None) is not None:
            self.interactive.close()
        self.env_patch.stop()
        self._tmp.cleanup()

    def test_profile_switch_reuses_same_middleware_instances(self):
        from harness_code_agent.core.interactive import InteractiveSession

        recorders = [RecordingMiddleware(), RecordingMiddleware()]
        with patch(
            "harness_code_agent.core.interactive.load_user_middlewares",
            return_value=recorders,
        ) as loader:
            self.interactive = InteractiveSession(
                cwd=self.temp_dir,
                enable_turn_summary=False,
            )
            first_agent = self.interactive.agent
            self.assertTrue(
                all(any(m is r for m in first_agent.middlewares) for r in recorders)
            )

            self.interactive._switch_profile("plan")

            second_agent = self.interactive.agent
            self.assertIsNot(second_agent, first_agent)
            self.assertTrue(
                all(any(m is r for m in second_agent.middlewares) for r in recorders)
            )
            loader.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
