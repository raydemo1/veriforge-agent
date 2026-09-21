import importlib
import json
import os
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from harness_code_agent.runtime.builtins.registry import (
    BUILTIN_TOOL_REGISTRY,
    TOOL_SCHEMAS,
)
from harness_code_agent.runtime.builtins.schemas import BROWSER_TOOL_SCHEMAS
from harness_code_agent.runtime.builtins.shell import (
    list_shell_jobs,
    read_shell_output,
    run_bash,
    stop_shell_job,
)
from harness_code_agent.runtime.tool_registry import (
    ToolRegistry,
    tool_schemas_for_profile,
)
from harness_code_agent.runtime.tool_result import ToolResult
from harness_code_agent.runtime.tool_runner import execute_tool


def _result(text: str, *, status: str | None = None) -> ToolResult:
    """Build a ToolResult from the legacy text conventions used in these tests."""
    if status is None:
        status = "failed" if text.startswith(("[error]", "[blocked]")) else "success"
    metadata = {"status_source": "permission"} if text.startswith("[blocked]") else {}
    error = text.removeprefix("[error] ").removeprefix("[blocked] ") if status == "failed" else None
    return ToolResult(tool="run_bash", status=status, output=text, error=error, metadata=metadata)



class ProductRuntimeTests(unittest.TestCase):
    def test_deepseek_model_profiles_use_intensity_defaults(self):
        from harness_code_agent import config

        try:
            with (
                patch.dict(os.environ, {
                    "OPENAI_BASE_URL": "https://api.deepseek.com",
                    "HARNESS_MODEL_INTENSITY": "hard",
                }, clear=True),
                patch("pathlib.Path.exists", return_value=False),
            ):
                importlib.reload(config)

            fast = config.resolve_model_profile("fast")
            normal = config.resolve_model_profile("normal")
            hard = config.resolve_model_profile("hard")
            max_profile = config.resolve_model_profile("max")

            self.assertEqual(config.MODEL, "deepseek-v4-pro")
            self.assertEqual(config.MODEL_INTENSITY, "hard")
            self.assertEqual((fast.model, fast.thinking, fast.reasoning_effort), ("deepseek-v4-flash", False, None))
            self.assertEqual((normal.model, normal.thinking, normal.reasoning_effort), ("deepseek-v4-flash", True, "high"))
            self.assertEqual((hard.model, hard.thinking, hard.reasoning_effort), ("deepseek-v4-pro", True, "high"))
            self.assertEqual((max_profile.model, max_profile.thinking, max_profile.reasoning_effort), ("deepseek-v4-pro", True, "max"))
        finally:
            importlib.reload(config)

    def test_model_intensity_and_profile_model_overrides(self):
        from harness_code_agent import config

        try:
            with (
                patch.dict(os.environ, {
                    "OPENAI_BASE_URL": "https://api.deepseek.com",
                    "HARNESS_MODEL_INTENSITY": "max",
                    "HARNESS_MODEL_FAST": "custom-fast",
                    "HARNESS_MODEL_MAX": "custom-max",
                }, clear=True),
                patch("pathlib.Path.exists", return_value=False),
            ):
                importlib.reload(config)

            self.assertEqual(config.MODEL_INTENSITY, "max")
            self.assertEqual(config.MODEL, "custom-max")
            self.assertEqual(config.resolve_model_profile("fast").model, "custom-fast")
            self.assertEqual(config.resolve_model_profile("max").model, "custom-max")
            self.assertEqual(config.resolve_model_profile("max").reasoning_effort, "max")
        finally:
            importlib.reload(config)

    def test_runtime_model_override_applies_to_non_fast_lanes(self):
        from harness_code_agent import config

        try:
            with (
                patch.dict(os.environ, {
                    "OPENAI_BASE_URL": "https://api.deepseek.com",
                    "HARNESS_MODEL_INTENSITY": "hard",
                }, clear=True),
                patch("pathlib.Path.exists", return_value=False),
            ):
                importlib.reload(config)

            config.set_model_override(model="deepseek-v4-flash-vision-exp", reasoning_effort="max")
            profile = config.resolve_model_profile("normal")
            self.assertEqual((profile.model, profile.thinking, profile.reasoning_effort), ("deepseek-v4-flash-vision-exp", True, "max"))
            fast = config.resolve_model_profile("fast")
            self.assertEqual((fast.model, fast.thinking, fast.reasoning_effort), ("deepseek-v4-flash", False, None))

            config.set_model_override(reasoning_effort="low")
            profile = config.resolve_model_profile("hard")
            self.assertEqual((profile.model, profile.reasoning_effort), ("deepseek-v4-pro", "low"))

            with self.assertRaises(ValueError):
                config.set_model_override(model="gpt-4o")
            with self.assertRaises(ValueError):
                config.set_model_override(reasoning_effort="medium")

            config.set_model_override()
            profile = config.resolve_model_profile("hard")
            self.assertEqual((profile.model, profile.reasoning_effort), ("deepseek-v4-pro", "high"))
        finally:
            importlib.reload(config)

    def test_deepseek_reasoning_content_round_trips(self):
        from harness_code_agent.agent.conversation import (
            _assistant_message_from_response,
        )

        cases = [
            ("direct_attr", SimpleNamespace(
                content=None,
                reasoning_content="think carefully",
                tool_calls=[
                    SimpleNamespace(
                        id="call_1",
                        function=SimpleNamespace(name="read_file", arguments='{"path":"README.md"}'),
                    ),
                ],
            ), "think carefully"),
            ("model_extra", SimpleNamespace(
                content=None,
                model_extra={"reasoning_content": "provider extra thinking"},
                tool_calls=[],
            ), "provider extra thinking"),
        ]

        for label, msg, expected in cases:
            with self.subTest(source=label):
                with (
                    patch("harness_code_agent.agent.conversation.config.BASE_URL", "https://api.deepseek.com"),
                    patch("harness_code_agent.agent.conversation.config.MODEL", "deepseek-v4-flash"),
                ):
                    assistant_msg = _assistant_message_from_response(msg)

                self.assertEqual(assistant_msg["reasoning_content"], expected)
                if label == "direct_attr":
                    self.assertEqual(assistant_msg["tool_calls"][0]["function"]["name"], "read_file")

    def test_non_deepseek_assistant_message_omits_reasoning_content(self):
        from harness_code_agent.agent.conversation import (
            _assistant_message_from_response,
        )

        msg = SimpleNamespace(content="ok", reasoning_content="hidden", tool_calls=None)

        with (
            patch("harness_code_agent.agent.conversation.config.BASE_URL", "https://api.openai.com/v1"),
            patch("harness_code_agent.agent.conversation.config.MODEL", "gpt-4o"),
        ):
            assistant_msg = _assistant_message_from_response(msg)

        self.assertNotIn("reasoning_content", assistant_msg)

    def test_provider_auto_detection_distinguishes_openai_deepseek_and_compatible(self):
        from harness_code_agent.agent.providers import resolve_provider_name

        self.assertEqual(
            resolve_provider_name(provider="auto", base_url="https://api.openai.com/v1", model="gpt-4o"),
            "openai",
        )
        self.assertEqual(
            resolve_provider_name(provider="auto", base_url="https://api.deepseek.com", model="deepseek-chat"),
            "deepseek",
        )
        self.assertEqual(
            resolve_provider_name(provider="auto", base_url="https://example.invalid/v1", model="custom"),
            "openai-compatible",
        )
        self.assertEqual(
            resolve_provider_name(provider="auto", base_url="https://example.invalid/v1", model="my-deepseek-fork-v1"),
            "openai-compatible",
        )

    def test_provider_clients_are_independent_per_owner(self):
        from harness_code_agent.agent import providers

        created = []

        def fake_openai(**kwargs):
            client = SimpleNamespace(kwargs=kwargs, closed=False)

            def close():
                client.closed = True

            client.close = close
            created.append(client)
            return client

        with (
            patch("harness_code_agent.agent.providers.OpenAI", side_effect=fake_openai),
            patch.object(providers.config, "API_KEY", "key-a"),
            patch.object(providers.config, "BASE_URL", "https://one.example/v1"),
        ):
            first = providers.get_client()
            second = providers.get_client()

        self.assertIsNot(first, second)
        self.assertEqual(len(created), 2)
        self.assertEqual(created[0].kwargs["base_url"], "https://one.example/v1")
        self.assertEqual(created[1].kwargs["base_url"], "https://one.example/v1")

    def test_provider_streaming_normalizes_content_reasoning_and_tool_calls(self):
        from harness_code_agent.agent.providers import ProviderAdapter

        def chunk(delta, finish_reason=None):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=delta,
                        finish_reason=finish_reason,
                    )
                ]
            )

        deltas = []
        chunks = [
            chunk(SimpleNamespace(content="hel")),
            chunk(SimpleNamespace(content="lo", reasoning_content="think ")),
            chunk(
                SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id="call_1",
                            type="function",
                            function=SimpleNamespace(name="read_file", arguments='{"pa'),
                        )
                    ]
                )
            ),
            chunk(
                SimpleNamespace(
                    reasoning_content="carefully",
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            function=SimpleNamespace(arguments='th":"README.md"}'),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            ),
        ]

        result = ProviderAdapter("deepseek").assistant_message_from_stream(chunks, on_text_delta=deltas.append)

        self.assertEqual(deltas, ["hel", "lo"])
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(result.assistant_message["content"], "hello")
        self.assertEqual(result.assistant_message["reasoning_content"], "think carefully")
        self.assertEqual(result.assistant_message["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(result.assistant_message["tool_calls"][0]["function"]["arguments"], '{"path":"README.md"}')

    def test_provider_chat_kwargs_strip_response_only_reasoning_content(self):
        from harness_code_agent.agent.providers import ProviderAdapter

        messages = [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "provider-only thinking",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            }
        ]

        kwargs = ProviderAdapter("deepseek").chat_kwargs(
            model="deepseek-v4-pro",
            messages=messages,
            max_tokens=10,
        )

        self.assertNotIn("reasoning_content", kwargs["messages"][0])
        self.assertEqual(messages[0]["reasoning_content"], "provider-only thinking")

    def test_provider_streaming_checks_cancellation_between_chunks(self):
        from harness_code_agent.agent.cancellation import (
            CancellationToken,
            CancelledError,
        )
        from harness_code_agent.agent.providers import ProviderAdapter

        def chunk(text):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=text),
                        finish_reason=None,
                    )
                ]
            )

        token = CancellationToken()
        deltas = []

        def chunks():
            yield chunk("hel")
            token.cancel()
            yield chunk("lo")

        with self.assertRaises(CancelledError):
            ProviderAdapter("openai-compatible").assistant_message_from_stream(
                chunks(),
                on_text_delta=deltas.append,
                cancellation_token=token,
            )

        self.assertEqual(deltas, ["hel"])

    def test_cancelling_while_waiting_for_stream_closes_the_active_client(self):
        from harness_code_agent.agent.cancellation import (
            CancellationToken,
            CancelledError,
        )
        from harness_code_agent.agent.conversation import Agent, AgentConversation

        started = threading.Event()
        released = threading.Event()

        class BlockingCompletions:
            def create(self, **kwargs):
                started.set()
                released.wait(timeout=2)
                raise RuntimeError("request closed")

        class BlockingClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=BlockingCompletions())
                self.closed = False

            def close(self):
                self.closed = True
                released.set()

        client = BlockingClient()
        with patch("harness_code_agent.agent.conversation.get_client", return_value=client):
            conversation = AgentConversation(Agent("test", "system", use_tools=False, stream_callback=lambda _: None))
        token = CancellationToken()
        errors = []

        def request():
            try:
                conversation.llm.request_assistant_message(
                    conversation.provider.chat_kwargs(model="m", messages=[], max_tokens=10),
                    cancellation_token=token,
                )
            except CancelledError as exc:
                errors.append(exc)

        worker = threading.Thread(target=request)
        worker.start()
        self.assertTrue(started.wait(timeout=1))
        token.cancel()
        worker.join(timeout=1)
        if worker.is_alive():
            released.set()
            worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertTrue(client.closed)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], CancelledError)

    def test_streaming_request_falls_back_to_non_stream_before_first_chunk(self):
        from harness_code_agent.agent.conversation import Agent, AgentConversation

        class FakeCompletions:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                if kwargs.get("stream"):
                    raise RuntimeError("stream unavailable")
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="fallback", tool_calls=None),
                            finish_reason="stop",
                        )
                    ]
                )

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        fake_client = FakeClient()
        with patch("harness_code_agent.agent.conversation.get_client", return_value=fake_client):
            conversation = AgentConversation(Agent("test", "system", use_tools=False, stream_callback=lambda _: None))

        with patch.object(conversation.trace, "error") as trace_error:
            completion = conversation.llm.request_assistant_message(
                conversation.provider.chat_kwargs(model="m", messages=[], max_tokens=10)
            )

        self.assertEqual(completion[0]["content"], "fallback")
        self.assertTrue(fake_client.chat.completions.calls[0]["stream"])
        self.assertNotIn("stream", fake_client.chat.completions.calls[1])
        trace_error.assert_called_once()
        self.assertEqual(trace_error.call_args.args[0], "stream_fallback")
        self.assertIn("stream unavailable", trace_error.call_args.args[1])

    def test_streaming_request_collects_text_deltas_without_non_stream_fallback(self):
        from harness_code_agent.agent.conversation import Agent, AgentConversation

        def chunk(text, finish_reason=None):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=text),
                        finish_reason=finish_reason,
                    )
                ]
            )

        class FakeCompletions:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return [chunk("hel"), chunk("lo", finish_reason="stop")]

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        deltas = []
        fake_client = FakeClient()
        with patch("harness_code_agent.agent.conversation.get_client", return_value=fake_client):
            conversation = AgentConversation(Agent("test", "system", use_tools=False, stream_callback=deltas.append))
            completion = conversation.llm.request_assistant_message(
                conversation.provider.chat_kwargs(model="m", messages=[], max_tokens=10)
            )

        self.assertEqual(completion[0]["content"], "hello")
        self.assertEqual(completion[1], "stop")
        self.assertEqual(deltas, ["hel", "lo"])
        self.assertTrue(conversation.last_run_streamed_text)
        self.assertEqual(len(fake_client.chat.completions.calls), 1)

    def test_tool_enabled_agent_builds_chat_kwargs_once(self):
        from harness_code_agent.agent.conversation import Agent, AgentConversation
        from harness_code_agent.agent.providers import ProviderAdapter

        class FakeCompletions:
            def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="done", tool_calls=None),
                            finish_reason="stop",
                        )
                    ]
                )

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        class CountingProvider:
            def __init__(self):
                self.calls = []
                self.delegate = ProviderAdapter("openai")

            @property
            def supports_prompt_cache_key(self):
                return self.delegate.supports_prompt_cache_key

            def chat_kwargs(self, **kwargs):
                chat_kwargs = self.delegate.chat_kwargs(**kwargs)
                self.calls.append(chat_kwargs)
                return chat_kwargs

            def assistant_message_from_response(self, msg):
                return self.delegate.assistant_message_from_response(msg)

        schema = [{"type": "function", "function": {"name": "read_file"}}]
        provider = CountingProvider()
        with patch("harness_code_agent.agent.conversation.get_client", return_value=FakeClient()):
            conversation = AgentConversation(Agent("test", "system", use_tools=True, tool_schemas=schema))
        conversation.provider = provider

        with (
            patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=1),
        ):
            result = conversation.run_until_idle()

        self.assertEqual(result, "done")
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0]["tools"], schema)
        self.assertEqual(provider.calls[0]["tool_choice"], "auto")

    def test_agent_loop_uses_configured_model_intensity_profile(self):
        from harness_code_agent.agent.conversation import Agent, AgentConversation
        from harness_code_agent.agent.providers import ProviderAdapter

        class FakeCompletions:
            def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="done", tool_calls=None),
                            finish_reason="stop",
                        )
                    ]
                )

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        class CapturingProvider:
            def __init__(self):
                self.calls = []
                self.delegate = ProviderAdapter("deepseek")

            @property
            def supports_prompt_cache_key(self):
                return self.delegate.supports_prompt_cache_key

            def chat_kwargs(self, **kwargs):
                chat_kwargs = self.delegate.chat_kwargs(**kwargs)
                self.calls.append(chat_kwargs)
                return chat_kwargs

            def assistant_message_from_response(self, msg):
                return self.delegate.assistant_message_from_response(msg)

        provider = CapturingProvider()
        with patch("harness_code_agent.agent.conversation.get_client", return_value=FakeClient()):
            conversation = AgentConversation(Agent("test", "system", use_tools=False))
        conversation.provider = provider

        with (
            patch("harness_code_agent.agent.conversation.config.BASE_URL", "https://api.deepseek.com"),
            patch("harness_code_agent.agent.conversation.config.MODEL_INTENSITY", "hard"),
            patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=1),
        ):
            conversation.run_until_idle()

        self.assertEqual(provider.calls[0]["model"], "deepseek-v4-pro")
        self.assertEqual(provider.calls[0]["reasoning_effort"], "high")
        self.assertEqual(provider.calls[0]["extra_body"], {"thinking": {"type": "enabled"}})

    def test_llm_call_simple_uses_fast_profile(self):
        from harness_code_agent.agent import conversation as loop

        class FakeCompletions:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))]
                )

        completions = FakeCompletions()
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        with (
            patch("harness_code_agent.agent.providers.get_client", return_value=fake_client),
            patch.object(loop.config, "BASE_URL", "https://api.deepseek.com"),
        ):
            result = loop.llm_call_simple([{"role": "user", "content": "summarize"}])

        self.assertEqual(result, "summary")
        self.assertEqual(completions.calls[0]["model"], "deepseek-v4-flash")
        self.assertNotIn("reasoning_effort", completions.calls[0])
        self.assertEqual(completions.calls[0]["extra_body"], {"thinking": {"type": "disabled"}})

    def test_local_turn_router_conservatively_routes_product_profiles(self):
        from harness_code_agent.profiles import router

        cases = [
            ("你是谁", "general", "general", "stay", "normal"),
            ("帮我修复这个 bug 并跑测试", "general", "coding-agent", "switch_profile", "normal"),
            ("帮我 review 这段代码有没有问题", "general", "review", "switch_profile", "normal"),
            ("先给我一个实现方案，不要改代码", "general", "plan", "switch_profile", "normal"),
            ("做一个好看的 todo 网页", "general", "app-builder", "switch_profile", "normal"),
            ("你是谁", "coding-agent", "coding-agent", "direct_answer", "direct_answer"),
            ("审阅这个 PR 的改动，然后修掉问题并跑测试", "general", "coding-agent", "switch_profile", "normal"),
            ("只审查这个分支，不要修改任何文件", "general", "review", "switch_profile", "normal"),
            ("检查为什么启动慢并直接优化", "general", "coding-agent", "switch_profile", "normal"),
            ("解释一下这个报错是什么意思，不要改代码", "general", "general", "stay", "normal"),
            ("先设计方案，等我确认后再实现", "general", "plan", "switch_profile", "normal"),
            ("设计并实现一个新的缓存模块", "general", "coding-agent", "switch_profile", "normal"),
            ("做一个漂亮的 TUI 界面", "general", "coding-agent", "switch_profile", "normal"),
            ("检查当前 Ruff 修改，没修完的一并修掉", "general", "coding-agent", "switch_profile", "normal"),
            ("帮我看下代码有没有安全问题，发现问题直接修复", "general", "coding-agent", "switch_profile", "normal"),
            ("review the PR, fix the findings, and run tests", "general", "coding-agent", "switch_profile", "normal"),
            ("review this patch only; do not edit files", "general", "review", "switch_profile", "normal"),
            ("build a polished terminal UI", "general", "coding-agent", "switch_profile", "normal"),
            ("build a polished React dashboard", "general", "app-builder", "switch_profile", "normal"),
            ("plan the migration but do not implement it", "general", "plan", "switch_profile", "normal"),
            ("不要修复，只解释这个异常", "general", "general", "stay", "normal"),
        ]

        for prompt, current, expected, expected_action, expected_turn_mode in cases:
            with self.subTest(prompt=prompt, current=current):
                decision = router.route_profile_for_turn(prompt, current_profile=current)
                self.assertEqual(decision.profile_name, expected)
                self.assertEqual(decision.action, expected_action)
                self.assertEqual(decision.turn_mode, expected_turn_mode)
                self.assertEqual(decision.source, "local")
                self.assertFalse(decision.fallback_used)

    def test_high_precision_local_route_hops_between_specialized_profiles(self):
        from harness_code_agent.profiles import router

        decision = router.route_profile_for_turn(
            "帮我 review 这段代码",
            current_profile="coding-agent",
            llm_classifier=lambda **_: (_ for _ in ()).throw(
                AssertionError("high precision review route should stay local")
            ),
        )

        self.assertEqual(decision.profile_name, "review")
        self.assertEqual(decision.action, "switch_profile")
        self.assertFalse(decision.fallback_used)
        self.assertEqual(decision.source, "local")
        self.assertFalse(decision.llm_called)

    def test_permission_policy_does_not_block_format_substrings_in_safe_commands(self):
        from harness_code_agent.runtime.permissions import PermissionPolicy

        policy = PermissionPolicy(mode="workspace-write")

        safe_commands = [
            "ruff check --diff --no-fix --output-format=text",
            "git log --format=%s -1",
            "git show --format=fuller HEAD",
        ]

        for command in safe_commands:
            with self.subTest(command=command):
                decision = policy.decide_tool_call("run_bash", {"command": command})
                self.assertNotEqual(decision.risk, "shell_blocked")

    def test_turn_summary_event_serializes_payload(self):
        from harness_code_agent.sessions.events import TurnSummaryEvent

        event = TurnSummaryEvent(
            turn=2,
            summary="- changed app.py",
            duration_seconds=12.5,
            tool_counts={"read_file": 1},
            changed_files=["app.py"],
            checkpoint="checkpoint created: abc",
            generated_by={"intensity": "fast", "model": "custom-fast"},
        ).to_event()

        self.assertEqual(event.type, "turn_summary")
        self.assertTrue(event.payload["long_task"])
        self.assertTrue(event.payload["fold_details"])
        self.assertEqual(event.payload["turn"], 2)
        self.assertEqual(event.payload["tool_counts"], {"read_file": 1})
        self.assertEqual(event.payload["changed_files"], ["app.py"])
        self.assertEqual(event.payload["generated_by"]["intensity"], "fast")

    def test_turn_summary_long_task_detection_rules(self):
        from harness_code_agent.sessions.turn_summary import should_summarize_turn

        simple = [{"type": "assistant_message", "payload": {"text": "hello"}}]
        three_tools = [
            {"type": "tool_result", "payload": {"tool": "read_file"}},
            {"type": "tool_result", "payload": {"tool": "read_file"}},
            {"type": "tool_result", "payload": {"tool": "read_file"}},
        ]

        self.assertFalse(should_summarize_turn(simple, profile_name="coding-agent", duration_seconds=1))
        self.assertFalse(should_summarize_turn(three_tools, profile_name="plan", duration_seconds=1))
        self.assertTrue(should_summarize_turn(three_tools, profile_name="coding-agent", duration_seconds=1))
        self.assertTrue(should_summarize_turn([{"type": "file_change", "payload": {"path": "app.py"}}], profile_name="coding-agent", duration_seconds=1))
        self.assertTrue(should_summarize_turn([{"type": "tool_result", "payload": {"tool": "run_bash"}}], profile_name="coding-agent", duration_seconds=1))
        self.assertTrue(should_summarize_turn([{"type": "agent_fallback", "payload": {"reason": "tool_call_budget_exceeded"}}], profile_name="coding-agent", duration_seconds=1))
        self.assertTrue(should_summarize_turn(simple, profile_name="coding-agent", duration_seconds=45))

    def test_generate_turn_summary_uses_configured_fast_profile(self):
        from harness_code_agent import config
        from harness_code_agent.sessions import turn_summary

        calls = []

        def fake_create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="- summary from fast"))]
            )

        profile = config.ModelProfile(
            provider="deepseek",
            model="custom-fast",
            thinking=False,
            reasoning_effort=None,
        )
        with patch.object(turn_summary.config, "resolve_model_profile", return_value=profile):
            result = turn_summary.generate_turn_summary(
                [{"type": "tool_result", "payload": {"tool": "read_file"}}],
                user_prompt="fix",
                assistant_text="done",
                checkpoint="",
                llm_create=fake_create,
            )

        self.assertEqual(result.summary, "- summary from fast")
        self.assertEqual(result.generated_by["model"], "custom-fast")
        self.assertEqual(calls[0]["model"], "custom-fast")
        self.assertEqual(calls[0]["extra_body"], {"thinking": {"type": "disabled"}})

    def test_generate_turn_summary_falls_back_when_llm_fails(self):
        from harness_code_agent.sessions import turn_summary

        def broken_create(**kwargs):
            raise RuntimeError("nope")

        result = turn_summary.generate_turn_summary(
            [
                {"type": "tool_result", "payload": {"tool": "write_file"}},
                {"type": "file_change", "payload": {"path": "app.py"}},
            ],
            user_prompt="fix app",
            assistant_text="updated app.py",
            checkpoint="checkpoint created: abc",
            llm_create=broken_create,
        )

        self.assertIn("fix app", result.summary)
        self.assertIn("app.py", result.summary)
        self.assertEqual(result.tool_counts, {"write_file": 1})

    def test_openai_provider_accepts_prompt_cache_key_and_stream_usage_options(self):
        from harness_code_agent.agent.providers import ProviderAdapter

        kwargs = ProviderAdapter("openai").chat_kwargs(
            model="m",
            messages=[],
            max_tokens=10,
            prompt_cache_key="cache-key",
            stream_options={"include_usage": True},
        )

        self.assertEqual(kwargs["prompt_cache_key"], "cache-key")
        self.assertEqual(kwargs["stream_options"], {"include_usage": True})

    def test_provider_adapter_maps_model_profile_kwargs(self):
        from harness_code_agent import config
        from harness_code_agent.agent.providers import ProviderAdapter

        deepseek = ProviderAdapter("deepseek").chat_kwargs(
            profile=config.ModelProfile(
                provider="deepseek",
                model="deepseek-v4-pro",
                thinking=True,
                reasoning_effort="high",
            ),
            messages=[],
            max_tokens=10,
        )
        openai = ProviderAdapter("openai").chat_kwargs(
            profile=config.ModelProfile(
                provider="openai",
                model="gpt-4o",
                thinking=True,
                reasoning_effort="high",
            ),
            messages=[],
            max_tokens=10,
        )

        self.assertEqual(deepseek["model"], "deepseek-v4-pro")
        self.assertEqual(deepseek["reasoning_effort"], "high")
        self.assertEqual(deepseek["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertEqual(openai["reasoning_effort"], "high")
        self.assertNotIn("extra_body", openai)

    def test_agent_loop_uses_prompt_cache_key_only_for_openai_provider(self):
        from harness_code_agent.agent.conversation import Agent, AgentConversation
        from harness_code_agent.agent.providers import ProviderAdapter

        class FakeCompletions:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="done", tool_calls=None),
                            finish_reason="stop",
                        )
                    ],
                    usage=None,
                )

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        class CapturingProvider:
            def __init__(self, name):
                self.name = name
                self.calls = []
                self.delegate = ProviderAdapter(name)

            @property
            def supports_prompt_cache_key(self):
                return self.delegate.supports_prompt_cache_key

            def chat_kwargs(self, **kwargs):
                self.calls.append(kwargs)
                return self.delegate.chat_kwargs(**kwargs)

            def assistant_message_from_response(self, msg):
                return self.delegate.assistant_message_from_response(msg)

        with patch("harness_code_agent.agent.conversation.get_client", return_value=FakeClient()):
            openai_conv = AgentConversation(Agent("test", "system", use_tools=False))
            compatible_conv = AgentConversation(Agent("test", "system", use_tools=False))
        openai_conv.provider = CapturingProvider("openai")
        compatible_conv.provider = CapturingProvider("openai-compatible")

        with (
            patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=1),
        ):
            openai_conv.run_until_idle()
            compatible_conv.run_until_idle()

        self.assertIn("prompt_cache_key", openai_conv.provider.calls[0])
        self.assertNotIn("prompt_cache_key", compatible_conv.provider.calls[0])

    def test_prompt_cache_key_changes_when_system_prompt_changes(self):
        from harness_code_agent.agent.conversation import Agent
        from harness_code_agent.agent.utils import _prompt_cache_key

        first = _prompt_cache_key(
            Agent("test", "system\nHARNESS A", use_tools=False), None, model="m"
        )
        second = _prompt_cache_key(
            Agent("test", "system\nHARNESS B", use_tools=False), None, model="m"
        )

        self.assertNotEqual(first, second)

    def test_prompt_cache_key_changes_when_model_changes(self):
        from harness_code_agent.agent.conversation import Agent
        from harness_code_agent.agent.utils import _prompt_cache_key

        first = _prompt_cache_key(Agent("test", "system", use_tools=False), None, model="gpt-5.6")
        second = _prompt_cache_key(Agent("test", "system", use_tools=False), None, model="gpt-6")

        self.assertNotEqual(first, second)

    def test_prompt_cache_key_uses_stable_prefix_identity_and_tools_hash(self):
        from harness_code_agent.agent.conversation import Agent
        from harness_code_agent.agent.utils import _prompt_cache_key

        first = _prompt_cache_key(
            Agent(
                "test",
                "rendered system",
                use_tools=False,
                prompt_cache_identity={"global_rules_hash": "a"},
            ),
            [{"type": "function", "function": {"name": "read_file"}}],
            model="m",
        )
        second = _prompt_cache_key(
            Agent(
                "test",
                "rendered system",
                use_tools=False,
                prompt_cache_identity={"global_rules_hash": "b"},
            ),
            [{"type": "function", "function": {"name": "read_file"}}],
            model="m",
        )
        third = _prompt_cache_key(
            Agent(
                "test",
                "rendered system",
                use_tools=False,
                prompt_cache_identity={"global_rules_hash": "a"},
            ),
            [{"type": "function", "function": {"name": "write_file"}}],
            model="m",
        )

        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)

    def test_prompt_cache_key_canonicalizes_tool_schema_order(self):
        from harness_code_agent.agent.conversation import Agent
        from harness_code_agent.agent.utils import _prompt_cache_key

        agent = Agent(
            "test",
            "rendered system",
            use_tools=True,
            prompt_cache_identity={"global_rules_hash": "a"},
        )
        read_schema = {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "required": ["path", "max_lines"],
                    "properties": {
                        "path": {"type": "string"},
                        "max_lines": {"type": "integer"},
                    },
                },
            },
        }
        write_schema = {
            "type": "function",
            "function": {
                "name": "write_file",
                "parameters": {
                    "type": "object",
                    "required": ["content", "path"],
                    "properties": {
                        "content": {"type": "string"},
                        "path": {"type": "string"},
                    },
                },
            },
        }

        first = _prompt_cache_key(agent, [read_schema, write_schema], model="m")
        second = _prompt_cache_key(agent, [write_schema, read_schema], model="m")

        self.assertEqual(first, second)

    def test_full_system_prompt_keeps_stable_prefix_and_memory_index_separate(self):
        from harness_code_agent.agent.conversation import Agent

        without_memory = Agent("test", "stable", use_tools=False)
        with_memory = Agent("test", "stable", use_tools=False, memory_index=" memory block \n")

        self.assertEqual(without_memory.full_system_prompt, "stable")
        self.assertIsNone(without_memory.memory_index)
        self.assertEqual(with_memory.system_prompt, "stable")
        self.assertEqual(with_memory.memory_index, "memory block")
        self.assertEqual(with_memory.full_system_prompt, "stable\n\nmemory block")

    def test_openai_cache_breakpoint_model_gate(self):
        from harness_code_agent.agent.providers import (
            ProviderAdapter,
            openai_model_supports_cache_breakpoint,
            supports_system_cache_breakpoint,
        )

        supported = {"gpt-5.6", "gpt-5.6-mini", "GPT-5.6", "gpt-6", "gpt-6.1"}
        unsupported = {"gpt-5.5", "gpt-5", "gpt-4o", "gpt-4.1", "o3", "deepseek-v4-pro", ""}
        for model in supported:
            self.assertTrue(openai_model_supports_cache_breakpoint(model), model)
        for model in unsupported:
            self.assertFalse(openai_model_supports_cache_breakpoint(model), model)

        openai = ProviderAdapter("openai")
        deepseek = ProviderAdapter("deepseek")
        compatible = ProviderAdapter("openai-compatible")
        self.assertTrue(supports_system_cache_breakpoint(openai, "gpt-5.6"))
        self.assertFalse(supports_system_cache_breakpoint(openai, "gpt-4o"))
        self.assertFalse(supports_system_cache_breakpoint(deepseek, "gpt-5.6"))
        self.assertFalse(supports_system_cache_breakpoint(compatible, "gpt-5.6"))
        self.assertFalse(supports_system_cache_breakpoint(SimpleNamespace(), "gpt-5.6"))

    def test_outbound_messages_split_memory_at_cache_breakpoint_only_for_supported_path(self):
        from harness_code_agent.agent.conversation import Agent, AgentConversation
        from harness_code_agent.agent.providers import ProviderAdapter

        memory_block = "[HARNESS_MEMORY_INDEX]\n- [project:doc1 v1 active] topic"
        agent = Agent("test", "stable-prefix", use_tools=False, memory_index=memory_block)
        with patch("harness_code_agent.agent.conversation.get_client"):
            conversation = AgentConversation(agent)

        conversation.provider = ProviderAdapter("openai")
        split = conversation._outbound_messages(conversation.messages, "gpt-5.6")
        too_old = conversation._outbound_messages(conversation.messages, "gpt-4o")
        conversation.provider = ProviderAdapter("deepseek")
        deepseek_view = conversation._outbound_messages(conversation.messages, "gpt-5.6")

        self.assertEqual(
            split[0]["content"],
            [
                {
                    "type": "text",
                    "text": "stable-prefix",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                },
                {"type": "text", "text": memory_block},
            ],
        )
        self.assertIs(too_old, conversation.messages)
        self.assertIs(deepseek_view, conversation.messages)
        # The durable log keeps the plain-string system message.
        self.assertEqual(
            conversation.messages[0]["content"],
            f"stable-prefix\n\n{memory_block}",
        )
        self.assertIsInstance(conversation.messages[0]["content"], str)

    def test_outbound_messages_without_memory_index_are_not_rendered(self):
        from harness_code_agent.agent.conversation import Agent, AgentConversation
        from harness_code_agent.agent.providers import ProviderAdapter

        agent = Agent("test", "stable", use_tools=False)
        with patch("harness_code_agent.agent.conversation.get_client"):
            conversation = AgentConversation(agent)
        conversation.provider = ProviderAdapter("openai")
        self.assertIs(
            conversation._outbound_messages(conversation.messages, "gpt-5.6"),
            conversation.messages,
        )

    def test_agent_loop_sends_memory_breakpoint_parts_on_supported_openai_model(self):
        from harness_code_agent import config
        from harness_code_agent.agent.conversation import Agent, AgentConversation
        from harness_code_agent.agent.providers import ProviderAdapter

        memory_block = "[HARNESS_MEMORY_INDEX]\n- [project:doc1 v1 active] topic"

        class FakeCompletions:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="done", tool_calls=None),
                            finish_reason="stop",
                        )
                    ],
                    usage=None,
                )

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        class CapturingProvider:
            def __init__(self, name):
                self.name = name
                self.calls = []
                self.delegate = ProviderAdapter(name)

            @property
            def supports_prompt_cache_key(self):
                return self.delegate.supports_prompt_cache_key

            def chat_kwargs(self, **kwargs):
                self.calls.append(kwargs)
                return self.delegate.chat_kwargs(**kwargs)

            def assistant_message_from_response(self, msg):
                return self.delegate.assistant_message_from_response(msg)

        agent = Agent("test", "stable-prefix", use_tools=False, memory_index=memory_block)
        with patch("harness_code_agent.agent.conversation.get_client", return_value=FakeClient()):
            conversation = AgentConversation(agent)
        conversation.provider = CapturingProvider("openai")

        profile = config.ModelProfile(provider="openai", model="gpt-5.6")
        with (
            patch(
                "harness_code_agent.agent.conversation.config.resolve_model_profile",
                return_value=profile,
            ),
            patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=1),
        ):
            conversation.run_until_idle()

        system_content = conversation.provider.calls[0]["messages"][0]["content"]
        self.assertEqual(
            system_content,
            [
                {
                    "type": "text",
                    "text": "stable-prefix",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                },
                {"type": "text", "text": memory_block},
            ],
        )
        self.assertEqual(
            conversation.messages[0]["content"],
            f"stable-prefix\n\n{memory_block}",
        )

    def test_cache_shape_tracks_memory_index_change_separately(self):
        from harness_code_agent.agent.conversation import Agent
        from harness_code_agent.agent.utils import (
            capture_prompt_cache_shape,
            compare_prompt_cache_shapes,
        )

        v1 = Agent("test", "stable", use_tools=False, memory_index="mem v1")
        v2 = Agent("test", "stable", use_tools=False, memory_index="mem v2")
        changed_prefix = Agent("test", "stable v2", use_tools=False, memory_index="mem v2")

        shape_v1 = capture_prompt_cache_shape(v1, None)
        shape_v2 = capture_prompt_cache_shape(v2, None)
        shape_prefix = capture_prompt_cache_shape(changed_prefix, None)

        self.assertEqual(
            compare_prompt_cache_shapes(shape_v1, shape_v2, None)["prefix_change_reasons"],
            ["memory_index"],
        )
        self.assertEqual(
            compare_prompt_cache_shapes(shape_v2, shape_prefix, None)["prefix_change_reasons"],
            ["system"],
        )

    def test_apply_model_override_invalidates_conversation_cache_key(self):
        from harness_code_agent import config
        from harness_code_agent.core.interactive import InteractiveSession

        self.addCleanup(config.set_model_override, None, None)
        config.set_model_override(model=None, reasoning_effort=None)

        session = InteractiveSession.__new__(InteractiveSession)
        session.conversation = SimpleNamespace(_cached_prompt_cache_key="cache-key")
        session.apply_model_override(model="deepseek-v4-pro")

        self.assertEqual(config.get_model_override()["model"], "deepseek-v4-pro")
        self.assertIsNone(session.conversation._cached_prompt_cache_key)

        session.conversation = None
        session.apply_model_override(reasoning_effort="low")
        self.assertEqual(config.get_model_override()["reasoning_effort"], "low")

    def test_usage_to_dict_normalizes_deepseek_cache_hit_and_miss_tokens(self):
        from harness_code_agent.agent.utils import _usage_to_dict

        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_cache_hit_tokens=70,
            prompt_cache_miss_tokens=30,
        )

        result = _usage_to_dict(usage)

        self.assertEqual(result["cached_tokens"], 70)
        self.assertEqual(result["cache_hit_tokens"], 70)
        self.assertEqual(result["cache_miss_tokens"], 30)

    def test_agent_loop_records_llm_cached_token_usage_event(self):
        from harness_code_agent.agent.conversation import Agent, AgentConversation
        from harness_code_agent.sessions.events import EventBus

        class FakeCompletions:
            def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="done", tool_calls=None),
                            finish_reason="stop",
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=100,
                        completion_tokens=20,
                        total_tokens=120,
                        prompt_tokens_details=SimpleNamespace(cached_tokens=80),
                    ),
                )

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        events = []
        with patch("harness_code_agent.agent.conversation.get_client", return_value=FakeClient()):
            conversation = AgentConversation(Agent("test", "system", use_tools=False))
        conversation.event_bus = EventBus(listener=events.append)
        conversation.emitter.event_bus = conversation.event_bus

        with (
            patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=1),
        ):
            conversation.run_until_idle()

        usage = next(event for event in events if event.type == "llm_usage")
        self.assertEqual(usage.payload["cached_tokens"], 80)
        self.assertEqual(usage.payload["prompt_tokens"], 100)
        self.assertEqual(usage.payload["cache_hit_ratio"], 0.8)
        self.assertTrue([event for event in events if event.type == "llm_request_started"])
        finished = [event for event in events if event.type == "llm_response_finished"]
        self.assertEqual(len(finished), 1)
        self.assertGreaterEqual(finished[0].payload["duration_ms"], 0)
        self.assertFalse(finished[0].payload["streamed"])

    def test_agent_loop_emits_cache_diagnostics_when_tool_schema_changes(self):
        from harness_code_agent.agent.conversation import Agent, AgentConversation
        from harness_code_agent.sessions.events import EventBus

        class FakeCompletions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=f"done {self.calls}", tool_calls=None),
                            finish_reason="stop",
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=100,
                        completion_tokens=20,
                        total_tokens=120,
                        prompt_cache_hit_tokens=80 if self.calls > 1 else 0,
                        prompt_cache_miss_tokens=20 if self.calls > 1 else 100,
                    ),
                )

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        read_schema = {"type": "function", "function": {"name": "read_file"}}
        write_schema = {"type": "function", "function": {"name": "write_file"}}
        events = []
        agent = Agent("test", "system", use_tools=True, tool_schemas=[read_schema])
        with patch("harness_code_agent.agent.conversation.get_client", return_value=FakeClient()):
            conversation = AgentConversation(agent)
        conversation.event_bus = EventBus(listener=events.append)
        conversation.emitter.event_bus = conversation.event_bus

        with (
            patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=1),
        ):
            conversation.run_until_idle()
            agent.update_tool_schemas([write_schema])
            conversation.run_until_idle()

        usage_events = [event for event in events if event.type == "llm_usage"]
        self.assertEqual(len(usage_events), 2)
        first_diag = usage_events[0].payload["cache_diagnostics"]
        second_diag = usage_events[1].payload["cache_diagnostics"]
        self.assertFalse(first_diag["prefix_changed"])
        self.assertTrue(second_diag["prefix_changed"])
        self.assertEqual(second_diag["prefix_change_reasons"], ["tools"])
        self.assertEqual(second_diag["cache_hit_tokens"], 80)
        self.assertEqual(second_diag["cache_miss_tokens"], 20)

    def test_invalidated_observation_keeps_original_message_and_appends_notice(self):
        from harness_code_agent.agent.conversation import Agent, AgentConversation

        cases = [
            ("long", "SECRET_FULL_CONTENT_UNSAFE" * 700, True),
            ("short", "SHORT_STALE_CONTENT", False),
        ]

        for label, file_content, expect_observation_files in cases:
            with self.subTest(observation_length=label):

                class FakeCompletions:
                    def __init__(self):
                        self.calls = []

                    def create(self, **kwargs):
                        self.calls.append(json.loads(json.dumps(kwargs["messages"])))
                        call_count = len(self.calls)
                        if call_count == 1:
                            message = SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        id="tc_read",
                                        type="function",
                                        function=SimpleNamespace(
                                            name="read_file",
                                            arguments='{"path":"note.txt"}',
                                        ),
                                    )
                                ],
                            )
                            finish_reason = "tool_calls"
                        elif call_count == 2:
                            message = SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        id="tc_write",
                                        type="function",
                                        function=SimpleNamespace(
                                            name="write_file",
                                            arguments='{"path":"note.txt","content":"updated"}',
                                        ),
                                    )
                                ],
                            )
                            finish_reason = "tool_calls"
                        else:
                            message = SimpleNamespace(content="done", tool_calls=None)
                            finish_reason = "stop"
                        return SimpleNamespace(
                            choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
                            usage=None,
                        )

                class FakeClient:
                    def __init__(self):
                        self.chat = SimpleNamespace(completions=FakeCompletions())

                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    (root / "note.txt").write_text(file_content, encoding="utf-8")
                    fake_client = FakeClient()
                    with (
                        patch("harness_code_agent.agent.conversation.get_client", return_value=fake_client),
                        patch("harness_code_agent.agent.conversation.Path.cwd", return_value=root),
                    ):
                        conversation = AgentConversation(Agent("test", "system", use_tools=True))

                    with (
                        patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=1),
                        patch("harness_code_agent.config.WORKSPACE", str(root)),
                    ):
                        conversation.run_until_idle()

                    third_prompt = json.dumps(fake_client.chat.completions.calls[2], ensure_ascii=False)

                    self.assertIn(label == "long" and "SECRET_FULL_CONTENT_UNSAFE" or "SHORT_STALE_CONTENT", third_prompt)
                    self.assertIn("FACT INVALIDATION", third_prompt)
                    self.assertNotIn("Compressed stale long observations", third_prompt)

                    if label == "long":
                        second_prompt = json.dumps(fake_client.chat.completions.calls[1], ensure_ascii=False)
                        self.assertIn("SECRET_FULL_CONTENT_UNSAFE", second_prompt)
                        self.assertIn("[OBS obs_0001 observed]", third_prompt)
                        self.assertNotIn("[OBS obs_0001 stale]", third_prompt)
                        self.assertIn("Stale observations: obs_0001", third_prompt)
                    if expect_observation_files:
                        self.assertTrue(list((root / ".harness" / "observations").rglob("*.txt")))

    def test_trace_writer_stores_traces_under_harness_directory_without_stderr_by_default(self):
        from harness_code_agent.agent.conversation import TraceWriter

        with tempfile.TemporaryDirectory() as tmp:
            stderr = StringIO()
            with (
                patch("harness_code_agent.agent.conversation.config.TRACE_STDERR", False),
                redirect_stderr(stderr),
            ):
                writer = TraceWriter("main_agent", workspace=tmp)
                writer.iteration(1, 42)

            trace_path = Path(tmp) / ".harness" / "traces" / "trace_main_agent.jsonl"
            self.assertTrue(trace_path.exists())
            self.assertFalse((Path(tmp) / "_trace_main_agent.jsonl").exists())
            self.assertEqual(stderr.getvalue(), "")

    def test_builtin_tool_registry_exposes_schema_and_dispatch_exports(self):

        registry_names = {
            schema["function"]["name"]
            for schema in BUILTIN_TOOL_REGISTRY.schemas()
        }
        exported_schema_names = {
            schema["function"]["name"]
            for schema in TOOL_SCHEMAS + BROWSER_TOOL_SCHEMAS
        }

        self.assertEqual(registry_names, exported_schema_names)
        self.assertEqual(BUILTIN_TOOL_REGISTRY.permission_for("web_search"), "network_read")
        self.assertEqual(BUILTIN_TOOL_REGISTRY.permission_for("list_shell_jobs"), "read")
        self.assertEqual(BUILTIN_TOOL_REGISTRY.permission_for("read_shell_output"), "read")
        self.assertEqual(BUILTIN_TOOL_REGISTRY.permission_for("stop_shell_job"), "control")
        self.assertTrue(BUILTIN_TOOL_REGISTRY.effect_for("list_shell_jobs", {}).barrier)
        self.assertTrue(BUILTIN_TOOL_REGISTRY.effect_for("read_shell_output", {}).barrier)
        self.assertTrue(BUILTIN_TOOL_REGISTRY.effect_for("stop_shell_job", {}).barrier)
        self.assertTrue(all(spec.permission for spec in BUILTIN_TOOL_REGISTRY.specs()))
        self.assertTrue(all(spec.capabilities for spec in BUILTIN_TOOL_REGISTRY.specs()))
        self.assertIsNone(BUILTIN_TOOL_REGISTRY.get("missing_tool"))

    def test_run_bash_long_running_uses_shell_job_manager(self):

        class FakeJobs:
            def start(self, command, *, early_exit_seconds=0.5):
                self.request = (command, early_exit_seconds)
                return SimpleNamespace(
                    job_id="shell-job-abc123",
                    command=command,
                    pid=456,
                    status="running",
                    exit_code=None,
                    output_tail="",
                )

        fake_jobs = FakeJobs()
        runtime_state = SimpleNamespace(shell_session=None, shell_job_manager=fake_jobs)

        result = run_bash(
            "npm run dev",
            timeout=300,
            runtime_state=runtime_state,
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(fake_jobs.request, ("npm run dev", 0.5))
        self.assertIn("shell-job-abc123", result.output)
        self.assertEqual(result.metadata["job_id"], "shell-job-abc123")

    def test_run_bash_does_not_reuse_runtime_shell_session(self):

        class DeadShell:
            def __init__(self):
                self.closed = False

            def run(self, command, timeout=300, artifact_dir=None):
                raise RuntimeError("Shell failed to become ready")

            def close(self):
                self.closed = True

        dead_shell = DeadShell()
        runtime_state = SimpleNamespace(shell_session=dead_shell, shell_job_manager=None)

        fresh_shell = SimpleNamespace(
            run=lambda command, timeout=300, artifact_dir=None: SimpleNamespace(
                stdout="hi", stderr="", exit_code=0, timed_out=False, output_spilled=False
            ),
            close=lambda: None,
        )
        with patch("harness_code_agent.workspace.shell_session.PersistentShellSession", return_value=fresh_shell):
            result = run_bash("echo hi", runtime_state=runtime_state)

        self.assertEqual(result.status, "success")
        self.assertFalse(dead_shell.closed)

    def test_run_bash_reports_nonzero_exit_as_success_with_exit_code_footer(self):

        class ExpectedFailureShell:
            def run(self, command, timeout=300, artifact_dir=None):
                return SimpleNamespace(
                    stdout="",
                    stderr="error: minutes must be between 1 and 120",
                    exit_code=2,
                    timed_out=False,
                    output_spilled=False,
                )

        shell = ExpectedFailureShell()
        shell.close = lambda: None
        with patch("harness_code_agent.workspace.shell_session.PersistentShellSession", return_value=shell):
            result = run_bash(
                'python focusflow.py "Task" 0',
                runtime_state=SimpleNamespace(shell_job_manager=None),
            )

        self.assertEqual(result.status, "success")
        self.assertIsNone(result.error)
        self.assertEqual(result.return_code, 2)
        self.assertIn("[exit_code: 2]", result.output)

    def test_run_bash_uses_one_shot_shell_for_powershell_exit(self):
        from unittest.mock import Mock


        persistent_shell = Mock()
        runtime_state = SimpleNamespace(shell_session=persistent_shell, shell_job_manager=None)
        shell_result = SimpleNamespace(
            stdout="ready",
            stderr="",
            exit_code=0,
            timed_out=False,
        )

        with (
            patch("harness_code_agent.workspace.shell_session.PersistentShellSession", return_value=Mock()),
            patch(
                "harness_code_agent.runtime.builtins.shell._requires_one_shot_powershell",
                return_value=True,
            ),
            patch(
                "harness_code_agent.runtime.builtins.shell._run_one_shot_powershell",
                return_value=shell_result,
            ) as one_shot,
        ):
            result = run_bash(
                "Write-Output ready; exit 0",
                runtime_state=runtime_state,
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.output, "ready\n\n[exit_code: 0]")
        one_shot.assert_called_once()
        persistent_shell.run.assert_not_called()

    def test_shell_job_tools_handle_list_read_stop(self):

        class FakeJob:
            def __init__(self, job_id="shell-job-1", status="running"):
                self.job_id = job_id
                self.command = "npm run dev"
                self.pid = 111
                self.status = status
                self.exit_code = None
                self.started_at = 10.0
                self.ended_at = None

            def uptime_seconds(self):
                return 2.0

        class FakeJobs:
            def list_jobs(self):
                return [FakeJob()]

            def read_output(self, job_id, max_chars=12000):
                self.read_request = (job_id, max_chars)
                return "ready"

            def stop(self, job_id):
                self.stop_request = job_id
                return FakeJob(job_id, status="stopped")

        fake_jobs = FakeJobs()
        runtime_state = SimpleNamespace(shell_job_manager=fake_jobs)

        listed = list_shell_jobs(runtime_state=runtime_state)
        read = read_shell_output("shell-job-1", max_chars=50, runtime_state=runtime_state)
        stopped = stop_shell_job("shell-job-1", runtime_state=runtime_state)

        self.assertEqual(listed.status, "success")
        self.assertIn("shell-job-1", listed.output)
        self.assertEqual(read.status, "success")
        self.assertEqual(fake_jobs.read_request, ("shell-job-1", 50))
        self.assertIn("ready", read.output)
        self.assertEqual(stopped.status, "success")
        self.assertEqual(fake_jobs.stop_request, "shell-job-1")
        self.assertIn("stopped", stopped.output)

    def test_agent_conversation_close_closes_shell_job_manager(self):
        from harness_code_agent.agent.conversation import Agent, AgentConversation

        class FakeJobs:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        fake_jobs = FakeJobs()
        with patch("harness_code_agent.agent.conversation.get_client", return_value=SimpleNamespace(chat=SimpleNamespace(completions=None))):
            conversation = AgentConversation(Agent("test", "system", use_tools=False))
        conversation.runtime_state.shell_job_manager = fake_jobs

        conversation.close()

        self.assertTrue(fake_jobs.closed)

    def test_agent_conversation_runs_lifecycle_middleware_hooks(self):
        from harness_code_agent.agent.conversation import Agent, AgentConversation
        from harness_code_agent.runtime.middleware import AgentMiddleware

        class LifecycleMiddleware(AgentMiddleware):
            def __init__(self):
                self.closed = False

            def on_conversation_start(self, messages, runtime_state=None, agent_name=None):
                return [{"role": "system", "content": f"startup context for {agent_name}"}]

            def on_conversation_close(self, messages, runtime_state=None, agent_name=None):
                self.closed = True

        middleware = LifecycleMiddleware()
        with patch("harness_code_agent.agent.conversation.get_client", return_value=SimpleNamespace(chat=SimpleNamespace(completions=None))):
            conversation = AgentConversation(
                Agent("test", "system", use_tools=False, middlewares=[middleware])
            )

        self.assertEqual(conversation.messages[-1]["role"], "system")
        self.assertEqual(conversation.messages[-1]["content"], "startup context for test")

        conversation.close()

        self.assertTrue(middleware.closed)

    def test_tool_registry_requires_explicit_permission_classification(self):

        registry = ToolRegistry()
        schema = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location.",
                "parameters": {
                    "type": "object",
                    "required": ["location"],
                    "properties": {"location": {"type": "string"}},
                },
            },
        }

        with self.assertRaisesRegex(ValueError, "permission"):
            registry.register(schema, lambda **_: "sunny")

        with self.assertRaisesRegex(ValueError, "unknown permission"):
            registry.register(schema, lambda **_: "sunny", permission="weatherish")

    def test_agent_update_tool_schemas_invalidates_conversation_prompt_cache(self):
        from harness_code_agent.agent.conversation import Agent

        class DummyConversation:
            pass

        agent = Agent("test", "system", use_tools=True, tool_schemas=[])
        conversation = DummyConversation()
        conversation._cached_prompt_cache_key = "old-cache-key"
        agent._conversations.add(conversation)

        agent.update_tool_schemas(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "dynamic_tool",
                        "description": "Dynamic tool",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        )

        self.assertEqual(agent.allowed_tool_names, {"dynamic_tool"})
        self.assertIsNone(conversation._cached_prompt_cache_key)

    def test_agent_update_tool_schemas_preserves_prompt_cache_when_unchanged(self):
        from harness_code_agent.agent.conversation import Agent

        class DummyConversation:
            pass

        schemas = [
            {
                "type": "function",
                "function": {
                    "name": "dynamic_tool",
                    "description": "Dynamic tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        agent = Agent("test", "system", use_tools=True, tool_schemas=schemas)
        conversation = DummyConversation()
        conversation._cached_prompt_cache_key = "old-cache-key"
        agent._conversations.add(conversation)

        agent.update_tool_schemas(list(schemas))

        self.assertEqual(agent.allowed_tool_names, {"dynamic_tool"})
        self.assertEqual(conversation._cached_prompt_cache_key, "old-cache-key")

    def test_network_read_tool_permission_is_allowed_without_approval(self):
        from harness_code_agent.runtime.permissions import PermissionPolicy

        policy = PermissionPolicy(mode="workspace-write")

        decision = policy.decide_tool_call(
            "get_weather",
            {"location": "Hong Kong"},
            tool_permission="network_read",
        )

        self.assertTrue(decision.allowed)
        self.assertFalse(decision.requires_approval)
        self.assertEqual(decision.risk, "network_read")

    def test_ask_user_tool_appends_other_and_returns_structured_choice(self):
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.questions import StaticQuestionProvider
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.sessions.events import EventBus
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
                permission_policy=PermissionPolicy(mode="workspace-write"),
                event_bus=EventBus(root / ".harness" / "events.jsonl"),
                question_provider=StaticQuestionProvider(index=1),
            )

            result = execute_tool(
                "ask_user",
                {"question": "Pick a path", "options": ["Fast path"]},
                tool_context=context,
                agent_name="main_agent",
            )

            data = json.loads(result)
            self.assertEqual(data["selected_index"], 1)
            self.assertEqual(data["label"], "其他")
            self.assertTrue(data["is_other"])
            self.assertIn("ask_user", [schema["function"]["name"] for schema in TOOL_SCHEMAS])

    def test_structured_event_schema_covers_mvp_event_types(self):
        from harness_code_agent.sessions.events import (
            AgentBudgetWarningEvent,
            AgentFallbackEvent,
            AssistantMessageEvent,
            FailureEvent,
            FileChangeEvent,
            FinalReportEvent,
            SessionFinishedEvent,
            TaskOutcomeEvent,
            ToolCallEvent,
            ToolResultEvent,
            UserInputEvent,
        )

        event_types = {
            AgentBudgetWarningEvent(limit_type="total_tokens", used=80, limit=100).to_event().type,
            AgentFallbackEvent(reason="loop_detected").to_event().type,
            UserInputEvent(text="fix").to_event().type,
            AssistantMessageEvent(text="done").to_event().type,
            ToolCallEvent(tool="read_file", args={"path": "README.md"}).to_event().type,
            ToolResultEvent(tool="read_file", status="success", output="ok").to_event().type,
            FileChangeEvent(path="app.py").to_event().type,
            FailureEvent(category="tool_error", message="boom").to_event().type,
            FinalReportEvent(status="success", reason="completed", summary="done").to_event().type,
            SessionFinishedEvent(reason="user_exit", status="closed").to_event().type,
            TaskOutcomeEvent(status="success", evidence=["tests_passed"], summary="done").to_event().type,
        }

        self.assertEqual(event_types, {
            "agent_budget_warning",
            "agent_fallback",
            "user_input",
            "assistant_message",
            "tool_call",
            "tool_result",
            "file_change",
            "failure",
            "final_report",
            "session_finished",
            "task_outcome",
        })

    def test_assistant_message_event_records_streamed_flag_when_present(self):
        from harness_code_agent.sessions.events import AssistantMessageEvent

        event = AssistantMessageEvent(text="done", turn=1, streamed=True).to_event()

        self.assertEqual(event.payload["text"], "done")
        self.assertEqual(event.payload["turn"], 1)
        self.assertTrue(event.payload["streamed"])

    def test_failure_classification_uses_stable_sources_before_text(self):
        from harness_code_agent.runtime.tool_result import ToolResult
        from harness_code_agent.sessions.events import classify_tool_failure

        cases = [
            (
                ToolResult(tool="read_file", status="failed", error="missing", metadata={"status_source": "native"}),
                "tool_error",
            ),
            (
                ToolResult(tool="write_file", status="failed", error="empty path", metadata={"status_source": "validation"}),
                "validation_error",
            ),
            (
                ToolResult(tool="run_bash", status="failed", error="Command exited with code 1", metadata={"status_source": "shell"}),
                "runtime_error",
            ),
            (
                ToolResult(tool="write_file", status="failed", error="user said no", metadata={"status_source": "approval"}),
                "user_cancelled",
            ),
            (
                ToolResult(tool="custom", status="failed", error="", metadata={}),
                "unknown",
            ),
        ]

        for result, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(classify_tool_failure(result), expected)

    def test_tool_result_serializes_and_tool_execution_records_structured_events(self):
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.runtime.tool_result import ToolResult
        from harness_code_agent.sessions.events import EventBus
        from harness_code_agent.workspace.service import WorkspaceService

        result = ToolResult(
            tool="read_file",
            status="failed",
            output="",
            error="missing",
            return_code=2,
            metadata={"path": "missing.txt"},
        )

        self.assertEqual(result.to_dict()["tool"], "read_file")
        self.assertEqual(result.to_dict()["status"], "failed")
        self.assertFalse(result.to_dict()["ok"])
        self.assertEqual(result.to_dict()["error"], "missing")
        self.assertEqual(result.to_text(), "[error] missing")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("hello", encoding="utf-8")
            events_path = root / ".harness" / "events.jsonl"
            context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
                permission_policy=PermissionPolicy(mode="workspace-write"),
                event_bus=EventBus(events_path),
            )

            output = execute_tool(
                "read_file",
                {"path": "note.txt"},
                tool_context=context,
                agent_name="main_agent",
            )
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(output, "hello")
            event_types = [event["type"] for event in events]
            self.assertEqual(event_types, ["tool_call", "tool_result"])
            tool_result = next(event for event in events if event["type"] == "tool_result")
            self.assertEqual(tool_result["payload"]["tool"], "read_file")
            self.assertEqual(tool_result["payload"]["status"], "success")
            self.assertTrue(tool_result["payload"]["ok"])
            self.assertEqual(tool_result["payload"]["output"], "[redacted read_file output: 5 chars]")
            self.assertTrue(tool_result["payload"]["metadata"]["output_redacted"])
            self.assertEqual(tool_result["payload"]["metadata"]["output_length"], 5)

    def test_read_file_supports_line_ranges_and_line_numbers(self):
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.sessions.events import EventBus
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
            context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
                permission_policy=PermissionPolicy(mode="workspace-write"),
                event_bus=EventBus(root / ".harness" / "events.jsonl"),
            )

            output = execute_tool(
                "read_file",
                {
                    "path": "note.txt",
                    "start_line": 2,
                    "max_lines": 2,
                    "include_line_numbers": True,
                },
                tool_context=context,
                agent_name="main_agent",
            )

        self.assertEqual(output, "2: two\n3: three")

    def test_read_file_rejects_invalid_range_arguments_without_exception(self):
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.sessions.events import EventBus
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("one\ntwo\n", encoding="utf-8")
            context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
                permission_policy=PermissionPolicy(mode="workspace-write"),
                event_bus=EventBus(root / ".harness" / "events.jsonl"),
            )

            start_output = execute_tool(
                "read_file",
                {"path": "note.txt", "start_line": "abc", "max_lines": 1},
                tool_context=context,
                agent_name="main_agent",
            )
            max_output = execute_tool(
                "read_file",
                {"path": "note.txt", "start_line": 1, "max_lines": 0},
                tool_context=context,
                agent_name="main_agent",
            )

        self.assertIn("[error]", start_output)
        self.assertIn("start_line must be an integer", start_output)
        self.assertIn("[error]", max_output)
        self.assertIn("max_lines must be an integer", max_output)

    def test_read_file_requires_bounded_ranges_for_files_over_limit_lines(self):
        from harness_code_agent.runtime.builtins.filesystem import READ_FILE_MAX_LINES
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.sessions.events import EventBus
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "big.txt").write_text(
                "\n".join(f"line {i}" for i in range(1, READ_FILE_MAX_LINES + 2)),
                encoding="utf-8",
            )
            context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
                permission_policy=PermissionPolicy(mode="workspace-write"),
                event_bus=EventBus(root / ".harness" / "events.jsonl"),
            )

            output = execute_tool(
                "read_file",
                {"path": "big.txt"},
                tool_context=context,
                agent_name="main_agent",
            )

        self.assertIn("[error]", output)
        self.assertIn(f"{READ_FILE_MAX_LINES} lines", output)
        self.assertIn("start_line", output)
        self.assertIn("max_lines", output)

    def test_read_file_rejects_ranges_over_limit_lines(self):
        from harness_code_agent.runtime.builtins.filesystem import READ_FILE_MAX_LINES
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.sessions.events import EventBus
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "big.txt").write_text(
                "\n".join(f"line {i}" for i in range(1, READ_FILE_MAX_LINES + 200)),
                encoding="utf-8",
            )
            context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
                permission_policy=PermissionPolicy(mode="workspace-write"),
                event_bus=EventBus(root / ".harness" / "events.jsonl"),
            )

            output = execute_tool(
                "read_file",
                {"path": "big.txt", "start_line": 1, "max_lines": READ_FILE_MAX_LINES + 1},
                tool_context=context,
                agent_name="main_agent",
            )

        self.assertIn("[error]", output)
        self.assertIn(f"max_lines must be <= {READ_FILE_MAX_LINES}", output)

    def test_read_file_rejects_windows_with_too_much_output(self):
        from harness_code_agent.runtime.builtins.filesystem import (
            READ_FILE_MAX_OUTPUT_TOKENS,
        )
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.sessions.events import EventBus
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Use diverse text so BPE cannot merge tokens as aggressively as a
            # single repeated character (~6 tokens per 26-char fragment after
            # cl100k encoding; ~4 chars/token for the char-based fallback).
            fragment = "the quick brown fox jumps "
            repeats = (READ_FILE_MAX_OUTPUT_TOKENS // 5) + 10
            (root / "wide.txt").write_text(fragment * repeats, encoding="utf-8")
            context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
                permission_policy=PermissionPolicy(mode="workspace-write"),
                event_bus=EventBus(root / ".harness" / "events.jsonl"),
            )

            output = execute_tool(
                "read_file",
                {"path": "wide.txt", "start_line": 1, "max_lines": 1},
                tool_context=context,
                agent_name="main_agent",
            )

        self.assertIn("[error]", output)
        self.assertIn("too large", output)
        self.assertIn("tokens", output)
        self.assertNotIn("[TRUNCATED]", output)

    def test_tool_result_does_not_infer_status_from_raw_tool_text(self):
        from unittest.mock import patch

        from harness_code_agent.runtime.approvals import StaticApprovalProvider
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.sessions.events import EventBus
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / ".harness" / "events.jsonl"
            context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
                permission_policy=PermissionPolicy(mode="workspace-write"),
                event_bus=EventBus(events_path),
                approval_provider=StaticApprovalProvider(approved=True, reason="test approval"),
            )

            with patch.object(
                BUILTIN_TOOL_REGISTRY,
                "get",
                return_value=lambda **kwargs: "[error] this is domain output, not execution status",
            ):
                output = execute_tool(
                    "custom_tool",
                    {},
                    tool_context=context,
                    agent_name="main_agent",
                )

            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]
            tool_result = next(event for event in events if event["type"] == "tool_result")

            self.assertEqual(output, "[error] this is domain output, not execution status")
            self.assertEqual(tool_result["payload"]["status"], "unknown")
            self.assertIsNone(tool_result["payload"]["ok"])
            self.assertEqual(tool_result["payload"]["metadata"]["status_source"], "unstructured")
            self.assertFalse(any(event["type"] == "failure" for event in events))

    def test_unknown_tool_records_structured_failure_events(self):
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.sessions.events import EventBus
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / ".harness" / "events.jsonl"
            context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
                permission_policy=PermissionPolicy(mode="workspace-write"),
                event_bus=EventBus(events_path),
            )

            output = execute_tool(
                "missing_tool",
                {"secret": "nope"},
                tool_context=context,
                agent_name="main_agent",
            )

            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]
            event_types = [event["type"] for event in events]
            tool_result = next(event for event in events if event["type"] == "tool_result")

            self.assertEqual(output, "[error] Unknown tool: missing_tool")
            self.assertIn("tool_call", event_types)
            self.assertIn("failure", event_types)
            self.assertEqual(tool_result["payload"]["status"], "failed")
            self.assertFalse(tool_result["payload"]["ok"])
            failure = next(event for event in events if event["type"] == "failure")
            self.assertEqual(failure["payload"]["category"], "tool_error")
            self.assertEqual(failure["payload"]["tool"], "missing_tool")

    def test_tool_validation_failures_return_typed_failed_results(self):
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.sessions.events import EventBus
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / ".harness" / "events.jsonl"
            context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
                permission_policy=PermissionPolicy(mode="workspace-write"),
                event_bus=EventBus(events_path),
            )

            missing = execute_tool(
                "read_file",
                {"path": "missing.txt"},
                tool_context=context,
                agent_name="main_agent",
            )
            empty_write = execute_tool(
                "write_file",
                {"path": "", "content": "x"},
                tool_context=context,
                agent_name="main_agent",
            )
            empty_patch = execute_tool(
                "apply_patch",
                {"path": "", "search": "x", "replace": "y"},
                tool_context=context,
                agent_name="main_agent",
            )

            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]
            failed_results = [
                event for event in events
                if event["type"] == "tool_result"
                and event["payload"].get("status") == "failed"
            ]

            self.assertIn("[error] File not found: missing.txt", missing)
            self.assertIn("Empty file path", empty_write)
            self.assertIn("kind=invalid_arguments", empty_write)
            self.assertIn("Empty file path", empty_patch)
            self.assertIn("kind=invalid_arguments", empty_patch)
            self.assertEqual(len(failed_results), 3)
            failures = [event for event in events if event["type"] == "failure"]
            self.assertEqual(len(failures), 3)
            self.assertEqual(
                [event["payload"]["category"] for event in failures],
                ["tool_error", "validation_error", "validation_error"],
            )

    def test_session_store_creates_metadata_and_jsonl_events(self):
        from harness_code_agent.sessions.store import SessionStore

        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp) / ".harness")
            session = store.create(
                profile="terminal",
                cwd=Path(tmp),
                model="test-model",
                permission_mode="workspace-write",
            )
            bus = store.event_bus(session)
            bus.emit("session_started", agent="main_agent", payload={"task": "fix bug"})

            metadata = json.loads(session.metadata_path.read_text(encoding="utf-8"))
            events = [
                json.loads(line)
                for line in session.events_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(metadata["profile"], "terminal")
            self.assertEqual(metadata["model"], "test-model")
            self.assertEqual(metadata["permission_mode"], "workspace-write")
            self.assertEqual(events[0]["type"], "session_started")
            self.assertEqual(events[0]["sequence"], 1)

    def test_session_store_lists_and_reads_sessions(self):
        from harness_code_agent.sessions.store import SessionStore

        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp) / ".harness")
            first = store.create(
                profile="terminal",
                cwd=Path(tmp),
                model="model-a",
                permission_mode="workspace-write",
            )
            second = store.create(
                profile="plan",
                cwd=Path(tmp),
                model="model-b",
                permission_mode="read-only",
            )
            store.event_bus(second).emit("session_finished", agent="main_agent", payload={})

            sessions = store.list_sessions()
            metadata = store.read_metadata(second.id)
            events = store.read_events(second.id)

            self.assertEqual([item["id"] for item in sessions], [second.id, first.id])
            self.assertEqual(metadata["profile"], "plan")
            self.assertEqual(events[0]["type"], "session_finished")

    def test_session_store_forks_session_metadata_and_lineage_event(self):
        from harness_code_agent.sessions.store import SessionStore

        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp) / ".harness")
            source = store.create(
                profile="coding-agent",
                cwd=Path(tmp),
                model="model-a",
                permission_mode="workspace-write",
            )
            store.event_bus(source).emit("session_started", agent="main_agent", payload={})
            store.event_bus(source).emit("session_finished", agent="main_agent", payload={})

            fork = store.fork(source.id)
            metadata = store.read_metadata(fork.id)
            events = store.read_events(fork.id)

            self.assertNotEqual(fork.id, source.id)
            self.assertEqual(metadata["profile"], "coding-agent")
            self.assertEqual(metadata["model"], "model-a")
            self.assertEqual(metadata["permission_mode"], "workspace-write")
            self.assertEqual(metadata["forked_from"], source.id)
            self.assertEqual(metadata["forked_from_event_count"], 2)
            self.assertEqual(events[0]["type"], "session_forked")
            self.assertEqual(events[0]["payload"]["source_session_id"], source.id)

    def test_session_store_reads_fork_lineage_and_resumed_metadata(self):
        from harness_code_agent.sessions.store import SessionStore

        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp) / ".harness")
            source = store.create(
                profile="coding-agent",
                cwd=Path(tmp),
                model="model-a",
                permission_mode="workspace-write",
            )
            fork = store.fork(source.id)
            resumed = store.create(
                profile="coding-agent",
                cwd=Path(tmp),
                model="model-a",
                permission_mode="workspace-write",
                resumed_from=fork.id,
            )

            lineage = store.read_lineage(fork.id)
            resumed_metadata = store.read_metadata(resumed.id)

            self.assertEqual([item["id"] for item in lineage], [source.id, fork.id])
            self.assertEqual(resumed_metadata["resumed_from"], fork.id)

    def test_session_store_reads_latest_session_and_persisted_summary(self):
        from harness_code_agent.sessions.store import SessionStore

        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp) / ".harness")
            store.create(
                profile="coding-agent",
                cwd=Path(tmp),
                model="model-a",
                permission_mode="workspace-write",
            )
            second = store.create(
                profile="plan",
                cwd=Path(tmp),
                model="model-b",
                permission_mode="read-only",
            )
            store.event_bus(second).emit("user_input", agent="main_agent", payload={"text": "plan it"})
            summary = store.write_summary(second.id)

            latest = store.latest_session()

            self.assertEqual(latest["id"], second.id)
            self.assertIn("Session summary", summary)
            self.assertIn("profile: plan", summary)

    def test_session_summary_formats_human_readable_event_overview(self):
        from harness_code_agent.sessions.summary import format_session_summary

        metadata = {
            "id": "session-a",
            "profile": "coding-agent",
            "model": "model-a",
            "permission_mode": "workspace-write",
            "status": "running",
            "cwd": "C:/workspace",
            "created_at": "2026-05-20T00:00:00+00:00",
            "forked_from": "session-parent",
        }
        events = [
            {"sequence": 1, "type": "session_started", "agent": "main_agent", "payload": {}},
            {"sequence": 2, "type": "turn_started", "agent": "main_agent", "payload": {"turn": 1}},
            {"sequence": 3, "type": "tool_result", "agent": "main_agent", "payload": {"tool": "write_file", "status": "success", "ok": True}},
            {"sequence": 4, "type": "file_change", "agent": "main_agent", "payload": {"path": "app.py"}},
            {"sequence": 5, "type": "approval_requested", "agent": "main_agent", "payload": {"tool": "run_bash"}},
            {"sequence": 6, "type": "approval_decided", "agent": "main_agent", "payload": {"tool": "run_bash", "approved": False}},
            {
                "sequence": 7,
                "type": "profile_switched",
                "agent": "main_agent",
                "payload": {"previous_profile": "coding-agent", "profile": "plan", "reason": "slash command"},
            },
            {"sequence": 8, "type": "plan_ready", "agent": "main_agent", "payload": {"profile": "plan"}},
            {"sequence": 9, "type": "task_outcome", "agent": "main_agent", "payload": {"status": "success", "summary": "done"}},
            {"sequence": 10, "type": "agent_fallback", "agent": "main_agent", "payload": {"reason": "loop_detected"}},
            {"sequence": 11, "type": "session_finished", "agent": "main_agent", "payload": {"status": "closed", "reason": "user_exit"}},
        ]

        summary = format_session_summary(metadata, events)

        self.assertIn("Session summary", summary)
        self.assertIn("id: session-a", summary)
        self.assertIn("status: closed", summary)
        self.assertIn("forked_from: session-parent", summary)
        self.assertIn("turns: 1 started, 0 finished", summary)
        self.assertIn("tools: 1 call(s): write_file=1", summary)
        self.assertIn("changed_files: app.py", summary)
        self.assertIn("approvals: 1 requested, 0 approved, 1 denied", summary)
        self.assertIn("profile_switches: coding-agent -> plan (slash command)", summary)
        self.assertIn("plans_ready: 1", summary)
        self.assertIn("fallbacks: 1 (latest: loop_detected)", summary)
        self.assertIn("task_outcome: success - done", summary)
        self.assertIn("recent_events:", summary)

    def test_session_summary_uses_final_report_for_phase_two_status_and_categories(self):
        from harness_code_agent.sessions.summary import format_session_summary

        metadata = {"id": "session-final", "profile": "coding-agent", "status": "running"}
        events = [
            {"sequence": 1, "type": "user_input", "agent": "main_agent", "payload": {"text": "fix"}},
            {"sequence": 2, "type": "tool_result", "agent": "main_agent", "payload": {"tool": "read_file", "status": "failed"}},
            {"sequence": 3, "type": "failure", "agent": "main_agent", "payload": {"category": "tool_error", "message": "missing"}},
            {
                "sequence": 4,
                "type": "final_report",
                "agent": "main_agent",
                "payload": {
                    "status": "failed",
                    "reason": "verification_failed",
                    "summary": "tests still fail",
                    "statistics": {
                        "events": 3,
                        "user_inputs": 1,
                        "assistant_messages": 0,
                        "tool_calls": 1,
                        "failures": 1,
                        "file_changes": 0,
                    },
                    "failure_categories": {"tool_error": 1},
                    "tool_counts": {"read_file": 1},
                    "changed_files": [],
                },
            },
        ]

        summary = format_session_summary(metadata, events)

        self.assertIn("status: failed", summary)
        self.assertIn("final_report: failed - tests still fail", summary)
        self.assertIn("failure_categories: tool_error=1", summary)

    def test_session_summary_handles_empty_or_sparse_events(self):
        from harness_code_agent.sessions.summary import format_session_summary

        summary = format_session_summary(
            {"id": "empty-session", "profile": "plan", "status": "running"},
            [{"type": "tool_result", "payload": None}],
        )

        self.assertIn("id: empty-session", summary)
        self.assertIn("profile: plan", summary)
        self.assertIn("events: 1", summary)
        self.assertIn("tools: 1 call(s): unknown=1", summary)
        self.assertIn("changed_files: unknown", summary)
        self.assertIn("task_outcome: unknown", summary)

    def test_final_report_payload_is_statistics_ready_for_replay_and_evaluation(self):
        from harness_code_agent.sessions.events import FinalReportEvent
        from harness_code_agent.sessions.report import build_final_report

        metadata = {"id": "session-a", "created_at": "2026-05-21T00:00:00+00:00"}
        events = [
            {"sequence": 1, "type": "user_input", "agent": "main_agent", "payload": {"text": "fix"}},
            {"sequence": 2, "type": "assistant_message", "agent": "main_agent", "payload": {"text": "I changed app.py"}},
            {"sequence": 3, "type": "tool_result", "agent": "main_agent", "payload": {"tool": "write_file", "status": "success"}},
            {"sequence": 4, "type": "file_change", "agent": "main_agent", "payload": {"path": "app.py"}},
            {"sequence": 5, "type": "failure", "agent": "main_agent", "payload": {"category": "validation_error", "message": "empty"}},
            {"sequence": 6, "type": "agent_fallback", "agent": "main_agent", "payload": {"reason": "token_budget_exceeded"}},
        ]

        report = build_final_report(
            metadata,
            events,
            status="closed",
            reason="user_exit",
            summary="I changed app.py",
        )
        event = FinalReportEvent(**report).to_event()

        self.assertEqual(event.type, "final_report")
        self.assertEqual(event.payload["session_id"], "session-a")
        self.assertEqual(event.payload["status"], "closed")
        self.assertEqual(event.payload["reason"], "user_exit")
        self.assertEqual(event.payload["summary"], "I changed app.py")
        self.assertEqual(event.payload["statistics"]["user_inputs"], 1)
        self.assertEqual(event.payload["statistics"]["assistant_messages"], 1)
        self.assertEqual(event.payload["statistics"]["tool_calls"], 1)
        self.assertEqual(event.payload["statistics"]["failures"], 1)
        self.assertEqual(event.payload["statistics"]["fallbacks"], 1)
        self.assertEqual(event.payload["statistics"]["latest_fallback"], "token_budget_exceeded")
        self.assertEqual(event.payload["failure_categories"], {"validation_error": 1})
        self.assertEqual(event.payload["tool_counts"], {"write_file": 1})
        self.assertEqual(event.payload["changed_files"], ["app.py"])

    def test_changed_files_helpers_filter_verify_cache_paths(self):
        from harness_code_agent.sessions._event_helpers import changed_files

        events = [
            {"type": "file_change", "payload": {"path": "app.py"}},
            {"type": "file_change", "payload": {"path": ".pytest_cache/v/cache/nodeids"}},
            {"type": "file_change", "payload": {"path": ".ruff_cache/0.13.0/123"}},
            {"type": "file_change", "payload": {"path": ".mypy_cache/3.11/app.meta.json"}},
            {"type": "file_change", "payload": {"path": "__pycache__/app.cpython-311.pyc"}},
            {"type": "file_change", "payload": {"path": "build/temp.txt"}},
        ]

        self.assertEqual(changed_files(events), ["app.py"])

    def test_workspace_service_resolves_paths_and_snapshots_before_write(self):
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots")
            target = root / "src" / "app.py"
            target.parent.mkdir()
            target.write_text("old", encoding="utf-8")

            result = workspace.write_text("src/app.py", "new")

            self.assertEqual(target.read_text(encoding="utf-8"), "new")
            self.assertTrue(result.snapshot_path.exists())
            self.assertEqual(result.snapshot_path.read_text(encoding="utf-8"), "old")
            with self.assertRaises(ValueError):
                workspace.resolve("../outside.txt")

    def test_workspace_service_read_text_uses_same_path_lock_as_writes(self):
        import threading

        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("old", encoding="utf-8")
            workspace = WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots")
            results = []
            reader_started = threading.Event()

            def read_file():
                reader_started.set()
                results.append(workspace.read_text("app.py"))

            with workspace._path_lock(workspace.resolve("app.py")):
                reader = threading.Thread(target=read_file)
                reader.start()
                self.assertTrue(reader_started.wait(1))
                self.assertEqual(results, [])

            reader.join(timeout=1)

        self.assertEqual(results, ["old"])

    def test_workspace_service_different_file_writes_do_not_share_a_lock(self):
        import threading

        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots")
            write_finished = threading.Event()

            def write_other_file():
                workspace.write_text("b.txt", "b")
                write_finished.set()

            with workspace._path_lock(workspace.resolve("a.txt")):
                writer = threading.Thread(target=write_other_file)
                writer.start()
                self.assertTrue(write_finished.wait(1))
            writer.join(timeout=1)
            self.assertEqual((root / "b.txt").read_text(encoding="utf-8"), "b")

    def test_workspace_service_applies_unique_text_patch_and_rejects_ambiguous_patch(self):
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots")
            target = root / "app.py"
            target.write_text("alpha\nbeta\n", encoding="utf-8")

            result = workspace.apply_text_patch("app.py", search="beta\n", replace="gamma\n")

            self.assertEqual(target.read_text(encoding="utf-8"), "alpha\ngamma\n")
            self.assertTrue(result.snapshot_path.exists())

            target.write_text("same\nsame\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                workspace.apply_text_patch("app.py", search="same\n", replace="once\n")
            self.assertEqual(target.read_text(encoding="utf-8"), "same\nsame\n")

    def test_workspace_service_rolls_back_latest_snapshot_for_file(self):
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots")
            target = root / "app.py"
            target.write_text("old\n", encoding="utf-8")
            workspace.write_text("app.py", "new\n")

            result = workspace.rollback_latest_snapshot("app.py")

            self.assertEqual(target.read_text(encoding="utf-8"), "old\n")
            self.assertTrue(result.snapshot_path.exists())

    def test_permission_policy_uses_codex_sandbox_modes(self):
        from harness_code_agent.runtime.permissions import PermissionPolicy

        workspace_policy = PermissionPolicy(mode="workspace-write")
        read_decision = workspace_policy.decide_tool_call("read_file", {"path": "x.txt"})
        repo_search_decision = workspace_policy.decide_tool_call("repo_search", {"pattern": "needle"})
        agent_decision = workspace_policy.decide_tool_call("spawn_agent", {"role": "explorer"})
        edit_decision = workspace_policy.decide_tool_call("write_file", {"path": "x.txt"})
        todo_decision = workspace_policy.decide_tool_call(
            "update_todo",
            {"items": [{"text": "do work", "status": "in_progress"}]},
        )
        safe_shell_decision = workspace_policy.decide_tool_call(
            "run_bash",
            {"command": "git status --short"},
        )
        risky_shell_decision = workspace_policy.decide_tool_call(
            "run_bash",
            {"command": "npm install"},
        )
        mkdir_decision = workspace_policy.decide_tool_call(
            "run_bash",
            {"command": "mkdir generated"},
        )
        reset_decision = workspace_policy.decide_tool_call(
            "run_bash",
            {"command": "git commit --allow-empty -m test"},
        )
        blocked_commands = [
            "rm -rf /",
            "rm -rf ~",
            "Remove-Item C:\\ -Recurse",
            "git reset --hard",
            "git clean -fd",
            "git push --force origin main",
            "mkfs.ext4 /dev/sda",
            "dd if=/dev/zero of=/dev/sda",
        ]
        blocked_decisions = [
            workspace_policy.decide_tool_call("run_bash", {"command": command})
            for command in blocked_commands
        ]
        workspace_delete_decision = workspace_policy.decide_tool_call(
            "run_bash", {"command": "rm -rf build"}
        )
        glob_delete_decision = workspace_policy.decide_tool_call(
            "run_bash", {"command": "rm -rf *"}
        )
        unknown_decision = workspace_policy.decide_tool_call("new_tool", {})

        llm_auto_policy = PermissionPolicy(mode="llm-auto")
        llm_read_decision = llm_auto_policy.decide_tool_call("read_file", {"path": "x.txt"})
        llm_repo_search_decision = llm_auto_policy.decide_tool_call("repo_search", {"pattern": "needle"})
        llm_edit_decision = llm_auto_policy.decide_tool_call("write_file", {"path": "x.txt"})
        llm_todo_decision = llm_auto_policy.decide_tool_call(
            "update_todo",
            {"items": [{"text": "do work", "status": "in_progress"}]},
        )
        llm_safe_shell_decision = llm_auto_policy.decide_tool_call(
            "run_bash",
            {"command": "git status --short"},
        )
        llm_risky_shell_decision = llm_auto_policy.decide_tool_call(
            "run_bash",
            {"command": "npm install"},
        )
        llm_unknown_decision = llm_auto_policy.decide_tool_call("new_tool", {})
        llm_dangerous_decision = llm_auto_policy.decide_tool_call(
            "mcp_tool",
            {},
            tool_permission="dangerous",
        )
        llm_blocked_decision = llm_auto_policy.decide_tool_call(
            "run_bash",
            {"command": "rm -rf /"},
        )

        full_access_policy = PermissionPolicy(mode="danger-full-access")
        full_access_decision = full_access_policy.decide_tool_call(
            "run_bash",
            {"command": "npm install"},
        )
        full_access_blocked_decision = full_access_policy.decide_tool_call(
            "run_bash",
            {"command": "dd if=/dev/zero of=/dev/sda"},
        )
        overwrite_decision = full_access_policy.decide_tool_call(
            "run_bash",
            {"command": "Set-Content out.txt bad"},
        )

        self.assertTrue(read_decision.allowed)
        self.assertTrue(repo_search_decision.allowed)
        self.assertTrue(agent_decision.allowed)
        self.assertTrue(edit_decision.allowed)
        self.assertTrue(todo_decision.allowed)
        self.assertTrue(safe_shell_decision.allowed)
        self.assertTrue(risky_shell_decision.requires_approval)
        self.assertEqual(risky_shell_decision.risk, "shell_risky")
        self.assertTrue(mkdir_decision.requires_approval)
        self.assertEqual(mkdir_decision.risk, "shell_risky")
        self.assertTrue(reset_decision.requires_approval)
        self.assertEqual(reset_decision.risk, "shell_risky")
        for command, blocked_decision in zip(blocked_commands, blocked_decisions):
            with self.subTest(command=command):
                self.assertFalse(blocked_decision.allowed)
                self.assertFalse(blocked_decision.requires_approval)
                self.assertEqual(blocked_decision.risk, "shell_blocked")
        # Recursive deletion inside the workspace is not catastrophic: it
        # follows the workspace-write action (ask) rather than a hard deny.
        self.assertTrue(workspace_delete_decision.requires_approval)
        self.assertTrue(glob_delete_decision.requires_approval)
        self.assertTrue(unknown_decision.requires_approval)
        self.assertTrue(llm_read_decision.allowed)
        self.assertTrue(llm_repo_search_decision.allowed)
        self.assertTrue(llm_edit_decision.allowed)
        self.assertTrue(llm_todo_decision.allowed)
        self.assertTrue(llm_safe_shell_decision.allowed)
        self.assertTrue(llm_risky_shell_decision.requires_approval)
        self.assertTrue(llm_unknown_decision.requires_approval)
        self.assertTrue(llm_dangerous_decision.requires_approval)
        self.assertFalse(llm_blocked_decision.allowed)
        self.assertFalse(llm_blocked_decision.requires_approval)
        self.assertEqual(llm_blocked_decision.risk, "shell_blocked")
        self.assertTrue(full_access_decision.allowed)
        self.assertFalse(full_access_blocked_decision.allowed)
        self.assertTrue(overwrite_decision.allowed)
        self.assertEqual(full_access_blocked_decision.risk, "shell_blocked")

    def test_read_only_mode_denies_mutations(self):
        from harness_code_agent.runtime.permissions import PermissionPolicy

        policy = PermissionPolicy(mode="read-only")
        self.assertTrue(
            policy.decide_tool_call("read_file", {"path": "x.txt"}).allowed
        )
        edit_decision = policy.decide_tool_call("write_file", {"path": "x.txt"})
        self.assertFalse(edit_decision.allowed)
        shell_decision = policy.decide_tool_call(
            "run_bash", {"command": "mkdir generated"}
        )
        self.assertFalse(shell_decision.allowed)
        read_shell = policy.decide_tool_call("run_bash", {"command": "git status"})
        self.assertTrue(read_shell.allowed)

    def test_permission_policy_rejects_unknown_mode_names(self):
        from harness_code_agent.runtime.permissions import PermissionPolicy

        with self.assertRaises(ValueError):
            PermissionPolicy(mode="unsupported-mode")

    def test_llm_auto_approval_provider_approves_only_high_confidence_json(self):
        from harness_code_agent.runtime.approvals import (
            ApprovalRequest,
            LlmAutoApprovalProvider,
        )

        class FakeCompletions:
            def __init__(self, content: str):
                self.content = content

            def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=self.content),
                        )
                    ]
                )

        class FakeClient:
            def __init__(self, content: str):
                self.chat = SimpleNamespace(completions=FakeCompletions(content))

        request = ApprovalRequest(
            tool_name="run_bash",
            args={"command": "npm install"},
            risk="shell_risky",
            reason="llm-auto mode requires automatic LLM approval",
        )

        with patch("harness_code_agent.agent.providers.get_client", return_value=FakeClient('{"approved": true, "confidence": 0.9, "reason": "bounded install"}')):
            approved = LlmAutoApprovalProvider().request(request)
        with patch("harness_code_agent.agent.providers.get_client", return_value=FakeClient('{"approved": true, "confidence": 0.5, "reason": "not sure"}')):
            low_confidence = LlmAutoApprovalProvider().request(request)
        with patch("harness_code_agent.agent.providers.get_client", return_value=FakeClient('{"approved": false, "confidence": 0.95, "reason": "too broad"}')):
            denied = LlmAutoApprovalProvider().request(request)
        with patch("harness_code_agent.agent.providers.get_client", return_value=FakeClient("not json")):
            invalid = LlmAutoApprovalProvider().request(request)

        self.assertTrue(approved.approved)
        self.assertEqual(approved.metadata["approval_source"], "llm_auto")
        self.assertEqual(approved.metadata["confidence"], 0.9)
        self.assertFalse(low_confidence.approved)
        self.assertFalse(denied.approved)
        self.assertFalse(invalid.approved)
        self.assertEqual(invalid.metadata["approval_source"], "llm_auto")
        self.assertIn("error", invalid.metadata)

    def test_llm_auto_approval_provider_rejects_model_exceptions(self):
        from harness_code_agent.runtime.approvals import (
            ApprovalRequest,
            LlmAutoApprovalProvider,
        )

        request = ApprovalRequest(
            tool_name="run_bash",
            args={"command": "npm install"},
            risk="shell_risky",
            reason="llm-auto mode requires automatic LLM approval",
        )

        with patch("harness_code_agent.agent.providers.get_client", side_effect=RuntimeError("network down")):
            result = LlmAutoApprovalProvider().request(request)

        self.assertFalse(result.approved)
        self.assertEqual(result.reason, "llm-auto approval failed")
        self.assertEqual(result.metadata["approval_source"], "llm_auto")
        self.assertIn("network down", result.metadata["error"])

    def test_execute_tool_records_events_and_snapshots(self):
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.sessions.events import EventBus
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("old", encoding="utf-8")
            events_path = root / ".harness" / "events.jsonl"
            workspace = WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots")
            context = ToolContext(
                workspace=workspace,
                permission_policy=PermissionPolicy(mode="workspace-write"),
                event_bus=EventBus(events_path),
            )

            result = execute_tool(
                "write_file",
                {"path": "note.txt", "content": "new"},
                tool_context=context,
                agent_name="main_agent",
            )

            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertIn("Wrote", result)
            self.assertEqual((root / "note.txt").read_text(encoding="utf-8"), "new")
            self.assertEqual(len(list((root / ".harness" / "snapshots").rglob("*.*"))), 1)
            event_types = [event["type"] for event in events]
            self.assertIn("tool_call", event_types)
            self.assertIn("tool_result", event_types)
            self.assertIn("file_change", event_types)

    def test_permission_middleware_denies_approval_and_emits_events(self):
        from harness_code_agent.runtime.permission_middleware import (
            PermissionMiddleware,
        )
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.sessions.events import EventBus
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / ".harness" / "events.jsonl"
            context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
                permission_policy=PermissionPolicy(mode="workspace-write"),
                event_bus=EventBus(events_path),
            )
            middleware = PermissionMiddleware(
                tool_context=context,
                tool_registry=BUILTIN_TOOL_REGISTRY,
            )

            blocked = middleware.before_tool(
                "run_bash",
                {"command": "npm install"},
                messages=[],
                agent_name="main_agent",
            )

            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertIsNotNone(blocked)
            self.assertEqual(blocked.metadata["status_source"], "approval")
            self.assertIn("[approval_denied]", blocked.output)
            event_types = [event["type"] for event in events]
            self.assertIn("approval_requested", event_types)
            self.assertIn("approval_decided", event_types)
            approval = next(
                event for event in events
                if event["type"] == "approval_decided" and event["payload"].get("tool") == "run_bash"
            )
            self.assertFalse(approval["payload"]["approved"])

    def test_permission_middleware_denial_in_agent_loop_emits_tool_result_and_failure_events(self):
        from harness_code_agent.agent.conversation import Agent, AgentConversation
        from harness_code_agent.runtime.permission_middleware import (
            PermissionMiddleware,
        )
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.sessions.events import EventBus
        from harness_code_agent.workspace.service import WorkspaceService

        class FakeCompletions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    message = SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="tc_shell",
                                type="function",
                                function=SimpleNamespace(
                                    name="run_bash",
                                    arguments='{"command":"rm -rf /"}',
                                ),
                            )
                        ],
                    )
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
                        usage=None,
                    )
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content="done", tool_calls=None),
                        finish_reason="stop",
                    )],
                    usage=None,
                )

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
                permission_policy=PermissionPolicy(mode="workspace-write"),
                event_bus=EventBus(),
            )
            middleware = PermissionMiddleware(
                tool_context=context,
                tool_registry=BUILTIN_TOOL_REGISTRY,
            )
            shell_schemas = tool_schemas_for_profile(allowed_permissions={"shell"})

            with patch("harness_code_agent.agent.conversation.get_client", return_value=FakeClient()):
                conversation = AgentConversation(
                    Agent(
                        "main_agent",
                        "system",
                        use_tools=True,
                        tool_schemas=shell_schemas,
                        middlewares=[middleware],
                        tool_context=context,
                    )
                )
            conversation.run_until_idle()

            event_types = [event.type for event in context.event_bus.events]
            self.assertIn("tool_call", event_types)
            self.assertIn("tool_result", event_types)
            self.assertIn("failure", event_types)
            self.assertNotIn("approval_requested", event_types)
            self.assertNotIn("approval_decided", event_types)
            tool_result = next(event for event in context.event_bus.events if event.type == "tool_result")
            self.assertEqual(tool_result.payload["status"], "failed")
            self.assertEqual(tool_result.payload["metadata"]["status_source"], "permission")

    def test_agent_loop_blocks_tool_calls_not_advertised_in_schema(self):
        from harness_code_agent.agent.conversation import Agent, AgentConversation
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.sessions.events import EventBus
        from harness_code_agent.workspace.service import WorkspaceService

        class FakeCompletions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    message = SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="tc_write",
                                type="function",
                                function=SimpleNamespace(
                                    name="write_file",
                                    arguments='{"path":"should_not_exist.txt","content":"bad"}',
                                ),
                            )
                        ],
                    )
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
                        usage=None,
                    )
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content="done", tool_calls=None),
                        finish_reason="stop",
                    )],
                    usage=None,
                )

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
                permission_policy=PermissionPolicy(mode="workspace-write"),
                event_bus=EventBus(),
            )
            read_only_schemas = tool_schemas_for_profile(allowed_permissions={"read"})
            with patch("harness_code_agent.agent.conversation.get_client", return_value=FakeClient()):
                conversation = AgentConversation(
                    Agent(
                        "consult_test",
                        "system",
                        use_tools=True,
                        tool_schemas=read_only_schemas,
                        tool_context=context,
                    )
                )
            conversation.run_until_idle()

            self.assertFalse((root / "should_not_exist.txt").exists())
            tool_result = next(event for event in context.event_bus.events if event.type == "tool_result")
            self.assertEqual(tool_result.payload["status"], "failed")
            self.assertEqual(tool_result.payload["metadata"]["status_source"], "permission")
            self.assertIn("not available", tool_result.payload["output"])

    def test_agent_loop_token_budget_fallback_blocks_pending_tool_calls(self):
        from harness_code_agent.agent.conversation import Agent, AgentConversation
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.sessions.events import EventBus
        from harness_code_agent.workspace.service import WorkspaceService

        class FakeCompletions:
            def create(self, **kwargs):
                message = SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="tc_write",
                            type="function",
                            function=SimpleNamespace(
                                name="write_file",
                                arguments='{"path":"note.txt","content":"should not write"}',
                            ),
                        )
                    ],
                )
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
                    usage=SimpleNamespace(prompt_tokens=7, completion_tokens=5, total_tokens=12),
                )

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
                permission_policy=PermissionPolicy(mode="danger-full-access"),
                event_bus=EventBus(),
            )
            write_schemas = tool_schemas_for_profile(allowed_permissions={"edit"})
            with patch("harness_code_agent.agent.conversation.get_client", return_value=FakeClient()):
                conversation = AgentConversation(
                    Agent(
                        "main_agent",
                        "system",
                        use_tools=True,
                        tool_schemas=write_schemas,
                        tool_context=context,
                    )
                )
            with (
                patch("harness_code_agent.agent.conversation.config.MAX_AGENT_TOTAL_TOKENS", 10),
                patch("harness_code_agent.agent.conversation.config.MAX_AGENT_TOOL_CALLS", 100),
                patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=1),
            ):
                text = conversation.run_until_idle()

            self.assertFalse((root / "note.txt").exists())
            self.assertIn("Agent fallback triggered", text)
            fallback = next(event for event in context.event_bus.events if event.type == "agent_fallback")
            self.assertEqual(fallback.payload["reason"], "token_budget_exceeded")
            self.assertEqual(fallback.payload["limit_type"], "total_tokens")
            tool_result = next(event for event in context.event_bus.events if event.type == "tool_result")
            self.assertEqual(tool_result.payload["status"], "failed")
            self.assertEqual(tool_result.payload["metadata"]["status_source"], "budget")

    def test_agent_loop_token_budget_covers_response_only_pre_exit_turns(self):
        from harness_code_agent.agent.conversation import Agent, AgentConversation

        class AlwaysInject:
            def per_iteration(self, iteration, messages, runtime_state=None, agent_name=None):
                return None

            def pre_exit(self, messages, runtime_state=None, agent_name=None):
                return "keep going"

        class FakeCompletions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls > 20:
                    # Safety net if the budget stop regresses: fail instead
                    # of hanging the suite forever (there is no iteration cap).
                    raise AssertionError("response-only turn was not stopped by token budget")
                message = SimpleNamespace(content="still thinking", tool_calls=None)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=message, finish_reason="stop")],
                    usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10),
                )

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        with patch("harness_code_agent.agent.conversation.get_client", return_value=FakeClient()):
            conversation = AgentConversation(Agent("main_agent", "system", use_tools=True))
        conversation.agent.middlewares = [AlwaysInject()]
        with (
            patch("harness_code_agent.agent.conversation.config.MAX_AGENT_TOTAL_TOKENS", 15),
            patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=1),
        ):
            text = conversation.run_until_idle()

        fallback = conversation.runtime_state.fallback
        self.assertEqual(fallback.stop_reason, "token_budget_exceeded")
        self.assertEqual(fallback.stop_limit_type, "total_tokens")
        self.assertIn("Agent fallback triggered", text)

    def test_agent_loop_tool_call_budget_blocks_unexecuted_pending_calls(self):
        from harness_code_agent.agent.conversation import Agent, AgentConversation
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.sessions.events import EventBus
        from harness_code_agent.workspace.service import WorkspaceService

        class FakeCompletions:
            def create(self, **kwargs):
                message = SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="tc_first",
                            type="function",
                            function=SimpleNamespace(
                                name="write_file",
                                arguments='{"path":"first.txt","content":"one"}',
                            ),
                        ),
                        SimpleNamespace(
                            id="tc_second",
                            type="function",
                            function=SimpleNamespace(
                                name="write_file",
                                arguments='{"path":"second.txt","content":"two"}',
                            ),
                        ),
                    ],
                )
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
                permission_policy=PermissionPolicy(mode="danger-full-access"),
                event_bus=EventBus(),
            )
            write_schemas = tool_schemas_for_profile(allowed_permissions={"edit"})
            with patch("harness_code_agent.agent.conversation.get_client", return_value=FakeClient()):
                conversation = AgentConversation(
                    Agent(
                        "main_agent",
                        "system",
                        use_tools=True,
                        tool_schemas=write_schemas,
                        tool_context=context,
                    )
                )
            with (
                patch("harness_code_agent.agent.conversation.config.MAX_AGENT_TOTAL_TOKENS", 100),
                patch("harness_code_agent.agent.conversation.config.MAX_AGENT_TOOL_CALLS", 1),
                patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=1),
            ):
                text = conversation.run_until_idle()

            self.assertTrue((root / "first.txt").exists())
            self.assertFalse((root / "second.txt").exists())
            self.assertIn("Agent fallback triggered", text)
            self.assertEqual(len([msg for msg in conversation.messages if msg.get("role") == "tool"]), 2)
            fallback = next(event for event in context.event_bus.events if event.type == "agent_fallback")
            self.assertEqual(fallback.payload["reason"], "tool_call_budget_exceeded")
            results = [event for event in context.event_bus.events if event.type == "tool_result"]
            self.assertEqual([event.payload["status"] for event in results], ["success", "failed"])
            self.assertEqual(results[-1].payload["metadata"]["status_source"], "budget")

    def test_subagent_message_injected_at_next_safe_boundary(self):
        from harness_code_agent.agent.conversation import Agent, AgentConversation

        class FakeCompletions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                text = "waiting for the child" if self.calls == 1 else "incorporating the report"
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content=text, tool_calls=None),
                        finish_reason="stop",
                    )],
                    usage=None,
                )

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        class QueueOnceMiddleware:
            # Stands in for the coordinator sink: the child's report is
            # queued while the main turn is already in flight.
            def __init__(self, conversation):
                self.conversation = conversation
                self.done = False

            def per_iteration(self, iteration, messages, runtime_state=None, agent_name=None):
                if not self.done:
                    self.done = True
                    self.conversation.queue_message(
                        "evidence contradicts the current plan",
                        tag="SUBAGENT MESSAGE from explorer",
                    )
                return None

            def pre_exit(self, messages, runtime_state=None, agent_name=None):
                return None

        with patch("harness_code_agent.agent.conversation.get_client", return_value=FakeClient()):
            conversation = AgentConversation(Agent("main_agent", "system", use_tools=True))
        conversation.agent.middlewares = [QueueOnceMiddleware(conversation)]

        with patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=1):
            text = conversation.run_until_idle()

        self.assertEqual(text, "incorporating the report")
        self.assertEqual(
            conversation.messages[-2],
            {
                "role": "user",
                "content": "[SUBAGENT MESSAGE from explorer]\nevidence contradicts the current plan",
            },
        )

    def test_agent_loop_time_budget_stops_run(self):
        from harness_code_agent.agent.conversation import Agent, AgentConversation
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.sessions.events import EventBus
        from harness_code_agent.workspace.service import WorkspaceService

        class FakeCompletions:
            def create(self, **kwargs):
                message = SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="tc_read",
                            type="function",
                            function=SimpleNamespace(
                                name="read_file",
                                arguments='{"path":"README.md"}',
                            ),
                        )
                    ],
                )
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("hello", encoding="utf-8")
            context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
                permission_policy=PermissionPolicy(mode="workspace-write"),
                event_bus=EventBus(),
            )
            read_schemas = tool_schemas_for_profile(allowed_permissions={"read"})
            with patch("harness_code_agent.agent.conversation.get_client", return_value=FakeClient()):
                conversation = AgentConversation(
                    Agent(
                        "main_agent",
                        "system",
                        use_tools=True,
                        tool_schemas=read_schemas,
                        tool_context=context,
                        time_budget=0.0,
                    )
                )
            with patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=1):
                text = conversation.run_until_idle()

            self.assertIn("Agent fallback triggered", text)
            fallback = next(event for event in context.event_bus.events if event.type == "agent_fallback")
            self.assertEqual(fallback.payload["reason"], "time_budget_exhausted")
            self.assertEqual(fallback.payload["limit_type"], "seconds")

    def test_agent_loop_budget_warning_emits_once(self):
        from harness_code_agent.agent.conversation import Agent, AgentConversation
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.sessions.events import EventBus
        from harness_code_agent.workspace.service import WorkspaceService

        class FakeCompletions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    message = SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="tc_read",
                                type="function",
                                function=SimpleNamespace(
                                    name="read_file",
                                    arguments='{"path":"README.md"}',
                                ),
                            )
                        ],
                    )
                else:
                    message = SimpleNamespace(content="done", tool_calls=None)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if self.calls == 1 else "stop")],
                    usage=SimpleNamespace(prompt_tokens=30, completion_tokens=30, total_tokens=60),
                )

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("hello", encoding="utf-8")
            context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
                permission_policy=PermissionPolicy(mode="workspace-write"),
                event_bus=EventBus(),
            )
            read_schemas = tool_schemas_for_profile(allowed_permissions={"read"})
            with patch("harness_code_agent.agent.conversation.get_client", return_value=FakeClient()):
                conversation = AgentConversation(
                    Agent(
                        "main_agent",
                        "system",
                        use_tools=True,
                        tool_schemas=read_schemas,
                        tool_context=context,
                    )
                )
            with (
                patch("harness_code_agent.agent.conversation.config.MAX_AGENT_TOTAL_TOKENS", 200),
                patch("harness_code_agent.agent.conversation.config.MAX_AGENT_TOOL_CALLS", 100),
                patch("harness_code_agent.agent.conversation.config.AGENT_BUDGET_WARN_FRACTION", 0.25),
                patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=1),
            ):
                conversation.run_until_idle()

            warnings = [event for event in context.event_bus.events if event.type == "agent_budget_warning"]
            self.assertEqual(len(warnings), 1)
            self.assertEqual(warnings[0].payload["limit_type"], "total_tokens")

    def test_permission_middleware_blocks_blacklisted_shell_without_approval(self):
        from harness_code_agent.runtime.permission_middleware import (
            PermissionMiddleware,
        )
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.sessions.events import EventBus
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
                permission_policy=PermissionPolicy(mode="workspace-write"),
                event_bus=EventBus(),
            )
            middleware = PermissionMiddleware(
                tool_context=context,
                tool_registry=BUILTIN_TOOL_REGISTRY,
            )

            blocked = middleware.before_tool(
                "run_bash",
                {"command": "rm -rf /"},
                messages=[],
                agent_name="main_agent",
            )

            self.assertIsNotNone(blocked)
            self.assertEqual(blocked.metadata["status_source"], "permission")
            self.assertIn("[blocked]", blocked.output)
            self.assertIn("安全黑名单", blocked.output)

    def test_env_shell_command_is_treated_as_read(self):
        from harness_code_agent.runtime.permissions import PermissionPolicy

        policy = PermissionPolicy(mode="workspace-write")
        decision = policy.decide_tool_call("run_bash", {"command": "env"})

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.risk, "shell_safe")

    def test_execute_tool_apply_patch_records_snapshot_and_rejects_ambiguous_patch(self):
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.sessions.events import EventBus
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("old\n", encoding="utf-8")
            events_path = root / ".harness" / "events.jsonl"
            context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
                permission_policy=PermissionPolicy(mode="workspace-write"),
                event_bus=EventBus(events_path),
            )

            result = execute_tool(
                "apply_patch",
                {"path": "note.txt", "search": "old\n", "replace": "new\n"},
                tool_context=context,
                agent_name="main_agent",
            )
            ambiguous = execute_tool(
                "apply_patch",
                {"path": "note.txt", "search": "", "replace": "x"},
                tool_context=context,
                agent_name="main_agent",
            )

            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertIn("Patched note.txt", result)
            self.assertIn("Patch search text must not be empty", ambiguous)
            self.assertIn("kind=invalid_arguments", ambiguous)
            self.assertEqual((root / "note.txt").read_text(encoding="utf-8"), "new\n")
            self.assertTrue(any(event["type"] == "file_change" for event in events))
            self.assertFalse(any(event["type"] == "file_changed" for event in events))

    def test_permission_middleware_allows_approved_tool_call(self):
        from harness_code_agent.runtime.approvals import StaticApprovalProvider
        from harness_code_agent.runtime.permission_middleware import (
            PermissionMiddleware,
        )
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.sessions.events import EventBus
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / ".harness" / "events.jsonl"
            context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
                permission_policy=PermissionPolicy(mode="workspace-write"),
                event_bus=EventBus(events_path),
                approval_provider=StaticApprovalProvider(approved=True, reason="test approval"),
            )
            middleware = PermissionMiddleware(
                tool_context=context,
                tool_registry=BUILTIN_TOOL_REGISTRY,
            )

            result = middleware.before_tool(
                "run_bash",
                {"command": "npm install"},
                messages=[],
                agent_name="main_agent",
            )
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertIsNone(result)
            requested = next(event for event in events if event["type"] == "approval_requested")
            decided = next(event for event in events if event["type"] == "approval_decided")
            self.assertEqual(requested["payload"]["tool"], "run_bash")
            self.assertEqual(requested["payload"]["risk"], "shell_risky")
            self.assertEqual(decided["payload"]["tool"], "run_bash")
            self.assertTrue(decided["payload"]["approved"])

    def test_permission_middleware_records_llm_auto_approval_metadata(self):
        from harness_code_agent.runtime.approvals import StaticApprovalProvider
        from harness_code_agent.runtime.permission_middleware import (
            PermissionMiddleware,
        )
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.sessions.events import EventBus
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / ".harness" / "events.jsonl"
            context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
                permission_policy=PermissionPolicy(mode="llm-auto"),
                event_bus=EventBus(events_path),
                approval_provider=StaticApprovalProvider(
                    approved=True,
                    reason="llm-auto approved: bounded",
                ),
            )
            context.approval_provider.request = lambda request: SimpleNamespace(
                approved=True,
                reason="llm-auto approved: bounded",
                metadata={
                    "approval_source": "llm_auto",
                    "model": "fast-model",
                    "confidence": 0.9,
                    "raw_reason": "bounded",
                },
            )
            middleware = PermissionMiddleware(
                tool_context=context,
                tool_registry=BUILTIN_TOOL_REGISTRY,
            )

            result = middleware.before_tool(
                "run_bash",
                {"command": "npm install"},
                messages=[],
                agent_name="main_agent",
            )
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertIsNone(result)
            requested = next(event for event in events if event["type"] == "approval_requested")
            decided = next(event for event in events if event["type"] == "approval_decided")
            self.assertEqual(requested["payload"]["risk"], "shell_risky")
            self.assertTrue(decided["payload"]["approved"])
            self.assertEqual(decided["payload"]["metadata"]["approval_source"], "llm_auto")
            self.assertEqual(decided["payload"]["metadata"]["confidence"], 0.9)

    def test_static_verifier_passes_clean_python_file(self):
        import subprocess

        from harness_code_agent.runtime.middleware import StaticVerifierMiddleware

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init"], cwd=root, capture_output=True, check=False)
            (root / "ok.py").write_text("x = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, capture_output=True, check=False)
            subprocess.run(["git", "commit", "-m", "init"], cwd=root, capture_output=True, check=False)
            (root / "ok.py").write_text("x = 2\n", encoding="utf-8")

            mw = StaticVerifierMiddleware(workspace_root=str(root))
            result = mw.pre_exit(messages=[])

            self.assertIsNone(result)

    def test_static_verifier_ignores_preexisting_dirty_python_files(self):
        from harness_code_agent.runtime.middleware import StaticVerifierMiddleware
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bad.py").write_text("def f(\n", encoding="utf-8")

            workspace = WorkspaceService(root=root)
            mw = StaticVerifierMiddleware(workspace_root=str(root), workspace=workspace)
            mw.begin_turn("task", messages=[])
            result = mw.pre_exit(messages=[])

            self.assertIsNone(result)

    def test_static_verifier_git_baseline_ignores_preexisting_dirty_files(self):
        import subprocess

        from harness_code_agent.runtime.middleware import StaticVerifierMiddleware

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init"], cwd=root, capture_output=True, check=False)
            (root / "ok.py").write_text("x = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, capture_output=True, check=False)
            subprocess.run(["git", "commit", "-m", "init"], cwd=root, capture_output=True, check=False)
            # Broken file is already dirty *before* the turn starts.
            (root / "bad.py").write_text("def f(\n", encoding="utf-8")

            mw = StaticVerifierMiddleware(workspace_root=str(root))
            mw.begin_turn("task", messages=[])
            result = mw.pre_exit(messages=[])

            self.assertIsNone(result)

    def test_static_verifier_catches_shell_edit_of_preexisting_dirty_file(self):
        import subprocess

        from harness_code_agent.runtime.middleware import StaticVerifierMiddleware

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init"], cwd=root, capture_output=True, check=False)
            (root / "good.py").write_text("x = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, capture_output=True, check=False)
            subprocess.run(["git", "commit", "-m", "init"], cwd=root, capture_output=True, check=False)
            # Already dirty (but valid) before the turn begins.
            (root / "good.py").write_text("x = 2\n", encoding="utf-8")

            mw = StaticVerifierMiddleware(workspace_root=str(root))
            mw.begin_turn("task", messages=[])
            # Shell-like out-of-band rewrite: the file stays inside the
            # baseline dirty set, so only content fingerprints can see it.
            (root / "good.py").write_text("def f(\n", encoding="utf-8")

            with patch(
                "harness_code_agent.runtime.middleware.verification._check_ruff",
                return_value=[],
            ):
                result = mw.pre_exit(messages=[])

            self.assertIsNotNone(result)
            self.assertIn("LINT CHECK FAILED", result)
            self.assertIn("good.py", result)

    def test_static_verifier_catches_shell_written_file_via_git_delta(self):
        import subprocess

        from harness_code_agent.runtime.middleware import StaticVerifierMiddleware

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init"], cwd=root, capture_output=True, check=False)
            (root / "ok.py").write_text("x = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, capture_output=True, check=False)
            subprocess.run(["git", "commit", "-m", "init"], cwd=root, capture_output=True, check=False)

            mw = StaticVerifierMiddleware(workspace_root=str(root))
            mw.begin_turn("task", messages=[])
            # A file created out-of-band (e.g. via run_bash) — no workspace
            # service, no change journal, only the git delta can see it.
            (root / "bad.py").write_text("def f(\n", encoding="utf-8")

            with patch(
                "harness_code_agent.runtime.middleware.verification._check_ruff",
                return_value=[],
            ):
                result = mw.pre_exit(messages=[])

            self.assertIsNotNone(result)
            self.assertIn("LINT CHECK FAILED", result)
            self.assertIn("bad.py", result)

    def test_static_verifier_blocks_syntax_error_from_current_turn_workspace_change(self):
        from harness_code_agent.runtime.middleware import StaticVerifierMiddleware
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceService(root=root)
            mw = StaticVerifierMiddleware(workspace_root=str(root), workspace=workspace)
            mw.begin_turn("task", messages=[])
            workspace.write_text("bad.py", "def f(\n")

            with patch(
                "harness_code_agent.runtime.middleware.verification._check_ruff",
                return_value=[],
            ):
                result = mw.pre_exit(messages=[])

            self.assertIsNotNone(result)
            self.assertIn("LINT CHECK FAILED", result)
            self.assertIn("bad.py", result)

    def test_static_verifier_warns_only_once_then_allows_exit(self):
        from harness_code_agent.runtime.middleware import StaticVerifierMiddleware
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceService(root=root)
            mw = StaticVerifierMiddleware(workspace_root=str(root), workspace=workspace)
            mw.begin_turn("task", messages=[])
            workspace.write_text("warn.py", "x = 1\n")

            with patch(
                "harness_code_agent.runtime.middleware.verification._check_ruff",
                return_value=[("warn.py", "W292", "no newline at end of file", 1)],
            ):
                first = mw.pre_exit(messages=[])
                second = mw.pre_exit(messages=[])

            self.assertIsNotNone(first)
            self.assertIn("Lint warnings", first)
            self.assertIsNone(second)

    def test_static_verifier_ruff_blocks_error_findings_scoped_to_turn_files(self):
        import json
        from unittest.mock import MagicMock

        from harness_code_agent.runtime.middleware import StaticVerifierMiddleware
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "old.py").write_text("x = 1\n", encoding="utf-8")  # pre-existing
            workspace = WorkspaceService(root=root)
            mw = StaticVerifierMiddleware(workspace_root=str(root), workspace=workspace)
            mw.begin_turn("task", messages=[])
            workspace.write_text("new.py", "value = 1\n")

            payload = json.dumps([
                {
                    "code": "F821",
                    "message": "Undefined name `value`",
                    "filename": str(root / "new.py"),
                    "location": {"row": 3, "column": 1},
                }
            ])
            fake_run = MagicMock(return_value=SimpleNamespace(
                returncode=1, stdout=payload, stderr="",
            ))
            with patch(
                "harness_code_agent.runtime.middleware.verification.subprocess.run",
                fake_run,
            ):
                result = mw.pre_exit(messages=[])

            self.assertIsNotNone(result)
            self.assertIn("LINT CHECK FAILED", result)
            self.assertIn("F821", result)
            self.assertIn("new.py:3", result)
            # Ruff only ever sees the current-turn file, never the whole tree.
            argv = fake_run.call_args.args[0]
            self.assertIn("new.py", argv)
            self.assertNotIn("old.py", argv)

    def test_static_verifier_ruff_unusable_output_does_not_block_exit(self):
        from unittest.mock import MagicMock

        from harness_code_agent.runtime.middleware import StaticVerifierMiddleware
        from harness_code_agent.workspace.service import WorkspaceService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceService(root=root)
            mw = StaticVerifierMiddleware(workspace_root=str(root), workspace=workspace)
            mw.begin_turn("task", messages=[])
            workspace.write_text("ok.py", "x = 1\n")

            fake_run = MagicMock(return_value=SimpleNamespace(
                returncode=1, stdout="this is not json", stderr="",
            ))
            with patch(
                "harness_code_agent.runtime.middleware.verification.subprocess.run",
                fake_run,
            ):
                self.assertIsNone(mw.pre_exit(messages=[]))

    def test_static_verifier_skips_non_python_files(self):
        import subprocess

        from harness_code_agent.runtime.middleware import StaticVerifierMiddleware

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init"], cwd=root, capture_output=True, check=False)
            subprocess.run(["git", "add", "."], cwd=root, capture_output=True, check=False)
            subprocess.run(["git", "commit", "-m", "init"], cwd=root, capture_output=True, check=False)
            (root / "data.json").write_text("{}", encoding="utf-8")

            mw = StaticVerifierMiddleware(workspace_root=str(root))
            result = mw.pre_exit(messages=[])

            self.assertIsNone(result)

    def test_check_ruff_not_installed_gracefully_skips(self):
        from unittest.mock import patch as _patch

        from harness_code_agent.runtime.middleware import _check_ruff

        def fake_run(*a, **kw):
            raise FileNotFoundError

        with _patch("subprocess.run", side_effect=fake_run):
            result = _check_ruff("/tmp", ["x.py"])

        self.assertEqual(result, [])

    def test_check_ruff_timeout_is_non_blocking_warning(self):
        import subprocess

        from unittest.mock import patch as _patch

        from harness_code_agent.runtime.middleware import _check_ruff

        def fake_run(*a, **kw):
            raise subprocess.TimeoutExpired(cmd="ruff", timeout=30)

        with _patch("subprocess.run", side_effect=fake_run):
            result = _check_ruff("/tmp", ["x.py"])

        self.assertEqual(len(result), 1)
        self.assertTrue(result[0][1].startswith("RUFF"))
        self.assertNotIn("E", result[0][1][:1])
        self.assertNotIn("F", result[0][1][:1])

    # ------------------------------------------------------------------
    # safe_args_preview
    # ------------------------------------------------------------------

    def test_safe_args_preview_masks_sensitive_fields(self):
        from harness_code_agent.agent.conversation import safe_args_preview

        result = safe_args_preview({"path": "x.py", "content": "print('hello' * 999)"})
        self.assertNotIn("print", result)
        self.assertIn("chars", result)
        self.assertIn("path", result)

        result2 = safe_args_preview({"path": "x.py", "patch": "+def foo():\n    pass"})
        self.assertNotIn("def foo", result2)
        self.assertIn("chars", result2)

    def test_safe_args_preview_handles_large_fields(self):
        from harness_code_agent.agent.conversation import safe_args_preview

        large = "x" * 500
        result = safe_args_preview({"key": large})
        self.assertIn("chars", result)
        # JSON-serialized string includes quotes, so the char count is 500 + 2
        self.assertIn("502 chars", result)

    def test_safe_args_preview_sorts_keys_stably(self):
        from harness_code_agent.agent.conversation import safe_args_preview

        # Call multiple times — result should be identical
        args = {"z": 3, "a": 1, "path": "f.py"}
        r1 = safe_args_preview(args)
        r2 = safe_args_preview(args)
        self.assertEqual(r1, r2)
        # 'a' should come before 'z'
        self.assertLess(r1.index("a"), r1.index("z"))

    def test_safe_args_preview_respects_max_chars(self):
        from harness_code_agent.agent.conversation import safe_args_preview

        args = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}
        result = safe_args_preview(args, max_chars=30)
        self.assertLessEqual(len(result), 33)  # 30 + "..."

    def test_safe_args_preview_redacts_sensitive_key_names(self):
        from harness_code_agent.agent.conversation import safe_args_preview

        result = safe_args_preview({
            "path": "x.py",
            "api_key": "sk-1234567890",
            "jwt": "header.payload.signature",
            "token": "short-secret",
        })

        self.assertIn('"api_key": "[redacted]"', result)
        self.assertIn('"jwt": "[redacted]"', result)
        self.assertIn('"token": "[redacted]"', result)
        self.assertNotIn("short-secret", result)


class SessionResumeTests(unittest.TestCase):
    """Resuming a history session forks it instead of replacing the live one."""

    def setUp(self):
        import shutil

        self.shutil = shutil
        self.temp_dir = Path(tempfile.mkdtemp())
        self.env_patch = patch.dict(os.environ, {
            "HARNESS_MEMORY_GENERATION_DISABLED": "1",
        })
        self.env_patch.start()
        from harness_code_agent.core.interactive import InteractiveSession

        self.interactive = InteractiveSession(
            cwd=self.temp_dir,
            enable_turn_summary=False,
        )

    def tearDown(self):
        self.interactive.close()
        self.env_patch.stop()
        self.shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _seed_source_session(self):
        from harness_code_agent.sessions.events import UserInputEvent
        from harness_code_agent.sessions.journal import SessionJournal

        source = self.interactive.session
        journal = SessionJournal(source.journal_path)
        journal.append_message({"role": "user", "content": "记住：提交前跑测试"})
        journal.append_message({"role": "assistant", "content": "好的，我会跑测试。"})
        self.interactive.event_bus.emit_event(
            UserInputEvent(text="记住：提交前跑测试", turn=1).to_event()
        )
        return source

    def test_resume_forks_without_duplicating_messages(self):
        from harness_code_agent.sessions.journal import SessionJournal

        source = self._seed_source_session()
        source_journal_size = source.journal_path.stat().st_size
        # The source must be a *history* session: branch off it first.
        self.interactive.fork_current_session()
        middle = self.interactive.session

        self.interactive.resume_from_session(source.id)

        active = self.interactive.session
        self.assertNotEqual(active.id, source.id)
        self.assertNotEqual(active.id, middle.id)
        # Live messages: exactly one system + one user + one assistant.
        roles = [m["role"] for m in self.interactive.conversation.messages]
        self.assertEqual(roles, ["system", "user", "assistant"])
        # The fork journal file keeps one copy of each message: the system
        # message persisted at session start + user + assistant.
        entries = [
            entry for entry in SessionJournal(active.journal_path).read()
            if entry.kind == "message"
        ]
        self.assertEqual(len(entries), 3)
        # A fresh recovery from the fork journal matches the live messages.
        recovered = SessionJournal(active.journal_path).recovery_messages(
            self.interactive.agent.system_prompt
        )
        self.assertEqual(recovered, self.interactive.conversation.messages)
        # The source session journal is untouched by the resume.
        self.assertEqual(source.journal_path.stat().st_size, source_journal_size)

    def test_repeated_resume_does_not_multiply_messages(self):
        from harness_code_agent.sessions.journal import SessionJournal

        source = self._seed_source_session()
        self.interactive.fork_current_session()  # leave source behind
        self.interactive.resume_from_session(source.id)
        fork1 = self.interactive.session
        self.interactive.resume_from_session(source.id)
        fork2 = self.interactive.session
        self.assertNotEqual(fork1.id, fork2.id)
        roles = [m["role"] for m in self.interactive.conversation.messages]
        self.assertEqual(roles, ["system", "user", "assistant"])
        entries = [
            entry for entry in SessionJournal(fork2.journal_path).read()
            if entry.kind == "message"
        ]
        self.assertEqual(len(entries), 3)

    def test_resuming_current_session_is_noop(self):
        current = self.interactive.session
        self.interactive.resume_from_session(current.id)
        self.assertEqual(self.interactive.session.id, current.id)

    def test_forked_session_starts_as_running(self):
        source = self._seed_source_session()
        forked = self.interactive.session_store.fork(source.id)
        metadata = self.interactive.session_store.read_metadata(forked.id)
        self.assertEqual(metadata["status"], "running")
        self.assertEqual(metadata["forked_from"], source.id)

    def test_sessions_panel_excludes_current_session(self):
        from harness_code_agent.tui_bridge import BridgeServer

        source = self._seed_source_session()
        bridge = BridgeServer.__new__(BridgeServer)
        bridge._session = self.interactive
        bridge._session_error = None
        panel = bridge._sessions_panel()
        ids = {option["id"] for option in panel["options"]}
        self.assertNotIn(source.id, ids)


class ContextCommandTests(unittest.TestCase):
    def setUp(self):
        import shutil

        self.shutil = shutil
        self.temp_dir = Path(tempfile.mkdtemp())
        self.env_patch = patch.dict(os.environ, {
            "HARNESS_MEMORY_GENERATION_DISABLED": "1",
        })
        self.env_patch.start()
        from harness_code_agent.core.interactive import InteractiveSession

        self.interactive = InteractiveSession(
            cwd=self.temp_dir,
            enable_turn_summary=False,
            output_sink=lambda _text: None,
        )

    def tearDown(self):
        self.interactive.close()
        self.env_patch.stop()
        self.shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_context_status_returns_token_breakdown(self):
        result = self.interactive.context_status()

        self.assertIn("上下文估算", result)
        self.assertIn("系统指令", result)


if __name__ == "__main__":
    unittest.main()
