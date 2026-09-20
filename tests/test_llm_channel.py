from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from harness_code_agent.agent import llm_channel as ch


class _RateLimited(Exception):
    def __init__(self):
        super().__init__("429 rate limit")
        self.status_code = 429
        self.response = SimpleNamespace(headers={"retry-after": "0"})


class _StreamUnsupported(Exception):
    pass


class _FakeProvider:
    name = "fake"
    supports_prompt_cache_key = False

    def assistant_message_from_stream(self, stream, **kwargs):  # pragma: no cover
        raise AssertionError("stream should fail before parsing")

    def assistant_message_from_response(self, message):
        return {"role": "assistant", "content": "non-stream ok"}


class _FakeCompletions:
    def __init__(self, on_stream_create):
        self._on_stream_create = on_stream_create
        self.nonstream_calls = 0

    def create(self, **kwargs):
        if kwargs.get("stream"):
            return self._on_stream_create()
        self.nonstream_calls += 1
        choice = SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content="non-stream ok"),
        )
        return SimpleNamespace(choices=[choice], usage=None)


class _FakeClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)
        self.closed = False

    def with_options(self, **kwargs):
        return self

    def close(self):
        self.closed = True


class _FakeEmitter:
    def __getattr__(self, _name):
        return lambda *a, **k: None


class _FakeTrace:
    def __init__(self):
        self.errors = []

    def error(self, label, detail):
        self.errors.append((label, detail))


def _make_conversation(rate_limit: bool):
    def stream_create():
        raise _RateLimited() if rate_limit else _StreamUnsupported("no such stream mode")

    completions = _FakeCompletions(stream_create)
    client = _FakeClient(completions)
    return (
        SimpleNamespace(
            provider=_FakeProvider(),
            client=client,
            agent=SimpleNamespace(stream_callback=lambda delta: None),
            emitter=_FakeEmitter(),
            trace=_FakeTrace(),
            event_bus=None,
            _client_needs_refresh=False,
            last_run_streamed_text=False,
            next_call_id=lambda: "call-1",
            refresh_client=lambda: None,
            record_llm_usage=lambda *a, **k: None,
            _check_cancelled=lambda token=None: None,
        ),
        completions,
    )


class StreamFallbackPolicyTests(unittest.TestCase):
    def test_exhausted_retryable_error_does_not_fall_back_to_nonstream(self):
        conv, completions = _make_conversation(rate_limit=True)
        channel = ch.LlmChannel(conv)
        with patch.object(ch.config, "LLM_MAX_RETRIES", 0):
            with self.assertRaises(_RateLimited):
                channel.request_assistant_message(
                    {"model": "m", "messages": []}, cancellation_token=None
                )
        # No second retry budget spent on the non-streaming path.
        self.assertEqual(completions.nonstream_calls, 0)
        self.assertEqual(conv.trace.errors, [])

    def test_non_retryable_stream_error_falls_back_to_nonstream(self):
        conv, completions = _make_conversation(rate_limit=False)
        channel = ch.LlmChannel(conv)
        with patch.object(ch.config, "LLM_MAX_RETRIES", 0):
            message, finish = channel.request_assistant_message(
                {"model": "m", "messages": []}, cancellation_token=None
            )
        self.assertEqual(completions.nonstream_calls, 1)
        self.assertEqual(finish, "stop")
        self.assertEqual(message["content"], "non-stream ok")
        self.assertTrue(conv.trace.errors)


class _EmptyChoicesCompletions:
    def __init__(self, empty_calls):
        self.empty_calls = empty_calls
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.empty_calls:
            return SimpleNamespace(choices=[], usage=None)
        choice = SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content="non-stream ok"),
        )
        return SimpleNamespace(choices=[choice], usage=None)


def _make_nonstream_conversation(completions):
    client = _FakeClient(completions)
    return SimpleNamespace(
        provider=_FakeProvider(),
        client=client,
        agent=SimpleNamespace(stream_callback=None),
        emitter=_FakeEmitter(),
        trace=_FakeTrace(),
        event_bus=None,
        _client_needs_refresh=False,
        last_run_streamed_text=False,
        next_call_id=lambda: "call-1",
        refresh_client=lambda: None,
        record_llm_usage=lambda *a, **k: None,
        _check_cancelled=lambda token=None: None,
    )


class EmptyChoicesRetryTests(unittest.TestCase):
    def test_empty_choices_retried_within_budget_then_succeeds(self):
        completions = _EmptyChoicesCompletions(empty_calls=2)
        conv = _make_nonstream_conversation(completions)
        channel = ch.LlmChannel(conv)
        with patch.object(ch.config, "LLM_MAX_RETRIES", 2):
            message, finish = channel.request_assistant_message(
                {"model": "m", "messages": []}
            )
        self.assertEqual(completions.calls, 3)
        self.assertEqual(finish, "stop")
        self.assertEqual(message["content"], "non-stream ok")

    def test_persistent_empty_choices_raises_and_never_returns_none(self):
        completions = _EmptyChoicesCompletions(empty_calls=99)
        conv = _make_nonstream_conversation(completions)
        channel = ch.LlmChannel(conv)
        with patch.object(ch.config, "LLM_MAX_RETRIES", 2):
            with self.assertRaises(ch._EmptyChoicesError):
                channel.request_assistant_message(
                    {"model": "m", "messages": []}
                )
        self.assertEqual(completions.calls, 3)


class ConversationErrorBoundaryTests(unittest.TestCase):
    def test_channel_failure_ends_turn_with_a_single_request(self):
        from harness_code_agent.agent.conversation import Agent

        conv = Agent("test_agent", "sys", use_tools=False).start_conversation("task")
        finishes = []
        with (
            patch.object(
                conv.llm,
                "request_assistant_message",
                side_effect=ValueError("auth failed"),
            ) as request,
            patch.object(conv.trace, "finish", side_effect=lambda *a: finishes.append(a)),
        ):
            result = conv.run_until_idle()
        request.assert_called_once()
        self.assertEqual(result, "")
        self.assertEqual(finishes[0][0], "api_error")


if __name__ == "__main__":
    unittest.main()
