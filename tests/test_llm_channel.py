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


if __name__ == "__main__":
    unittest.main()
