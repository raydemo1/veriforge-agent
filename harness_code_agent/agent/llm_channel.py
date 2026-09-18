"""LLM request channel: streaming and non-streaming calls with timing events.

Separated from AgentConversation so request mechanics (retries, stream
fallback, thought events) live apart from conversation orchestration.
"""
from __future__ import annotations

import logging
import random
import time

from .. import config
from .cancellation import CancelledError
from .providers import ProviderAdapter, client_scope
from .utils import _usage_to_dict

log = logging.getLogger("harness")


class LlmStreamTimeoutError(TimeoutError):
    """Raised when a streaming model response stops making progress."""


class LlmRequestTimeoutError(TimeoutError):
    """Raised when a non-streaming model request exceeds the timeout."""


class MultimodalRequestError(RuntimeError):
    """Configured endpoint rejected an image request."""


_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_RETRYABLE_ERROR_TOKENS = (
    "rate_limit",
    "429",
    "apiconnectionerror",
    "connectionerror",
    "remoteprotocolerror",
)


def _is_retryable_llm_error(exc: BaseException) -> bool:
    """Provider-side 429/5xx or a transport connection failure."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _RETRYABLE_STATUS_CODES:
        return True
    haystack = f"{type(exc).__name__} {exc}".lower()
    return any(token in haystack for token in _RETRYABLE_ERROR_TOKENS)


def _retry_delay(attempt: int) -> float:
    """Exponential backoff capped at 20s plus a small jitter."""
    return min(2.0 ** (attempt + 1), 20.0) + random.uniform(0.0, 0.5)


def _call_with_retry(action, *, operation: str, attempts: int, cancellation_token=None):
    """Run ``action`` with bounded retries for retryable provider errors.

    Application-level retries complement the OpenAI client's transport
    retries: they cover error shapes the SDK surfaces directly (rate-limit
    JSON, reset connections) and keep backoff visible in the logs.
    """
    attempts = max(1, int(attempts))
    last_exc: Exception | None = None
    for attempt in range(attempts):
        if cancellation_token is not None and cancellation_token.is_cancelled:
            raise CancelledError("Turn cancelled by user")
        try:
            return action()
        except CancelledError:
            raise
        except Exception as exc:
            last_exc = exc
            if cancellation_token is not None and cancellation_token.is_cancelled:
                raise CancelledError("Turn cancelled by user") from exc
            if attempt + 1 >= attempts or not _is_retryable_llm_error(exc):
                raise
            delay = _retry_delay(attempt)
            log.warning(
                "%s hit a retryable provider error (attempt %d/%d), waiting %.1fs: %s",
                operation,
                attempt + 1,
                attempts,
                delay,
                exc,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _is_multimodal_rejection(exc: Exception, messages: object) -> bool:
    if not isinstance(messages, list) or not any(
        isinstance(message, dict)
        and isinstance(message.get("content"), list)
        and any(
            isinstance(block, dict) and block.get("type") in {"image_url", "file"}
            for block in message["content"]
        )
        for message in messages
    ):
        return False
    status_code = getattr(exc, "status_code", None)
    if status_code in {400, 415, 422}:
        return True
    detail = str(exc).lower()
    return any(token in detail for token in ("unsupported", "image", "pdf", "file", "content type", "multimodal"))


def _is_timeout_error(exc: BaseException) -> bool:
    error_name = type(exc).__name__.lower()
    return (
        isinstance(exc, TimeoutError)
        or "timeout" in error_name
        or "timed out" in str(exc).lower()
    )


def _stream_client(client):
    timeout = max(1.0, config.LLM_STREAM_IDLE_TIMEOUT_SECONDS)
    with_options = getattr(client, "with_options", None)
    if not callable(with_options):
        return client
    return with_options(timeout=timeout, max_retries=0)


def _reset_client_after_stream_timeout(conv) -> None:
    try:
        conv.client.close()
    except (OSError, RuntimeError):
        pass
    conv._client_needs_refresh = True


def llm_call_simple(messages: list[dict]) -> str:
    """Simple LLM call without tools — used for summarization.

    Retries 429/5xx/transport errors with bounded backoff so context
    compaction does not crash the agent; a terminal failure still returns
    a placeholder summary instead of propagating.
    """
    profile = config.resolve_model_profile("fast")
    adapter = ProviderAdapter(profile.provider)
    attempts = max(1, int(config.LLM_MAX_RETRIES) + 1)

    def _invoke() -> str:
        with client_scope() as client:
            resp = client.chat.completions.create(**adapter.chat_kwargs(
                profile=profile,
                messages=messages,
                max_tokens=10000,
            ))
        return resp.choices[0].message.content or ""

    try:
        return _call_with_retry(_invoke, operation="llm_call_simple", attempts=attempts)
    except CancelledError:
        raise
    except Exception as e:
        log.error("llm_call_simple failed: %s", e)
        # Return a minimal summary rather than crashing
        return "[context summarization failed — continuing with truncated context]"


class LlmChannel:
    """Performs one assistant request (streaming when the agent asks for it)."""

    def __init__(self, conversation):
        # The channel reads provider/client/trace/stream state through the
        # conversation it serves; it owns no state of its own.
        self.conversation = conversation

    def request_assistant_message(self, kwargs: dict, cancellation_token=None) -> tuple[dict, str | None] | None:
        conv = self.conversation
        if getattr(conv, "_client_needs_refresh", False):
            conv.refresh_client()
        if conv.agent.stream_callback is not None:
            stream_kwargs = dict(kwargs)
            stream_kwargs["stream"] = True
            call_id = conv.next_call_id()
            request_started = time.perf_counter()
            first_token_ms: int | None = None
            conv.emitter.emit_llm_request_started(call_id, streamed=True, model=str(stream_kwargs.get("model") or config.MODEL))
            saw_chunk = False
            thought_start_time: float | None = None
            thought_finished = False

            def finish_thought() -> None:
                nonlocal thought_finished
                if thought_start_time is None or thought_finished:
                    return
                thought_finished = True
                if conv.event_bus is not None:
                    from ..sessions.events import ThoughtFinishedEvent
                    conv.event_bus.emit_event(
                        ThoughtFinishedEvent(
                            duration_seconds=time.time() - thought_start_time,
                            source=conv.provider.name,
                        ).to_event()
                    )

            def on_chunk() -> None:
                nonlocal saw_chunk
                saw_chunk = True

            def on_text_delta(delta: str) -> None:
                nonlocal first_token_ms
                finish_thought()
                if first_token_ms is None:
                    first_token_ms = int((time.perf_counter() - request_started) * 1000)
                    conv.emitter.emit_llm_first_token(call_id, first_token_ms, model=str(stream_kwargs.get("model") or config.MODEL))
                conv.last_run_streamed_text = True
                conv.agent.stream_callback(delta)

            def on_reasoning_start() -> None:
                nonlocal thought_start_time
                if thought_start_time is not None or thought_finished:
                    return
                thought_start_time = time.time()
                if conv.event_bus is not None:
                    from ..sessions.events import ThoughtStartedEvent
                    conv.event_bus.emit_event(ThoughtStartedEvent().to_event())

            def on_reasoning_delta(delta: str) -> None:
                pass  # Reasoning content collected silently, not displayed

            remove_cancel_callback = self._interrupt_client_on_cancel(cancellation_token)
            try:
                if conv.provider.supports_prompt_cache_key:
                    stream_kwargs.setdefault("stream_options", {"include_usage": True})
                stream = _stream_client(conv.client).chat.completions.create(**stream_kwargs)
                result = conv.provider.assistant_message_from_stream(
                    stream,
                    on_text_delta=on_text_delta,
                    on_chunk=on_chunk,
                    on_reasoning_start=on_reasoning_start,
                    on_reasoning_delta=on_reasoning_delta,
                    cancellation_token=cancellation_token,
                )
                finish_thought()
                conv.emitter.emit_llm_response_finished(
                    call_id,
                    int((time.perf_counter() - request_started) * 1000),
                    finish_reason=result.finish_reason,
                    streamed=True,
                    first_token_ms=first_token_ms,
                    model=str(stream_kwargs.get("model") or config.MODEL),
                )
                conv.record_llm_usage(result.usage, kwargs.get("prompt_cache_key"))
                return result.assistant_message, result.finish_reason
            except CancelledError:
                raise
            except Exception as exc:
                if cancellation_token is not None and cancellation_token.is_cancelled:
                    raise CancelledError("Turn cancelled by user") from exc
                if _is_timeout_error(exc):
                    _reset_client_after_stream_timeout(conv)
                    raise LlmStreamTimeoutError("模型响应等待超时，请重试") from exc
                if _is_multimodal_rejection(exc, kwargs.get("messages")):
                    raise MultimodalRequestError(
                        "当前 endpoint 拒绝了图片输入。请检查模型的多模态能力和 "
                        "HARNESS_MODEL_INPUT_MODE 配置；VeriForge 不会自动切换模型或本地降级。"
                    ) from exc
                if saw_chunk:
                    raise
                conv.trace.error("stream_fallback", str(exc))
            finally:
                remove_cancel_callback()

        conv._check_cancelled(cancellation_token)
        call_id = conv.next_call_id()
        request_started = time.perf_counter()
        conv.emitter.emit_llm_request_started(call_id, streamed=False, model=str(kwargs.get("model") or config.MODEL))
        remove_cancel_callback = self._interrupt_client_on_cancel(cancellation_token)
        try:
            response = conv.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if cancellation_token is not None and cancellation_token.is_cancelled:
                raise CancelledError("Turn cancelled by user") from exc
            if _is_timeout_error(exc):
                raise LlmRequestTimeoutError("模型请求超时，请重试") from exc
            if _is_multimodal_rejection(exc, kwargs.get("messages")):
                raise MultimodalRequestError(
                    "当前 endpoint 拒绝了图片输入。请检查模型的多模态能力和 "
                    "HARNESS_MODEL_INPUT_MODE 配置；VeriForge 不会自动切换模型或本地降级。"
                ) from exc
            raise
        finally:
            remove_cancel_callback()
        conv._check_cancelled(cancellation_token)
        if not response.choices:
            return None
        choice = response.choices[0]
        conv.emitter.emit_llm_response_finished(
            call_id,
            int((time.perf_counter() - request_started) * 1000),
            finish_reason=choice.finish_reason,
            streamed=False,
            first_token_ms=None,
            model=str(kwargs.get("model") or config.MODEL),
        )
        conv.record_llm_usage(_usage_to_dict(getattr(response, "usage", None)), kwargs.get("prompt_cache_key"))

        return conv.provider.assistant_message_from_response(choice.message), choice.finish_reason
    def _interrupt_client_on_cancel(self, cancellation_token):
        if cancellation_token is None:
            return lambda: None
        conv = self.conversation
        client = conv.client

        def interrupt() -> None:
            try:
                client.close()
            except (OSError, RuntimeError):
                pass
            finally:
                conv._client_needs_refresh = True

        return cancellation_token.add_callback(interrupt)
