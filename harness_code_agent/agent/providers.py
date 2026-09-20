"""Provider adapters for OpenAI-compatible chat completions."""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from contextlib import contextmanager, suppress
from dataclasses import dataclass

from openai import OpenAI

from .. import config
from ..provider_resolution import resolve_provider_name
from .utils import _get, _usage_to_dict

TextDeltaCallback = Callable[[str], None]
ChunkCallback = Callable[[], None]

# Explicit prompt-cache breakpoints ship on the OpenAI GPT-5.6 family and
# later model families only.
_OPENAI_BREAKPOINT_MODEL_RE = re.compile(r"^gpt-(\d+)(?:[.-](\d+))?(?:[.-].*)?$")


def openai_model_supports_cache_breakpoint(model: str | None) -> bool:
    """Return True for OpenAI models that accept prompt_cache_breakpoint."""
    match = _OPENAI_BREAKPOINT_MODEL_RE.fullmatch((model or "").strip().lower())
    if match is None:
        return False
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return (major, minor) >= (5, 6)


def supports_system_cache_breakpoint(provider, model: str | None) -> bool:
    """Capability gate: only the native OpenAI path on GPT-5.6+ models."""
    return (
        getattr(provider, "name", None) == "openai"
        and openai_model_supports_cache_breakpoint(model)
    )


def build_system_cache_breakpoint_content(
    stable_prefix: str,
    dynamic_suffix: str,
) -> list[dict]:
    """Render the system message as two text parts split at the cache boundary.

    The breakpoint marks the end of the reusable prefix; content after it
    (the memory index) may change without invalidating the cached prefix.
    """
    return [
        {
            "type": "text",
            "text": stable_prefix,
            "prompt_cache_breakpoint": {"mode": "explicit"},
        },
        {
            "type": "text",
            "text": dynamic_suffix,
        },
    ]


def get_client() -> OpenAI:
    client_config = _current_client_config()
    return OpenAI(
        api_key=client_config[0],
        base_url=client_config[1],
        timeout=client_config[2],
        max_retries=client_config[3],
    )


@contextmanager
def client_scope():
    """Yield one independently owned client and always release its transport."""
    client = get_client()
    try:
        yield client
    finally:
        with suppress(Exception):
            client.close()


def _current_client_config() -> tuple[str | None, str | None, float, int]:
    # The SDK never retries: LlmChannel is the single owner of 429/5xx/
    # connection retries, backoff and Retry-After handling.
    return (
        config.API_KEY,
        config.BASE_URL,
        float(config.LLM_REQUEST_TIMEOUT_SECONDS),
        0,
    )


def current_adapter() -> ProviderAdapter:
    return ProviderAdapter(
        name=resolve_provider_name(
            provider=config.PROVIDER,
            base_url=config.BASE_URL,
            model=config.MODEL,
        )
    )


@dataclass(frozen=True)
class StreamResult:
    assistant_message: dict
    finish_reason: str | None
    usage: dict | None = None


@dataclass(frozen=True)
class ProviderAdapter:
    name: str

    def chat_kwargs(
        self,
        *,
        model: str | None = None,
        profile: config.ModelProfile | None = None,
        messages: list[dict],
        max_tokens: int,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        stream: bool = False,
        prompt_cache_key: str | None = None,
        stream_options: dict | None = None,
    ) -> dict:
        if profile is not None:
            model = profile.model
        if model is None:
            raise ValueError("model or profile is required")
        kwargs = {
            "model": model,
            "messages": _strip_response_only_message_fields(messages, provider_name=self.name),
            "max_tokens": max_tokens,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if stream:
            kwargs["stream"] = True
        if prompt_cache_key:
            kwargs["prompt_cache_key"] = prompt_cache_key
        if stream_options:
            kwargs["stream_options"] = stream_options
        if profile is not None:
            if profile.reasoning_effort and self.supports_reasoning_effort:
                kwargs["reasoning_effort"] = profile.reasoning_effort
            if self.name == "deepseek" and profile.thinking is not None:
                thinking_type = "enabled" if profile.thinking else "disabled"
                kwargs["extra_body"] = {"thinking": {"type": thinking_type}}
        return kwargs

    def assistant_message_from_response(self, msg) -> dict:
        assistant_msg = {"role": "assistant", "content": _get(msg, "content")}
        reasoning_content = _reasoning_content_from(msg)
        if self.requires_reasoning_content_roundtrip and reasoning_content is not None:
            assistant_msg["reasoning_content"] = reasoning_content

        tool_calls = [_normalize_tool_call(tc) for tc in (_get(msg, "tool_calls") or [])]
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        return assistant_msg

    def assistant_message_from_stream(
        self,
        chunks: Iterable,
        *,
        on_text_delta: TextDeltaCallback | None = None,
        on_chunk: ChunkCallback | None = None,
        on_reasoning_start: Callable[[], None] | None = None,
        on_reasoning_delta: Callable[[str], None] | None = None,
        cancellation_token=None,
    ) -> StreamResult:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_by_index: dict[int, dict] = {}
        finish_reason: str | None = None
        usage: dict | None = None

        for chunk in chunks:
            _check_cancelled(cancellation_token)
            if on_chunk is not None:
                on_chunk()
            _check_cancelled(cancellation_token)
            chunk_usage = _usage_to_dict(_get(chunk, "usage"))
            if chunk_usage is not None:
                usage = chunk_usage
            choices = _get(chunk, "choices") or []
            if not choices:
                continue
            choice = choices[0]
            finish = _get(choice, "finish_reason")
            if finish is not None:
                finish_reason = finish
            delta = _get(choice, "delta") or {}

            text_delta = _get(delta, "content")
            if text_delta:
                content_parts.append(text_delta)
                if on_text_delta is not None:
                    on_text_delta(text_delta)

            reasoning_delta = _reasoning_content_from(delta)
            if reasoning_delta:
                if not reasoning_parts and on_reasoning_start is not None:
                    on_reasoning_start()
                reasoning_parts.append(reasoning_delta)
                if on_reasoning_delta is not None:
                    on_reasoning_delta(reasoning_delta)

            for tc_delta in _get(delta, "tool_calls") or []:
                index = _get(tc_delta, "index")
                if index is None:
                    index = len(tool_calls_by_index)
                entry = tool_calls_by_index.setdefault(
                    int(index),
                    {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                )
                call_id = _get(tc_delta, "id")
                if call_id:
                    entry["id"] = call_id
                call_type = _get(tc_delta, "type")
                if call_type:
                    entry["type"] = call_type
                fn_delta = _get(tc_delta, "function") or {}
                name_delta = _get(fn_delta, "name")
                if name_delta:
                    entry["function"]["name"] += name_delta
                args_delta = _get(fn_delta, "arguments")
                if args_delta:
                    entry["function"]["arguments"] += args_delta

        assistant_msg = {
            "role": "assistant",
            "content": "".join(content_parts) or None,
        }
        if self.requires_reasoning_content_roundtrip and reasoning_parts:
            assistant_msg["reasoning_content"] = "".join(reasoning_parts)
        tool_calls = [
            call
            for _, call in sorted(tool_calls_by_index.items())
            if call.get("id") or call["function"].get("name") or call["function"].get("arguments")
        ]
        if tool_calls:
            for idx, call in enumerate(tool_calls):
                if not call.get("id"):
                    call["id"] = f"call_{idx}"
            assistant_msg["tool_calls"] = tool_calls
        return StreamResult(assistant_message=assistant_msg, finish_reason=finish_reason, usage=usage)

    @property
    def requires_reasoning_content_roundtrip(self) -> bool:
        return self.name == "deepseek"

    @property
    def supports_prompt_cache_key(self) -> bool:
        return self.name == "openai"

    @property
    def supports_reasoning_effort(self) -> bool:
        return self.name in {"deepseek", "openai"}


def _check_cancelled(cancellation_token) -> None:
    if cancellation_token is not None and getattr(cancellation_token, "is_cancelled", False):
        from .cancellation import CancelledError
        raise CancelledError("Turn cancelled by user")


def _normalize_tool_call(tc) -> dict:
    fn = _get(tc, "function") or {}
    return {
        "id": _get(tc, "id"),
        "type": _get(tc, "type") or "function",
        "function": {
            "name": _get(fn, "name"),
            "arguments": _get(fn, "arguments") or "",
        },
    }


def _reasoning_content_from(value) -> str | None:
    reasoning_content = _get(value, "reasoning_content")
    if reasoning_content is None:
        reasoning_content = _get(_get(value, "model_extra") or {}, "reasoning_content")
    return reasoning_content


def _strip_response_only_message_fields(
    messages: list[dict],
    *,
    provider_name: str | None = None,
) -> list[dict]:
    """Return provider-bound messages without response-only bookkeeping fields."""
    cleaned: list[dict] = []
    for message in messages:
        if not isinstance(message, dict):
            cleaned.append(message)
            continue
        content = message.get("content")
        has_internal_attachment_metadata = isinstance(content, list) and any(
            isinstance(block, dict) and "attachment" in block for block in content
        )
        if "reasoning_content" not in message and not has_internal_attachment_metadata:
            cleaned.append(message)
            continue
        outbound = dict(message)
        outbound.pop("reasoning_content", None)
        if has_internal_attachment_metadata:
            outbound["content"] = [_provider_content_block(block, provider_name) for block in content]
        cleaned.append(outbound)
    return cleaned


def _provider_content_block(block, provider_name: str | None):
    if not isinstance(block, dict):
        return block
    cleaned = {key: value for key, value in block.items() if key != "attachment"}
    if provider_name == "deepseek" and cleaned.get("type") == "file":
        file_payload = cleaned.pop("file", None)
        if isinstance(file_payload, dict):
            cleaned.update({
                key: file_payload[key]
                for key in ("file_id", "file_data", "filename")
                if file_payload.get(key) is not None
            })
    return cleaned
