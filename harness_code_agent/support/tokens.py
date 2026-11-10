"""Token accounting shared by the agent loop, tools and memory.

tiktoken is optional: without it the estimate falls back to ~4 chars per
token, which keeps Terminal-Bench style environments installable.
"""
from __future__ import annotations

import json

from .. import config

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

# Try tiktoken for accurate counting; fall back to char-based estimation.
# This removes tiktoken as a hard dependency — critical for TB2 environments
# where pip install may be slow or unavailable.
_encoder = None
_use_tiktoken = False

try:
    import tiktoken
    _use_tiktoken = True
except ImportError:
    pass


def _get_encoder():
    global _encoder
    if not _use_tiktoken:
        return None
    if _encoder is None:
        try:
            _encoder = tiktoken.encoding_for_model(config.MODEL)
        except (KeyError, ValueError, OSError):
            _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(messages: list[dict]) -> int:
    """Rough token count for a message list.
    Uses tiktoken if available, otherwise estimates ~4 chars per token."""
    enc = _get_encoder()
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, list):
            text, attachment_tokens = _content_text_and_attachment_tokens(content)
            content = text
            total += attachment_tokens
        text = str(content) + str(msg.get("reasoning_content") or "")
        if enc:
            total += len(enc.encode(text, disallowed_special=())) + 4
        else:
            # ~4 chars per token is a reasonable approximation
            total += len(text) // 4 + 4
        for tc in msg.get("tool_calls", []):
            args = str(tc.get("function", {}).get("arguments", ""))
            if enc:
                total += len(enc.encode(args, disallowed_special=()))
            else:
                total += len(args) // 4
    return total


def count_text_tokens(text: str) -> int:
    """Token count for a single text string.
    Uses tiktoken if available, otherwise estimates ~4 chars per token.
    Shared by context accounting and tool-output size limits."""
    enc = _get_encoder()
    if enc:
        return len(enc.encode(text, disallowed_special=()))
    # ~4 chars per token is a reasonable approximation
    return len(text) // 4


def count_request_tokens(messages: list[dict], *, tool_schemas: list[dict] | None = None) -> int:
    """Estimate the full request size, including tool schema overhead."""
    total = count_tokens(messages)
    if not tool_schemas:
        return total
    schema_text = json.dumps(tool_schemas, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    enc = _get_encoder()
    if enc:
        return total + len(enc.encode(schema_text, disallowed_special=()))
    return total + len(schema_text) // 4


def _content_text_and_attachment_tokens(content: list) -> tuple[str, int]:
    """Flatten multimodal content without ever copying encoded binary data."""
    parts: list[str] = []
    attachment_tokens = 0
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
            continue
        metadata = block.get("attachment")
        if not isinstance(metadata, dict):
            parts.append("[binary attachment omitted]")
            attachment_tokens += 1_000
            continue
        name = str(metadata.get("name") or "attachment")
        mime_type = str(metadata.get("mimeType") or metadata.get("mime_type") or "application/octet-stream")
        size = int(metadata.get("size") or 0)
        sha256 = str(metadata.get("sha256") or "")
        parts.append(
            f"[attachment name={name} mime={mime_type} size={size} sha256={sha256[:12]}]"
        )
        # Binary inputs have provider-specific token costs. Use a bounded estimate
        # derived from metadata, never from the Base64 payload itself.
        attachment_tokens += min(16_384, max(256, size // 4_096))
    return " ".join(part for part in parts if part), attachment_tokens
