"""Context lifecycle helpers for token-safe structured compaction."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field

from .. import config

log = logging.getLogger("harness")

DEFAULT_RECENT_TAIL_TOKENS = 16_384
DEFAULT_MIN_RECENT_MESSAGES = 2
MIN_COMPACT_REGION_TOKENS = 400

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
        text = str(content)
        if enc:
            total += len(enc.encode(text)) + 4
        else:
            # ~4 chars per token is a reasonable approximation
            total += len(text) // 4 + 4
        for tc in msg.get("tool_calls", []):
            args = str(tc.get("function", {}).get("arguments", ""))
            if enc:
                total += len(enc.encode(args))
            else:
                total += len(args) // 4
    return total


def count_text_tokens(text: str) -> int:
    """Token count for a single text string.
    Uses tiktoken if available, otherwise estimates ~4 chars per token.
    Shared by context accounting and tool-output size limits."""
    enc = _get_encoder()
    if enc:
        return len(enc.encode(text))
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
        return total + len(enc.encode(schema_text))
    return total + len(schema_text) // 4


# ---------------------------------------------------------------------------
# Context anxiety detection
# ---------------------------------------------------------------------------

@dataclass
class ContextAnxietySignal:
    detected: bool = False
    score: int = 0
    reasons: list[str] = field(default_factory=list)
    source: str = "assistant_recent_messages"

    def __bool__(self) -> bool:
        return self.detected


# Patterns that indicate the model is trying to wrap up prematurely
_ANXIETY_PATTERNS = [
    r"(?i)let me wrap up",
    r"(?i)i('ll| will) finalize",
    r"(?i)that should be (enough|sufficient)",
    r"(?i)i('ll| will) stop here",
    r"(?i)due to (the )?(context |token )?limit",
    r"(?i)running (low on|out of) (context|space|tokens)",
    r"(?i)to (save|conserve) (context|space|tokens)",
    r"(?i)i('ve| have) covered the (main|key|essential)",
    r"(?i)in the interest of (time|space|brevity)",
    r"上下文.*(快满|快没|不够|用完|空间)",
    r"(快没|没有|没).*上下文",
    r"先收尾",
    r"只覆盖关键",
]


def detect_anxiety(messages: list[dict]) -> ContextAnxietySignal:
    """
    Check recent assistant messages for signs of context anxiety —
    the model trying to wrap up work prematurely because it thinks
    it's running out of context space.
    """
    # Only check the last few assistant messages
    recent_texts = []
    for msg in reversed(messages[-10:]):
        if msg.get("role") == "assistant" and msg.get("content"):
            recent_texts.append(msg["content"])
        if len(recent_texts) >= 3:
            break

    combined = " ".join(recent_texts)
    reasons: list[str] = []
    for pattern in _ANXIETY_PATTERNS:
        match = re.search(pattern, combined)
        if match:
            reasons.append(match.group(0))
    if len(reasons) >= 2:
        log.warning(f"Context anxiety detected ({len(reasons)} signals found)")
        return ContextAnxietySignal(
            detected=True,
            score=len(reasons),
            reasons=reasons,
        )
    return ContextAnxietySignal()


def _compaction_economics(messages: list[dict]) -> bool:
    return count_tokens(messages) >= MIN_COMPACT_REGION_TOKENS


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _messages_to_text(messages: list[dict]) -> str:
    """Flatten messages into readable text for summarization."""
    parts = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content") or ""
        if isinstance(content, list):
            content, _attachment_tokens = _content_text_and_attachment_tokens(content)
        if content:
            parts.append(f"[{role}] {content[:3000]}")
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            parts.append(f"[tool_call] {fn.get('name', '?')}({fn.get('arguments', '')[:500]})")
    return "\n".join(parts)


def _sanitize_message_for_rebuild(message: dict) -> dict:
    msg = dict(message)
    content = msg.get("content")
    if isinstance(content, list):
        safe_content, _attachment_tokens = _content_text_and_attachment_tokens(content)
        msg["content"] = safe_content
    elif msg.get("role") == "tool" and isinstance(content, str):
        msg["content"] = _strip_tool_output_detail(content)
    elif isinstance(content, str) and len(content) > 2_000:
        msg["content"] = _fold_long_text(content, 2_000, label="REBUILD_CONTEXT_MESSAGE_SUMMARY")
    return msg


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


def _strip_tool_output_detail(content: str) -> str:
    """Strip the detail body from a tool output message.

    For OBS-formatted messages, keeps the metadata header lines only.
    For plain-text tool outputs, replaces with a compact placeholder."""
    # OBS format: keep header up to the observation:/preview boundary
    lines = content.split("\n")
    header_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("observation:", "--- preview ---")):
            break
        header_lines.append(line)
    if len(header_lines) < len(lines):
        return (
            "\n".join(header_lines)
            + "\ndetail: older full output discarded during context rebuild."
        )
    # Plain text — drop the body entirely
    digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:12]
    return (
        f"[tool output discarded during context rebuild]\n"
        f"original_chars: {len(content)}\n"
        f"sha256: {digest}"
    )


def _last_turn_messages(messages: list[dict], *, max_turns: int) -> list[dict]:
    if max_turns <= 0:
        return []
    user_seen = 0
    start = 1 if messages and messages[0].get("role") == "system" else 0
    for idx in range(len(messages) - 1, start - 1, -1):
        if messages[idx].get("role") == "user":
            user_seen += 1
            if user_seen >= max_turns:
                return messages[idx:]
    return messages[start:]


def _state_value(state: dict, key: str) -> str:
    value = state.get(key)
    if value is None or value == "":
        return "none"
    if isinstance(value, list):
        return _format_list(value)
    return str(value)


def _state_list(state: dict, key: str) -> str:
    value = state.get(key)
    if value is None or value == "":
        return "none"
    if isinstance(value, list):
        return _format_list(value)
    return str(value)


def _format_list(values: list) -> str:
    items = [str(item).strip() for item in values if str(item).strip()]
    return "\n".join(f"- {item}" for item in items) if items else "none"


def _fold_long_text(text: str, limit: int, *, label: str) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head_budget = max(200, limit // 2)
    tail_budget = max(200, limit - head_budget)
    omitted = max(0, len(text) - head_budget - tail_budget)
    return (
        f"[{label}]\n"
        f"original_chars: {len(text)}\n"
        f"omitted_chars: {omitted}\n\n"
        f"{text[:head_budget]}"
        f"\n\n...[{omitted} chars omitted]...\n\n"
        f"{text[-tail_budget:]}"
    )
