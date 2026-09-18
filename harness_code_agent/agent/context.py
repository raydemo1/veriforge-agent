"""Context lifecycle helpers for lightweight auto-compaction.

Auto-compaction starts near the context limit by summarizing older conversation.
If the context rapidly refills across turns, the session is reset from a
persisted handoff document. It never rewrites the system prompt or tools schema.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------

def summarize_older_conversation(
    messages: list[dict],
    llm_call,
    *,
    current_turn_start_index: int,
) -> list[dict]:
    """Summarize conversation before the current turn and keep current turn raw."""
    if not messages:
        return messages

    system = [messages[0]] if messages[0].get("role") == "system" else []
    system_len = len(system)
    split = max(system_len, min(current_turn_start_index, len(messages)))
    older = messages[system_len:split]
    current = messages[split:]
    if not older:
        return messages
    if not _compaction_economics(older):
        return messages

    old_text = _messages_to_text(older)
    summary = llm_call([
        {"role": "system", "content": (
            "You are summarizing older conversation history for a coding agent. "
            "Preserve decisions, active constraints, changed files, files touched, "
            "failed commands, recent errors, current status, and next action. "
            "Discard old full tool output, repeated logs, and resolved branches."
        )},
        {"role": "user", "content": old_text},
    ])
    summary_msg = {
        "role": "user",
        "content": f"[COMPACTED CONTEXT — summary of older conversation]\n{summary}",
    }
    return system + [summary_msg] + current


def create_handoff_reset(
    messages: list[dict],
    state: dict,
    llm_call,
    *,
    session_id: str = "default",
    profile: str = "",
    workspace: str = "",
    max_turns: int = 5,
) -> tuple[str, Path]:
    """Create a handoff document for a fresh context reset and persist it to temp."""
    if not messages:
        handoff = _fallback_handoff_text(state, session_id=session_id, profile=profile, workspace=workspace)
        return _persist_handoff_reset(handoff, session_id=session_id)

    recent = _last_turn_messages(messages, max_turns=max_turns)
    recent_text = _messages_to_text([
        _sanitize_message_for_rebuild(msg)
        for msg in recent
        if msg.get("role") != "system"
    ])
    reset_input = (
        "Create a handoff document for a fresh coding-agent session after an automatic context reset.\n\n"
        "Requirements:\n"
        "- Save-worthy content only; the next model context will be empty except for this handoff.\n"
        "- Include a Suggested Skills section.\n"
        "- Reference existing artifacts by path instead of duplicating them.\n"
        "- Redact secrets, credentials, tokens, and personal data.\n"
        "- Preserve current task, status, constraints, changed files, errors, and exact next action.\n"
        "- Mention that shell state may persist but cwd/env should be verified before relying on it.\n\n"
        f"Session: {session_id or 'default'}\n"
        f"Profile: {profile or 'unknown'}\n"
        f"Workspace: {workspace or config.WORKSPACE}\n\n"
        f"## Current User Task\n{_state_value(state, 'current_user_task')}\n\n"
        f"## Current Todo\n{_state_value(state, 'current_todo')}\n\n"
        f"## Changed Files\n{_state_list(state, 'changed_files')}\n\n"
        f"## Files Touched\n{_state_list(state, 'files_touched')}\n\n"
        f"## Recent Errors\n{_state_list(state, 'recent_errors')}\n\n"
        f"## Failed Commands\n{_state_list(state, 'failed_commands')}\n\n"
        f"## Active Constraints\n{_state_list(state, 'active_constraints')}\n\n"
        f"## Latest Checkpoint Summary\n{_state_value(state, 'latest_checkpoint_summary')}\n\n"
        f"## Last {max_turns} Turns\n{recent_text or 'none'}\n\n"
        f"## Next Recommended Action\n{_state_value(state, 'next_recommended_action')}\n\n"
        "Avoid duplicating old full tool output, read_file content, repeated logs, and resolved branches."
    )
    handoff = llm_call([
        {"role": "system", "content": (
            "You are writing a concise but complete handoff document for a fresh agent context. "
            "Use Markdown headings. Include: Summary, Current State, Suggested Skills, "
            "Important Files/Artifacts, Known Issues, and Next Steps."
        )},
        {"role": "user", "content": reset_input},
    ])
    if not isinstance(handoff, str) or not handoff.strip():
        handoff = _fallback_handoff_text(state, session_id=session_id, profile=profile, workspace=workspace)
    return _persist_handoff_reset(handoff.strip(), session_id=session_id)


def restore_from_handoff_reset(handoff: str, system_prompt: str, handoff_path: str | Path) -> list[dict]:
    """Build a fresh message list from a persisted handoff reset document."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            "[HANDOFF RESET]\n"
            "The previous context was intentionally cleared after an automatic context reset. "
            "Continue this same session using the handoff document below as the source of continuity.\n\n"
            f"Handoff document path: {handoff_path}\n\n"
            "## Handoff Document\n"
            + handoff.strip()
            + "\n\nDo not assume old tool outputs are current facts. Re-read files or rerun commands before relying on exact current state."
        )},
    ]


def _compaction_economics(messages: list[dict]) -> bool:
    return count_tokens(messages) >= MIN_COMPACT_REGION_TOKENS


def _persist_handoff_reset(handoff: str, *, session_id: str) -> tuple[str, Path]:
    safe_session = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in (session_id or "default"))
    root = Path(tempfile.gettempdir()) / "harness-code-agent" / "handoffs" / safe_session
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = root / f"handoff-{stamp}-{int(time.time() * 1000) % 1000:03d}.md"
    path.write_text(handoff.rstrip() + "\n", encoding="utf-8")
    return handoff.rstrip(), path


def _fallback_handoff_text(state: dict, *, session_id: str, profile: str, workspace: str) -> str:
    return (
        "# Handoff Reset\n\n"
        "## Summary\n"
        "The previous context was cleared after repeated auto-compaction pressure. "
        "Continue the same session from the state below.\n\n"
        "## Current State\n"
        f"- Session: {session_id or 'default'}\n"
        f"- Profile: {profile or 'unknown'}\n"
        f"- Workspace: {workspace or config.WORKSPACE}\n"
        f"- Current task: {_state_value(state, 'current_user_task')}\n"
        f"- Current todo: {_state_value(state, 'current_todo')}\n"
        f"- Changed files: {_state_list(state, 'changed_files')}\n"
        f"- Files touched: {_state_list(state, 'files_touched')}\n\n"
        "## Suggested Skills\n"
        "- Use repo-specific skills only if the next task mentions them or requires their workflow.\n\n"
        "## Known Issues\n"
        f"{_state_list(state, 'recent_errors')}\n\n"
        "## Failed Commands\n"
        f"{_state_list(state, 'failed_commands')}\n\n"
        "## Active Constraints\n"
        f"{_state_list(state, 'active_constraints')}\n\n"
        "## Next Steps\n"
        f"{_state_value(state, 'next_recommended_action')}\n\n"
        "## Notes\n"
        "Shell state may persist across this reset, but verify cwd/env before relying on it. "
        "Re-read files or rerun commands before relying on exact current facts."
    )


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
