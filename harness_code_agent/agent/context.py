"""Context lifecycle helpers for token-safe structured compaction."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ..support.tokens import _content_text_and_attachment_tokens, count_tokens

log = logging.getLogger("harness")

DEFAULT_RECENT_TAIL_TOKENS = 16_384
DEFAULT_MIN_RECENT_MESSAGES = 2
MIN_COMPACT_REGION_TOKENS = 400

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
            parts.append(f"[{role}] {content}")
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            parts.append(f"[tool_call] {fn.get('name', '?')}({fn.get('arguments', '')})")
    return "\n".join(parts)
