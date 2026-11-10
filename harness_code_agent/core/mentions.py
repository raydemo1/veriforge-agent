from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..sessions.store import SessionStore
from ..sessions.summary import load_session_summary
from ..workspace.paths import is_relative_to


@dataclass(frozen=True)
class Mention:
    raw: str
    kind: str
    target: str


@dataclass(frozen=True)
class ResolvedMention:
    raw: str
    kind: str
    target: str
    resolved: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MentionResolutionError(ValueError):
    pass


def parse_mentions(text: str) -> list[Mention]:
    mode = os.environ.get("HARNESS_MENTION_MODE", "explicit").strip().lower()
    if mode == "off":
        return []
    if mode != "explicit":
        mode = "explicit"
    mentions: list[Mention] = []
    seen: set[tuple[str, str]] = set()
    for raw, token in _iter_mention_tokens(text):
        if not token:
            continue
        if token.startswith("session:"):
            target = token.removeprefix("session:")
            kind = "session"
        elif token.startswith("file:"):
            target = token.removeprefix("file:")
            kind = "file"
        else:
            continue
        key = (kind, target)
        if key in seen:
            continue
        seen.add(key)
        mentions.append(Mention(raw=raw, kind=kind, target=target))
    return mentions


def _iter_mention_tokens(text: str):
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "@" or (i > 0 and not text[i - 1].isspace()):
            i += 1
            continue
        start = i
        i += 1
        typed_prefix = None
        for prefix in ("file:", "session:"):
            if text.startswith(prefix, i):
                typed_prefix = prefix
                break
        if typed_prefix is None:
            while i < n and not text[i].isspace():
                i += 1
            continue

        i += len(typed_prefix)
        if i < n and text[i] == '"':
            i += 1
            chars = []
            escaped = False
            while i < n:
                ch = text[i]
                if escaped:
                    chars.append(ch)
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    i += 1
                    break
                else:
                    chars.append(ch)
                i += 1
            token = typed_prefix + "".join(chars).strip()
            raw = text[start:i]
            if token.removeprefix(typed_prefix):
                yield raw, token
            continue

        token_start = i
        while i < n and not text[i].isspace():
            i += 1
        token = text[token_start:i].rstrip(".,;:!?)]}")
        if not token or token.startswith("@"):
            continue
        raw = text[start:i]
        yield raw, typed_prefix + token



def resolve_mentions(
    text: str,
    *,
    workspace_root: str | Path,
    session_store: SessionStore,
) -> list[ResolvedMention]:
    root = Path(workspace_root).resolve()
    return [
        _resolve_file_mention(item, root)
        if item.kind == "file"
        else _resolve_session_mention(item, session_store)
        for item in parse_mentions(text)
    ]


def render_mention_context(resolved: list[ResolvedMention]) -> str:
    if not resolved:
        return ""
    parts = ["Mention context:"]
    for item in resolved:
        parts.append("")
        parts.append(f"- {item.raw} resolved as {item.kind}: {item.resolved}")
        parts.append("```text")
        parts.append(item.content)
        parts.append("```")
    return "\n".join(parts)


def _resolve_file_mention(mention: Mention, root: Path) -> ResolvedMention:
    if not mention.target:
        raise MentionResolutionError(f"Empty file mention: {mention.raw}")
    raw_path = Path(mention.target)
    if raw_path.is_absolute():
        candidate = raw_path.resolve()
    else:
        candidate = (root / raw_path).resolve()
    if not is_relative_to(candidate, root):
        raise MentionResolutionError(f"File mention escapes workspace: {mention.raw}")
    if not candidate.exists() or not (candidate.is_file() or candidate.is_dir()):
        raise MentionResolutionError(f"File mention not found: {mention.raw}")
    is_dir = candidate.is_dir()
    kind = "directory" if is_dir else "file"
    instruction = (
        "Use list_files to inspect this directory if needed."
        if is_dir
        else "Use read_file to inspect this file if needed."
    )
    content = "\n".join(
        [
            f"kind: {kind}",
            f"path: {candidate}",
            instruction,
        ]
    )
    return ResolvedMention(
        raw=mention.raw,
        kind=kind,
        target=mention.target,
        resolved=str(candidate),
        content=content,
        metadata={"path": str(candidate), "is_dir": is_dir},
    )


def _resolve_session_mention(
    mention: Mention,
    session_store: SessionStore,
) -> ResolvedMention:
    if not mention.target:
        raise MentionResolutionError(f"Empty session mention: {mention.raw}")
    try:
        content = load_session_summary(session_store, mention.target)
    except (FileNotFoundError, ValueError) as e:
        raise MentionResolutionError(f"Session mention not found: {mention.raw}") from e
    return ResolvedMention(
        raw=mention.raw,
        kind="session",
        target=mention.target,
        resolved=mention.target,
        content=content,
    )
