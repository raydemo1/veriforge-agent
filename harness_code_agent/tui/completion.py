"""Completion logic for slash commands and @mentions (UI-agnostic)."""
from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

MENTION_CACHE_TTL_SECONDS = 30.0

EXCLUDED_DIRS = {
    ".git",
    ".harness",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    ".next",
}


@dataclass(frozen=True)
class MentionCandidate:
    insert_text: str
    display: str
    description: str
    kind: str


def current_mention_query(text_before_cursor: str) -> tuple[str, int] | None:
    """Extract the current @mention query from text before cursor."""
    quote_idx = text_before_cursor.rfind('@"')
    plain_idx = text_before_cursor.rfind("@")
    if quote_idx >= 0 and quote_idx + 2 >= plain_idx:
        prefix = text_before_cursor[quote_idx + 2:]
        if '"' in prefix:
            return None
        return prefix, -(len(prefix) + 2)
    if plain_idx == -1:
        return None
    if plain_idx > 0 and not text_before_cursor[plain_idx - 1].isspace():
        return None
    prefix = text_before_cursor[plain_idx + 1:]
    if any(ch.isspace() for ch in prefix):
        return None
    return prefix, -(len(prefix) + 1)


def mention_candidates(
    root: Path,
    prefix: str,
    session_store,
    *,
    limit: int = 50,
) -> list[MentionCandidate]:
    """Return file/session candidates matching the @mention prefix."""
    prefix = prefix.strip()
    if prefix.startswith("session:"):
        session_prefix = prefix.removeprefix("session:")
        return _session_candidates(session_store, session_prefix, limit=limit)
    if prefix.startswith("file:"):
        file_prefix = prefix.removeprefix("file:")
        return _file_candidates(root, file_prefix, limit=limit)

    # Keep recent conversations visible before workspace files. A global score
    # sort lets a large file tree crowd the history section out entirely.
    sessions = _session_candidates(session_store, prefix, limit=min(8, limit))
    files = _file_candidates(root, prefix, limit=max(0, limit - len(sessions)))
    return sessions + files


def _file_candidates(root: Path, prefix: str, *, limit: int) -> list[MentionCandidate]:
    file_rels = [
        rel for rel, is_dir in iter_workspace_paths(root, limit=2000) if not is_dir
    ]
    return _file_candidates_from_rels(file_rels, prefix, limit=limit)


def _file_candidates_from_rels(
    file_rels: list[str], prefix: str, *, limit: int
) -> list[MentionCandidate]:
    candidates = _scored_file_candidates(file_rels, prefix)
    candidates.sort(key=lambda item: (-item[0], item[1].display.lower()))
    return [candidate for _score, candidate in candidates[:limit]]


def _scored_file_candidates(
    file_rels: list[str], prefix: str
) -> list[tuple[int, MentionCandidate]]:
    prefix = prefix.strip().strip('"')
    file_candidates: list[tuple[int, MentionCandidate]] = []
    for rel in file_rels:
        score = fuzzy_score(prefix, rel)
        if score <= 0 and prefix:
            continue
        insert = f"file:{quote_mention_path(rel)}"
        file_candidates.append((
            score or 1,
            MentionCandidate(
                insert_text=insert,
                display=rel,
                description="当前工作区文件",
                kind="file",
            ),
        ))
    return file_candidates


def _session_candidates(session_store, prefix: str, *, limit: int) -> list[MentionCandidate]:
    session_items = [
        (item.get("id", ""), _session_preview(session_store, item.get("id", "")))
        for item in session_store.list_sessions()[:100]
    ]
    session_items = [(sid, preview) for sid, preview in session_items if preview]
    return _session_candidates_from_items(session_items, prefix, limit=limit)


def _session_candidates_from_items(
    session_items: list[tuple[str, str]], prefix: str, *, limit: int
) -> list[MentionCandidate]:
    candidates = _scored_session_candidates(session_items, prefix)
    candidates.sort(key=lambda item: -item[0])
    return [candidate for _score, candidate in candidates[:limit]]


def _scored_session_candidates(
    session_items: list[tuple[str, str]], prefix: str
) -> list[tuple[int, MentionCandidate]]:
    candidates: list[tuple[int, MentionCandidate]] = []
    for session_id, preview in session_items:
        score = max(fuzzy_score(prefix, session_id), fuzzy_score(prefix, preview))
        if score <= 0 and prefix:
            continue
        insert = f"session:{session_id}"
        candidates.append((
            score or 1,
            MentionCandidate(
                insert_text=insert,
                display=preview,
                description="历史会话",
                kind="session",
            ),
        ))
    return candidates


class MentionIndex:
    """Small session-level cache for @mention candidates.

    Holds only a file-list cache and a session-preview cache, both refreshed
    purely by a short TTL. Picking a candidate still goes through the normal
    attachment/resolve validation, so a briefly stale entry cannot cause a
    wrong result.
    """

    def __init__(
        self,
        root: str | Path,
        session_store,
        *,
        ttl_seconds: float = MENTION_CACHE_TTL_SECONDS,
    ) -> None:
        self._root = Path(root)
        self._session_store = session_store
        self._ttl = ttl_seconds
        self._file_rels: list[str] | None = None
        self._files_expire = 0.0
        self._session_items: list[tuple[str, str]] | None = None
        self._sessions_expire = 0.0

    def _cached_file_rels(self) -> list[str]:
        now = time.monotonic()
        if self._file_rels is None or now >= self._files_expire:
            self._file_rels = [
                rel
                for rel, is_dir in iter_workspace_paths(self._root, limit=2000)
                if not is_dir
            ]
            self._files_expire = now + self._ttl
        return self._file_rels

    def _cached_session_items(self) -> list[tuple[str, str]]:
        now = time.monotonic()
        if self._session_items is None or now >= self._sessions_expire:
            items: list[tuple[str, str]] = []
            for meta in self._session_store.list_sessions()[:100]:
                session_id = str(meta.get("id") or "")
                preview = _session_preview(self._session_store, session_id)
                if preview:
                    items.append((session_id, preview))
            self._session_items = items
            self._sessions_expire = now + self._ttl
        return self._session_items

    def candidates(self, prefix: str, *, limit: int = 50) -> list[MentionCandidate]:
        """Return file/session candidates matching the @mention prefix."""
        prefix = prefix.strip()
        if prefix.startswith("session:"):
            session_prefix = prefix.removeprefix("session:")
            return _session_candidates_from_items(
                self._cached_session_items(), session_prefix, limit=limit
            )
        if prefix.startswith("file:"):
            file_prefix = prefix.removeprefix("file:")
            return _file_candidates_from_rels(
                self._cached_file_rels(), file_prefix, limit=limit
            )

        # Keep recent conversations visible before workspace files. A global score
        # sort lets a large file tree crowd the history section out entirely.
        sessions = _session_candidates_from_items(
            self._cached_session_items(), prefix, limit=min(8, limit)
        )
        files = _file_candidates_from_rels(
            self._cached_file_rels(), prefix, limit=max(0, limit - len(sessions))
        )
        return sessions + files


def _session_preview(session_store, session_id: str) -> str:
    try:
        for event in reversed(session_store.read_events(session_id)):
            if event.get("type") != "user_input":
                continue
            preview = str((event.get("payload") or {}).get("text") or "").replace("\n", " ").strip()
            if preview:
                return preview[:72] + ("…" if len(preview) > 72 else "")
    except (OSError, ValueError, KeyError, TypeError):
        return ""
    return ""


def replace_mention_fragment(text: str, insert_text: str) -> str:
    """Replace the active @mention fragment while preserving surrounding text."""
    start = _active_mention_start(text)
    if start is None:
        return text
    end = start + 1
    if text.startswith('@"', start):
        end = _quoted_mention_end(text, start + 2)
    elif any(text.startswith(prefix + '"', start + 1) for prefix in ("file:", "session:")):
        quote_start = text.find('"', start + 1)
        end = _quoted_mention_end(text, quote_start + 1)
    else:
        while end < len(text) and not text[end].isspace():
            end += 1
    replacement = "@" + insert_text
    suffix = text[end:]
    if not suffix:
        suffix = " "
    return text[:start] + replacement + suffix


def _active_mention_start(text: str) -> int | None:
    for index in range(len(text) - 1, -1, -1):
        if text[index] != "@":
            continue
        if index > 0 and not text[index - 1].isspace():
            continue
        return index
    return None


def _quoted_mention_end(text: str, start: int) -> int:
    escaped = False
    i = start
    while i < len(text):
        ch = text[i]
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            return i + 1
        i += 1
    return i


def iter_workspace_paths(root: Path, *, limit: int = 2000) -> Iterable[tuple[str, bool]]:
    """Iterate workspace files and directories, excluding common dirs."""
    import os
    root = Path(root)
    results = []

    def walk(curr_dir: Path):
        try:
            for entry in os.scandir(curr_dir):
                if entry.name in EXCLUDED_DIRS:
                    continue
                if entry.is_symlink():
                    continue
                is_dir = entry.is_dir(follow_symlinks=False)
                entry_path = Path(entry.path)
                try:
                    rel = entry_path.relative_to(root).as_posix()
                except ValueError:
                    continue
                if not rel:
                    continue
                results.append((rel, is_dir))
                if is_dir:
                    walk(entry_path)
        except OSError:
            pass

    walk(root)
    results.sort(key=lambda item: item[0].lower())
    yield from results[:limit]


def quote_mention_path(path: str) -> str:
    """Quote a path containing spaces for @mention."""
    if any(ch.isspace() for ch in path):
        escaped = path.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return path


def fuzzy_score(query: str, candidate: str) -> int:
    """Score how well a candidate matches the query. Higher = better."""
    query = query.lower()
    candidate_lower = candidate.lower()
    if not query:
        return 1
    if candidate_lower.startswith(query):
        return 1000 - len(candidate)
    if query in candidate_lower:
        return 700 - candidate_lower.index(query)
    score = 0
    pos = 0
    streak = 0
    for ch in query:
        idx = candidate_lower.find(ch, pos)
        if idx == -1:
            return 0
        if idx == pos:
            streak += 1
            score += 15 + streak
        else:
            streak = 0
            score += 5
        pos = idx + 1
    return score
