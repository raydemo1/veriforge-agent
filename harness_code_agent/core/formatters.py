"""Formatting helpers for CLI commands."""
from __future__ import annotations

from ..profiles import list_profiles
from ..sessions.journal import SessionJournal
from ..sessions.store import SessionStore
from ..sessions.summary import load_session_summary


def format_profiles() -> str:
    lines = ["Available profiles:", ""]
    for profile in list_profiles():
        lines.append(f"  {profile['name']:15s} {profile['description']}")
    return "\n".join(lines)


def print_profiles() -> None:
    print(format_profiles())


def print_session(store: SessionStore, session_id: str) -> None:
    print(load_session_summary(store, session_id))


def _build_resume_context(
    store: SessionStore,
    session_id: str,
    *,
    max_recent_events: int = 8,
) -> str:
    lineage = store.read_lineage(session_id)
    current = lineage[-1]
    journal_path = store.root / "sessions" / session_id / "journal.jsonl"
    if journal_path.exists() and journal_path.stat().st_size:
        recovered = SessionJournal(journal_path).recovery_messages("")
        lines = [
            f"Resuming session: {current.get('id', session_id)}",
            "The following state was rebuilt from the append-only session journal.",
            "Re-check current files before relying on old tool output.",
            "",
        ]
        for message in recovered[1:]:
            role = message.get("role", "unknown")
            content = str(message.get("content") or "")
            lines.append(f"[{role}] {content}")
        return "\n\n".join(lines)
    lines = [
        f"Resuming session: {current.get('id', session_id)}",
        "Lineage: " + " -> ".join(item.get("id", "") for item in lineage),
        f"Workspace: {current.get('cwd', '')}",
        f"Profile: {current.get('profile', '')}",
        f"Permission mode: {current.get('permission_mode', '')}",
    ]
    if current.get("forked_from"):
        lines.append(f"Forked from: {current.get('forked_from')}")
    lines.append("")
    lines.append("Recent session events:")
    for metadata in lineage:
        events = store.read_events(metadata["id"])
        if not events:
            lines.append(f"- {metadata['id']}: no events")
            continue
        lines.append(f"- {metadata['id']}:")
        for event in events[-max_recent_events:]:
            lines.append(f"  - {_event_summary(event)}")
    return "\n".join(lines)


def _event_summary(event: dict) -> str:
    payload = event.get("payload") or {}
    payload_bits = []
    for key in sorted(payload)[:4]:
        value = payload[key]
        text = str(value).replace("\n", " ")
        if len(text) > 80:
            text = text[:77] + "..."
        payload_bits.append(f"{key}={text}")
    suffix = f" ({', '.join(payload_bits)})" if payload_bits else ""
    return f"#{event.get('sequence')} {event.get('type')} agent={event.get('agent')}{suffix}"
