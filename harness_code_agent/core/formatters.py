"""Formatting helpers for CLI commands."""
from __future__ import annotations

from ..profiles import list_profiles
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
