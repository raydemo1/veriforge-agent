"""Formatting helpers for CLI commands."""
from __future__ import annotations

from ..sessions.store import SessionStore
from ..sessions.summary import load_session_summary


def print_session(store: SessionStore, session_id: str) -> None:
    print(load_session_summary(store, session_id))
