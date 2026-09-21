from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .events import EventBus


@dataclass
class Session:
    id: str
    root: Path
    metadata_path: Path
    events_path: Path
    snapshots_dir: Path
    summary_path: Path
    compacted_dir: Path = Path(".")
    journal_path: Path = Path(".")


class SessionStore:
    """Durable local session storage under a Harness state directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.sessions_dir = self.root / "sessions"

    def create(
        self,
        *,
        profile: str,
        cwd: str | Path,
        model: str,
        permission_mode: str,
        resumed_from: str | None = None,
        profile_source: str | None = None,
    ) -> Session:
        session_id = self._new_session_id()
        session_root = self.sessions_dir / session_id
        snapshots_dir = session_root / "snapshots"
        compacted_dir = session_root / "compacted"
        session_root.mkdir(parents=True, exist_ok=False)
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        compacted_dir.mkdir(parents=True, exist_ok=True)
        (compacted_dir / "history").mkdir(parents=True, exist_ok=True)

        session = Session(
            id=session_id,
            root=session_root,
            metadata_path=session_root / "session.json",
            events_path=session_root / "events.jsonl",
            snapshots_dir=snapshots_dir,
            summary_path=session_root / "summary.md",
            compacted_dir=compacted_dir,
            journal_path=session_root / "journal.jsonl",
        )
        metadata = {
            "id": session.id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cwd": str(Path(cwd).resolve()),
            "profile": profile,
            "initial_profile": profile,
            "profile_source": profile_source or "explicit",
            "model": model,
            "permission_mode": permission_mode,
            "status": "running",
        }
        if resumed_from:
            metadata["resumed_from"] = resumed_from
        session.metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        session.events_path.write_text("", encoding="utf-8")
        session.journal_path.write_text("", encoding="utf-8")
        return session

    def fork(self, source_session_id: str) -> Session:
        source_metadata = self.read_metadata(source_session_id)
        source_events = self.read_events(source_session_id)
        session = self.create(
            profile=source_metadata["profile"],
            cwd=source_metadata["cwd"],
            model=source_metadata["model"],
            permission_mode=source_metadata["permission_mode"],
        )
        source_journal = self._session_root(source_session_id) / "journal.jsonl"
        if source_journal.exists():
            session.journal_path.write_text(
                source_journal.read_text(encoding="utf-8", errors="replace"),
                encoding="utf-8",
            )

        metadata = json.loads(session.metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                # Lineage lives in forked_from; status follows the lifecycle,
                # and a fresh branch starts as a running session.
                "status": "running",
                "forked_from": source_metadata.get("id", source_session_id),
                "forked_from_event_count": len(source_events),
                "forked_at": metadata["created_at"],
            }
        )
        session.metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.event_bus(session).emit(
            "session_forked",
            agent=None,
            payload={
                "source_session_id": source_metadata.get("id", source_session_id),
                "source_event_count": len(source_events),
            },
        )
        return session

    def event_bus(
        self,
        session: Session,
        *,
        listener: Callable[[Any], None] | None = None,
    ) -> EventBus:
        return EventBus(session.events_path, listener=listener)

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions = []
        if not self.sessions_dir.exists():
            return sessions
        for metadata_path in self.sessions_dir.glob("*/session.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            session_root = metadata_path.parent
            metadata.setdefault("id", session_root.name)
            metadata["events_path"] = str(session_root / "events.jsonl")
            sessions.append(metadata)
        sessions.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return sessions

    def latest_session(self) -> dict[str, Any]:
        sessions = self.list_sessions()
        if not sessions:
            raise FileNotFoundError("No sessions found.")
        return sessions[0]

    def read_metadata(self, session_id: str) -> dict[str, Any]:
        session_root = self._session_root(session_id)
        metadata_path = session_root / "session.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def update_status(self, session_id: str, status: str) -> dict[str, Any]:
        metadata = self.read_metadata(session_id)
        metadata["status"] = status
        metadata_path = self._session_root(session_id) / "session.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return metadata

    def update_permission_mode(self, session_id: str, permission_mode: str) -> dict[str, Any]:
        metadata = self.read_metadata(session_id)
        metadata["permission_mode"] = permission_mode
        metadata_path = self._session_root(session_id) / "session.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return metadata

    def update_profile(self, session_id: str, profile: str, profile_source: str | None = None) -> dict[str, Any]:
        metadata = self.read_metadata(session_id)
        metadata.setdefault("initial_profile", metadata.get("profile", profile))
        metadata["profile"] = profile
        if profile_source:
            metadata["profile_source"] = profile_source
        metadata_path = self._session_root(session_id) / "session.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return metadata

    def update_routing_mode(self, session_id: str, routing_mode: str) -> dict[str, Any]:
        metadata = self.read_metadata(session_id)
        metadata["routing_mode"] = routing_mode
        metadata_path = self._session_root(session_id) / "session.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return metadata

    def read_events(self, session_id: str) -> list[dict[str, Any]]:
        session_root = self._session_root(session_id)
        events_path = session_root / "events.jsonl"
        if not events_path.exists():
            raise FileNotFoundError(f"Session events not found: {session_id}")
        events = []
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            events.append(json.loads(line))
        return events

    def write_summary(self, session_id: str) -> str:
        from .summary import format_session_summary

        metadata = self.read_metadata(session_id)
        events = self.read_events(session_id)
        summary = format_session_summary(metadata, events, session_id=session_id)
        summary_path = self._summary_path(session_id)
        summary_path.write_text(summary + "\n", encoding="utf-8")
        return summary

    def read_lineage(self, session_id: str) -> list[dict[str, Any]]:
        lineage = []
        seen = set()
        current_id = session_id
        while current_id:
            if current_id in seen:
                raise ValueError(f"Session lineage cycle detected: {current_id}")
            seen.add(current_id)
            metadata = self.read_metadata(current_id)
            lineage.append(metadata)
            current_id = metadata.get("forked_from")
        lineage.reverse()
        return lineage

    def _session_root(self, session_id: str) -> Path:
        if "/" in session_id or "\\" in session_id or session_id in {"", ".", ".."}:
            raise ValueError(f"Invalid session id: {session_id}")
        return self.sessions_dir / session_id

    def _summary_path(self, session_id: str) -> Path:
        return self._session_root(session_id) / "summary.md"

    @staticmethod
    def _new_session_id() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{uuid.uuid4().hex[:8]}"
