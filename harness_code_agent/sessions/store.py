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
        profile_source: str | None = None,
        session_id: str | None = None,
    ) -> Session:
        session_id = session_id or self._new_session_id()
        session_root = self._session_root(session_id)
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
        session.metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        session.events_path.write_text("", encoding="utf-8")
        session.journal_path.write_text("", encoding="utf-8")
        return session

    def load(self, session_id: str) -> Session:
        self.read_metadata(session_id)
        root = self._session_root(session_id)
        return Session(session_id, root, root / "session.json", root / "events.jsonl",
                       root / "snapshots", root / "summary.md", root / "compacted", root / "journal.jsonl")

    def fork(self, source_session_id: str) -> Session:
        from ..runtime.recovery import atomic_json, atomic_bytes
        source = self.read_metadata(source_session_id)
        boundary = source.get("fork_boundary")
        if boundary is None:
            raise ValueError("会话尚无完整、稳定的回合，暂时无法创建分支")
        fork = self.create(profile=source["profile"], cwd=source["cwd"], model=source["model"],
                           permission_mode=source["permission_mode"], profile_source=source["profile_source"])
        journal = []
        source_path = self._session_root(source_session_id) / "journal.jsonl"
        with source_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry["sequence"] > boundary["journal_sequence"]:
                    break
                journal.append(line)
        atomic_bytes(fork.journal_path, "".join(journal).encode("utf-8"))
        events = []
        with (self._session_root(source_session_id) / "events.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if len(events) >= boundary["event_count"]:
                    break
                if line.strip():
                    events.append(line)
        atomic_bytes(fork.events_path, "".join(events).encode("utf-8"))
        metadata = self.read_metadata(fork.id)
        for key in ("model_selection", "routing_mode", "environment"):
            if key in source:
                metadata[key] = source[key]
        metadata.update(forked_from=source_session_id,
                        history_boundary=dict(boundary), fork_boundary=dict(boundary),
                        title=source.get("title", "新会话"))
        atomic_json(fork.metadata_path, metadata)
        return fork

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
            if metadata.get("status") in {"restoring", "discarded"}:
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
        from ..runtime.recovery import atomic_json

        metadata = self.read_metadata(session_id)
        metadata["status"] = status
        metadata_path = self._session_root(session_id) / "session.json"
        atomic_json(metadata_path, metadata)
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

    def read_history_events(self, session_id: str, *, event_count: int | None = None) -> list[dict[str, Any]]:
        events = self.read_events(session_id)
        if event_count is not None:
            events = events[:event_count]
        kept = []
        for index, event in enumerate(events):
            if event["type"] == "session_rewound":
                boundary = event["payload"]["event_count"]
                if not 0 <= boundary <= index:
                    raise ValueError("回撤历史边界无效")
                kept = [(i, item) for i, item in kept if i < boundary]
            kept.append((index, event))
        return [event for _, event in kept]

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
