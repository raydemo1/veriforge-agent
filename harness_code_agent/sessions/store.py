from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any

from .events import EventBus
from .jsonl import read_records


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
        cwd: str | Path,
        model: str,
        permission_mode: str,
        session_id: str | None = None,
        environment: str = "local",
        status: str = "running",
    ) -> Session:
        from ..runtime.recovery import atomic_json

        session_id = session_id or self._new_session_id()
        session_root = self._session_root(session_id)
        snapshots_dir = session_root / "snapshots"
        compacted_dir = session_root / "compacted"

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
            "created_at": datetime.now(UTC).isoformat(),
            "cwd": str(Path(cwd).resolve()),
            "model": model,
            "permission_mode": permission_mode,
            "status": status,
            "environment": environment,
        }
        session_root.mkdir(parents=True, exist_ok=False)
        with ExitStack() as cleanup:
            cleanup.callback(self.remove_created, session)
            snapshots_dir.mkdir()
            (compacted_dir / "history").mkdir(parents=True)
            session.events_path.touch()
            session.journal_path.touch()
            atomic_json(session.metadata_path, metadata)
            cleanup.pop_all()
        return session

    def remove_created(self, session: Session) -> None:
        root = self._session_root(session.id)
        if session.root != root or root.resolve() != root:
            raise ValueError(f"Invalid new session directory: {session.id}")
        shutil.rmtree(root)

    def load(self, session_id: str) -> Session:
        self.read_metadata(session_id)
        root = self._session_root(session_id)
        return Session(session_id, root, root / "session.json", root / "events.jsonl",
                       root / "snapshots", root / "summary.md", root / "compacted", root / "journal.jsonl")

    def fork(self, source_session_id: str, *, status: str = "running") -> Session:
        from ..runtime.recovery import atomic_bytes, atomic_json
        source = self.read_metadata(source_session_id)
        boundary = source.get("fork_boundary")
        if boundary is None:
            raise ValueError("会话尚无完整、稳定的回合，暂时无法创建分支")
        source_root = self._session_root(source_session_id)
        journal = []
        sequence = boundary["journal_sequence"]
        if sequence:
            for entry in read_records(source_root / "journal.jsonl"):
                if entry["sequence"] > sequence:
                    break
                journal.append(entry)
                if entry["sequence"] == sequence:
                    break
            if not journal or journal[-1]["sequence"] != sequence:
                raise ValueError("会话记录未到达分支边界")
        events = list(islice(read_records(source_root / "events.jsonl"), boundary["event_count"]))
        if len(events) != boundary["event_count"]:
            raise ValueError("会话事件未到达分支边界")
        fork = self.create(cwd=source["cwd"], model=source["model"],
                           permission_mode=source["permission_mode"],
                           environment=source.get("environment", "local"), status=status)
        with ExitStack() as cleanup:
            cleanup.callback(self.remove_created, fork)
            for path, records in ((fork.journal_path, journal), (fork.events_path, events)):
                atomic_bytes(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records).encode("utf-8"))
            metadata = self.read_metadata(fork.id)
            if "model_selection" in source:
                metadata["model_selection"] = source["model_selection"]
            metadata.update(forked_from=source_session_id,
                            history_boundary=dict(boundary), fork_boundary=dict(boundary),
                            title=source.get("title", "新会话"))
            atomic_json(fork.metadata_path, metadata)
            cleanup.pop_all()
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

    def _update_metadata(self, session_id: str, **fields: Any) -> dict[str, Any]:
        from ..runtime.recovery import atomic_json

        metadata = self.read_metadata(session_id)
        metadata.update(fields)
        metadata_path = self._session_root(session_id) / "session.json"
        atomic_json(metadata_path, metadata)
        return metadata

    def update_status(self, session_id: str, status: str) -> dict[str, Any]:
        return self._update_metadata(session_id, status=status)

    def update_permission_mode(self, session_id: str, permission_mode: str) -> dict[str, Any]:
        return self._update_metadata(session_id, permission_mode=permission_mode)

    def read_events(self, session_id: str) -> list[dict[str, Any]]:
        session_root = self._session_root(session_id)
        events_path = session_root / "events.jsonl"
        if not events_path.exists():
            raise FileNotFoundError(f"Session events not found: {session_id}")
        return list(read_records(events_path))

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
        if not isinstance(session_id, str) or session_id in {"", ".", ".."} or any(char in session_id for char in "/\\:"):
            raise ValueError(f"Invalid session id: {session_id}")
        return self.sessions_dir / session_id

    def _summary_path(self, session_id: str) -> Path:
        return self._session_root(session_id) / "summary.md"

    @staticmethod
    def _new_session_id() -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{uuid.uuid4().hex[:8]}"
