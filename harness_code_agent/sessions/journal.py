from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JournalEntry:
    sequence: int
    kind: str
    timestamp: float
    payload: dict[str, Any]


class SessionJournal:
    """Append-only logical conversation journal used for recovery and audit."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()
        self._sequence = self._last_sequence()

    def append_message(self, message: dict) -> JournalEntry:
        return self.append("message", {"message": _json_safe(message)})

    def append_compaction(
        self,
        *,
        summary: str,
        first_kept_sequence: int,
        phase: str,
        state: dict | None = None,
    ) -> JournalEntry:
        return self.append(
            "compaction",
            {
                "summary": summary,
                "first_kept_sequence": first_kept_sequence,
                "phase": phase,
                "state": _json_safe(state or {}),
            },
        )

    def append(self, kind: str, payload: dict[str, Any]) -> JournalEntry:
        with self._lock:
            self._sequence += 1
            entry = JournalEntry(self._sequence, kind, time.time(), payload)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(entry.__dict__, ensure_ascii=False) + "\n")
            return entry

    def read(self) -> list[JournalEntry]:
        entries = []
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                data = json.loads(line)
                entries.append(JournalEntry(**data))
            except (json.JSONDecodeError, TypeError):
                continue
        return entries

    def recovery_messages(self, system_prompt: str) -> list[dict]:
        entries = self.read()
        compactions = [entry for entry in entries if entry.kind == "compaction"]
        if compactions:
            latest = compactions[-1]
            boundary = int(latest.payload.get("first_kept_sequence") or latest.sequence + 1)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "[COMPACTED CONTEXT]\n" + str(latest.payload.get("summary") or "")},
            ]
            messages.extend(
                entry.payload["message"]
                for entry in entries
                if entry.kind == "message"
                and entry.sequence >= boundary
                and entry.payload.get("message", {}).get("role") != "system"
            )
            return messages
        return [
            {"role": "system", "content": system_prompt},
            *[
                entry.payload["message"]
                for entry in entries
                if entry.kind == "message"
                and entry.payload.get("message", {}).get("role") != "system"
            ],
        ]

    @property
    def sequence(self) -> int:
        return self._sequence

    def first_sequence_of_recent_messages(self, count: int) -> int:
        messages = [entry for entry in self.read() if entry.kind == "message"]
        if count <= 0 or not messages:
            return self._sequence + 1
        return messages[max(0, len(messages) - count)].sequence

    def _last_sequence(self) -> int:
        try:
            lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
            return int(json.loads(lines[-1]).get("sequence", 0)) if lines else 0
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return 0


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_json_safe(item) for item in value]
        return str(value)
