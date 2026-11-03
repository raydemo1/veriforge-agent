"""Agent trace writer."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .. import config


def _truncate(s: str, n: int) -> str:
    return s[:n] + "..." if len(s) > n else s


class TraceWriter:
    """Appends structured events to a JSONL trace file in the harness directory.

    Each writer has its own file beneath the session's trace directory.
    """

    def __init__(self, agent_name: str, *, workspace: str | Path | None = None,
                 session_id: str | None = None, turn_id=None, request_id=None):
        self.agent_name = agent_name
        self.session_id = session_id or "standalone_" + uuid.uuid4().hex
        self._turn_id = turn_id or (lambda: None)
        self._request_id = request_id or (lambda: None)
        self._start_time = time.monotonic()
        self._lock = threading.Lock()
        self._path = None
        safe = lambda value: re.sub(r"[^a-zA-Z0-9_-]", "_", value)
        trace_dir = Path(workspace or Path.cwd()).resolve() / ".harness" / "traces" / safe(self.session_id)
        try:
            trace_dir.mkdir(parents=True, exist_ok=True)
            self._path = trace_dir / f"{safe(agent_name)}_{uuid.uuid4().hex}.jsonl"
        except OSError as exc:
            logging.getLogger(__name__).warning("Cannot create session trace directory: %s", exc)

    def _write(self, event_type: str, data: dict):
        with self._lock:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "t": round(time.monotonic() - self._start_time, 3),
                "agent": self.agent_name,
                "event": event_type,
                "session_id": self.session_id,
                "turn_id": self._turn_id(),
                "ui_request_id": self._request_id(),
                "tool_call_id": None,
                **data,
            }
            line = json.dumps(entry, ensure_ascii=False)
            if self._path is not None:
                try:
                    with self._path.open("a", encoding="utf-8") as stream:
                        stream.write(line + "\n")
                except OSError as exc:
                    self._path = None
                    logging.getLogger(__name__).warning("Session trace write failed: %s", exc)
            if config.TRACE_STDERR:
                import sys
                print(f"[TRACE] {line}", file=sys.stderr)

    def iteration(self, n: int, tokens: int):
        self._write("iteration", {"n": n, "tokens": tokens})

    def llm_response(self, content: str | None, tool_calls: list | None, finish_reason: str | None):
        self._write("llm_response", {
            "content": (content or "")[:500],
            "tool_calls": [tc["function"]["name"] for tc in (tool_calls or [])],
            "finish_reason": finish_reason,
        })

    def tool_call(self, name: str, args: dict, result: str, *, tool_call_id=None, shell_job_id=None):
        self._write("tool_call", {
            "tool": name,
            "args": _truncate(json.dumps(args, ensure_ascii=False), 300),
            "result": _truncate(result, 500),
            "tool_call_id": tool_call_id,
            "shell_job_id": shell_job_id,
        })

    def middleware_inject(self, source: str, hook: str, message: str):
        self._write("middleware", {
            "source": source,
            "hook": hook,
            "message": message[:300],
        })

    def tool_failure_policy(
        self,
        *,
        source: str,
        tool: str,
        tool_call_id: str,
        kind: str,
        category: str,
        visibility: str,
        attempt: int,
        mode: str,
        intercepted: bool,
        turn_failure_count: int,
        stop_reason: str = "",
    ):
        self._write("tool_failure_policy", {
            "source": source,
            "tool": tool,
            "tool_call_id": tool_call_id,
            "kind": kind,
            "category": category,
            "visibility": visibility,
            "attempt": attempt,
            "mode": mode,
            "intercepted": intercepted,
            "turn_failure_count": turn_failure_count,
            "stop_reason": stop_reason,
        })

    def context_event(self, event_type: str, reason: str = ""):
        self._write("context", {"type": event_type, "reason": reason})

    def error(self, error_type: str, message: str):
        self._write("error", {"type": error_type, "message": message[:500]})

    def finish(self, reason: str, iterations: int):
        self._write("finish", {"reason": reason, "iterations": iterations})
