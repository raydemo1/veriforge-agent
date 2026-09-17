"""Agent trace writer."""
from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

from .. import config


def _truncate(s: str, n: int) -> str:
    return s[:n] + "..." if len(s) > n else s


class TraceWriter:
    """Appends structured events to a JSONL trace file in the harness directory.

    Each line is a JSON object with: timestamp, agent, event_type, and data.
    Trace file: {WORKSPACE}/.harness/traces/trace_{agent_name}.jsonl
    """

    def __init__(self, agent_name: str, *, workspace: str | Path | None = None):
        self.agent_name = agent_name
        self._start_time = time.time()
        trace_dir = Path(workspace or Path.cwd()).resolve() / ".harness" / "traces"
        try:
            trace_dir.mkdir(parents=True, exist_ok=True)
            test_file = trace_dir / f"trace_test_{agent_name}"
            test_file.write_text("test")
            test_file.unlink()
            self._path = trace_dir / f"trace_{agent_name}.jsonl"
        except OSError:
            # Workspace not writable, use harness-agent dir
            self._path = Path(__file__).parent / f"trace_{agent_name}.jsonl"

    def _write(self, event_type: str, data: dict):
        with contextlib.suppress(Exception):  # never let tracing break the agent
            entry = {
                "t": round(time.time() - self._start_time, 2),
                "agent": self.agent_name,
                "event": event_type,
                **data,
            }
            line = json.dumps(entry, ensure_ascii=False)[:10000]
            # Write to file
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
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

    def tool_call(self, name: str, args: dict, result: str):
        self._write("tool_call", {
            "tool": name,
            "args": _truncate(json.dumps(args, ensure_ascii=False), 300),
            "result": _truncate(result, 500),
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
