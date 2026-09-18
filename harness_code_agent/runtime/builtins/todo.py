"""Todo list tool implementation."""
from __future__ import annotations

import contextlib
import json
import os
import time
import uuid
from pathlib import Path

from ... import config
from ...agent.runtime_state import TodoList, normalize_todo_items
from ..tool_context import ToolContext
from ..tool_result import ToolResult


def update_todo(
    items: list[dict],
    *,
    runtime_state=None,
    agent_name: str | None = None,
    tool_context: ToolContext | None = None,
) -> ToolResult:
    """Create or replace the current execution todo list."""
    if runtime_state is None:
        return ToolResult(
            tool="update_todo",
            status="failed",
            output="[error] update_todo requires runtime state",
            error="update_todo requires runtime state",
            metadata={"status_source": "runtime"},
        )

    current = getattr(runtime_state, "todo", None)
    next_seq = current.next_seq if current is not None else 0
    revision = (current.revision if current is not None else 0) + 1
    try:
        normalized, next_seq = normalize_todo_items(items, next_seq=next_seq)
    except (TypeError, ValueError) as exc:
        return ToolResult(
            tool="update_todo",
            status="failed",
            output=f"[error] {exc}",
            error=str(exc),
            metadata={"status_source": "validation"},
        )

    updated_at = _utc_timestamp()
    todo = TodoList(
        items=normalized,
        revision=revision,
        updated_at=updated_at,
        next_seq=next_seq,
    )
    payload = todo.snapshot()

    workspace = tool_context.workspace.root if tool_context is not None else Path(config.WORKSPACE)
    state_path = _todo_state_path(workspace, runtime_state, tool_context)
    ok, error = _atomic_write_json(state_path, payload)
    if not ok:
        return ToolResult(
            tool="update_todo",
            status="failed",
            output=f"[error] Failed to write state.json atomically: {error}",
            error=f"Failed to write state.json atomically: {error}",
            metadata={"status_source": "native"},
        )

    runtime_state.todo = todo
    written = [str(state_path.relative_to(workspace))]
    counts = {status: sum(1 for item in normalized if item.status == status) for status in
              ("completed", "in_progress", "pending", "cancelled")}
    summary = ", ".join(f"{status} {count}" for status, count in counts.items() if count)
    return ToolResult(
        tool="update_todo",
        status="success",
        output=(
            f"Updated todo list: {', '.join(written)}. Revision {revision} ({summary}).\n"
            + todo.render()
        ),
        metadata={
            "status_source": "native",
            "todo_state": payload,
            "file_changes": [
                {"path": path, "operation": "write_file", "snapshot_path": None}
                for path in written
            ],
        },
    )


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _todo_state_path(workspace: Path, runtime_state, tool_context: ToolContext | None) -> Path:
    session_id = (
        (tool_context.session_id if tool_context is not None else None)
        or getattr(runtime_state, "session_id", None)
        or "default"
    )
    safe_session_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(session_id))
    return workspace / ".harness" / "sessions" / safe_session_id / "todo" / "state.json"


def _atomic_write_json(path: Path, payload: dict) -> tuple[bool, str | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    json.loads(text)
    temp_path = path.parent / f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        with open(temp_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        json.loads(temp_path.read_text(encoding="utf-8"))
        os.replace(temp_path, path)
        return True, None
    except OSError as exc:
        with contextlib.suppress(OSError):
            if temp_path.exists():
                temp_path.unlink()
        return False, str(exc)
