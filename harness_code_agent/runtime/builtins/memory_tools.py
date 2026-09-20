from __future__ import annotations

import json

from ...memory import MemoryService, MemoryWriteCommand
from ..tool_context import ToolContext
from ..tool_result import ToolResult


def _service(tool_context: ToolContext | None) -> MemoryService:
    if tool_context is None or tool_context.workspace is None:
        raise ValueError("memory tools require a workspace tool context")
    return MemoryService(tool_context.workspace.root)


def memory_search(
    query: str,
    scope: str = "both",
    paths: list[str] | None = None,
    tool_context: ToolContext | None = None,
) -> ToolResult:
    try:
        if tool_context is not None and not getattr(tool_context, "memory_use_enabled", True):
            raise ValueError("memory use is disabled for this session")
        if scope not in {"project", "user", "both"}:
            raise ValueError("scope must be project, user, or both")
        hits = _service(tool_context).search(query, scope=scope, paths=paths or [])
        output = MemoryService.format_hits(hits) or "No relevant memory found."
        return ToolResult(
            tool="memory_search", status="success", output=output,
            metadata={"status_source": "native", "result_count": len(hits)},
        )
    except (KeyError, OSError, ValueError) as exc:
        return _failure("memory_search", exc)


def memory_read(
    memory_id: str,
    scope: str = "project",
    tool_context: ToolContext | None = None,
) -> ToolResult:
    try:
        if tool_context is not None and not getattr(tool_context, "memory_use_enabled", True):
            raise ValueError("memory use is disabled for this session")
        if scope not in {"project", "user"}:
            raise ValueError("scope must be project or user")
        doc = _service(tool_context).read(memory_id, scope=scope)
        payload = {**doc.metadata(), "body": doc.body}
        return ToolResult(
            tool="memory_read", status="success",
            output=json.dumps(payload, ensure_ascii=False, indent=2),
            metadata={"status_source": "native", "memory_id": doc.id, "version": doc.version},
        )
    except (KeyError, OSError, ValueError) as exc:
        return _failure("memory_read", exc)


def memory_write(
    topic: str,
    body: str,
    scope: str = "project",
    applicability: str = "",
    source_paths: list[str] | None = None,
    memory_id: str = "",
    expected_version: int | None = None,
    supersedes: str = "",
    tool_context: ToolContext | None = None,
) -> ToolResult:
    try:
        if tool_context is not None and not getattr(tool_context, "memory_auto_extract_enabled", True):
            raise ValueError("memory generation is disabled for this session")
        if scope not in {"project", "user"}:
            raise ValueError("scope must be project or user")
        session_id = getattr(tool_context, "session_id", None)
        doc = _service(tool_context).write(
            MemoryWriteCommand(
                topic=topic, body=body, scope=scope, applicability=applicability,
                source_sessions=[session_id] if session_id else [],
                source_paths=source_paths or [], memory_id=memory_id or None,
                expected_version=expected_version, supersedes=supersedes or None,
            )
        )
        return ToolResult(
            tool="memory_write", status="success",
            output=f"Saved memory {doc.id} v{doc.version} ({doc.scope}, {doc.status}).",
            metadata={"status_source": "native", "memory_id": doc.id, "version": doc.version},
        )
    except (KeyError, OSError, ValueError) as exc:
        return _failure("memory_write", exc)


def memory_validate(
    memory_id: str,
    expected_version: int,
    scope: str = "project",
    tool_context: ToolContext | None = None,
) -> ToolResult:
    try:
        if tool_context is not None and not getattr(tool_context, "memory_auto_extract_enabled", True):
            raise ValueError("memory generation is disabled for this session")
        if scope not in {"project", "user"}:
            raise ValueError("scope must be project or user")
        doc = _service(tool_context).validate(memory_id, expected_version, scope=scope)
        return ToolResult(
            tool="memory_validate", status="success",
            output=f"Validated memory {doc.id}; now active at v{doc.version}.",
            metadata={"status_source": "native", "memory_id": doc.id, "version": doc.version},
        )
    except (KeyError, OSError, ValueError) as exc:
        return _failure("memory_validate", exc)


def memory_forget(
    memory_id: str,
    expected_version: int,
    scope: str = "project",
    tool_context: ToolContext | None = None,
) -> ToolResult:
    try:
        if tool_context is not None and not getattr(tool_context, "memory_auto_extract_enabled", True):
            raise ValueError("memory generation is disabled for this session")
        if scope not in {"project", "user"}:
            raise ValueError("scope must be project or user")
        _service(tool_context).forget(memory_id, expected_version, scope=scope)
        return ToolResult(
            tool="memory_forget", status="success",
            output=f"Forgot memory {memory_id}; its content was removed.",
            metadata={"status_source": "native", "memory_id": memory_id},
        )
    except (KeyError, OSError, ValueError) as exc:
        return _failure("memory_forget", exc)


def _failure(tool: str, exc: Exception) -> ToolResult:
    return ToolResult(
        tool=tool, status="failed", output=f"[error] {exc}", error=str(exc),
        metadata={"status_source": "validation"},
    )
