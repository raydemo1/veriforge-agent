from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from ...memory import EvidenceReference, MemoryService, MemoryWriteCommand
from ..tool_context import ToolContext
from ..tool_result import ToolResult


def _service(context, *, write=False):
    if context is None or context.workspace is None:
        raise ValueError("memory requires a workspace context")
    flag = "memory_auto_extract_enabled" if write else "memory_use_enabled"
    if not getattr(context, flag, True):
        raise ValueError("memory generation is disabled" if write else "memory use is disabled")
    service = getattr(context, "memory_service", None)
    if service is None:
        service = MemoryService(context.workspace.root)
        context.memory_service = service
    return service


def source_evidence(context, quote):
    path = getattr(context, "memory_journal_path", None)
    sequence = getattr(context, "memory_user_sequence", None)
    if not path or not sequence or not quote.strip():
        raise ValueError("An exact quote from the current user's input is required")
    entry = None
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if value.get("sequence") == sequence and value.get("kind") == "user_input":
                entry = value
                break
    if entry is None:
        raise ValueError("Current user source unavailable")
    offset = entry["payload"]["text"].find(quote)
    if offset < 0:
        raise ValueError("Evidence quote is not in the original user input")
    service = _service(context, write=True)
    if entry["payload"].get("memory_generation") != service.extraction.generation:
        raise ValueError("User source belongs to a different memory generation")
    return [EvidenceReference(context.session_id, sequence, "user_input", hashlib.sha256(quote.encode()).hexdigest(),
                              start_offset=offset, end_offset=offset + len(quote),
                              source_namespace=service.stores["project"].namespace, observed_at=entry["timestamp"])]


def _result(name, operation):
    try:
        payload = operation()
        return ToolResult(tool=name, status="success", output=json.dumps(payload, ensure_ascii=False),
                          metadata={"status_source": "native"})
    except (KeyError, OSError, ValueError, sqlite3.Error) as exc:
        return ToolResult(tool=name, status="failed", output=f"[error] {exc}", error=str(exc),
                          metadata={"status_source": "validation"})


def memory_search(query: str, scope: str = "both", paths: list[str] | None = None, limit: int = 8,
                  offset: int = 0, max_tokens: int = 2000, kind: str | None = None,
                  tool_context: ToolContext | None = None) -> ToolResult:
    return _result("memory_search", lambda: _service(tool_context).search(
        query, scope=scope, paths=paths, limit=limit, offset=offset, max_tokens=max_tokens, kind=kind))


def memory_read(memory_id: str = "", scope: str = "project", topic_id: str = "", history: bool = False,
                offset: int = 0, limit: int = 10, tool_context: ToolContext | None = None) -> ToolResult:
    def read():
        service = _service(tool_context)
        if memory_id:
            return service.read(memory_id, scope=scope, history=history, offset=offset, limit=limit)
        if topic_id:
            return service.topic(topic_id, scope=scope, offset=offset, limit=limit)
        return service.topics(scope=scope, offset=offset, limit=limit)
    return _result("memory_read", read)


def memory_write(topic: str, fact_key: str, kind: str, body: str, source_quote: str, scope: str = "project",
                 applicability: str = "", source_paths: list[str] | None = None, memory_id: str = "",
                 expected_version: int | None = None, restore_forgotten: bool = False, tool_context: ToolContext | None = None) -> ToolResult:
    def write():
        service = _service(tool_context, write=True)
        evidence = source_evidence(tool_context, source_quote)
        if restore_forgotten and not any(term in source_quote.casefold() for term in ("重新记住", "重新保存", "remember again", "restore memory")):
            raise ValueError("Quote the explicit request to remember the forgotten fact again")
        record = service.write(MemoryWriteCommand(topic=topic, fact_key=fact_key, kind=kind, body=body,
                               scope=scope, evidence=evidence, applicability=applicability, source_paths=source_paths or [],
                               memory_id=memory_id or None, expected_version=expected_version, restore_forgotten=restore_forgotten))
        return {"id": record.id, "version": record.version, "status": record.status, "scope": scope}
    return _result("memory_write", write)


def memory_validate(memory_id: str, expected_version: int, source_quote: str, scope: str = "project",
                    tool_context: ToolContext | None = None) -> ToolResult:
    def validate():
        service = _service(tool_context, write=True)
        record = service.validate(memory_id, expected_version, source_evidence(tool_context, source_quote), scope=scope)
        return {"id": record.id, "version": record.version, "status": record.status}
    return _result("memory_validate", validate)


def memory_forget(memory_id: str, expected_version: int, source_quote: str, scope: str = "project",
                  tool_context: ToolContext | None = None) -> ToolResult:
    def forget():
        service = _service(tool_context, write=True)
        source_evidence(tool_context, source_quote)
        if not any(word in source_quote.casefold() for word in ("忘记", "遗忘", "清除记忆", "删除记忆", "forget", "remove memory")):
            raise ValueError("Quote the user's explicit request to forget this memory")
        service.forget(memory_id, expected_version, scope=scope)
        return {"id": memory_id, "forgotten": True}
    return _result("memory_forget", forget)
