"""Read-only semantic code navigation."""

from __future__ import annotations

import json

from ..tool_context import ToolContext
from ..tool_result import ToolResult


def code_intelligence(
    operation: str,
    path: str,
    line: int | None = None,
    column: int | None = None,
    *,
    tool_context: ToolContext,
) -> ToolResult:
    service = tool_context.get_language_service()
    result = service.query(operation, path, line, column)
    if not result["available"]:
        result["hint"] = "Use repo_search and read_file to continue code inspection."
    return ToolResult(
        tool="code_intelligence",
        status="success",
        output=json.dumps(result, ensure_ascii=False),
        metadata={"status_source": "native", "available": result["available"]},
    )
