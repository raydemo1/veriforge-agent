from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ToolStatus = Literal["success", "failed", "unknown"]


@dataclass(frozen=True)
class ToolResult:
    tool: str
    status: ToolStatus
    output: str = ""
    error: str | None = None
    return_code: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"success", "failed", "unknown"}:
            raise ValueError(f"Invalid tool result status: {self.status}")

    @property
    def ok(self) -> bool | None:
        if self.status == "success":
            return True
        if self.status == "failed":
            return False
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "status": self.status,
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
            "return_code": self.return_code,
            "metadata": self.metadata,
        }

    def to_text(self) -> str:
        if self.output:
            return self.output
        if self.error:
            return f"[error] {self.error}"
        return ""


def unstructured_tool_result_from_text(
    *,
    tool: str,
    text: str,
    return_code: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> ToolResult:
    result_metadata = dict(metadata or {})
    result_metadata.setdefault("status_source", "unstructured")
    return ToolResult(
        tool=tool,
        status="unknown",
        output=text,
        error=None,
        return_code=return_code,
        metadata=result_metadata,
    )
