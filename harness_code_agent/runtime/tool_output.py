"""Shared model-facing tool output accounting."""
from __future__ import annotations

import json
from pathlib import Path

from .. import config
from ..agent.context import count_text_tokens
from .tool_result import ToolResult


def output_limit() -> int:
    return config.TOOL_OUTPUT_MAX_TOKENS


def result_text(result: ToolResult) -> str:
    text = result.to_text()
    if result.error and result.error not in text:
        text += f"\nerror: {result.error}"
    return text


def inline_message(result: ToolResult, output: str, observation_id: str) -> str:
    lines = [f"[OBS {observation_id} observed]", f"tool: {result.tool}", f"status: {result.status}"]
    if result.return_code is not None:
        lines.append(f"return_code: {result.return_code}")
    if result.metadata.get("timed_out"):
        lines.append("timed_out: true")
    if result.metadata.get("cancelled"):
        lines.append("cancelled: true")
    if result.metadata.get("source_log"):
        lines.append(f"source_log: {result.metadata['source_log']}")
    if result.metadata.get("next_read"):
        lines.append("next_read: " + json.dumps(result.metadata["next_read"], ensure_ascii=False))
    if result.metadata.get("next_offset") is not None:
        lines.append(f"next_offset: {result.metadata['next_offset']}")
    if result.metadata.get("output_complete") is False:
        lines.append("output_complete: false")
    return "\n".join(lines) + "\n\n" + output


def fits_message(text: str) -> bool:
    return count_text_tokens(text) + 8 <= output_limit()


def fitting_prefix(text: str, budget: int) -> str:
    if count_text_tokens(text) <= budget:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if count_text_tokens(text[:middle]) <= budget:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def fitting_suffix(text: str, budget: int) -> str:
    if count_text_tokens(text) <= budget:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if count_text_tokens(text[-middle:]) <= budget:
            low = middle
        else:
            high = middle - 1
    return text[-low:] if low else ""


def artifact_preview(path: Path, edge_tokens: int = 500) -> str:
    with path.open("rb") as stream:
        head = stream.read(32768).decode("utf-8", errors="ignore")
        stream.seek(max(0, path.stat().st_size - 32768))
        tail = stream.read(32768).decode("utf-8", errors="ignore")
    return ("--- output head ---\n" + fitting_prefix(head, edge_tokens)
            + "\n\n[middle omitted; read the output file for full details]\n\n--- output tail ---\n"
            + fitting_suffix(tail, edge_tokens))
