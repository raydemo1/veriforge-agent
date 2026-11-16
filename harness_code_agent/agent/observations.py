from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..runtime.tool_output import (
    artifact_preview,
    fits_message,
    inline_message,
    output_limit,
    result_text,
)
from ..runtime.tool_result import ToolResult
from ..support.tokens import count_text_tokens

log = logging.getLogger("harness")

@dataclass
class ToolObservation:
    id: str
    tool: str
    raw_output_path: Path | None
    #: File paths this observation is about, used by context compaction.
    resource_keys: list[str]
    output_chars: int
    output_hash: str
    created_at: float = field(default_factory=time.time)
    artifact_adopted: bool = False
    message: str = ""
    artifact_reference: dict | None = None
    inline_output: str | None = None


class ObservationStore:
    MAX_OBSERVATIONS = 300

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.observations: list[ToolObservation] = []

    def create(self, *, tool: str, args: dict[str, Any], result: ToolResult) -> ToolObservation:
        obs_id = "obs_" + uuid.uuid4().hex
        output = result_text(result)
        raw_path = None
        adopted = False
        reference = None
        inline_output = None
        artifact = result.metadata.get("artifact_path")
        try:
            if artifact:
                candidate = Path(artifact)
                if not candidate.is_absolute():
                    raise ValueError("tool output artifact path must be absolute")
                candidate = candidate.resolve()
                if not candidate.is_relative_to(self.root) or not candidate.is_file():
                    raise ValueError("tool output artifact is missing or outside its observation directory")
                hasher = hashlib.sha256()
                output_chars = 0
                small = ""
                oversize = False
                with candidate.open("r", encoding="utf-8", newline="") as stream:
                    while chunk := stream.read(4096):
                        output_chars += len(chunk)
                        hasher.update(chunk.encode("utf-8"))
                        if not oversize:
                            small += chunk
                            oversize = not fits_message(inline_message(result, small, obs_id))
                output_hash = hasher.hexdigest()
                if result.error and result.error not in small and not oversize:
                    small += f"\nerror: {result.error}"
                if not oversize and fits_message(inline_message(result, small, obs_id)):
                    output = small
                    candidate.unlink()
                    result.metadata.pop("artifact_path", None)
                    result.metadata.pop("output_spilled", None)
                else:
                    raw_path, adopted = candidate, True
            else:
                output_chars = len(output)
                output_hash = hashlib.sha256(output.encode("utf-8")).hexdigest()
                if not fits_message(inline_message(result, output, obs_id)):
                    self.root.mkdir(parents=True, exist_ok=True)
                    raw_path = self.root / f"{obs_id}.txt"
                    with raw_path.open("x", encoding="utf-8", newline="") as stream:
                        stream.write(output)
            if raw_path:
                relative = raw_path.as_posix()
                reference = {"path": relative, "sha256": output_hash,
                             "bytes": raw_path.stat().st_size, "chars": output_chars}
                message = inline_message(result, "", obs_id).rstrip() + (
                    f"\noutput_chars: {output_chars}\noutput_bytes: {reference['bytes']}"
                    f"\nraw_output: {relative}"
                    "\nOutput exceeds the inline budget. Read it with "
                    + "read_file(" + json.dumps({"path": relative, "start_line": 1, "max_lines": 2000}) + ")."
                )
                edge_tokens = min(500, max(0, (output_limit() - count_text_tokens(message) - 100) // 2))
                message += "\n\n" + artifact_preview(raw_path, edge_tokens)
            else:
                message = inline_message(result, output, obs_id)
                inline_output = output
        except (OSError, ValueError) as exc:
            if raw_path is not None and not adopted:
                try:
                    raw_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raw_path, reference = None, None
            result.metadata.pop("artifact_path", None)
            result.metadata.pop("output_spilled", None)
            output_chars = len(output)
            output_hash = hashlib.sha256(output.encode("utf-8")).hexdigest()
            result.metadata["output_complete"] = False
            message = inline_message(result, f"[error] Output is incomplete: {type(exc).__name__}; artifact unavailable.", obs_id)
            log.warning("Could not preserve tool output: %s", exc)
        observation = ToolObservation(
            id=obs_id, tool=tool, raw_output_path=raw_path,
            resource_keys=_resource_keys_for(tool, args),
            output_chars=output_chars, output_hash=output_hash, artifact_adopted=adopted,
            message=message, artifact_reference=reference,
            inline_output=inline_output,
        )
        self.observations.append(observation)
        del self.observations[:-self.MAX_OBSERVATIONS]
        return observation


def _resource_keys_for(tool: str, args: dict[str, Any]) -> list[str]:
    """Paths this observation is about; used only for compaction hints."""
    if tool in {"read_file", "write_file", "apply_patch"} and args.get("path"):
        return [f"file:{_norm_path(args['path'])}"]
    if tool == "bash":
        return ["workspace"]
    return []


def _norm_path(path: object) -> str:
    return str(path).replace("\\", "/").strip()
