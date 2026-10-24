from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..runtime.shell_classification import (
    ShellEffect,
    ShellTrait,
    analyze_shell_command,
)
from ..runtime.tool_output import (
    artifact_preview,
    fits_message,
    inline_message,
    output_limit,
    result_text,
)
from ..runtime.tool_result import ToolResult
from .context import count_text_tokens

log = logging.getLogger("harness")

def observation_dir_for(root: str | Path, session_id: str | None, agent_id: str | None = None) -> Path:
    """Directory for observation files and streamed tool-output artifacts."""
    safe_session_id = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in str(session_id or "default")
    )
    safe_agent_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(agent_id or "main"))
    return Path(root) / ".harness" / "observations" / safe_session_id / safe_agent_id


@dataclass
class ToolObservation:
    id: str
    tool: str
    raw_output_path: Path | None
    resource_keys: list[str]
    observed_file_generations: dict[str, int]
    observed_workspace_generation: int
    output_chars: int
    output_hash: str
    created_at: float = field(default_factory=time.time)
    stale: bool = False
    artifact_adopted: bool = False
    message: str = ""
    artifact_reference: dict | None = None
    inline_output: str | None = None


class FactTracker:
    def __init__(self) -> None:
        self.file_generation: dict[str, int] = {}
        self.workspace_generation = 0
        self.invalidation_notices: list[str] = []

    def resource_keys_for(self, tool: str, args: dict[str, Any]) -> list[str]:
        if tool in {"read_file", "write_file", "apply_patch"} and args.get("path"):
            return [f"file:{_norm_path(args['path'])}"]
        if tool == "run_bash":
            return ["workspace"]
        return []

    def observed_file_generations(self, resource_keys: list[str]) -> dict[str, int]:
        result: dict[str, int] = {}
        for key in resource_keys:
            if key.startswith("file:"):
                path = key.removeprefix("file:")
                result[path] = self.file_generation.get(path, 0)
        return result

    def apply_mutation(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        result: ToolResult,
        observations: list[ToolObservation],
        exclude_ids: set[str] | None = None,
    ) -> str:
        if result.status != "success":
            return ""
        exclude_ids = exclude_ids or set()
        changed: list[str] = []
        if tool in {"write_file", "apply_patch"} and args.get("path"):
            path = _norm_path(args["path"])
            old = self.file_generation.get(path, 0)
            self.file_generation[path] = old + 1
            changed.append(f"{path}: file_generation {old} -> {old + 1}")
        elif tool == "run_bash" and self._shell_may_mutate(str(args.get("command", ""))):
            old = self.workspace_generation
            self.workspace_generation = old + 1
            changed.append(f"workspace_generation {old} -> {old + 1}")
        if not changed:
            return ""

        stale_ids: list[str] = []
        for obs in observations:
            if obs.id in exclude_ids or obs.stale:
                continue
            if self.is_stale(obs):
                obs.stale = True
                stale_ids.append(obs.id)
        notice = (
            "[FACT INVALIDATION]\n"
            + "\n".join(changed)
            + "\nStale observations: "
            + (", ".join(stale_ids) if stale_ids else "none")
            + "\nTreat stale observations only as historical notes. Re-read files or rerun commands before relying on current facts."
        )
        self.invalidation_notices.append(notice)
        del self.invalidation_notices[:-5]
        if not stale_ids:
            return ""
        return notice

    def is_stale(self, observation: ToolObservation) -> bool:
        if observation.observed_workspace_generation < self.workspace_generation:
            return True
        for path, generation in observation.observed_file_generations.items():
            if generation < self.file_generation.get(path, 0):
                return True
        return False

    def _shell_may_mutate(self, command: str) -> bool:
        analysis = analyze_shell_command(command)
        return bool(
            analysis.effects & {
                ShellEffect.WRITE,
                ShellEffect.DELETE,
                ShellEffect.GIT_MUTATION,
            }
            or ShellTrait.UNKNOWN_EFFECT in analysis.traits
        )


class ObservationStore:
    MAX_OBSERVATIONS = 300

    def __init__(self, root: str | Path, workspace_root: str | Path | None = None):
        self.root = Path(root).resolve()
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else next(
            (parent.parent for parent in self.root.parents if parent.name == ".harness"), self.root.parent
        )
        self.observations: list[ToolObservation] = []

    def create(self, *, tool: str, args: dict[str, Any], result: ToolResult,
               fact_tracker: FactTracker) -> ToolObservation:
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
                    candidate = self.workspace_root / candidate
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
                relative = raw_path.relative_to(self.workspace_root).as_posix()
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
            resource_keys=fact_tracker.resource_keys_for(tool, args),
            observed_file_generations=fact_tracker.observed_file_generations(fact_tracker.resource_keys_for(tool, args)),
            observed_workspace_generation=fact_tracker.workspace_generation,
            output_chars=output_chars, output_hash=output_hash, artifact_adopted=adopted,
            message=message, artifact_reference=reference,
            inline_output=inline_output,
        )
        self.observations.append(observation)
        del self.observations[:-self.MAX_OBSERVATIONS]
        return observation

    def observed_message(self, observation: ToolObservation, result: ToolResult) -> str:
        return observation.message


def _norm_path(path: object) -> str:
    return str(path).replace("\\", "/").strip()
