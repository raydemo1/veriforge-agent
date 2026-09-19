from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..runtime.shell_classification import (
    ShellEffect,
    ShellTrait,
    analyze_shell_command,
)
from ..runtime.tool_result import ToolResult

log = logging.getLogger("harness")

FRESH_DETAIL_LIMIT = 12_000
_OBS_ID_PREFIX = "[OBS "

# Preview budget for compact tool-output refs: head + tail chars shown inline
# when the full output is stored to file.
PREVIEW_TOTAL_CHARS = 1000


def observation_dir_for(root: str | Path, session_id: str | None) -> Path:
    """Directory for observation files and streamed tool-output artifacts."""
    safe_session_id = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in str(session_id or "default")
    )
    return Path(root) / ".harness" / "observations" / safe_session_id


def _output_preview(output: str) -> str:
    """Head + tail preview so the agent can see both context and final results
    without reading the full output file."""
    if len(output) <= PREVIEW_TOTAL_CHARS:
        return output
    head_chars = PREVIEW_TOTAL_CHARS // 2  # 500
    tail_chars = PREVIEW_TOTAL_CHARS - head_chars  # 500
    head = output[:head_chars]
    tail = output[-tail_chars:]
    omitted = len(output) - head_chars - tail_chars
    return (
        f"{head}\n\n"
        f"...[{omitted} chars omitted — raw observation output is internal; rerun a narrower command if more detail is needed]...\n\n"
        f"{tail}"
    )


@dataclass
class ToolObservation:
    id: str
    tool: str
    args_summary: str
    raw_output_path: Path
    summary: str
    resource_keys: list[str]
    observed_file_generations: dict[str, int]
    observed_workspace_generation: int
    output_chars: int
    output_hash: str
    created_at: float = field(default_factory=time.time)
    stale: bool = False
    # True when raw_output_path is a tool-managed artifact adopted by the
    # store (located inside the store directory, so safe to unlink later).
    artifact_adopted: bool = False


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

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        self.observations: list[ToolObservation] = []

    def create(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        result: ToolResult,
        fact_tracker: FactTracker,
    ) -> ToolObservation:
        self._counter += 1
        obs_id = f"obs_{self._counter:04d}"
        output = result.to_text()
        resource_keys = fact_tracker.resource_keys_for(tool, args)
        artifact_path = result.metadata.get("artifact_path")
        adopted_path = self._adoptable_artifact(artifact_path)
        if adopted_path is not None:
            # The tool (e.g. run_bash) already streamed the full output to an
            # artifact while it ran.  Adopt it instead of spilling a second
            # copy; the short ToolResult text is only the preview.
            raw_path = adopted_path
            output_chars = int(result.metadata.get("output_chars") or len(output))
            output_hash = str(
                result.metadata.get("artifact_sha256")
                or hashlib.sha256(output.encode("utf-8", errors="replace")).hexdigest()
            )[:16]
        else:
            if artifact_path:
                # Untrusted/out-of-tree path: never adopt (the store unlinks
                # old observations), fall back to the normal spill path.
                log.warning(
                    "Ignoring artifact_path outside observation store (%s not under %s)",
                    artifact_path,
                    self.root,
                )
            output_hash = hashlib.sha256(output.encode("utf-8", errors="replace")).hexdigest()[:16]
            raw_path = self.root / f"{obs_id}.txt"
            raw_path.write_text(output, encoding="utf-8")
            output_chars = len(output)
        observation = ToolObservation(
            id=obs_id,
            tool=tool,
            args_summary=_args_summary(args),
            raw_output_path=raw_path,
            summary=_summary_for(tool, args, result, output_hash),
            resource_keys=resource_keys,
            observed_file_generations=fact_tracker.observed_file_generations(resource_keys),
            observed_workspace_generation=fact_tracker.workspace_generation,
            output_chars=output_chars,
            output_hash=output_hash,
            artifact_adopted=adopted_path is not None,
        )
        self.observations.append(observation)
        self._cleanup_old()
        return observation

    def _adoptable_artifact(self, artifact_path: object) -> Path | None:
        """Return a safe artifact path to adopt, else None.

        Adoption means the store later owns and may unlink the file during
        observation eviction, so the path must resolve to an existing file
        inside the store directory — never an arbitrary external path.
        """
        if not artifact_path:
            return None
        try:
            candidate = Path(str(artifact_path)).resolve(strict=False)
            root = self.root.resolve()
        except (OSError, ValueError):
            return None
        if not candidate.is_relative_to(root) or not candidate.is_file():
            return None
        return candidate

    def _cleanup_old(self) -> None:
        excess = len(self.observations) - self.MAX_OBSERVATIONS
        if excess <= 0:
            return
        for obs in self.observations[:excess]:
            try:
                obs.raw_output_path.unlink(missing_ok=True)
            except OSError:
                pass
        del self.observations[:excess]

    def observed_message(self, observation: ToolObservation, result: ToolResult) -> str:
        output = result.to_text()
        from .. import config as _cfg
        inline_limit = getattr(_cfg, "TOOL_OUTPUT_INLINE_LIMIT", 4000)

        if observation.artifact_adopted:
            # Full output was streamed to the adopted artifact while the
            # command ran; the ToolResult text is a bounded head+tail preview.
            total_bytes = result.metadata.get("output_bytes", observation.output_chars)
            return (
                f"[OBS {observation.id} observed]\n"
                f"tool: {observation.tool}\n"
                f"args: {observation.args_summary}\n"
                f"output_chars: {observation.output_chars}\n"
                f"output_sha256: {observation.output_hash}\n"
                f"resource_keys: {', '.join(observation.resource_keys) or 'none'}\n"
                f"raw_output: {observation.raw_output_path}\n"
                f"output_bytes: {total_bytes}\n"
                f"summary: {observation.summary}\n"
                "observation: Full output is stored in this artifact and the "
                "command was not interrupted by its output volume; the "
                "preview below shows head and tail only. "
                "Read the artifact path if you need the middle, or rerun a "
                "narrower command.\n\n"
                + output
            )

        if len(output) <= inline_limit:
            # Small output — full inline (existing path, unchanged)
            detail = output
            if len(detail) > FRESH_DETAIL_LIMIT:
                omitted = len(detail) - FRESH_DETAIL_LIMIT
                detail = detail[:FRESH_DETAIL_LIMIT] + (
                    f"\n\n[TRUNCATED observation detail: {omitted} chars omitted]"
                )
            return (
                f"[OBS {observation.id} observed]\n"
                f"tool: {observation.tool}\n"
                f"args: {observation.args_summary}\n"
                f"output_chars: {observation.output_chars}\n"
                f"output_sha256: {observation.output_hash}\n"
                f"resource_keys: {', '.join(observation.resource_keys) or 'none'}\n"
                "observation: This is the tool result as observed when the tool ran.\n\n"
                + detail
            )

        # Large output — compact file ref + head+tail preview
        return (
            f"[OBS {observation.id} observed]\n"
            f"tool: {observation.tool}\n"
            f"args: {observation.args_summary}\n"
            f"output_chars: {observation.output_chars}\n"
            f"output_sha256: {observation.output_hash}\n"
            f"resource_keys: {', '.join(observation.resource_keys) or 'none'}\n"
            f"raw_output: {observation.raw_output_path}\n"
            f"summary: {observation.summary}\n"
            "observation: Full output is stored as an internal artifact. "
            "Use the preview, or rerun a narrower command if more detail is needed.\n\n"
            "--- preview ---\n"
            + _output_preview(output)
            + "\n--- end preview ---"
        )

def _args_summary(args: dict[str, Any]) -> str:
    redacted = dict(args or {})
    if "content" in redacted:
        redacted["content"] = f"[{len(str(redacted['content']))} chars]"
    return json.dumps(redacted, ensure_ascii=False, sort_keys=True)


def _summary_for(tool: str, args: dict[str, Any], result: ToolResult, output_hash: str) -> str:
    if tool == "read_file":
        return f"read_file observed path={args.get('path', '')!s}; status={result.status}; output hash={output_hash}."
    if tool == "run_bash":
        return f"run_bash observed command={args.get('command', '')!s}; status={result.status}; return_code={result.return_code}."
    if tool in {"write_file", "apply_patch"}:
        return f"{tool} changed path={args.get('path', '')!s}; status={result.status}."
    return f"{tool} returned status={result.status}; output hash={output_hash}."


def _norm_path(path: object) -> str:
    return str(path).replace("\\", "/").strip()
