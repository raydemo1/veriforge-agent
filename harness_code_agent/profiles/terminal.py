"""
Terminal task profile — optimized for Terminal-Bench 2.1.

Key constraints:
  - 30 min (1800s) hard timeout per task
  - Tasks are well-defined CLI problems, not open-ended
  - No UI, no browser testing needed
  - Correctness is binary: tests pass or fail

The hard timeout is configurable without touching this file:

  # Via environment variables:
  PROFILE_TERMINAL_TASK_BUDGET=1800
  # Or via ProfileConfig in code:
  from harness_code_agent.profiles.base import ProfileConfig
  cfg = ProfileConfig(task_budget=1200)
  profile = TerminalProfile(cfg=cfg)
"""
from __future__ import annotations

import os
from typing import ClassVar

from ..tracking_policy import TASK_TRACKING_POLICY
from .base import (
    AgentConfig,
    BaseProfile,
    build_profile_prompt,
)


class TerminalProfile(BaseProfile):

    # --- Default values (overridable via ProfileConfig or env vars) ---
    _DEFAULTS: ClassVar[dict] = {
        "task_budget": 1800,
    }

    def _get(self, key: str):
        """Resolve a config value: env var > ProfileConfig > default."""
        return self.cfg.resolve(key, self.name(), self._DEFAULTS[key])

    def name(self) -> str:
        return "terminal"

    def description(self) -> str:
        return "Solve terminal/CLI tasks (Terminal-Bench 2.1)"

    def main_agent(self) -> AgentConfig:
        return AgentConfig(
            system_prompt=build_profile_prompt(
                role="Solve the terminal task non-interactively and demonstrate the result with commands.",
                working_style=(
                    "Read the task and examples, then use targeted reads or commands to resolve uncertainty. "
                    "Preserve exact names, paths, shapes, file counts, formats, output, URLs, hosts, ports, "
                    "protocols, branches, process signals, and cleanup state. Replay literal user workflows "
                    "and use checks that can fail.\n\n"
                    f"{TASK_TRACKING_POLICY}\n\n"
                    "Use the active shell's syntax. Each run_bash starts at the workspace root with fresh "
                    "shell state, so group dependent steps. Verify background service readiness separately "
                    "with process, port, or path evidence. Read [exit_code: N], stdout, and stderr even when "
                    "the tool reports success. Retry only to test a hypothesis or gather new evidence. "
                    "For widespread edits, search the workspace, preview a deterministic rewrite, then "
                    "check for remaining occurrences."
                ),
                boundaries=(
                    "Resolve ambiguity from the task and repository without asking questions. Use "
                    "write_file or apply_patch for small edits and run_bash for generation, formatting, "
                    "or deterministic batch edits. Keep shell writes inside the workspace. Review, "
                    "integrate, and verify worker proposals yourself."
                ),
                completion=(
                    "Back each requirement with fresh passing command output. The last foreground run_bash "
                    "must succeed, with a passing foreground check after the final structured edit."
                ),
            ),
            time_budget=self._get("task_budget"),
        )

    # --- TB2 task metadata for dynamic timeout ---
    _tb2_tasks: dict | None = None

    @classmethod
    def _load_tb2_tasks(cls) -> dict:
        """Load TB2 task metadata from bundled JSON."""
        if cls._tb2_tasks is None:
            import json
            from pathlib import Path
            tb2_path = Path(__file__).resolve().parent / "tb2_tasks.json"
            if tb2_path.exists():
                cls._tb2_tasks = json.loads(tb2_path.read_text(encoding="utf-8"))
            else:
                cls._tb2_tasks = {}
        return cls._tb2_tasks

    def resolve_task_timeout(self, user_prompt: str) -> float | None:
        """Look up TB2 task timeout by matching task name in prompt or workspace path."""
        meta = self._lookup_task_meta(user_prompt)
        return meta.get("agent_timeout_sec") if meta else None

    def resolve_task_metadata(self, user_prompt: str) -> dict | None:
        """Look up TB2 task metadata for runtime policy and reporting."""
        meta = self._lookup_task_meta(user_prompt)
        return dict(meta) if meta else None

    def _lookup_task_meta(self, user_prompt: str) -> dict | None:
        """Look up full TB2 task metadata (timeout, difficulty, category)."""
        from .. import config as _cfg
        tasks = self._load_tb2_tasks()
        if not tasks:
            return None

        for env_key in ("HARNESS_TERMINAL_TASK_NAME", "HCA_TERMINAL_TASK_NAME", "TERMINAL_BENCH_TASK_NAME"):
            meta = _task_meta_by_name(tasks, os.environ.get(env_key, ""))
            if meta:
                return meta

        # Check workspace path first (most reliable)
        ws_lower = _cfg.WORKSPACE.lower()
        for task_name, meta in tasks.items():
            if task_name in ws_lower:
                return _with_task_name(task_name, meta)

        # Check user prompt
        prompt_lower = user_prompt.lower()
        for task_name, meta in tasks.items():
            if len(task_name) > 6 and (
                task_name in prompt_lower or
                task_name.replace("-", " ") in prompt_lower or
                task_name.replace("-", "_") in prompt_lower
            ):
                return _with_task_name(task_name, meta)

        return None


def _task_meta_by_name(tasks: dict, raw_task_name: str) -> dict | None:
    task_name = str(raw_task_name or "").strip().lower()
    if not task_name:
        return None
    candidates = [task_name.replace("_", "-")]
    if "/" in task_name:
        candidates.append(task_name.rsplit("/", 1)[-1].replace("_", "-"))
    for candidate in candidates:
        meta = tasks.get(candidate)
        if meta:
            return _with_task_name(candidate, meta)
    return None


def _with_task_name(task_name: str, meta: dict) -> dict:
    resolved = dict(meta)
    resolved.setdefault("task_name", task_name)
    return resolved
