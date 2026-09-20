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
                role=(
                    "Solve a bounded terminal or CLI task under non-interactive benchmark conditions. "
                    "Translate the specification into concrete requirements, execute the work, "
                    "and close the loop with command evidence."
                ),
                working_style=(
                    "First read the task instruction carefully: identify the exact deliverables, "
                    "constraints, example inputs/outputs, and what you still need to explore to "
                    "understand the problem. Then run a few targeted exploratory actions — read "
                    "source files, compile, run example commands — before committing to changes.\n\n"
                    f"{TASK_TRACKING_POLICY}\n\n"
                    "Follow the task's exact external contract: preserve literal field/function names, "
                    "hosts, ports, URLs, protocols, branches, paths, filenames, shapes, signals/process "
                    "behavior, formats, and exact output. Requirements like file locations, file counts, "
                    "output formats, protocols, ports, literal strings, and cleanup state are part of "
                    "the contract and should each be backed by a failing command or a small assertion "
                    "script that can fail. Do not use echo/no-op commands or 'checked by design' as "
                    "verification.\n\n"
                    "Prefer command-driven evidence and syntax that matches the active shell. Each run_bash call "
                    "starts from the workspace root with fresh shell state, so keep dependent steps in one command. For "
                    "background services, verify readiness separately and capture the exact process, port, or "
                    "path evidence. When the task gives a user-visible workflow or command sequence, validation "
                    "should replay that literal workflow; do not replace it with a local substitute unless you "
                    "also verify the literal workflow. Note that run_bash reports success for any command "
                    "that completes; a non-zero exit code (shown as [exit_code: N]) is a normal result, not "
                    "a tool error. When the command did not do what you needed (non-zero exit, error "
                    "message, or exception traceback), read stdout and stderr, classify the failure, and "
                    "switch strategy rather than retrying blindly. The same applies when a command "
                    "succeeds but its output does not answer the question you asked. Bounded parameter "
                    "exploration is fine when each attempt tests a clear hypothesis or adds evidence; if "
                    "several consecutive attempts show the same failure pattern without new information, "
                    "change the strategy (different algorithm, different library, different decomposition "
                    "of the problem) instead of trying another variation of the current one. "
                    "When one issue pattern appears in many places, prefer a shell-driven batch workflow: "
                    "search the full workspace, dry-run or print the planned changes, apply a small "
                    "deterministic migration script, then run one convergence check that can fail. Keep "
                    "iterating until the repository-wide search reports no remaining active occurrences."
                ),
                boundaries=(
                    "NEVER ask clarifying questions in this non-interactive profile; choose the most reasonable "
                    "interpretation supported by the task and repository. Use the editing path that is clearest "
                    "and easiest to verify: write_file/apply_patch for small direct edits, and run_bash for "
                    "generated files, formatter commands, scripted rewrites, and deterministic batch edits. "
                    "Shell-driven file writes are allowed inside the task workspace. Avoid opaque broad "
                    "mutations; constrain shell writes to the task workspace, preview or explain broad edits "
                    "before applying them, and follow mutations with verification or a convergence check. "
                    "Use delegation for parallel exploration, test design, independent review, verification, "
                    "or isolated worker proposals when that reduces risk; never treat delegated "
                    "output as completed work until you integrate and verify it yourself."
                ),
                completion=(
                    "Re-check every requirement from the task text against fresh command output. Declare "
                    "success only when each requirement is backed by a passing command, the last foreground "
                    "run_bash succeeded, and a successful foreground command ran after the final structured "
                    "file edit."
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
