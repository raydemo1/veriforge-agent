"""Terminal-Bench task metadata and per-task time budgets.

This lives on the evaluation side because it is benchmark configuration, not
agent identity: the runtime receives a resolved budget and never looks up a
task name to decide how to think.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

TB2_TASKS_PATH = Path(__file__).resolve().parent / "data" / "tb2_tasks.json"
DEFAULT_TASK_BUDGET_SECONDS = 1800.0


@lru_cache(maxsize=1)
def load_task_metadata() -> dict:
    if not TB2_TASKS_PATH.is_file():
        return {}
    return json.loads(TB2_TASKS_PATH.read_text(encoding="utf-8"))


def resolve_task_metadata(user_prompt: str = "", *, workspace: str = "") -> dict | None:
    """Look up TB2 metadata by explicit task name, workspace path, or prompt."""
    tasks = load_task_metadata()
    if not tasks:
        return None

    for env_key in ("HARNESS_TERMINAL_TASK_NAME", "HCA_TERMINAL_TASK_NAME", "TERMINAL_BENCH_TASK_NAME"):
        meta = _task_meta_by_name(tasks, os.environ.get(env_key, ""))
        if meta:
            return meta

    workspace_lower = str(workspace or "").lower()
    if workspace_lower:
        for task_name, meta in tasks.items():
            if task_name in workspace_lower:
                return _with_task_name(task_name, meta)

    prompt_lower = str(user_prompt or "").lower()
    for task_name, meta in tasks.items():
        if len(task_name) > 6 and (
            task_name in prompt_lower
            or task_name.replace("-", " ") in prompt_lower
            or task_name.replace("-", "_") in prompt_lower
        ):
            return _with_task_name(task_name, meta)
    return None


def resolve_task_budget(user_prompt: str = "", *, workspace: str = "") -> tuple[float, dict | None]:
    """Return the wall-clock budget and the metadata that produced it."""
    metadata = resolve_task_metadata(user_prompt, workspace=workspace)
    if metadata and metadata.get("agent_timeout_sec"):
        try:
            return float(metadata["agent_timeout_sec"]), metadata
        except (TypeError, ValueError):
            pass
    return _default_budget(), metadata


def _default_budget() -> float:
    raw = os.environ.get("HARNESS_TASK_BUDGET")
    try:
        return float(raw) if raw else DEFAULT_TASK_BUDGET_SECONDS
    except ValueError:
        return DEFAULT_TASK_BUDGET_SECONDS


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
