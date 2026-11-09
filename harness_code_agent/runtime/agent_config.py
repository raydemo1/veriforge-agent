"""Runtime parameters for the single main agent.

These are execution settings, not an agent identity: they never select a
prompt, a tool set, or a way of thinking. A benchmark tightens a budget here;
an interactive session leaves it unset.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeOptions:
    # Hard wall-clock limit for one task, in seconds. None means no limit.
    time_budget: float | None = None
    # Benchmark metadata attached to the agent's runtime state for reporting.
    task_metadata: dict | None = None
