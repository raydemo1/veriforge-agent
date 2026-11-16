"""Single-task execution shared by CLI and benchmark adapters."""
from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .core.interactive import InteractiveSession, TurnResult, print_turn_result
from .runtime.approvals import NoApprovalProvider
from .runtime.questions import NoQuestionProvider


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    cwd: Path
    harness_root: Path


@dataclass(frozen=True)
class RunResult:
    cwd: Path
    session_id: str | None
    harness_root: Path | None
    status: Literal["completed", "stopped", "failed", "interrupted"]
    turn_result: TurnResult | None = None
    error: str = ""
    #: Set when the runtime stopped the turn itself (budget, context, tool
    #: failure policy).  A stopped turn is not a completed task.
    stop_reason: str = ""
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return {"completed": 0, "stopped": 1, "failed": 1, "interrupted": 130}[self.status]


def run_task(
    *,
    cwd: str | Path,
    task: str,
    time_budget: float | None = None,
    stream_sink: Callable[[str], None] | None = None,
    on_session_started: Callable[[SessionInfo], None] | None = None,
) -> RunResult:
    """Run one task and close its resources, including on errors or interrupts.

    Completed means the runtime finished the turn, not that a verifier accepted
    the task. Approvals and questions never read stdin. SystemExit propagates
    after cleanup so process signal handlers retain their exit codes.
    """
    workspace = Path(cwd).resolve()
    session = None
    session_id = None
    harness_root = None
    turn_result = None
    status: Literal["completed", "stopped", "failed", "interrupted"] = "completed"
    error = ""
    stop_reason = ""
    timings: dict[str, float] = {}
    started = time.perf_counter()
    try:
        if not task.strip():
            raise ValueError("No task provided")
        session = InteractiveSession(
            cwd=workspace,
            time_budget=time_budget,
            stream_sink=stream_sink,
            approval_provider=NoApprovalProvider(),
            question_provider=NoQuestionProvider(),
        )
        timings["init_seconds"] = round(time.perf_counter() - started, 3)
        session_id = session.session_id
        harness_root = session.session_store.root
        if session_id is None:
            raise RuntimeError("Session initialization did not produce a session ID")
        if on_session_started is not None:
            on_session_started(SessionInfo(session_id, workspace, harness_root))
        turn_started = time.perf_counter()
        if task.startswith("/"):
            from .tui.commands import default_command_registry

            registry = default_command_registry(skill_registry=session.skill_registry)
            if not registry.is_agent_command(task):
                session.handle_slash_command(task)
                turn_result = TurnResult(text="")
            else:
                turn_result = session.submit(task)
        else:
            turn_result = session.submit(task)
        timings["turn_seconds"] = round(time.perf_counter() - turn_started, 3)
        conversation = getattr(session, "conversation", None)
        fallback = getattr(getattr(conversation, "runtime_state", None), "fallback", None)
        if getattr(fallback, "stop_requested", False):
            stop_reason = str(getattr(fallback, "stop_reason", "") or "agent_fallback")
            status = "stopped"
    except KeyboardInterrupt:
        status = "interrupted"
        error = "Interrupted."
    except Exception:
        status = "failed"
        error = traceback.format_exc()
    finally:
        if session is not None:
            close_started = time.perf_counter()
            try:
                session.close()
            except Exception:
                status = "failed" if status in {"completed", "stopped"} else status
                error = "\n".join(part for part in (error.strip(), traceback.format_exc().strip()) if part)
            finally:
                timings["close_seconds"] = round(time.perf_counter() - close_started, 3)
        timings["total_seconds"] = round(time.perf_counter() - started, 3)
    return RunResult(workspace, session_id, harness_root, status, turn_result, error,
                     stop_reason, timings)


def print_run_result(result: RunResult) -> None:
    """Print session identity and the turn using the standard CLI format."""
    if result.session_id:
        print(f"veriforge session: {result.session_id}", flush=True)
        print(f"workspace: {result.cwd}", flush=True)
    if result.timings:
        print("veriforge timing: " + json.dumps(result.timings, sort_keys=True), flush=True)
    if result.stop_reason:
        print(f"veriforge stop: {result.stop_reason}", flush=True)
    if result.turn_result is not None:
        print_turn_result(result.turn_result)
