"""Shell execution and background shell job tools."""
from __future__ import annotations

import json
import os

from ... import config
from ...support.cancellation import CancelledError
from ...workspace import file_changes
from ...workspace.shell_jobs import ShellJobNotFound
from ..shell_classification import (
    ShellEffect,
    analyze_shell_command,
    is_long_running_shell_command,
)
from ..tool_output import fitting_prefix, output_limit
from ..tool_result import ToolResult


def run_bash(
    command: str,
    timeout: int = 300,
    write_paths: list[str] | None = None,
    runtime_state=None,
    agent_name: str | None = None,
    tool_context=None,
    cancellation_token=None,
    permission_decision=None,
    sandbox_permissions: str | None = None,
    additional_permissions: dict | None = None,
    justification: str | None = None,
    background: bool | None = None,
    execution_plan=None,
) -> ToolResult:
    """Run one self-contained shell command from the workspace root.

    ``sandbox_permissions`` / ``additional_permissions`` / ``justification``
    are inputs to the permission policy; they were already validated and
    approved before this function is called, and execution uses the resulting
    ``permission_decision.authorization`` snapshot.

    ``execution_plan`` is the preparation-time snapshot. When present, its
    background decision and permission decision are authoritative; the tool
    does not re-analyze the command or re-run the long-command heuristic.

    ``status`` reflects only whether the tool invocation worked: a process
    that started and completed is always ``success`` — including non-zero
    exit codes such as a failing test run. The raw output and exit code are
    returned for the model to interpret. Only transport/runtime problems
    (no job manager, timeout, shell exception) are ``failed``.
    """
    if cancellation_token is not None:
        cancellation_token.check()
    requested_timeout = timeout
    try:
        timeout = max(1, min(int(timeout), int(config.TOOL_MAX_TIMEOUT_SECONDS)))
    except (TypeError, ValueError):
        timeout = int(config.SHELL_DEFAULT_TIMEOUT_SECONDS)
    if execution_plan is not None:
        permission_decision = (
            permission_decision
            or getattr(execution_plan, "permission_decision", None)
        )
        planned_background = getattr(execution_plan, "background", None)
        if planned_background is not None:
            background = bool(planned_background)
    write_paths = _effective_write_paths(write_paths or [], tool_context)
    trace_root = _file_change_root(tool_context, runtime_state)
    before = (
        file_changes.snapshot(trace_root)
        if trace_root is not None and file_changes.enabled()
        else None
    )
    result = _run_shell(command, timeout, write_paths, runtime_state, cancellation_token,
                        requested_timeout, tool_context, permission_decision, background,
                        execution_plan)
    if before is not None:
        # Diagnostic summary only: ``file_changes`` is a list-shaped contract
        # used by the change-event pipeline, so the snapshot diff gets its own
        # key and never reaches the Changes projection.
        result.metadata["file_change_summary"] = file_changes.diff(
            before, file_changes.snapshot(trace_root)
        )
    return result


def _file_change_root(tool_context, runtime_state):
    workspace = getattr(tool_context, "workspace", None)
    root = getattr(workspace, "root", None)
    if root:
        return root
    manager = _shell_job_manager(runtime_state)
    return getattr(manager, "workspace", None)


def _effective_write_paths(write_paths: list[str], tool_context) -> list[str]:
    """Workers without an explicit narrowing get their allowed_paths grant."""
    if write_paths:
        return write_paths
    policy = getattr(tool_context, "permission_policy", None)
    if policy is None or getattr(policy, "role", None) is None:
        return write_paths
    allowed = list(getattr(policy, "allowed_paths", ()) or ())
    return allowed or write_paths


def _run_shell(command, timeout, write_paths, runtime_state, cancellation_token, requested_timeout,
               tool_context=None, permission_decision=None, background=None, execution_plan=None):
    import time

    manager = _shell_job_manager(runtime_state)
    if manager is None:
        return ToolResult(tool="bash", status="failed", output="[error] No shell job manager available",
                          error="No shell job manager available", metadata={"status_source": "runtime"})
    background = (
        bool(background)
        if background is not None
        else is_long_running_shell_command(command)
    )
    # The executor follows the permission decision's network grant. Only a
    # direct call without a prepared decision falls back to analyzing the text.
    granted_network = getattr(permission_decision, "network_access", None)
    allow_network = (
        bool(granted_network)
        if granted_network is not None
        else _network_allowed(command, tool_context)
    )
    trace_stages = os.environ.get("HARNESS_TRACE_STAGES", "").lower() in {
        "1", "true", "yes",
    }
    call_started = time.perf_counter()
    job = manager.start(command, write_paths=write_paths, background=background,
                        early_exit_seconds=0.5 if background else 0,
                        cancellation_token=cancellation_token, turn_id=getattr(runtime_state, "turn_id", ""),
                        allow_network=allow_network,
                        authorization=getattr(permission_decision, "authorization", None),
                        shell_analysis=getattr(execution_plan, "analysis", None))
    start_ms = (time.perf_counter() - call_started) * 1000
    metadata = {"status_source": "shell", "job_id": job.job_id, "pid": job.pid,
                "write_paths": write_paths, "job_status": job.status,
                "requested_timeout": requested_timeout, "effective_timeout": timeout}
    if trace_stages:
        metadata["stage_timings"] = {
            **job.stage_timings,
            "start_ms": round(start_ms, 2),
        }
    if background and job.status == "running":
        return ToolResult(tool="bash", status="success",
                          output=f"Started background shell job {job.job_id} (pid={job.pid}). Use read_shell_output to inspect logs and stop_shell_job to stop it.",
                          metadata=metadata)
    timed_out = False
    wait_started = time.perf_counter()
    try:
        deadline = time.monotonic() + timeout
        while not job._finished.wait(0.05):
            if cancellation_token is not None:
                cancellation_token.check()
            if time.monotonic() >= deadline:
                timed_out = True
                manager.stop(job.job_id)
                break
    except CancelledError as exc:
        manager.stop(job.job_id)
        exc.tool_result = _job_result(job, dict(metadata, cancelled=True), "Execution cancelled")
        raise
    wait_ms = (time.perf_counter() - wait_started) * 1000
    error = job.error or (f"Command timed out after {timeout}s" if timed_out else None)
    result_started = time.perf_counter()
    result = _job_result(job, dict(metadata, timed_out=timed_out), error)
    if trace_stages:
        stages = result.metadata.setdefault("stage_timings", {})
        stages["wait_ms"] = round(wait_ms, 2)
        stages["result_ms"] = round((time.perf_counter() - result_started) * 1000, 2)
        stages["total_ms"] = round((time.perf_counter() - call_started) * 1000, 2)
    return result


def _job_result(job, metadata, error=None):
    timed_out = bool(metadata.get("timed_out"))
    if job.output.path is None:
        output = job.output.read(0, job.output.total_chars)
    else:
        preview = max(2, int(config.SHELL_OUTPUT_PREVIEW_CHARS))
        output = (job.output.read(0, preview // 2) + "\n\n[Full output saved to artifact]\n\n"
                  + job.output.read_tail(preview - preview // 2))
    if job.exit_code == 127 and "native-sandbox:" in output and not timed_out:
        error = output
    lowered = output.lower()
    if (
        job.exit_code not in (None, 0)
        and "\\\\.\\pipe\\local\\dotnet_" in lowered
        and (
            "access is denied" in lowered
            or "拒绝访问" in output
            or ("access to the path" in lowered and "is denied" in lowered)
        )
    ):
        # Accurate failure semantics for the upstream restricted-token limit.
        # The command is not retried and permissions are not escalated here.
        metadata["failure_kind"] = "windows_named_pipe_denied"
        error = (
            "Windows sandbox IPC error:\n"
            "PowerShell failed to create a native-process pipe.\n"
            "This is an execution environment limitation, not a test failure.\n"
            "Retry using an equivalent command without PowerShell piping.\n"
            "Do not request escalated permissions."
        )
    metadata.update(
        job_status=job.status,
        # A timed-out wrapper may still have process exit code 0. Keep that
        # value as diagnostics, but do not report it as completion.
        exit_code=None if timed_out else job.exit_code,
        output_bytes=job.output.path.stat().st_size if job.output.path else len(output.encode("utf-8")),
        output_chars=job.output.total_chars,
    )
    if timed_out:
        metadata["process_exit_code"] = job.exit_code
    if job.warnings:
        metadata["warnings"] = list(job.warnings)
    if not job.readers_drained:
        # Readers are still draining: the captured output is not complete, and
        # the model must not treat it as the whole result.
        metadata["output_complete"] = False
    if job.output.path is not None:
        metadata.update(artifact_path=str(job.output.path), output_spilled=True)
    footer = "\n\n[timed out]" if timed_out else _exit_code_footer(job.exit_code)
    return ToolResult(tool="bash", status="failed" if error else "success",
                      output=(output or "(no output)") + footer,
                      error=error, return_code=None if timed_out else job.exit_code,
                      metadata=metadata)


def list_shell_jobs(runtime_state=None) -> ToolResult:
    manager = _shell_job_manager(runtime_state)
    if manager is None:
        return ToolResult(
            tool="list_shell_jobs",
            status="failed",
            output="[error] No shell job manager available",
            error="No shell job manager available",
            metadata={"status_source": "runtime"},
        )
    jobs = [
        {
            "job_id": job.job_id,
            "command": job.command,
            "pid": job.pid,
            "status": job.status,
            "exit_code": job.exit_code,
            "started_at": job.started_at,
            "ended_at": job.ended_at,
            "uptime_seconds": job.uptime_seconds(),
            # A background job keeps running while its descendants live, so the
            # launcher exit is reported next to the job state, not instead of it.
            "launcher_exited": job.launcher_exited,
            "warnings": list(job.warnings),
        }
        for job in manager.list_jobs()
        if job.background
    ]
    return ToolResult(
        tool="list_shell_jobs",
        status="success",
        output=json.dumps({"jobs": jobs}, ensure_ascii=False),
        metadata={"status_source": "shell_job", "job_count": len(jobs)},
    )


def read_shell_output(job_id: str, max_chars: int = 12_000, runtime_state=None, offset: int = 0,
                      tool_context=None) -> ToolResult:
    manager = _shell_job_manager(runtime_state)
    if manager is None:
        return ToolResult(
            tool="read_shell_output",
            status="failed",
            output="[error] No shell job manager available",
            error="No shell job manager available",
            metadata={"status_source": "runtime", "job_id": job_id},
        )
    max_chars = _clamp_shell_output_chars(max_chars)
    try:
        output = manager.read_output(job_id, max_chars=max_chars, offset=max(0, int(offset)))
        job = manager.get(job_id) if hasattr(manager, "get") else None
    except (ShellJobNotFound, OSError, ValueError) as exc:
        text = f"[error] {exc}"
        return ToolResult(
            tool="read_shell_output",
            status="failed",
            output=text,
            error=str(exc),
            metadata={"status_source": "shell_job", "job_id": job_id},
        )
    output = fitting_prefix(output, max(1, output_limit() - 800))
    header = f"Shell job {job_id}"
    metadata = {"status_source": "shell_job", "job_id": job_id, "max_chars": max_chars,
                "next_offset": max(0, int(offset)) + len(output)}
    if job is not None:
        header += f" status={job.status} pid={job.pid} exit_code={job.exit_code}"
        metadata.update({"job_status": job.status, "pid": job.pid, "exit_code": job.exit_code})
        buffer = getattr(job, "output", None)
        if buffer is not None and buffer.path is not None and tool_context is not None:
            path = buffer.path.as_posix()
            metadata["source_log"] = path
            tool_context.register_output_artifact({"path": path, "append_only": True})
    body = output or "(no output)"
    return ToolResult(
        tool="read_shell_output",
        status="success",
        output=f"{header}\n\n{body}",
        metadata=metadata,
    )


def stop_shell_job(job_id: str, runtime_state=None) -> ToolResult:
    manager = _shell_job_manager(runtime_state)
    if manager is None:
        return ToolResult(
            tool="stop_shell_job",
            status="failed",
            output="[error] No shell job manager available",
            error="No shell job manager available",
            metadata={"status_source": "runtime", "job_id": job_id},
        )
    try:
        job = manager.stop(job_id)
    except ShellJobNotFound as exc:
        text = f"[error] {exc}"
        return ToolResult(
            tool="stop_shell_job",
            status="failed",
            output=text,
            error=str(exc),
            metadata={"status_source": "shell_job", "job_id": job_id},
        )
    return ToolResult(
        tool="stop_shell_job",
        status="success",
        output=f"Shell job {job.job_id} is {job.status} (pid={job.pid}, exit_code={job.exit_code})",
        metadata={
            "status_source": "shell_job",
            "job_id": job.job_id,
            "job_status": job.status,
            "pid": job.pid,
            "exit_code": job.exit_code,
        },
    )


def _network_allowed(command: str, tool_context) -> bool:
    """Network grant for a direct call that has no prepared decision.

    The agent loop always passes the permission decision's ``network_access``;
    this fallback keeps direct integrations aligned with the same rule: only a
    command the analyzer can prove uses the network keeps connectivity.
    """
    workspace = getattr(tool_context, "workspace", None)
    root = str(workspace.root) if workspace is not None and getattr(workspace, "root", None) else None
    analysis = analyze_shell_command(command, root)
    return ShellEffect.NETWORK in analysis.effects


def _shell_job_manager(runtime_state):
    return getattr(runtime_state, "shell_job_manager", None) if runtime_state is not None else None


def _exit_code_footer(exit_code) -> str:
    if exit_code is None:
        return ""
    return f"\n\n[exit_code: {exit_code}]"


def _clamp_shell_output_chars(value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 12_000
    return max(1, min(100_000, parsed))
