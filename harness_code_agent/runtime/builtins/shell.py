"""Shell execution and background shell job tools."""
from __future__ import annotations

import json

from ... import config
from ...agent.cancellation import CancelledError
from ...workspace.shell_jobs import ShellJobNotFound
from ..shell_classification import is_long_running_shell_command
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
) -> ToolResult:
    """Run one self-contained shell command from the workspace root.

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
    return _run_shell(command, timeout, write_paths or [], runtime_state, cancellation_token, requested_timeout)


def _run_shell(command, timeout, write_paths, runtime_state, cancellation_token, requested_timeout):
    import time

    manager = _shell_job_manager(runtime_state)
    if manager is None:
        return ToolResult(tool="run_bash", status="failed", output="[error] No shell job manager available",
                          error="No shell job manager available", metadata={"status_source": "runtime"})
    background = is_long_running_shell_command(command)
    job = manager.start(command, write_paths=write_paths, background=background,
                        early_exit_seconds=0.5 if background else 0,
                        cancellation_token=cancellation_token, turn_id=getattr(runtime_state, "turn_id", ""))
    metadata = {"status_source": "shell", "job_id": job.job_id, "pid": job.pid,
                "write_paths": write_paths, "job_status": job.status,
                "requested_timeout": requested_timeout, "effective_timeout": timeout}
    if background and job.status == "running":
        return ToolResult(tool="run_bash", status="success",
                          output=f"Started background shell job {job.job_id} (pid={job.pid}). Use read_shell_output to inspect logs and stop_shell_job to stop it.",
                          metadata=metadata)
    timed_out = False
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
    error = job.error or (f"Command timed out after {timeout}s" if timed_out else None)
    return _job_result(job, dict(metadata, timed_out=timed_out), error)


def _job_result(job, metadata, error=None):
    if job.output.path is None:
        output = job.output.read(0, job.output.total_chars)
    else:
        preview = max(2, int(config.SHELL_OUTPUT_PREVIEW_CHARS))
        output = (job.output.read(0, preview // 2) + "\n\n[Full output saved to artifact]\n\n"
                  + job.output.read_tail(preview - preview // 2))
    if job.exit_code == 127 and "native-sandbox:" in output:
        error = output
    metadata.update(job_status=job.status, exit_code=job.exit_code,
                    output_bytes=job.output.path.stat().st_size if job.output.path else len(output.encode("utf-8")),
                    output_chars=job.output.total_chars)
    if job.output.path is not None:
        metadata.update(artifact_path=str(job.output.path), output_spilled=True)
    return ToolResult(tool="run_bash", status="failed" if error else "success",
                      output=(output or "(no output)") + _exit_code_footer(job.exit_code),
                      error=error, return_code=job.exit_code, metadata=metadata)


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
            relative = buffer.path.relative_to(tool_context.workspace.root).as_posix()
            metadata["source_log"] = relative
            tool_context.register_output_artifact({"path": relative, "append_only": True})
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
