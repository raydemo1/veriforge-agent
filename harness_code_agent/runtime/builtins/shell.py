"""Shell execution and background shell job tools."""
from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess

from ... import config
from ...agent.cancellation import CancelledError
from ...workspace.shell_jobs import ShellJobNotFound
from ..shell_classification import analyze_shell_command
from ..tool_result import ToolResult


def run_bash(
    command: str,
    timeout: int = 300,
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
    try:
        timeout = max(1, min(int(timeout), int(config.TOOL_MAX_TIMEOUT_SECONDS)))
    except (TypeError, ValueError):
        timeout = int(config.TOOL_DEFAULT_TIMEOUT_SECONDS)
    if analyze_shell_command(command).long_running:
        manager = _shell_job_manager(runtime_state)
        if manager is None:
            return ToolResult(
                tool="run_bash",
                status="failed",
                output="[error] No shell job manager available for long-running run_bash command",
                error="No shell job manager available",
                metadata={"status_source": "runtime"},
            )
        job = manager.start(command, early_exit_seconds=0.5)
        metadata = {
            "status_source": "shell_job",
            "job_id": job.job_id,
            "pid": job.pid,
            "job_status": job.status,
            "exit_code": job.exit_code,
        }
        if job.status == "running":
            output = (
                f"Started background shell job {job.job_id} (pid={job.pid}). "
                "Use read_shell_output to inspect logs and stop_shell_job to stop it."
            )
            return ToolResult(tool="run_bash", status="success", output=output, metadata=metadata)
        # The process started and ended on its own: an invocation success,
        # even though the "long-running" command exited early (often an error
        # the model needs to diagnose). Report it as content, not failure.
        output = f"Long-running command exited immediately as {job.status}."
        if job.output_tail:
            output += f"\n\nRecent output:\n{job.output_tail}"
        output += _exit_code_footer(job.exit_code)
        return ToolResult(
            tool="run_bash",
            status="success",
            output=output,
            return_code=job.exit_code,
            metadata=metadata,
        )

    from ...workspace.shell_session import PersistentShellSession

    one_shot_powershell = _requires_one_shot_powershell(command)
    shell_session = PersistentShellSession(_workspace_root(tool_context))
    remove_cancel_callback = lambda: None
    if runtime_state is not None and hasattr(runtime_state, "register_shell_session"):
        runtime_state.register_shell_session(shell_session)
    if cancellation_token is not None and not one_shot_powershell:
        remove_cancel_callback = cancellation_token.add_callback(
            lambda: _interrupt_shell_session(shell_session)
        )
    try:
        if cancellation_token is not None:
            cancellation_token.check()
        if one_shot_powershell:
            shell_result = _run_one_shot_powershell(
                command,
                timeout,
                tool_context,
                cancellation_token,
            )
        else:
            shell_result = shell_session.run(command, timeout=timeout)
        if cancellation_token is not None:
            cancellation_token.check()
        if shell_result.timed_out:
            output = (
                f"[error] Command timed out after {timeout}s. "
                f"If this command legitimately needs more time (e.g. compilation, training), "
                f"retry with a larger timeout parameter."
            )
            return ToolResult(
                tool="run_bash",
                status="failed",
                output=output,
                error=f"Command timed out after {timeout}s",
                return_code=shell_result.exit_code,
                metadata={"timed_out": True, "status_source": "shell"},
            )
        output = _build_shell_output(shell_result.stdout, shell_result.stderr)
        output = output or "(no output)"
        output += _exit_code_footer(shell_result.exit_code)
        truncated = bool(getattr(shell_result, "output_truncated", False))
        output_bytes = int(getattr(shell_result, "output_bytes", 0) or 0)
        if truncated:
            output += (
                f"\n\n[output truncated after {output_bytes} bytes; "
                "the command was interrupted. Narrow the command, paginate it, "
                "or redirect output to a file and read selected ranges.]"
            )
        return ToolResult(
            tool="run_bash",
            status="success",
            output=output,
            error=None,
            return_code=shell_result.exit_code,
            metadata={
                "timed_out": False,
                "output_truncated": truncated,
                "output_bytes": output_bytes,
                "status_source": "shell",
            },
        )
    except CancelledError:
        raise
    except Exception as e:
        metadata = {"status_source": "exception"}
        return ToolResult(
            tool="run_bash",
            status="failed",
            output=f"[error] {e}",
            error=str(e),
            metadata=metadata,
        )
    finally:
        remove_cancel_callback()
        if runtime_state is not None and hasattr(runtime_state, "unregister_shell_session"):
            runtime_state.unregister_shell_session(shell_session)
        shell_session.close()


def _requires_one_shot_powershell(command: str) -> bool:
    """Run PowerShell ``exit`` through an explicit one-shot process."""
    if os.name != "nt" or not re.search(r"(?i)(?:^|[;|&]\s*)exit(?:\s|$)", command):
        return False
    from ...workspace.shell_session import sandbox_mode, windows_shell_kind

    return sandbox_mode() == "host" and windows_shell_kind() == "pwsh"


def _interrupt_shell_session(shell_session) -> None:
    with contextlib.suppress(Exception):
        shell_session.interrupt()


def _run_one_shot_powershell(command: str, timeout: int, tool_context, cancellation_token=None):
    from ...workspace.shell_session import ShellResult, windows_shell_path

    executable = windows_shell_path()
    if executable is None:
        raise RuntimeError("PowerShell 7 (pwsh) was not found")
    process = subprocess.Popen(
        [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        cwd=_workspace_root(tool_context),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    remove_cancel_callback = lambda: None
    if cancellation_token is not None:
        remove_cancel_callback = cancellation_token.add_callback(
            lambda: _terminate_process_tree(process)
        )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_tree(process)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process)
            stdout = str(exc.stdout or "")
            stderr = str(exc.stderr or "")
            for stream in (process.stdout, process.stderr, process.stdin):
                if stream is not None:
                    with contextlib.suppress(Exception):
                        stream.close()
        return ShellResult(
            stdout=str(stdout or exc.stdout or ""),
            stderr=str(stderr or exc.stderr or ""),
            exit_code=130,
            timed_out=True,
        )
    finally:
        remove_cancel_callback()
    stdout, stderr, truncated, output_bytes = _cap_one_shot_output(stdout or "", stderr or "")
    return ShellResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=process.returncode,
        output_truncated=truncated,
        output_bytes=output_bytes,
    )


def _cap_one_shot_output(stdout: str, stderr: str) -> tuple[str, str, bool, int]:
    max_bytes = int(getattr(config, "SHELL_MAX_OUTPUT_BYTES", 0) or 0)
    total = len(stdout.encode("utf-8", errors="replace")) + len(
        stderr.encode("utf-8", errors="replace")
    )
    if not max_bytes or total <= max_bytes:
        return stdout, stderr, False, total
    encoding = "utf-8"
    keep = max(0, max_bytes // 2)
    stdout = stdout.encode(encoding, errors="replace")[:keep].decode(encoding, errors="replace")
    stderr = stderr.encode(encoding, errors="replace")[:keep].decode(encoding, errors="replace")
    return stdout, stderr, True, total


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        with contextlib.suppress(Exception):
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                timeout=5,
                check=False,
            )
    if process.poll() is None:
        with contextlib.suppress(Exception):
            process.kill()


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
    ]
    return ToolResult(
        tool="list_shell_jobs",
        status="success",
        output=json.dumps({"jobs": jobs}, ensure_ascii=False),
        metadata={"status_source": "shell_job", "job_count": len(jobs)},
    )


def read_shell_output(job_id: str, max_chars: int = 12_000, runtime_state=None) -> ToolResult:
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
        output = manager.read_output(job_id, max_chars=max_chars)
        job = manager.get(job_id) if hasattr(manager, "get") else None
    except ShellJobNotFound as exc:
        text = f"[error] {exc}"
        return ToolResult(
            tool="read_shell_output",
            status="failed",
            output=text,
            error=str(exc),
            metadata={"status_source": "shell_job", "job_id": job_id},
        )
    header = f"Shell job {job_id}"
    metadata = {"status_source": "shell_job", "job_id": job_id, "max_chars": max_chars}
    if job is not None:
        header += f" status={job.status} pid={job.pid} exit_code={job.exit_code}"
        metadata.update({"job_status": job.status, "pid": job.pid, "exit_code": job.exit_code})
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


def _workspace_root(tool_context=None) -> str:
    if tool_context is not None:
        return str(tool_context.workspace.root)
    return config.WORKSPACE


def _clamp_shell_output_chars(value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 12_000
    return max(1, min(100_000, parsed))


def _build_shell_output(stdout: str, stderr: str) -> str:
    """Build shell output string from stdout and stderr."""
    stderr = (stderr or "").strip()
    stdout = (stdout or "").strip()
    if stderr:
        if stdout:
            return stdout + "\n\n--- STDERR ---\n" + stderr
        return "--- STDERR ---\n" + stderr
    return stdout
