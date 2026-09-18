from __future__ import annotations

import contextlib
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import psutil

from .. import config
from .shell_session import (
    _docker_resource_args,
    _docker_user_arg,
    docker_cli_path,
    sandbox_mode,
    validate_shell_configuration,
    windows_shell_kind,
    windows_shell_path,
)

ShellJobStatus = Literal["running", "exited", "stopped", "failed"]


def _posix_rlimits_enabled() -> bool:
    return os.name != "nt" and any(
        int(getattr(config, name, 0) or 0) > 0
        for name in ("SHELL_CPU_SECONDS", "SHELL_FSIZE_BLOCKS", "SHELL_MEMORY_KB")
    )


def _apply_posix_rlimits() -> None:
    """Apply per-job rlimits in the forked child (POSIX host only).

    Failures are swallowed to mirror the ``ulimit ... || true`` semantics
    of the persistent-session path: an unsupported cap or a value above
    the hard limit must not prevent the job from starting.
    """
    import resource

    cpu = int(config.SHELL_CPU_SECONDS)
    fsize = int(config.SHELL_FSIZE_BLOCKS)
    memory_kb = int(config.SHELL_MEMORY_KB)
    settings = []
    if cpu > 0:
        settings.append((resource.RLIMIT_CPU, (cpu, cpu)))
    if fsize > 0:
        settings.append((resource.RLIMIT_FSIZE, (fsize, fsize)))
    if memory_kb > 0:
        bytes_value = memory_kb * 1024
        settings.append((resource.RLIMIT_AS, (bytes_value, bytes_value)))
    for rlimit, value in settings:
        try:
            resource.setrlimit(rlimit, value)
        except (ValueError, OSError):
            pass


class ShellJobNotFound(KeyError):
    pass


class RingBuffer:
    def __init__(self, max_chars: int = 1_000_000):
        self.max_chars = max(1, int(max_chars))
        self._text = ""
        self._lock = threading.RLock()

    def append(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._text += text
            if len(self._text) > self.max_chars:
                self._text = self._text[-self.max_chars:]

    def read_tail(self, max_chars: int) -> str:
        max_chars = max(1, int(max_chars))
        with self._lock:
            return self._text[-max_chars:]


@dataclass
class ShellJob:
    job_id: str
    command: str
    pid: int | None
    process: subprocess.Popen | None
    status: ShellJobStatus = "running"
    exit_code: int | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    output: RingBuffer = field(default_factory=RingBuffer)
    container_name: str | None = None
    error: str | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _reader_threads: list[threading.Thread] = field(default_factory=list, repr=False)

    @property
    def output_tail(self) -> str:
        return self.output.read_tail(12_000)

    def uptime_seconds(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.time()
        return max(0.0, end - self.started_at)

    def mark(self, status: ShellJobStatus, *, exit_code: int | None = None, error: str | None = None) -> None:
        with self._lock:
            self.status = status
            self.exit_code = exit_code
            self.error = error or self.error
            if status != "running" and self.ended_at is None:
                self.ended_at = time.time()


class ShellJobManager:
    def __init__(self, workspace: str | Path, *, ring_buffer_chars: int = 1_000_000):
        self.workspace = str(Path(workspace).resolve())
        self.ring_buffer_chars = ring_buffer_chars
        self._jobs: dict[str, ShellJob] = {}
        self._lock = threading.RLock()
        self._closed = False

    def start(self, command: str, *, early_exit_seconds: float = 0.0) -> ShellJob:
        command = str(command or "").strip()
        job_id = f"shell-job-{uuid.uuid4().hex[:8]}"
        output = RingBuffer(self.ring_buffer_chars)
        if not command:
            job = ShellJob(job_id=job_id, command=command, pid=None, process=None, status="failed", output=output)
            job.mark("failed", error="empty command")
            self._store(job)
            return job

        try:
            process, container_name = self._start_process(command)
        except Exception as exc:
            job = ShellJob(
                job_id=job_id,
                command=command,
                pid=None,
                process=None,
                status="failed",
                output=output,
                error=str(exc),
            )
            output.append(f"[error] Failed to start shell job: {exc}\n")
            job.mark("failed", error=str(exc))
            self._store(job)
            return job

        job = ShellJob(
            job_id=job_id,
            command=command,
            pid=process.pid,
            process=process,
            output=output,
            container_name=container_name,
        )
        self._store(job)
        self._start_reader(job, process.stdout, "stdout")
        self._start_reader(job, process.stderr, "stderr")
        threading.Thread(target=self._monitor, args=(job,), daemon=True).start()

        if early_exit_seconds > 0:
            deadline = time.time() + early_exit_seconds
            while time.time() < deadline:
                if process.poll() is not None:
                    time.sleep(0.05)
                    break
                time.sleep(0.05)
        return job

    def get(self, job_id: str) -> ShellJob:
        with self._lock:
            job = self._jobs.get(str(job_id))
        if job is None:
            raise ShellJobNotFound(f"Unknown shell job: {job_id}")
        return job

    def list_jobs(self) -> list[ShellJob]:
        with self._lock:
            return list(self._jobs.values())

    def running_jobs(self) -> list[ShellJob]:
        return [job for job in self.list_jobs() if job.status == "running"]

    def read_output(self, job_id: str, max_chars: int = 12_000) -> str:
        max_chars = _clamp_int(max_chars, minimum=1, maximum=100_000)
        return self.get(job_id).output.read_tail(max_chars)

    def stop(self, job_id: str, *, grace_seconds: float = 5.0) -> ShellJob:
        job = self.get(job_id)
        if job.status != "running":
            return job
        if job.container_name:
            self._stop_docker_container(job)
        self._terminate_process_tree(job, grace_seconds=grace_seconds)
        process = job.process
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=max(0.1, grace_seconds))
            except subprocess.TimeoutExpired:
                with contextlib.suppress(Exception):
                    process.kill()
        job.mark("stopped", exit_code=process.poll() if process is not None else None)
        return job

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.status == "running":
                with contextlib.suppress(Exception):
                    self.stop(job.job_id)

    def _store(self, job: ShellJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def _start_process(self, command: str) -> tuple[subprocess.Popen, str | None]:
        if sandbox_mode() == "docker":
            return self._start_docker_process(command)
        if os.name == "nt":
            return self._start_windows_process(command), None
        return self._start_posix_process(command), None

    def _start_posix_process(self, command: str) -> subprocess.Popen:
        return subprocess.Popen(
            ["/bin/bash", "-lc", command],
            cwd=self.workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
            preexec_fn=_apply_posix_rlimits if _posix_rlimits_enabled() else None,
        )

    def _start_windows_process(self, command: str) -> subprocess.Popen:
        # Windows Job Objects are the native enforcement mechanism; the host
        # backend intentionally applies no CPU/memory cap here — run with
        # HARNESS_SANDBOX_MODE=docker when those caps are required.
        validate_shell_configuration()
        path = windows_shell_path()
        assert path is not None
        kind = windows_shell_kind()
        if kind == "pwsh":
            wrapped = (
                f"& {{ {command} }}; "
                "if ($global:LASTEXITCODE -is [int]) { exit $global:LASTEXITCODE } "
                "elseif ($?) { exit 0 } else { exit 1 }"
            )
            args = [path, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", wrapped]
        else:
            args = [path, "--cd", self.workspace, "--exec", "bash", "-lc", command]
        return subprocess.Popen(
            args,
            cwd=self.workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )

    def _start_docker_process(self, command: str) -> tuple[subprocess.Popen, str]:
        docker = docker_cli_path()
        if not docker:
            raise RuntimeError("Docker sandbox requested but docker CLI was not found")
        container_name = f"hca-job-{uuid.uuid4().hex[:12]}"
        network = (config.DOCKER_NETWORK or "none").strip().lower()
        if network not in {"none", "bridge"}:
            raise ValueError("HARNESS_DOCKER_NETWORK must be none or bridge")
        args = [
            docker,
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            network,
            "--security-opt=no-new-privileges",
            "-v",
            f"{self.workspace}:/workspace",
            "-w",
            "/workspace",
        ]
        args.extend(_docker_resource_args())
        args.extend(_docker_user_arg())
        args.extend([config.DOCKER_IMAGE, "bash", "--noprofile", "--norc", "-lc", command])
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return process, container_name

    def _start_reader(self, job: ShellJob, pipe, stream_name: str) -> None:
        if pipe is None:
            return
        thread = threading.Thread(target=self._reader_loop, args=(job, pipe, stream_name), daemon=True)
        with job._lock:
            job._reader_threads.append(thread)
        thread.start()

    def _reader_loop(self, job: ShellJob, pipe, stream_name: str) -> None:
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                if stream_name == "stderr":
                    job.output.append(f"[stderr] {line}")
                else:
                    job.output.append(line)
        finally:
            with contextlib.suppress(Exception):
                pipe.close()

    def _monitor(self, job: ShellJob) -> None:
        process = job.process
        if process is None:
            return
        exit_code = process.wait()
        for thread in list(job._reader_threads):
            thread.join()
        if job.status == "running":
            job.mark("exited", exit_code=exit_code)

    def _terminate_process_tree(self, job: ShellJob, *, grace_seconds: float = 5.0) -> None:
        if job.pid is None:
            return
        try:
            parent = psutil.Process(job.pid)
            children = parent.children(recursive=True)
            processes = children + [parent]
            for proc in children:
                _safe_terminate(proc)
            _safe_terminate(parent)
            _, alive = psutil.wait_procs(processes, timeout=grace_seconds)
            for proc in alive:
                _safe_kill(proc)
            if alive:
                psutil.wait_procs(alive, timeout=2)
        except (psutil.NoSuchProcess, ProcessLookupError):
            return
        except Exception:
            self._terminate_process_tree_fallback(job)

    def _terminate_process_tree_fallback(self, job: ShellJob) -> None:
        if job.pid is None:
            return
        if os.name == "nt":
            with contextlib.suppress(Exception):
                subprocess.run(
                    ["taskkill", "/PID", str(job.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                )
        else:
            with contextlib.suppress(Exception):
                os.killpg(job.pid, 15)

    def _stop_docker_container(self, job: ShellJob) -> None:
        if not job.container_name:
            return
        docker = docker_cli_path()
        if not docker:
            return
        with contextlib.suppress(Exception):
            subprocess.run(
                [docker, "rm", "-f", job.container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )


def _safe_terminate(process) -> None:
    with contextlib.suppress(Exception):
        process.terminate()


def _safe_kill(process) -> None:
    with contextlib.suppress(Exception):
        process.kill()


def _clamp_int(value: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = minimum
    return max(minimum, min(maximum, parsed))
