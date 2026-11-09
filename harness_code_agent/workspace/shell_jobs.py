from __future__ import annotations

import codecs
import contextlib
import os
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import psutil

from .. import config
from ..agent.cancellation import CancellationToken, CancelledError
from ..agent.context import count_text_tokens
from ..runtime.tool_output import output_limit
from .shell_environment import shell_executable

ShellJobStatus = Literal["running", "stopping", "exited", "stopped", "failed"]


def _posix_rlimits_enabled() -> bool:
    return os.name != "nt" and any(
        int(getattr(config, name, 0) or 0) > 0
        for name in ("SHELL_CPU_SECONDS", "SHELL_FSIZE_BLOCKS", "SHELL_MEMORY_KB")
    )


def _apply_posix_rlimits() -> None:
    """Apply per-job rlimits in the forked child (POSIX host only).

    Unsupported caps leave the child process's inherited limit in place.
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


class ShellOutputBuffer:
    def __init__(self, artifact_dir: Path | None = None):
        self.artifact_dir = artifact_dir
        self.path: Path | None = None
        self._text = ""
        self.total_chars = 0
        self._lock = threading.RLock()
        self.error: str | None = None

    def append(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            if self.error:
                return
            try:
                if self.path is None:
                    self._text += text
                    if count_text_tokens(self._text) <= output_limit():
                        self.total_chars += len(text)
                        return
                    directory = self.artifact_dir or Path.cwd() / ".harness" / "observations" / "default" / "jobs"
                    directory.mkdir(parents=True, exist_ok=True)
                    self.path = directory / f"job_{uuid.uuid4().hex}.txt"
                    with self.path.open("x", encoding="utf-8", newline="") as stream:
                        stream.write(self._text)
                    self._text = ""
                else:
                    with self.path.open("a", encoding="utf-8", newline="") as stream:
                        stream.write(text)
                self.total_chars += len(text)
            except OSError as exc:
                self.error = f"Output incomplete: {exc}"

    def read(self, offset: int, max_chars: int) -> str:
        with self._lock:
            if self.error:
                raise OSError(self.error)
            if self.path is None:
                return self._text[offset:offset + max_chars]
            with self.path.open("r", encoding="utf-8", newline="") as stream:
                remaining = offset
                while remaining:
                    chunk = stream.read(min(4096, remaining))
                    if not chunk:
                        return ""
                    remaining -= len(chunk)
                return stream.read(max_chars)

    def read_tail(self, max_chars: int) -> str:
        return self.read(max(0, self.total_chars - max_chars), max_chars)


@dataclass
class ShellJob:
    job_id: str
    command: str
    pid: int | None
    process: subprocess.Popen | None
    status: ShellJobStatus = "running"
    background: bool = True
    exit_code: int | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    output: ShellOutputBuffer = field(default_factory=ShellOutputBuffer)
    error: str | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _reader_threads: list[threading.Thread] = field(default_factory=list, repr=False)
    _finished: threading.Event = field(default_factory=threading.Event, repr=False)
    _processes: dict = field(default_factory=dict, repr=False)
    _write_lease: object = field(default=None, repr=False)
    _temporary_directory: object = field(default=None, repr=False)

    @property
    def output_tail(self) -> str:
        return self.output.read_tail(12_000)

    def uptime_seconds(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.time()
        return max(0.0, end - self.started_at)

    def track_process(self, process):
        try:
            key = (process.pid, process.create_time())
        except psutil.NoSuchProcess:
            return
        with self._lock:
            if key in self._processes:
                return
            self._processes[key] = process

    def mark(self, status: ShellJobStatus, *, exit_code: int | None = None, error: str | None = None) -> None:
        with self._lock:
            self.status = status
            self.exit_code = exit_code
            self.error = error or self.error
            if status in {"exited", "stopped", "failed"}:
                try:
                    if self._temporary_directory is not None:
                        self._temporary_directory.cleanup()
                        self._temporary_directory = None
                except OSError as exc:
                    self.error = f"Private shell temp cleanup failed: {exc}"
                finally:
                    if self._write_lease is not None:
                        self._write_lease.release()
                        self._write_lease = None
                self.ended_at = self.ended_at or time.time()
                self._finished.set()


class ShellJobManager:
    def __init__(self, workspace: str | Path, *, artifact_dir: Path | None = None, environment="local", session_id=None):
        self.workspace = str(Path(workspace).resolve())
        self.environment = environment
        self.session_id = session_id
        self.artifact_dir = artifact_dir or Path(workspace) / ".harness" / "observations" / "default" / "jobs"
        self._jobs: dict[str, ShellJob] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._closing = False
        self._start_token = CancellationToken()
        self._starting = 0
        self._starts_idle = threading.Event()
        self._starts_idle.set()

    def start(self, command: str, *, early_exit_seconds: float = 0.0, write_paths=(),
              background=True, cancellation_token=None, turn_id="") -> ShellJob:
        with self._lock:
            if self._closing:
                raise RuntimeError("Shell job manager is closing")
            self._starting += 1
            self._starts_idle.clear()
        token = self._start_token.create_child()
        remove_callback = cancellation_token.add_callback(token.cancel) if cancellation_token else lambda: None
        try:
            return self._start_job(command, early_exit_seconds=early_exit_seconds, write_paths=write_paths,
                                   background=background, cancellation_token=token, turn_id=turn_id)
        finally:
            remove_callback()
            token.close()
            with self._lock:
                self._starting -= 1
                if not self._starting:
                    self._starts_idle.set()

    def _start_job(self, command, *, early_exit_seconds, write_paths, background, cancellation_token, turn_id):
        cancellation_token.check()
        command = str(command or "").strip()
        job_id = f"shell-job-{uuid.uuid4().hex[:8]}"
        output = ShellOutputBuffer(self.artifact_dir)
        if not command:
            job = ShellJob(job_id=job_id, command=command, pid=None, process=None, status="failed", output=output)
            job.mark("failed", error="empty command")
            self._store(job)
            return job

        lease, temporary = None, None
        try:
            if os.name == "nt":
                from .access import WorkspaceAccessCoordinator
                from .native_shell import (
                    launch_arguments,
                    write_claims,
                    write_directories,
                )

                root = Path(self.workspace)
                directories = write_directories(root, write_paths)
                if background and root in directories and self.environment != "worktree":
                    raise ValueError("后台任务不能申请整个共享 Local 的写入许可；请缩小 write_paths，或由用户主动创建 Worktree")
                access = WorkspaceAccessCoordinator(root)
                lease = access.lease(write_claims(root, write_paths), cancellation_token, owner=self.session_id)
                if directories and self.session_id is not None:
                    access.limit_recovery(self.session_id, turn_id, "Shell：目录内写入无法完整回撤")
                for directory in directories:
                    directory.mkdir(parents=True, exist_ok=True)
                temporary = tempfile.TemporaryDirectory(prefix="veriforge-shell-")
                args = launch_arguments(root, directories, temporary.name, command)
                if cancellation_token is not None:
                    cancellation_token.check()
                process = subprocess.Popen(
                    args, cwd=self.workspace, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env=dict(os.environ, TEMP=temporary.name, TMP=temporary.name, TMPDIR=temporary.name),
                    creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                )
            else:
                if write_paths:
                    raise ValueError("write_paths requires the Windows directory sandbox")
                process = self._start_posix_process(command)
        except BaseException as exc:
            if temporary is not None:
                temporary.cleanup()
            if lease is not None:
                lease.release()
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, CancelledError):
                raise
            job = ShellJob(job_id=job_id, command=command, pid=None, process=None,
                           status="failed", output=output, error=str(exc))
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
            _write_lease=lease,
            _temporary_directory=temporary,
            background=background,
        )
        self._store(job)
        self._track_process_tree(job)
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
        return [job for job in self.list_jobs() if job.status in {"running", "stopping"}]

    def read_output(self, job_id: str, max_chars: int = 12_000, *, offset: int | None = None) -> str:
        max_chars = _clamp_int(max_chars, minimum=1, maximum=100_000)
        output = self.get(job_id).output
        return output.read_tail(max_chars) if offset is None else output.read(max(0, offset), max_chars)

    def stop(self, job_id: str, *, grace_seconds: float = 5.0) -> ShellJob:
        job = self.get(job_id)
        with job._lock:
            if job.status not in {"running", "stopping"}:
                return job
            job.status = "stopping"
        deadline = time.monotonic() + max(0.0, grace_seconds) + 2.0
        self._terminate_process_tree(job, grace_seconds=grace_seconds)
        job._finished.wait(timeout=max(0.0, deadline - time.monotonic()))
        return job

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closing = True
        self._start_token.cancel()
        if not self._starts_idle.wait(5):
            raise RuntimeError("Shell startup has not exited; job resources retained")
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.status in {"running", "stopping"}:
                self.stop(job.job_id)
        if self.running_jobs():
            raise RuntimeError("Shell processes have not exited; job resources retained")
        self._closed = True

    def export_jobs(self) -> list[dict]:
        """Finished jobs and their output, kept across a runtime suspension."""
        exported = []
        for job in self.list_jobs():
            if job.status in {"running", "stopping"}:
                raise RuntimeError("cannot export running shell jobs")
            with job.output._lock:
                exported.append({
                    "job_id": job.job_id, "command": job.command, "pid": job.pid, "status": job.status,
                    "background": job.background, "exit_code": job.exit_code, "started_at": job.started_at,
                    "ended_at": job.ended_at, "error": job.error, "output_error": job.output.error,
                    "output_path": str(job.output.path) if job.output.path is not None else None,
                    "output_text": job.output._text, "output_chars": job.output.total_chars})
        return exported

    def import_jobs(self, entries: list[dict]) -> None:
        for entry in entries:
            output = ShellOutputBuffer(self.artifact_dir)
            output.path = Path(entry["output_path"]) if entry["output_path"] else None
            output._text = entry["output_text"]
            output.total_chars = entry["output_chars"]
            output.error = entry["output_error"]
            job = ShellJob(job_id=entry["job_id"], command=entry["command"], pid=entry["pid"], process=None,
                           status=entry["status"], background=entry["background"], exit_code=entry["exit_code"],
                           started_at=entry["started_at"], ended_at=entry["ended_at"], output=output,
                           error=entry["error"])
            job._finished.set()
            self._store(job)

    def _store(self, job: ShellJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def _start_posix_process(self, command: str) -> subprocess.Popen:
        return subprocess.Popen(
            [shell_executable(), "-lc", command],
            cwd=self.workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            start_new_session=True,
            # rlimits must be set between fork and exec; this is the only
            # stdlib hook for it and it stays None unless a limit is configured.
            preexec_fn=_apply_posix_rlimits if _posix_rlimits_enabled() else None,  # noqa: PLW1509
        )

    def _start_reader(self, job: ShellJob, pipe, stream_name: str) -> None:
        if pipe is None:
            return
        thread = threading.Thread(target=self._reader_loop, args=(job, pipe, stream_name), daemon=True)
        with job._lock:
            job._reader_threads.append(thread)
        thread.start()

    def _reader_loop(self, job: ShellJob, pipe, stream_name: str) -> None:
        try:
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            while chunk := pipe.read1(4096):
                text = decoder.decode(chunk)
                job.output.append(("[stderr] " if stream_name == "stderr" else "") + text)
            job.output.append(decoder.decode(b"", final=True))
        finally:
            with contextlib.suppress(Exception):
                pipe.close()

    def _monitor(self, job: ShellJob) -> None:
        try:
            self._wait_for_exit(job)
        except Exception as exc:
            with job._lock:
                job.error = f"Process exit could not be confirmed: {exc}"

    def _wait_for_exit(self, job: ShellJob) -> None:
        process = job.process
        if process is None:
            return
        while process.poll() is None:
            self._track_process_tree(job)
            with job._lock:
                job._processes = {key: proc for key, proc in job._processes.items() if _process_running(proc)}
            time.sleep(0.1)
        exit_code = process.wait()
        while True:
            self._track_process_tree(job)
            with job._lock:
                processes = list(job._processes.values())
            if not any(_process_running(proc) for proc in processes):
                break
            time.sleep(0.1)
        for thread in list(job._reader_threads):
            thread.join()
        if job.status in {"running", "stopping"}:
            job.mark("stopped" if job.status == "stopping" else "exited", exit_code=exit_code)

    def _track_process_tree(self, job):
        with job._lock:
            processes = list(job._processes.values())
        if not processes and job.pid is not None and job.process is not None and job.process.poll() is None:
            try:
                processes.append(psutil.Process(job.pid))
            except psutil.NoSuchProcess:
                pass
        if job.process is not None and job.process.poll() is None:
            processes = processes[:1]
        for process in processes:
            try:
                job.track_process(process)
                for child in process.children(recursive=True):
                    job.track_process(child)
            except psutil.NoSuchProcess:
                continue

    def _terminate_process_tree(self, job: ShellJob, *, grace_seconds: float = 5.0) -> None:
        if job.pid is None:
            return
        try:
            self._track_process_tree(job)
            with job._lock:
                processes = list(job._processes.values())
            for proc in processes:
                _safe_terminate(proc)
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
        if job.pid is None or job.process is None or job.process.poll() is not None:
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



def _safe_terminate(process) -> None:
    with contextlib.suppress(Exception):
        process.terminate()


def _safe_kill(process) -> None:
    with contextlib.suppress(Exception):
        process.kill()


def _process_running(process) -> bool:
    try:
        return process.is_running() and process.status() not in {psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD}
    except psutil.NoSuchProcess:
        return False


def _clamp_int(value: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = minimum
    return max(minimum, min(maximum, parsed))
