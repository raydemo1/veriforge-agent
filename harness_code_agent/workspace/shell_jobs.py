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
from ..runtime.tool_output import output_limit
from ..support.cancellation import CancellationToken, CancelledError
from ..support.tokens import count_text_tokens
from .shell_environment import shell_executable

ShellJobStatus = Literal["running", "stopping", "exited", "stopped", "failed"]

#: Sentinel for "caller did not supply this fact", distinct from an explicit None.
_UNSET = object()


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
    # The lifecycle facts stay separate: the launched command exited, the
    # managed processes drained, the output readers drained, and the write
    # lease plus private temp directory are gone.  Only the last one is safe to
    # assert to other agents.
    launcher_exited: bool = False
    processes_drained: bool = False
    readers_drained: bool = False
    resources_released: bool = False
    #: Set when a tracking pass could not read a process state.  A later
    #: successful pass clears it; until then the tree cannot be confirmed empty.
    tracking_failed: bool = False
    warnings: list[str] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _reader_threads: list[threading.Thread] = field(default_factory=list, repr=False)
    _finished: threading.Event = field(default_factory=threading.Event, repr=False)
    _processes: dict = field(default_factory=dict, repr=False)
    _write_lease: object = field(default=None, repr=False)
    _temporary_directory: object = field(default=None, repr=False)

    @property
    def drained(self) -> bool:
        """Both halves of the drain, never only the process tree."""
        return self.processes_drained and self.readers_drained

    @property
    def output_tail(self) -> str:
        return self.output.read_tail(12_000)

    def uptime_seconds(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.time()
        return max(0.0, end - self.started_at)

    def launcher_alive(self) -> bool:
        """Whether the launched process may still be running.

        An unreadable state counts as alive: releasing the write lease on a
        guess is worse than keeping it.
        """
        process = self.process
        if process is None:
            return False
        try:
            return process.poll() is None
        except Exception:
            return True

    def has_retained_resources(self) -> bool:
        """Whether the job still holds the write lease or the private temp dir.

        A terminal status is not enough: a job can report `exited` while a
        descendant keeps the lease, and such a job must stay stoppable.
        """
        with self._lock:
            if self.resources_released:
                return False
            if self._write_lease is not None or self._temporary_directory is not None:
                return True
            if self.launcher_alive():
                return True
            return any(_process_running(process) for process in self._processes.values())

    def track_process(self, process):
        try:
            key = (process.pid, process.create_time())
        except psutil.NoSuchProcess:
            return
        with self._lock:
            if key in self._processes:
                return
            self._processes[key] = process

    def mark(self, status: ShellJobStatus, *, exit_code: int | None | object = _UNSET,
             error: str | None = None, warning: str | None = None) -> None:
        """Publish a terminal state, then try to release resources.

        The state change and the resource release are deliberately separate:
        a job whose processes survived termination still becomes terminal for
        its caller, but keeps its write lease instead of pretending it is gone.
        """
        with self._lock:
            self.status = status
            if exit_code is not _UNSET:
                self.exit_code = exit_code
            self.error = error or self.error
            if warning and warning not in self.warnings:
                self.warnings.append(warning)
            if status in {"exited", "stopped", "failed"}:
                self.ended_at = self.ended_at or time.time()
                self._finished.set()
        if status in {"exited", "stopped", "failed"}:
            self.release_resources()

    def release_resources(self) -> bool:
        """Release the write lease and private temp dir, or keep them retryable.

        The lease *is* the shared-workspace write permission: releasing it
        while a managed process still runs would let another session write the
        same directories.  A handle is only dropped after its release actually
        succeeded, so a failed attempt leaves the job in a state that can be
        retried and reported.
        """
        with self._lock:
            if self.resources_released:
                return True
            if self.launcher_alive():
                self._retain("write lease retained: the launched process is still running")
                return False
            alive = [process for process in self._processes.values() if _process_running(process)]
            if alive:
                self._retain(f"write lease retained: {len(alive)} managed process(es) still running")
                return False
            if self.tracking_failed:
                self._retain(
                    "write lease retained: the process tree could not be confirmed after a tracking failure"
                )
                return False
            if self._temporary_directory is not None:
                try:
                    self._temporary_directory.cleanup()
                except OSError as exc:
                    self.error = f"Private shell temp cleanup failed: {exc}"
                    self._retain(f"private temp directory retained: {exc}")
                    return False
                self._temporary_directory = None
            if self._write_lease is not None:
                try:
                    self._write_lease.release()
                except Exception as exc:
                    self._retain(f"write lease release failed: {exc}")
                    return False
                self._write_lease = None
            self.resources_released = True
            return True

    def _retain(self, note: str) -> None:
        if note not in self.warnings:
            self.warnings.append(note)


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
        try:
            self._track_process_tree(job)
            self._start_reader(job, process.stdout, "stdout")
            self._start_reader(job, process.stderr, "stderr")
            threading.Thread(target=self._monitor, args=(job,), daemon=True).start()
        except BaseException as exc:
            # The process exists and is not being monitored: terminate it and
            # keep the lease rather than leaving an untracked child behind.
            self._fail_unconfirmed_exit(job, exc)
            return job

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

    def retained_jobs(self) -> list[ShellJob]:
        """Jobs that still hold a write lease or private temp dir.

        These block suspension and closing just like running jobs, even when
        their status is already terminal.
        """
        return [job for job in self.list_jobs() if job.has_retained_resources()]

    def read_output(self, job_id: str, max_chars: int = 12_000, *, offset: int | None = None) -> str:
        max_chars = _clamp_int(max_chars, minimum=1, maximum=100_000)
        output = self.get(job_id).output
        return output.read_tail(max_chars) if offset is None else output.read(max(0, offset), max_chars)

    def stop(self, job_id: str, *, grace_seconds: float = 5.0) -> ShellJob:
        """Stop a job, including one that already reported a terminal status.

        A job can be `exited` while a descendant still holds the write lease.
        Such a job has no monitor left to clean it up, so stopping it has to
        terminate the survivors and release the lease here.
        """
        job = self.get(job_id)
        with job._lock:
            was_running = job.status in {"running", "stopping"}
            if was_running:
                job.status = "stopping"
            elif not job.has_retained_resources():
                return job
        deadline = time.monotonic() + max(0.0, grace_seconds) + 2.0
        self._terminate_process_tree(job, grace_seconds=grace_seconds)
        if not was_running:
            # A clean tracking pass clears a transient tracking failure so the
            # lease can actually be released instead of staying retained.
            self._track_process_tree(job)
            self._prune_dead(job)
            with job._lock:
                job.processes_drained = not any(
                    _process_running(process) for process in job._processes.values()
                )
            job.release_resources()
            return job
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
        retained = []
        for job in jobs:
            if job.status in {"running", "stopping"} or job.has_retained_resources():
                self.stop(job.job_id)
            if not job.release_resources():
                retained.append(job.job_id)
        if self.running_jobs() or retained:
            raise RuntimeError("Shell processes have not exited; job resources retained")
        self._closed = True

    def export_jobs(self) -> list[dict]:
        """Finished jobs and their output, kept across a runtime suspension."""
        exported = []
        for job in self.list_jobs():
            if job.status in {"running", "stopping"}:
                raise RuntimeError("cannot export running shell jobs")
            if job.has_retained_resources():
                raise RuntimeError("cannot export shell jobs whose resources are retained")
            with job.output._lock:
                exported.append({
                    "job_id": job.job_id, "command": job.command, "pid": job.pid, "status": job.status,
                    "background": job.background, "exit_code": job.exit_code, "started_at": job.started_at,
                    "ended_at": job.ended_at, "error": job.error, "output_error": job.output.error,
                    "output_path": str(job.output.path) if job.output.path is not None else None,
                    "output_text": job.output._text, "output_chars": job.output.total_chars,
                    # Lifecycle facts survive a suspension: a restored background
                    # job must still say whether its launcher had exited.
                    "launcher_exited": job.launcher_exited,
                    "processes_drained": job.processes_drained,
                    "readers_drained": job.readers_drained,
                    "warnings": list(job.warnings)})
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
                           error=entry["error"],
                           launcher_exited=bool(entry.get("launcher_exited", entry["status"] != "running")),
                           processes_drained=bool(entry.get("processes_drained", True)),
                           readers_drained=bool(entry.get("readers_drained", True)),
                           warnings=list(entry.get("warnings") or []))
            job._finished.set()
            # A restored job has no lease and no tracked processes in this process.
            job.resources_released = True
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
        except BaseException as exc:
            self._fail_unconfirmed_exit(job, exc)

    def _fail_unconfirmed_exit(self, job: ShellJob, exc: BaseException) -> None:
        """Terminal state for a monitor that could not confirm the exit.

        Termination is best effort: whatever survives keeps the write lease, so
        a later ``close()`` still reports the retained resources instead of
        assuming they were freed.
        """
        message = f"Process exit could not be confirmed: {type(exc).__name__}: {exc}"
        with job._lock:
            job.error = job.error or message
        try:
            self._terminate_process_tree(job)
        except BaseException as cleanup_exc:  # cleanup must not hide the original failure
            with job._lock:
                job.warnings.append(f"terminating processes after an unconfirmed exit failed: {cleanup_exc}")
        try:
            self._prune_dead(job)
            with job._lock:
                job.processes_drained = not any(
                    _process_running(proc) for proc in job._processes.values()
                )
            job.mark("failed", error=message, warning=message)
        except BaseException:
            with job._lock:
                job.status = "failed"
                job.error = job.error or message
                job.ended_at = job.ended_at or time.time()
                job._finished.set()

    def _wait_for_exit(self, job: ShellJob) -> None:
        process = job.process
        if process is None:
            with job._lock:
                job.processes_drained = True
                job.readers_drained = True
            job.mark("stopped" if job.status == "stopping" else "exited")
            return
        while process.poll() is None:
            self._track_process_tree(job)
            self._prune_dead(job)
            time.sleep(0.1)
        exit_code = process.wait()
        with job._lock:
            job.launcher_exited = True
            job.exit_code = exit_code
        self._drain(job)
        if job.status in {"running", "stopping"}:
            job.mark("stopped" if job.status == "stopping" else "exited", exit_code=exit_code)
        else:
            job.release_resources()

    def _prune_dead(self, job: ShellJob) -> None:
        with job._lock:
            job._processes = {
                key: proc for key, proc in job._processes.items() if _process_running(proc)
            }

    def _drain(self, job: ShellJob) -> None:
        """Wait for managed processes and output readers, always with a bound.

        A background job's descendants *are* the job, so they are followed for
        as long as they live.  A foreground command has already produced its
        result; descendants that outlive it get a grace period and are then
        terminated, because otherwise the tool call would block until its own
        timeout with nothing left to report.
        """
        deadline = None
        if not job.background:
            deadline = time.monotonic() + config.SHELL_JOB_EXIT_DRAIN_SECONDS
        while True:
            self._track_process_tree(job)
            self._prune_dead(job)
            with job._lock:
                alive = [proc for proc in job._processes.values() if _process_running(proc)]
            if not alive:
                break
            if deadline is not None and time.monotonic() >= deadline:
                self._terminate_survivors(job, len(alive))
                break
            time.sleep(0.1)
        self._join_readers(job)
        self._prune_dead(job)
        with job._lock:
            job.processes_drained = not any(
                _process_running(proc) for proc in job._processes.values()
            )

    def _terminate_survivors(self, job: ShellJob, count: int) -> None:
        with job._lock:
            job.warnings.append(
                f"terminated {count} process(es) still running "
                f"{config.SHELL_JOB_EXIT_DRAIN_SECONDS:g}s after the command exited"
            )
        try:
            self._terminate_process_tree(job)
        except Exception as exc:
            with job._lock:
                job.warnings.append(f"terminating surviving processes failed: {exc}")
            return
        deadline = time.monotonic() + config.SHELL_JOB_TERMINATION_CONFIRM_SECONDS
        while time.monotonic() < deadline:
            self._prune_dead(job)
            with job._lock:
                if not any(_process_running(proc) for proc in job._processes.values()):
                    return
            time.sleep(0.05)

    def _join_readers(self, job: ShellJob) -> None:
        deadline = time.monotonic() + config.SHELL_JOB_READER_JOIN_SECONDS
        for thread in list(job._reader_threads):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(remaining)
        stuck = [thread.name for thread in job._reader_threads if thread.is_alive()]
        with job._lock:
            job.readers_drained = not stuck
            if stuck:
                job._retain(
                    f"output readers did not finish within "
                    f"{config.SHELL_JOB_READER_JOIN_SECONDS:g}s; captured output may be incomplete"
                )

    def _track_process_tree(self, job):
        """Refresh the tracked tree and decide whether a failure may be cleared.

        Only a pass that enumerated the children of the *live* launcher is
        authoritative: it re-reads the whole tree from its root.  Once the
        launcher has exited, an empty tracked set proves nothing — the
        descendants that could not be read earlier are exactly the ones missing
        from it — so a previous tracking failure stays set and keeps the lease
        instead of being cleared by "this pass raised nothing".
        """
        if job.launcher_alive():
            if job.pid is None:
                return
            try:
                root = psutil.Process(job.pid)
            except psutil.NoSuchProcess:
                # The launcher reports alive but cannot be resolved: no evidence either way.
                return
            except (psutil.Error, OSError) as exc:
                self._mark_tracking_failed(job, exc)
                return
            try:
                job.track_process(root)
                for child in root.children(recursive=True):
                    job.track_process(child)
            except psutil.NoSuchProcess:
                return
            except (psutil.Error, OSError) as exc:
                self._mark_tracking_failed(job, exc)
                return
            with job._lock:
                job.tracking_failed = False
            return
        with job._lock:
            processes = list(job._processes.values())
        for process in processes:
            try:
                job.track_process(process)
                for child in process.children(recursive=True):
                    job.track_process(child)
            except psutil.NoSuchProcess:
                continue
            except (psutil.Error, OSError) as exc:
                self._mark_tracking_failed(job, exc)
                return

    def _mark_tracking_failed(self, job: ShellJob, exc: BaseException) -> None:
        with job._lock:
            job.tracking_failed = True
            job._retain(f"process tracking failed: {type(exc).__name__}: {exc}")

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
    """Whether a managed process must still be considered alive.

    A state that cannot be read (AccessDenied, a recycled pid, an OS error)
    counts as running: an empty tracked set may only be trusted when every
    read succeeded.
    """
    try:
        if not process.is_running():
            return False
        return process.status() not in {psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD}
    except psutil.NoSuchProcess:
        return False
    except (psutil.Error, OSError):
        return True


def _clamp_int(value: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = minimum
    return max(minimum, min(maximum, parsed))
