from __future__ import annotations

import base64
import contextlib
import hashlib
import logging
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .. import config

log = logging.getLogger("harness")

STDERR_SEPARATOR = "\n\n--- STDERR ---\n"


def combine_output(stdout: str, stderr: str) -> str:
    """Canonical layout for stdout+stderr used by ToolResults and artifacts."""
    stderr = (stderr or "").strip()
    stdout = (stdout or "").strip()
    if stderr:
        if stdout:
            return stdout + STDERR_SEPARATOR + stderr
        return "--- STDERR ---\n" + stderr
    return stdout


@dataclass
class ShellResult:
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool = False
    # True when the full output was streamed to ``artifact_path`` and
    # stdout/stderr only carry a bounded head+tail preview.  The command
    # itself ran to completion; it is never interrupted for output volume.
    output_spilled: bool = False
    # Size of the NORMALISED combined artifact (UTF-8 bytes of stripped
    # stdout + the inserted "--- STDERR ---" separator + stripped stderr) —
    # a presentation-size figure, not a raw stdout+stderr byte accounting.
    output_bytes: int = 0
    output_chars: int = 0
    artifact_path: str | None = None
    artifact_sha256: str | None = None


class PersistentShellSession:
    """Persistent shell session that preserves cwd/env across commands."""

    def __init__(self, cwd: str):
        self.cwd = str(Path(cwd).resolve())
        self._backend: _BaseShellBackend
        mode = sandbox_mode()
        if mode == "docker":
            self._backend = _DockerShellBackend(self.cwd)
        elif os.name == "nt":
            self._backend = _make_windows_shell_backend(self.cwd)
        else:
            self._backend = _PosixShellBackend(self.cwd)

    def run(self, command: str, timeout: int = 300, artifact_dir: str | Path | None = None) -> ShellResult:
        return self._backend.run(command, timeout, artifact_dir=artifact_dir)

    def interrupt(self) -> None:
        self._backend.interrupt()

    def close(self) -> None:
        self._backend.close()


class _BaseShellBackend:
    _SYNC_TIMEOUT_SECONDS = 5

    def __init__(self, cwd: str):
        self.cwd = cwd
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._closed = False
        self._start()
        self._start_reader()
        self._sync()

    def _start(self) -> None:
        raise NotImplementedError

    def _reader_loop(self) -> None:
        raise NotImplementedError

    def _send(self, script: str) -> None:
        raise NotImplementedError

    def _interrupt_impl(self) -> None:
        raise NotImplementedError

    def _cleanup_impl(self) -> None:
        raise NotImplementedError

    def _build_script(self, command: str, marker: str) -> str:
        raise NotImplementedError

    def _build_sync_command(self, marker: str) -> str:
        raise NotImplementedError

    def _start_reader(self) -> None:
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _capture_until(
        self,
        token: str,
        timeout: float,
        *,
        spool_dir: Path,
        preview_bytes: int | None = None,
    ) -> "_OutputCapture | None":
        """Drain output until ``token`` (followed by a newline) or deadline.

        Draining never stops early because of output volume: the capture
        keeps memory bounded and streams overflow to a spool file.  Returns
        None on timeout; the caller owns killing the command in that case.
        """
        deadline = time.time() + timeout
        capture = _OutputCapture(
            token.encode("utf-8"),
            preview_bytes=int(config.SHELL_OUTPUT_PREVIEW_CHARS if preview_bytes is None else preview_bytes),
            spool_dir=spool_dir,
        )

        while time.time() < deadline:
            try:
                chunk = self._queue.get(timeout=min(0.1, max(deadline - time.time(), 0.01)))
            except queue.Empty:
                if self._process.poll() is not None:
                    break
                continue
            capture.feed(chunk)
            if capture.found:
                return capture

        return None

    def _sync(self) -> None:
        marker = f"__CODEX_SYNC_{uuid.uuid4().hex}__"
        self._drain_queue()
        self._send(self._build_sync_command(marker))
        capture = self._capture_until(
            marker,
            timeout=self._SYNC_TIMEOUT_SECONDS,
            spool_dir=Path(tempfile.gettempdir()),
            preview_bytes=4096,
        )
        if capture is None:
            raise RuntimeError("Shell failed to become ready")
        capture.discard()

    def run(self, command: str, timeout: int = 300, artifact_dir: str | Path | None = None) -> ShellResult:
        marker = uuid.uuid4().hex
        stdout_marker = f"__CODEX_STDOUT_{marker}__"
        stderr_marker = f"__CODEX_STDERR_{marker}__"
        exit_marker = f"__CODEX_EXIT_{marker}__"

        spool_dir = Path(artifact_dir) if artifact_dir else Path(tempfile.gettempdir()) / "hca-shell-output"
        self._drain_queue()
        self._send(self._build_script(command, marker))
        capture = self._capture_until(exit_marker, timeout=timeout, spool_dir=spool_dir)
        if capture is None:
            # The command exceeded run_bash's own timeout: the backend kills
            # the process tree on interrupt.  Output keeps draining until
            # then and is simply discarded.
            self.interrupt()
            return ShellResult(stdout="", stderr="", exit_code=130, timed_out=True)
        try:
            if not capture.spilled:
                return self._parse_result(
                    capture.buffered_text(), stdout_marker, stderr_marker, exit_marker
                )
            capture.finish_writing()
            return self._parse_spilled_capture(
                capture,
                stdout_marker=stdout_marker,
                stderr_marker=stderr_marker,
                exit_marker=exit_marker,
                artifact_dir=spool_dir,
            )
        finally:
            capture.discard()

    def interrupt(self) -> None:
        self._interrupt_impl()
        self._sync()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._cleanup_impl()
        finally:
            self._drain_queue()

    @staticmethod
    def _clean_section(text: str) -> str:
        return text.lstrip("\r\n").rstrip("\r\n")

    def _normalize_output(self, stdout: str, stderr: str) -> tuple[str, str]:
        return self._clean_section(stdout), self._clean_section(stderr)

    def _parse_result(
        self,
        raw: str,
        stdout_marker: str,
        stderr_marker: str,
        exit_marker: str,
    ) -> ShellResult:
        stdout_idx = raw.find(stdout_marker)
        stderr_idx = raw.find(stderr_marker)
        exit_idx = raw.find(exit_marker)
        if min(stdout_idx, stderr_idx, exit_idx) == -1:
            raise RuntimeError(f"Shell output missing command markers: {raw[-500:]}")

        stdout_text = raw[stdout_idx + len(stdout_marker):stderr_idx]
        stderr_text = raw[stderr_idx + len(stderr_marker):exit_idx]
        exit_tail = raw[exit_idx + len(exit_marker):]
        exit_tail = exit_tail.lstrip(":").strip()

        exit_code = 1
        if exit_tail:
            first_line = exit_tail.splitlines()[0].strip()
            try:
                exit_code = int(first_line)
            except ValueError:
                exit_code = 1

        stdout_text, stderr_text = self._normalize_output(stdout_text, stderr_text)
        combined = combine_output(stdout_text, stderr_text)
        return ShellResult(
            stdout=stdout_text,
            stderr=stderr_text,
            exit_code=exit_code,
            timed_out=False,
            output_bytes=len(combined.encode("utf-8", errors="replace")),
            output_chars=len(combined),
        )

    def _parse_spilled_capture(
        self,
        capture: "_OutputCapture",
        *,
        stdout_marker: str,
        stderr_marker: str,
        exit_marker: str,
        artifact_dir: Path,
    ) -> ShellResult:
        """Stream the raw spool once: strip harness markers, write the clean
        combined artifact, and return a bounded head+tail preview."""
        built = _stream_capture_to_artifact(
            capture.spool_path,
            artifact_dir=artifact_dir,
            stdout_marker=stdout_marker.encode("utf-8"),
            stderr_marker=stderr_marker.encode("utf-8"),
            exit_marker=exit_marker.encode("utf-8"),
            preview_bytes=capture.preview_bytes,
        )
        return ShellResult(
            stdout=built.preview,
            stderr="",
            exit_code=built.exit_code,
            output_spilled=True,
            output_bytes=built.total_bytes,
            output_chars=built.total_chars,
            artifact_path=built.path,
            artifact_sha256=built.sha256,
        )


class _OutputCapture:
    """Bounded-memory drain for one command's raw output.

    Up to ``preview_bytes`` the stream is buffered in full.  Beyond it the
    buffer is flushed to a spool file and all later chunks go straight to
    disk; memory only retains a frozen head and a rolling tail.  The command
    keeps running undisturbed no matter how much it prints.
    """

    def __init__(self, completion_token: bytes, *, preview_bytes: int, spool_dir: Path):
        self._token = bytes(completion_token)
        self.preview_bytes = max(256, int(preview_bytes))
        self._spool_dir = Path(spool_dir)
        self._buffer = bytearray()
        self._head = bytearray()
        self._tail = bytearray()
        self._spool = None
        self.spool_path: Path | None = None
        self.total_bytes = 0
        self.found = False
        self._carry = b""
        self._await_newline = False

    @property
    def spilled(self) -> bool:
        return self._spool is not None

    def feed(self, chunk: bytes) -> None:
        if self.found:
            return
        self._append(chunk)
        if self._await_newline:
            # The unique completion token was already seen; only its line
            # terminator can still arrive in a later chunk.
            if b"\n" in chunk or b"\r" in chunk:
                self.found = True
            return
        window = self._carry + chunk
        idx = window.find(self._token)
        if idx != -1:
            tail = window[idx + len(self._token):]
            if b"\n" in tail or b"\r" in tail:
                self.found = True
            else:
                self._await_newline = True
            return
        # Retain a token-length suffix so a token split across reads (chunks
        # can be as small as one byte) is matched on the next feed.
        self._carry = window[-(len(self._token) - 1):] if len(window) >= len(self._token) else window

    def _append(self, chunk: bytes) -> None:
        self.total_bytes += len(chunk)
        if self._spool is None:
            self._buffer.extend(chunk)
            if self.total_bytes <= self.preview_bytes:
                return
            self._spool_dir.mkdir(parents=True, exist_ok=True)
            self.spool_path = self._spool_dir / f"shell_output_{uuid.uuid4().hex}.raw"
            self._spool = self.spool_path.open("wb")
            self._spool.write(self._buffer)
            head_budget = self.preview_bytes // 2
            self._head = bytearray(self._buffer[:head_budget])
            self._tail = bytearray(
                self._buffer[max(0, len(self._buffer) - (self.preview_bytes - head_budget)):]
            )
            self._buffer = bytearray()
            return
        self._spool.write(chunk)
        tail_budget = self.preview_bytes - self.preview_bytes // 2
        self._tail.extend(chunk)
        excess = len(self._tail) - tail_budget
        if excess > 0:
            del self._tail[:excess]

    def finish_writing(self) -> None:
        """Flush and close the spool writer before the artifact pass reads it.

        No further bytes arrive once the completion marker has been observed.
        """
        if self._spool is not None:
            with contextlib.suppress(Exception):
                self._spool.flush()
                self._spool.close()
            self._spool = None

    def buffered_text(self) -> str:
        return self._buffer.decode("utf-8", errors="replace")

    def discard(self) -> None:
        """Release the spool file; called for every capture, success or not."""
        if self._spool is not None:
            with contextlib.suppress(Exception):
                self._spool.close()
            self._spool = None
        if self.spool_path is not None:
            with contextlib.suppress(OSError):
                self.spool_path.unlink(missing_ok=True)


@dataclass
class _BuiltArtifact:
    path: str
    preview: str
    total_bytes: int
    total_chars: int
    sha256: str
    exit_code: int | None


def _stream_capture_to_artifact(
    raw_path: Path,
    *,
    artifact_dir: Path,
    stdout_marker: bytes,
    stderr_marker: bytes,
    exit_marker: bytes,
    preview_bytes: int,
) -> _BuiltArtifact:
    """One streaming pass over the raw capture.

    Splits stdout/stderr at the marker lines, writes the clean combined
    artifact (same layout as :func:`combine_output`), and accumulates a
    bounded head+tail preview, total size and a sha256, all with bounded
    memory.

    Note the real data path for the persistent backends: the command writes
    into a *guest-side* temp file (POSIX/Docker/PowerShell capture script)
    which is then ``cat``-ed back over the session channel; the host-side
    streaming sink starts at that cat.  This keeps host memory bounded and
    never interrupts the command, but it is not a direct
    child-process-to-host pipe.
    """
    import codecs

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / f"shell_output_{uuid.uuid4().hex}.txt"
    markers = (stdout_marker, stderr_marker, exit_marker)
    state = 0  # 0=before stdout, 1=stdout, 2=stderr, 3=exit line pending/done
    head_budget = max(128, int(preview_bytes) // 2)
    tail_budget = max(128, int(preview_bytes) - head_budget)
    head = bytearray()
    tail = bytearray()
    total_bytes = 0
    hasher = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    total_chars = 0
    stdout_has_content = False
    stderr_header_written = False
    exit_code: int | None = None
    carry = b""
    exit_carry = b""

    out = out_path.open("wb")
    try:
        with Path(raw_path).open("rb") as src:

            def emit(data: bytes, section: int) -> None:
                nonlocal total_bytes, total_chars, stdout_has_content, stderr_header_written
                if not data or section not in (1, 2):
                    return
                if section == 2:
                    if not stderr_header_written:
                        # Match combine_output(): leading whitespace before
                        # the first real stderr byte is dropped.
                        data = data.lstrip()
                        if not data:
                            return
                        header = STDERR_SEPARATOR if stdout_has_content else "--- STDERR ---\n"
                        data = header.encode("utf-8") + data
                        stderr_header_written = True
                elif data.strip():
                    stdout_has_content = True
                out.write(data)
                hasher.update(data)
                total_bytes += len(data)
                total_chars += len(decoder.decode(data))
                room = head_budget - len(head)
                if room > 0:
                    head.extend(data[:room])
                tail.extend(data)
                excess = len(tail) - tail_budget
                if excess > 0:
                    del tail[:excess]

            while True:
                if state == 3:
                    chunk = src.read(65536)
                    if not chunk:
                        break
                    exit_carry += chunk
                    if b"\n" not in exit_carry and b"\r" not in exit_carry:
                        continue
                    code_bytes = exit_carry.lstrip(b":").splitlines()[0].strip()
                    try:
                        exit_code = int(code_bytes)
                    except ValueError:
                        exit_code = 1
                    break

                chunk = src.read(65536)
                if not chunk:
                    emit(carry, state)
                    carry = b""
                    break
                pending = carry + chunk
                pos = 0
                while state < 3:
                    marker = markers[state]
                    idx = pending.find(marker, pos)
                    if idx == -1:
                        break
                    emit(pending[pos:idx], state)
                    pos = idx + len(marker)
                    state += 1
                    if state == 3:
                        remainder = pending[pos:]
                        if b"\n" in remainder or b"\r" in remainder:
                            code_bytes = remainder.lstrip(b":").splitlines()[0].strip()
                            try:
                                exit_code = int(code_bytes)
                            except ValueError:
                                exit_code = 1
                        else:
                            exit_carry = remainder
                if state == 3 and exit_code is not None:
                    break
                if state == 3:
                    carry = b""
                    continue
                hold = len(markers[state])
                if pos > 0:
                    # Marker(s) consumed this chunk: only retain bytes that
                    # came after the last marker, never bytes before it.
                    safe = max(pos, len(pending) - hold)
                else:
                    safe = max(0, len(pending) - hold)
                if safe > pos:
                    emit(pending[pos:safe], state)
                carry = pending[safe:]
    finally:
        total_chars += len(decoder.decode(b"", final=True))
        out.close()

    head_text = head.decode("utf-8", errors="replace").strip()
    tail_text = tail.decode("utf-8", errors="replace").strip()
    omitted = max(0, total_bytes - len(head) - len(tail))
    preview = (
        f"{head_text}\n\n"
        f"...[{omitted} bytes omitted; full output streamed to artifact]...\n\n"
        f"{tail_text}"
    )
    return _BuiltArtifact(
        path=str(out_path),
        preview=preview,
        total_bytes=total_bytes,
        total_chars=total_chars,
        sha256=hasher.hexdigest(),
        exit_code=exit_code,
    )


class _BoundedPipeSink:
    """Bounded-memory drain for one decoded text pipe (stdout or stderr).

    Until ``preview_chars`` characters have arrived the whole stream is
    buffered.  Beyond that the full stream is spooled to a file and memory
    retains only a frozen head and a rolling tail — so draining a multi-GB
    one-shot process never grows host memory.  Used by reader threads around
    one-shot subprocesses (``communicate()`` would buffer it all).
    """

    def __init__(self, *, preview_chars: int, spool_dir: Path):
        self.preview_chars = max(256, int(preview_chars))
        self._spool_dir = Path(spool_dir)
        self._buffer: list[str] = []
        self._buffered_chars = 0
        self._head = ""
        self._tail = ""
        self._spool = None
        self._spilled = False
        self.spool_path: Path | None = None
        self.total_chars = 0

    @property
    def spilled(self) -> bool:
        # Survives finish_writing() closing the handle.
        return self._spilled

    def feed(self, chunk: str) -> None:
        if not chunk:
            return
        self.total_chars += len(chunk)
        if not self._spilled:
            self._buffer.append(chunk)
            self._buffered_chars += len(chunk)
            if self._buffered_chars <= self.preview_chars:
                return
            self._spool_dir.mkdir(parents=True, exist_ok=True)
            self.spool_path = self._spool_dir / f"shell_output_{uuid.uuid4().hex}.raw"
            self._spool = self.spool_path.open("w", encoding="utf-8", newline="")
            self._spool.write("".join(self._buffer))
            self._spilled = True
            head_budget = self.preview_chars // 2
            buffered = "".join(self._buffer)
            self._head = buffered[:head_budget]
            self._tail = buffered[max(0, len(buffered) - (self.preview_chars - head_budget)):]
            self._buffer = []
            self._buffered_chars = 0
            return
        self._spool.write(chunk)
        tail_budget = self.preview_chars - self.preview_chars // 2
        self._tail = (self._tail + chunk)[-tail_budget:]

    def finish_writing(self) -> None:
        if self._spool is not None:
            with contextlib.suppress(Exception):
                self._spool.flush()
                self._spool.close()
            self._spool = None

    def buffered_text(self) -> str:
        return "".join(self._buffer)

    @property
    def head_text(self) -> str:
        return self._head if self.spilled else self.buffered_text()

    @property
    def tail_text(self) -> str:
        return self._tail if self.spilled else self.buffered_text()

    def iter_content(self, block_chars: int = 65536):
        """Yield the full captured content, from memory or spool file."""
        if not self.spilled:
            text = self.buffered_text()
            if text:
                yield text
            return
        with self.spool_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            while True:
                chunk = f.read(block_chars)
                if not chunk:
                    break
                yield chunk

    def discard(self) -> None:
        if self._spool is not None:
            with contextlib.suppress(Exception):
                self._spool.close()
            self._spool = None
        if self.spool_path is not None:
            with contextlib.suppress(OSError):
                self.spool_path.unlink(missing_ok=True)


_TRAILING_WS_RE = re.compile(r"\s+$")


def _stream_sinks_to_artifact(
    stdout_sink: _BoundedPipeSink,
    stderr_sink: _BoundedPipeSink,
    *,
    artifact_dir: str | Path,
    preview_chars: int | None = None,
) -> tuple[str, str, int, int, str]:
    """Stream drained stdout/stderr sinks into one clean combined artifact.

    Layout matches :func:`combine_output` (both streams stripped, stderr under
    a separator).  Returns ``(path, preview, total_bytes, total_chars, sha)``.
    """
    budget = max(256, int(config.SHELL_OUTPUT_PREVIEW_CHARS if preview_chars is None else preview_chars))
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"shell_output_{uuid.uuid4().hex}.txt"
    hasher = hashlib.sha256()
    total_chars = 0

    def _emit_stream(f, chunks, *, at_start: bool) -> int:
        """Write chunks, stripping leading whitespace once and holding back
        trailing whitespace at the boundary. Returns kept character count."""
        pending = ""
        leading = at_start
        written = 0
        for chunk in chunks:
            chunk = pending + chunk
            if leading:
                chunk = chunk.lstrip()
                if chunk:
                    leading = False
            match = _TRAILING_WS_RE.search(chunk)
            if match:
                keep, pending = chunk[: match.start()], chunk[match.start():]
                # Bound pathological all-whitespace streams.
                if len(pending) > 4096:
                    keep += pending[:-4096]
                    pending = pending[-4096:]
            else:
                keep, pending = chunk, ""
            if keep:
                f.write(keep)
                hasher.update(keep.encode("utf-8", errors="replace"))
                written += len(keep)
        return written

    with path.open("w", encoding="utf-8", newline="") as f:
        total_chars = _emit_stream(f, stdout_sink.iter_content(), at_start=True)
        saw_out = total_chars > 0
        if stderr_sink.total_chars:
            # Decide whether stderr has non-whitespace content without
            # materialising it: head/tail text is already in memory.
            stderr_has_content = bool(
                (stderr_sink.head_text + stderr_sink.tail_text).strip()
            )
            if stderr_has_content:
                sep = STDERR_SEPARATOR if saw_out else "--- STDERR ---\n"
                f.write(sep)
                hasher.update(sep.encode("utf-8"))
                total_chars += len(sep) + _emit_stream(
                    f, stderr_sink.iter_content(), at_start=True
                )
    data_total = path.stat().st_size

    head_budget = budget // 2
    tail_budget = budget - head_budget
    # Whether the stderr section was emitted, derived from sink content.
    has_stderr_section = bool(
        stderr_sink.total_chars
        and (stderr_sink.head_text + stderr_sink.tail_text).strip()
    )
    if has_stderr_section and saw_out:
        head = stdout_sink.head_text.strip()[:head_budget]
        tail = stderr_sink.tail_text.strip()[-tail_budget:]
    else:
        head = stdout_sink.head_text.strip()[:head_budget]
        tail = stdout_sink.tail_text.strip()
        tail = tail[-tail_budget:] if len(tail) > budget else ""
    omitted = max(0, data_total - len(head.encode("utf-8")) - len(tail.encode("utf-8")))
    preview = (
        f"{head}\n\n"
        f"...[{omitted} bytes omitted; full output streamed to artifact]...\n\n"
        f"{tail}"
    )
    return str(path), preview, data_total, total_chars, hasher.hexdigest()


def sandbox_mode() -> str:
    requested = (config.SANDBOX_MODE or "host").strip().lower()
    if requested in {"host", "docker"}:
        return requested
    raise ValueError("HARNESS_SANDBOX_MODE must be host or docker")


def docker_cli_path() -> str | None:
    return shutil.which("docker")


def docker_shell_hint() -> str:
    if sandbox_mode() != "docker":
        return "host shell"
    path = docker_cli_path()
    if not path:
        return "Docker CLI not found"
    user_hint = ""
    user_arg = _docker_user_arg()
    if user_arg:
        user_hint = f", user={user_arg[1]}"
    return f"Docker sandbox ({config.DOCKER_IMAGE}, network={config.DOCKER_NETWORK}{user_hint})"


def docker_info_check() -> tuple[bool, str]:
    """Check Docker daemon connectivity for doctor diagnostics.

    Returns (ok, detail) tuple.
    """
    docker = docker_cli_path()
    if not docker:
        return False, "Docker CLI not found"
    try:
        completed = subprocess.run(
            [docker, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
            check=False,
        )
    except FileNotFoundError:
        return False, "Docker CLI not found"
    except subprocess.TimeoutExpired:
        return False, "Docker daemon unreachable (timeout)"
    except Exception as exc:
        return False, f"Docker connectivity check failed: {exc}"
    if completed.returncode == 0:
        version = completed.stdout.strip()
        if version:
            return True, f"Docker {version}"
        return True, "Docker available"
    detail = (completed.stderr or completed.stdout or "").strip()
    if detail:
        return False, f"Docker daemon unreachable: {detail[:120]}"
    return False, f"Docker daemon unreachable (exit code {completed.returncode})"


def _docker_user_arg() -> list[str]:
    explicit = (config.DOCKER_USER or "").strip()
    if explicit:
        return ["--user", explicit]
    # Auto-detect: on POSIX, avoid container root whenever the host UID is non-root.
    if os.name == "posix":
        try:
            uid = os.getuid()
            gid = os.getgid()
        except AttributeError:
            return []
        if uid != 0:
            return ["--user", f"{uid}:{gid}"]
    # Windows / Docker Desktop: do not force UID mapping by default
    return []


def _docker_run_args(container_name: str, host_cwd: str) -> list[str]:
    image = (config.DOCKER_IMAGE or "").strip()
    if not image:
        raise ValueError("HARNESS_DOCKER_IMAGE must not be empty")
    network = (config.DOCKER_NETWORK or "none").strip().lower()
    if network not in {"none", "bridge"}:
        raise ValueError("HARNESS_DOCKER_NETWORK must be none or bridge")
    args = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--network",
        network,
        "--security-opt=no-new-privileges",
        "-v",
        f"{host_cwd}:/workspace",
        "-w",
        "/workspace",
    ]
    args.extend(_docker_resource_args())
    args.extend(_docker_user_arg())
    args.extend([image, "sleep", "infinity"])
    return args


def _docker_resource_args() -> list[str]:
    """Container-level memory/CPU caps; zero means leave the Docker default."""
    args: list[str] = []
    memory_mb = int(getattr(config, "DOCKER_MEMORY_MB", 0) or 0)
    if memory_mb > 0:
        args.extend(["--memory", f"{memory_mb}m"])
    cpus = float(getattr(config, "DOCKER_CPUS", 0) or 0)
    if cpus > 0:
        args.extend(["--cpus", f"{cpus:g}"])
    return args


def _docker_exec_args(container_name: str) -> list[str]:
    return [
        "docker",
        "exec",
        "-i",
        "-w",
        "/workspace",
        container_name,
        "bash",
        "--noprofile",
        "--norc",
        "-s",
    ]


def _run_docker_checked(args: list[str], action: str) -> None:
    try:
        completed = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Docker sandbox requested but docker CLI was not found. "
            "Install Docker or set HARNESS_SANDBOX_MODE=host."
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        if detail:
            raise RuntimeError(
                f"Failed to {action}: {detail}\n"
                f"Check that Docker daemon is running and the image is available."
            )
        raise RuntimeError(
            f"Failed to {action}: docker exited with code {completed.returncode}. "
            f"Check that Docker daemon is running."
        )


def windows_shell_path() -> str | None:
    requested = (config.WINDOWS_SHELL or "pwsh").strip().lower()
    if requested == "pwsh":
        return shutil.which("pwsh")
    if requested == "wsl":
        return shutil.which("wsl.exe")
    raise ValueError("HARNESS_WINDOWS_SHELL must be pwsh or wsl")


def windows_shell_kind() -> str:
    requested = (config.WINDOWS_SHELL or "pwsh").strip().lower()
    if requested in {"pwsh", "wsl"}:
        return requested
    raise ValueError("HARNESS_WINDOWS_SHELL must be pwsh or wsl")


def windows_shell_hint() -> str:
    if os.name != "nt":
        return "POSIX shell"
    kind = windows_shell_kind()
    if kind == "pwsh":
        return "PowerShell 7 (pwsh)"
    if kind == "wsl":
        return "WSL Bash"
    return "no Windows shell"


def validate_shell_configuration() -> None:
    """Validate the explicitly selected host shell without choosing a fallback."""
    if sandbox_mode() == "docker" or os.name != "nt":
        return
    kind = windows_shell_kind()
    if windows_shell_path() is None:
        executable = "pwsh.exe" if kind == "pwsh" else "wsl.exe"
        raise RuntimeError(
            f"HARNESS_WINDOWS_SHELL={kind} selected, but {executable} was not found. "
            "Install the selected shell or choose the other explicit backend."
        )


def _make_windows_shell_backend(cwd: str) -> _BaseShellBackend:
    validate_shell_configuration()
    path = windows_shell_path()
    assert path is not None
    kind = windows_shell_kind()
    if kind == "pwsh":
        return _PowerShellBackend(cwd, executable=path)
    return _WslShellBackend(cwd, executable=path)


class _DockerShellBackend(_BaseShellBackend):
    def __init__(self, cwd: str):
        self.host_cwd = str(Path(cwd).resolve())
        self.container_name = f"hca-shell-{uuid.uuid4().hex[:12]}"
        self._container_started = False
        super().__init__("/workspace")

    def _start(self) -> None:
        if not self._container_started:
            _run_docker_checked(
                _docker_run_args(self.container_name, self.host_cwd),
                "start Docker sandbox",
            )
            self._container_started = True
        self._process = subprocess.Popen(
            _docker_exec_args(self.container_name),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("Failed to create Docker sandbox shell")

    def _reader_loop(self) -> None:
        assert self._process.stdout is not None
        while not self._closed:
            chunk = self._process.stdout.read(1)
            if not chunk:
                return
            self._queue.put(chunk)

    def _send(self, script: str) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(script.encode("utf-8"))
        self._process.stdin.flush()

    def _interrupt_impl(self) -> None:
        self._stop_exec_process(timeout=2)
        self._start()
        self._start_reader()

    def _cleanup_impl(self) -> None:
        self._stop_exec_process(timeout=5)
        if self._container_started:
            with contextlib.suppress(Exception):
                subprocess.run(
                    ["docker", "rm", "-f", self.container_name],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    check=False,
                    timeout=10,
                )
            self._container_started = False

    def _stop_exec_process(self, *, timeout: int) -> None:
        if getattr(self, "_process", None) is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if getattr(self, "_process", None) is not None:
            if self._process.stdin is not None:
                self._process.stdin.close()
            if self._process.stdout is not None:
                self._process.stdout.close()
        # Give the reader thread a moment to exit gracefully
        reader = getattr(self, "_reader", None)
        if reader is not None and reader.is_alive():
            reader.join(timeout=2)

    def _build_sync_command(self, marker: str) -> str:
        return f"printf '%s\\n' '{marker}'\n"

    def _build_script(self, command: str, marker: str) -> str:
        stdout_marker = f"__CODEX_STDOUT_{marker}__"
        stderr_marker = f"__CODEX_STDERR_{marker}__"
        exit_marker = f"__CODEX_EXIT_{marker}__"
        return _posix_capture_script(
            command,
            stdout_marker,
            stderr_marker,
            exit_marker,
            prefix="__codex",
        )


class _WslShellBackend(_BaseShellBackend):
    """Persistent Bash session launched through the explicitly selected WSL backend."""

    _SYNC_TIMEOUT_SECONDS = 20

    def __init__(self, cwd: str, *, executable: str):
        self.executable = executable
        super().__init__(cwd)

    def _start(self) -> None:
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self._process = subprocess.Popen(
            [
                self.executable,
                "--cd",
                self.cwd,
                "--exec",
                "bash",
                "--noprofile",
                "--norc",
                "-s",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("Failed to create persistent WSL shell")

    def _reader_loop(self) -> None:
        assert self._process.stdout is not None
        while not self._closed:
            chunk = self._process.stdout.read(1)
            if not chunk:
                return
            self._queue.put(chunk)

    def _send(self, script: str) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(script.encode("utf-8"))
        self._process.stdin.flush()

    def _interrupt_impl(self) -> None:
        self._stop_process(timeout=2)
        self._start()
        self._start_reader()

    def _cleanup_impl(self) -> None:
        self._stop_process(timeout=5)

    def _stop_process(self, *, timeout: int) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._process.stdin is not None:
            self._process.stdin.close()
        if self._process.stdout is not None:
            self._process.stdout.close()

    def _build_sync_command(self, marker: str) -> str:
        return f"printf '%s\\n' '{marker}'\n"

    def _build_script(self, command: str, marker: str) -> str:
        stdout_marker = f"__CODEX_STDOUT_{marker}__"
        stderr_marker = f"__CODEX_STDERR_{marker}__"
        exit_marker = f"__CODEX_EXIT_{marker}__"
        return _posix_capture_script(
            command,
            stdout_marker,
            stderr_marker,
            exit_marker,
            prefix="__hca",
        )


# --- POSIX script construction shared by the Docker/WSL/native backends ---


def _posix_ulimit_preamble() -> str:
    """Resource-limit lines applied inside the per-command subshell.

    Empty by default: address-space caps (-v) can false-kill JVM/Node
    workloads, so every limit is opt-in via configuration.
    """
    lines = []
    if config.SHELL_CPU_SECONDS > 0:
        lines.append(f"ulimit -t {int(config.SHELL_CPU_SECONDS)} 2>/dev/null || true")
    if config.SHELL_FSIZE_BLOCKS > 0:
        lines.append(f"ulimit -f {int(config.SHELL_FSIZE_BLOCKS)} 2>/dev/null || true")
    if config.SHELL_MEMORY_KB > 0:
        lines.append(f"ulimit -v {int(config.SHELL_MEMORY_KB)} 2>/dev/null || true")
    return "".join(f"{line}\n" for line in lines)


def _posix_capture_script(
    command: str,
    stdout_marker: str,
    stderr_marker: str,
    exit_marker: str,
    *,
    prefix: str,
) -> str:
    preamble = _posix_ulimit_preamble()
    # With resource limits configured the block runs in a subshell so the
    # ulimit cannot leak onto the persistent shell; the default brace form
    # preserves cwd/env side effects of the command as before.  Output is
    # always replayed in full: the host-side reader streams it to a bounded
    # sink/artifact, so there is no shell-side byte ceiling.
    block_open, block_close = ("(\n", ")\n") if preamble else ("{\n", "}\n")
    inner = preamble + f"{command}\n"
    return (
        f"{prefix}_out=$(mktemp)\n"
        f"{prefix}_err=$(mktemp)\n"
        f"{block_open}"
        f"{inner}"
        f"{block_close}"
        f" 1>\"${prefix}_out\" 2>\"${prefix}_err\"\n"
        f"{prefix}_status=$?\n"
        f"printf '%s\\n' '{stdout_marker}'\n"
        f"cat \"${prefix}_out\"\n"
        f"printf '\\n%s\\n' '{stderr_marker}'\n"
        f"cat \"${prefix}_err\"\n"
        f"printf '\\n%s:%s\\n' '{exit_marker}' \"${prefix}_status\"\n"
        f"rm -f \"${prefix}_out\" \"${prefix}_err\"\n"
    )


class _PowerShellBackend(_BaseShellBackend):
    def __init__(self, cwd: str, *, executable: str):
        self.executable = executable
        super().__init__(cwd)

    def _start(self) -> None:
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self._process = subprocess.Popen(
            [self.executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", _powershell_host_command()],
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("Failed to create persistent Windows shell")

    def _reader_loop(self) -> None:
        assert self._process.stdout is not None
        while not self._closed:
            chunk = self._process.stdout.read(1)
            if not chunk:
                return
            self._queue.put(chunk)

    def _send(self, script: str) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(script.encode("utf-8"))
        self._process.stdin.flush()

    def _interrupt_impl(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._process.stdin is not None:
            self._process.stdin.close()
        if self._process.stdout is not None:
            self._process.stdout.close()
        self._start()
        self._start_reader()

    def _cleanup_impl(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._process.stdin is not None:
            self._process.stdin.close()
        if self._process.stdout is not None:
            self._process.stdout.close()

    def _build_sync_command(self, marker: str) -> str:
        return f"Write-Output '{marker}'\n"

    def _build_script(self, command: str, marker: str) -> str:
        stdout_marker = f"__CODEX_STDOUT_{marker}__"
        stderr_marker = f"__CODEX_STDERR_{marker}__"
        exit_marker = f"__CODEX_EXIT_{marker}__"
        encoded_command = base64.b64encode(command.encode("utf-8")).decode("ascii")
        return (
            f"$__hca_out = Join-Path ([System.IO.Path]::GetTempPath()) '{marker}.out'; "
            f"$__hca_err = Join-Path ([System.IO.Path]::GetTempPath()) '{marker}.err'; "
            "function __hca_Show($p) { "
            "if (Test-Path -LiteralPath $p) { Get-Content -LiteralPath $p -Raw -Encoding utf8 -ErrorAction SilentlyContinue } }; "
            f"$__hca_command = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String('{encoded_command}')); "
            "$global:LASTEXITCODE = $null; "
            "$global:__hca_last_success = $null; "
            "try { "
            "$__hca_command += [Environment]::NewLine + '$global:__hca_last_success = $?'; "
            "$__hca_script = [scriptblock]::Create($__hca_command); "
            "$__hca_error_count = $Error.Count; "
            "& $__hca_script 1> $__hca_out 2> $__hca_err; "
            "$__hca_pipeline_ok = [bool]$global:__hca_last_success; "
            "$__hca_native_status = $global:LASTEXITCODE; "
            "if ($__hca_pipeline_ok) { $__hca_status = 0 } "
            "elseif ($__hca_native_status -is [int]) { $__hca_status = $__hca_native_status } "
            "elseif ($Error.Count -eq $__hca_error_count) { $__hca_status = 0 } "
            "else { $__hca_status = 1 } "
            "} catch { $_ | Out-String | Set-Content -LiteralPath $__hca_err -Encoding utf8NoBOM; $__hca_status = 1 }; "
            f"Write-Output '{stdout_marker}'; "
            " __hca_Show $__hca_out; "
            f"Write-Output '{stderr_marker}'; "
            " __hca_Show $__hca_err; "
            f"Write-Output ('{exit_marker}:' + $__hca_status); "
            "Remove-Item -LiteralPath $__hca_out, $__hca_err -Force -ErrorAction SilentlyContinue\n"
        )


def _powershell_host_command() -> str:
    return (
        "[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)\n"
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)\n"
        "$OutputEncoding = [System.Text.UTF8Encoding]::new($false)\n"
        "chcp 65001 > $null\n"
        "$ProgressPreference = 'SilentlyContinue'\n"
        "function global:prompt { '' }\n"
        "while (($__hca_line = [Console]::In.ReadLine()) -ne $null) {\n"
        "  try { Invoke-Expression $__hca_line }\n"
        "  catch { Write-Error $_ }\n"
        "}\n"
    )


class _PosixShellBackend(_BaseShellBackend):
    def _start(self) -> None:
        import pty
        import termios

        master_fd, slave_fd = pty.openpty()
        attrs = termios.tcgetattr(slave_fd)
        attrs[3] &= ~termios.ECHO
        termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)

        env = os.environ.copy()
        env["TERM"] = "dumb"
        env["PS1"] = ""
        env["PROMPT_COMMAND"] = ""

        self._master_fd = master_fd
        self._slave_fd = slave_fd
        self._process = subprocess.Popen(
            ["/bin/bash", "--noprofile", "--norc", "-s"],
            cwd=self.cwd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
            start_new_session=True,
            close_fds=True,
        )
        os.close(slave_fd)
        self._slave_fd = None

    def _reader_loop(self) -> None:
        while not self._closed:
            try:
                chunk = os.read(self._master_fd, 1024)
            except OSError:
                return
            if not chunk:
                return
            self._queue.put(chunk)

    def _send(self, script: str) -> None:
        os.write(self._master_fd, script.encode("utf-8"))

    def _interrupt_impl(self) -> None:
        os.write(self._master_fd, b"\x03")

    def _cleanup_impl(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        try:
            os.close(self._master_fd)
        except OSError:
            pass

    def _build_sync_command(self, marker: str) -> str:
        return f"printf '%s\\n' '{marker}'\n"

    def _build_script(self, command: str, marker: str) -> str:
        stdout_marker = f"__CODEX_STDOUT_{marker}__"
        stderr_marker = f"__CODEX_STDERR_{marker}__"
        exit_marker = f"__CODEX_EXIT_{marker}__"
        return _posix_capture_script(
            command,
            stdout_marker,
            stderr_marker,
            exit_marker,
            prefix="__codex",
        )

