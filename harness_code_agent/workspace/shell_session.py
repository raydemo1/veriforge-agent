from __future__ import annotations

import base64
import contextlib
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .. import config

log = logging.getLogger("harness")


@dataclass
class ShellResult:
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool = False
    output_truncated: bool = False
    output_bytes: int = 0


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

    def run(self, command: str, timeout: int = 300) -> ShellResult:
        return self._backend.run(command, timeout)

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

    def _read_until(
        self,
        token: str,
        timeout: float,
        *,
        max_bytes: int = 0,
    ) -> tuple[str | None, bool]:
        """Read until ``token`` (followed by a newline) or the deadline.

        Returns ``(text, truncated)``.  ``text`` is None on timeout.  When
        ``max_bytes`` > 0 and the capture buffer fills before the marker
        arrives, reading stops immediately with the captured head and
        ``truncated=True`` so the caller can interrupt the still-running
        command instead of buffering unbounded output in process memory.
        """
        deadline = time.time() + timeout
        buffer = bytearray()
        token_bytes = token.encode("utf-8")

        while time.time() < deadline:
            try:
                chunk = self._queue.get(timeout=min(0.1, max(deadline - time.time(), 0.01)))
            except queue.Empty:
                if self._process.poll() is not None:
                    break
                continue
            buffer.extend(chunk)
            if max_bytes and len(buffer) >= max_bytes:
                return buffer.decode("utf-8", errors="replace"), True
            token_idx = buffer.find(token_bytes)
            if token_idx != -1:
                tail = buffer[token_idx + len(token_bytes):]
                if b"\n" in tail or b"\r" in tail:
                    return buffer.decode("utf-8", errors="replace"), False

        return None, False

    def _sync(self) -> None:
        marker = f"__CODEX_SYNC_{uuid.uuid4().hex}__"
        self._drain_queue()
        self._send(self._build_sync_command(marker))
        synced, _ = self._read_until(marker, timeout=self._SYNC_TIMEOUT_SECONDS)
        if synced is None:
            raise RuntimeError("Shell failed to become ready")

    def run(self, command: str, timeout: int = 300) -> ShellResult:
        marker = uuid.uuid4().hex
        stdout_marker = f"__CODEX_STDOUT_{marker}__"
        stderr_marker = f"__CODEX_STDERR_{marker}__"
        exit_marker = f"__CODEX_EXIT_{marker}__"

        self._drain_queue()
        self._send(self._build_script(command, marker))
        max_bytes = max(0, int(config.SHELL_MAX_OUTPUT_BYTES))
        raw, truncated = self._read_until(exit_marker, timeout=timeout, max_bytes=max_bytes)
        if truncated:
            # The command is still producing output past the capture ceiling.
            # Stop it immediately; parse the captured head on a best-effort
            # basis so the model still sees the beginning of the output.
            with contextlib.suppress(Exception):
                self.interrupt()
            return self._parse_truncated_result(raw or "", stdout_marker, stderr_marker)
        if raw is None:
            self.interrupt()
            return ShellResult(stdout="", stderr="", exit_code=130, timed_out=True)

        return self._parse_result(raw, stdout_marker, stderr_marker, exit_marker)

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
        return ShellResult(
            stdout=stdout_text,
            stderr=stderr_text,
            exit_code=exit_code,
            timed_out=False,
            output_bytes=len(stdout_text.encode("utf-8", errors="replace"))
            + len(stderr_text.encode("utf-8", errors="replace")),
        )

    def _parse_truncated_result(
        self,
        raw: str,
        stdout_marker: str,
        stderr_marker: str,
    ) -> ShellResult:
        """Best-effort head extraction when capture hit the byte ceiling."""
        stdout_text = ""
        stderr_text = ""
        stdout_idx = raw.find(stdout_marker)
        if stdout_idx != -1:
            rest = raw[stdout_idx + len(stdout_marker):]
            stderr_idx = rest.find(stderr_marker)
            if stderr_idx != -1:
                stdout_text = rest[:stderr_idx]
                stderr_text = rest[stderr_idx + len(stderr_marker):]
            else:
                stdout_text = rest
        else:
            stdout_text = raw
        stdout_text, stderr_text = self._normalize_output(stdout_text, stderr_text)
        return ShellResult(
            stdout=stdout_text,
            stderr=stderr_text,
            exit_code=None,
            output_truncated=True,
            output_bytes=len(raw.encode("utf-8", errors="replace")),
        )


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
    max_bytes = max(0, int(config.SHELL_MAX_OUTPUT_BYTES))
    preamble = _posix_ulimit_preamble()
    # With resource limits configured the block runs in a subshell so the
    # ulimit cannot leak onto the persistent shell; the default brace form
    # preserves cwd/env side effects of the command as before.
    block_open, block_close = ("(\n", ")\n") if preamble else ("{\n", "}\n")
    inner = preamble + f"{command}\n"
    return (
        f"{prefix}_out=$(mktemp)\n"
        f"{prefix}_err=$(mktemp)\n"
        f"{prefix}_max={max_bytes}\n"
        f"{prefix}_show() {{ if [ \"${prefix}_max\" -gt 0 ]; then "
        f"head -c \"${prefix}_max\" \"$1\"; else cat \"$1\"; fi; }}\n"
        f"{block_open}"
        f"{inner}"
        f"{block_close}"
        f" 1>\"${prefix}_out\" 2>\"${prefix}_err\"\n"
        f"{prefix}_status=$?\n"
        f"printf '%s\\n' '{stdout_marker}'\n"
        f"\"${prefix}_show\" \"${prefix}_out\"\n"
        f"printf '\\n%s\\n' '{stderr_marker}'\n"
        f"\"${prefix}_show\" \"${prefix}_err\"\n"
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
        max_chars = max(0, int(config.SHELL_MAX_OUTPUT_BYTES))
        return (
            f"$__hca_out = Join-Path ([System.IO.Path]::GetTempPath()) '{marker}.out'; "
            f"$__hca_err = Join-Path ([System.IO.Path]::GetTempPath()) '{marker}.err'; "
            f"$__hca_max = {max_chars}; "
            "function __hca_Show($p) { "
            "$c = if (Test-Path -LiteralPath $p) { Get-Content -LiteralPath $p -Raw -Encoding utf8 -ErrorAction SilentlyContinue } else { '' }; "
            "if ($__hca_max -gt 0 -and $c.Length -gt $__hca_max) { $c = $c.Substring(0, $__hca_max) }; "
            "$c }; "
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
