"""Static shell syntax layer: one parse per command, no execution.

Two backends, dispatched by the shell that will actually run the command:

* PowerShell 7 — the native ``System.Management.Automation.Language.Parser``
  inside a reused helper process (``powershell_parser.ps1``). The command text
  travels as JSON data; it is never spliced into ``-Command``.
* Bash — the ``tree-sitter`` / ``tree-sitter-bash`` grammar.

Both produce the same minimal fact set consumed by
:mod:`runtime.shell_classification`:

``commands``      the commands that may run, with static argument text
``redirections``  output redirections that will create/modify a file
``dynamic``       some value is only known at run time
``complete``      every operation that may run was reliably extracted
``error``         why the result is not complete

``complete`` is deliberately conservative: it is true only when the executed
command set and their static arguments are all visible.  A dynamic *argument*
(``rm -rf $DIR``) keeps ``complete`` true — the delete operation is known — but
sets ``dynamic``; a dynamic *command* (``& $cmd``) or an unparsed construct
clears ``complete``. An unavailable parser or exceeded parsing budget also sets
``analysis_failed``, which prevents execution.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("harness")

SHELL_POWERSHELL = "powershell"
SHELL_BASH = "bash"

PARSER_SCRIPT = Path(__file__).with_name("powershell_parser.ps1")

#: Command text larger than this is not parsed at all (fail closed).
MAX_COMMAND_CHARS = 32_000
#: Parser process start-up budget.
STARTUP_TIMEOUT_SECONDS = 10.0
#: One request/response round trip budget.
REQUEST_TIMEOUT_SECONDS = 5.0
#: A single protocol line larger than this is rejected.
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
#: Stderr retained for diagnostics after a parser failure.
_MAX_DIAGNOSTIC_LINES = 20

#: Wrapper commands whose own name hides the real executable.
_WRAPPERS = frozenset({"sudo", "command", "nohup", "time", "nice", "stdbuf", "ionice", "env"})
_WRAPPER_FLAGS_WITH_VALUE = frozenset({"-u", "-g", "--user", "--group", "-n", "-p"})


@dataclass(frozen=True)
class ParsedCommand:
    executable: str
    args: tuple[str, ...] = ()
    wrappers: tuple[str, ...] = ()
    dynamic: bool = False
    #: One flag per argument: True when the parser saw literal text, False when
    #: the value is resolved only at run time.  ``args`` keeps the same text for
    #: readability; the flags are the authority, not markers inside the string.
    arg_literals: tuple[bool, ...] = ()


@dataclass(frozen=True)
class ParsedRedirection:
    target: str
    append: bool = False
    literal: bool = True


@dataclass(frozen=True)
class ShellParseResult:
    shell: str
    commands: tuple[ParsedCommand, ...] = ()
    redirections: tuple[ParsedRedirection, ...] = ()
    dynamic: bool = False
    complete: bool = True
    error: str | None = None
    analysis_failed: bool = False

    @property
    def failed(self) -> bool:
        return not self.complete


def current_shell() -> str:
    """The shell ``run_bash`` will hand this command to on this host."""
    return SHELL_POWERSHELL if os.name == "nt" else SHELL_BASH


def parse_shell_command(command: str, shell: str | None = None) -> ShellParseResult:
    """Parse ``command`` for ``shell`` (default: this host's shell).

    Cached: the same command string is parsed once per process so every
    consumer (policy, planner, middleware, allowlist) reads one fact set.
    """
    text = str(command or "")
    kind = shell or current_shell()
    result = _parse_cached(text, kind)
    if result.analysis_failed:
        # An infrastructure outage is not a property of this command: do not
        # memoize it, so a restarted parser can answer the next call.
        _parse_cached.cache_clear()
    return result


@lru_cache(maxsize=512)
def _parse_cached(text: str, shell: str) -> ShellParseResult:
    if not text.strip():
        return ShellParseResult(shell)
    if len(text) > MAX_COMMAND_CHARS:
        return ShellParseResult(
            shell, complete=False, error="command text exceeds the parser limit",
            analysis_failed=True,
        )
    try:
        if shell == SHELL_POWERSHELL:
            return _parse_powershell(text)
        return _parse_bash(text)
    except Exception as exc:  # parser outage must never become a silent allow
        log.debug("shell parser failed: %s", type(exc).__name__)
        return ShellParseResult(
            shell, complete=False, error=f"{type(exc).__name__}: {exc}", analysis_failed=True
        )


def reset_cache() -> None:
    """Drop cached parse results (tests, and after a parser restart)."""
    _parse_cached.cache_clear()


def shutdown_shell_parser() -> None:
    """Stop the helper process if one is running."""
    global _powershell_client
    client, _powershell_client = _powershell_client, None
    if client is not None:
        client.close()


atexit.register(shutdown_shell_parser)


# ---------------------------------------------------------------------------
# PowerShell 7 backend
# ---------------------------------------------------------------------------

_client_lock = threading.Lock()
_powershell_client: _PowerShellParser | None = None


@lru_cache(maxsize=1)
def _powershell_executable() -> str | None:
    # `shutil.which` stats every PATH entry for every PATHEXT suffix (~0.25s on
    # Windows); the resolved shell cannot change within a process.
    return shutil.which("pwsh") or shutil.which("pwsh.exe")


def _parse_powershell(text: str) -> ShellParseResult:
    global _powershell_client
    if _powershell_executable() is None:
        return ShellParseResult(
            SHELL_POWERSHELL, complete=False, error="pwsh is not available", analysis_failed=True
        )
    with _client_lock:
        if _powershell_client is None:
            _powershell_client = _PowerShellParser()
        client = _powershell_client
    payload, error = client.parse(text)
    if payload is None:
        return ShellParseResult(SHELL_POWERSHELL, complete=False, error=error, analysis_failed=True)
    return _result_from_payload(SHELL_POWERSHELL, payload)


def _result_from_payload(shell: str, payload: dict) -> ShellParseResult:
    commands = tuple(
        _normalize_command(
            str(item.get("executable") or ""),
            [str(arg) for arg in item.get("args") or ()],
            [str(wrapper) for wrapper in item.get("wrappers") or ()],
            bool(item.get("dynamic")),
            [bool(value) for value in item.get("arg_literals") or ()],
        )
        for item in payload.get("commands") or ()
        if isinstance(item, dict)
    )
    commands = tuple(command for command in commands if command is not None)
    redirections = tuple(
        ParsedRedirection(
            str(item.get("target") or ""),
            bool(item.get("append")),
            bool(item.get("literal", True)),
        )
        for item in payload.get("redirections") or ()
        if isinstance(item, dict) and str(item.get("target") or "").strip()
    )
    error = payload.get("error")
    return ShellParseResult(
        shell=shell,
        commands=commands,
        redirections=redirections,
        dynamic=bool(payload.get("dynamic")),
        complete=bool(payload.get("complete")),
        error=str(error) if error else None,
        analysis_failed=payload["analysis_failed"],
    )


class _PowerShellParser:
    """One reused ``pwsh`` process speaking JSON Lines.

    Requests are serialized by a lock (one process, one request at a time).  A
    timeout or a protocol violation kills the process so the next call starts a
    fresh one; the *current* call fails closed instead of being retried.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._stderr: queue.Queue[str] = queue.Queue(maxsize=_MAX_DIAGNOSTIC_LINES)
        self._counter = 0
        self._closed = False

    def parse(self, text: str) -> tuple[dict | None, str | None]:
        if self._closed:
            return None, "shell parser is closed"
        with self._lock:
            process = self._ensure_process()
            if process is None:
                return None, "shell parser unavailable"
            self._counter += 1
            request_id = self._counter
            line = json.dumps({"id": request_id, "command": text})
            timer = threading.Timer(REQUEST_TIMEOUT_SECONDS, self._kill, args=(process,))
            try:
                timer.start()
                try:
                    process.stdin.write(line + "\n")
                    process.stdin.flush()
                    response = process.stdout.readline()
                except (OSError, ValueError):
                    self._discard(process)
                    return None, "shell parser stopped"
            finally:
                timer.cancel()
            if not response:
                self._discard(process)
                return None, self._failure_reason("shell parser produced no response")
            if len(response) > MAX_RESPONSE_BYTES:
                self._discard(process)
                return None, "shell parser response too large"
            try:
                payload = json.loads(response)
            except ValueError:
                self._discard(process)
                return None, "shell parser protocol error"
            if (
                not isinstance(payload, dict)
                or payload.get("id") != request_id
                or not all(isinstance(payload.get(key), bool)
                           for key in ("complete", "dynamic", "analysis_failed"))
                or not all(isinstance(payload.get(key), list)
                           and all(isinstance(item, dict) for item in payload[key])
                           for key in ("commands", "redirections"))
            ):
                self._discard(process)
                return None, "shell parser protocol error"
            return payload, None

    # -- process lifecycle ------------------------------------------------

    def _ensure_process(self) -> subprocess.Popen | None:
        if self._process is not None and self._process.poll() is None:
            return self._process
        self._discard(self._process)
        executable = _powershell_executable()
        if executable is None:
            return None
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                [
                    executable,
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(PARSER_SCRIPT),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
        except OSError:
            return None
        self._process = process
        threading.Thread(target=self._drain_stderr, args=(process,), daemon=True).start()
        if not self._await_ready(process):
            self._discard(process)
            return None
        return process

    def _await_ready(self, process: subprocess.Popen) -> bool:
        """Round-trip an empty command so a slow cold start is paid up front."""
        timer = threading.Timer(STARTUP_TIMEOUT_SECONDS, self._kill, args=(process,))
        try:
            timer.start()
            try:
                process.stdin.write(json.dumps({"id": 0, "command": ""}) + "\n")
                process.stdin.flush()
                line = process.stdout.readline()
            except (OSError, ValueError):
                return False
        finally:
            timer.cancel()
        if not line:
            return False
        try:
            payload = json.loads(line)
        except ValueError:
            return False
        return isinstance(payload, dict) and payload.get("id") == 0

    def _drain_stderr(self, process: subprocess.Popen) -> None:
        stream = process.stderr
        if stream is None:
            return
        try:
            for line in stream:
                text = line.strip()
                if not text:
                    continue
                if self._stderr.full():
                    self._stderr.get_nowait()
                self._stderr.put_nowait(text)
        except (OSError, ValueError):
            pass

    def _failure_reason(self, fallback: str) -> str:
        details = []
        while not self._stderr.empty():
            details.append(self._stderr.get_nowait())
        if details:
            return f"{fallback}: {details[-1][:200]}"
        return fallback

    def _kill(self, process: subprocess.Popen) -> None:
        try:
            process.kill()
        except (OSError, ValueError):
            pass

    def _discard(self, process: subprocess.Popen | None) -> None:
        if process is None:
            return
        if self._process is process:
            self._process = None
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except (OSError, ValueError):
                pass
        if process.poll() is None:
            self._kill(process)
        try:
            process.wait(timeout=2)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._discard(self._process)


# ---------------------------------------------------------------------------
# Bash backend (tree-sitter)
# ---------------------------------------------------------------------------

_bash_parser = None
_bash_parser_lock = threading.Lock()
_bash_parser_failed = False
#: tree-sitter's Parser object is not safe for concurrent parse() calls; the
#: cached instance is shared, so every parse is serialized.
_bash_parse_lock = threading.Lock()

_DYNAMIC_MARKERS = ("$", "`")
_GLOB_MARKERS = ("*", "?", "[")

_BASH_WRITE_REDIRECTS = frozenset({">", ">>", ">|", "&>", "&>>", ">&"})


def _bash_language():
    global _bash_parser, _bash_parser_failed
    if _bash_parser is not None or _bash_parser_failed:
        return _bash_parser
    with _bash_parser_lock:
        if _bash_parser is None and not _bash_parser_failed:
            try:
                import tree_sitter
                import tree_sitter_bash

                language = tree_sitter.Language(tree_sitter_bash.language())
                _bash_parser = tree_sitter.Parser(language)
            except Exception as exc:  # missing grammar is a capability gap, not a crash
                log.debug("tree-sitter-bash unavailable: %s", exc)
                _bash_parser_failed = True
    return _bash_parser


def _parse_bash(text: str) -> ShellParseResult:
    parser = _bash_language()
    if parser is None:
        return ShellParseResult(
            SHELL_BASH,
            complete=False,
            error="tree-sitter-bash is not installed",
            analysis_failed=True,
        )
    source = text.encode("utf-8", "replace")
    with _bash_parse_lock:
        tree = parser.parse(source)
    root = tree.root_node
    state = _BashWalk(source)
    state.visit(root)
    complete = not root.has_error and not state.incomplete
    error = state.error
    if root.has_error and not error:
        error = "syntax error"
    return ShellParseResult(
        shell=SHELL_BASH,
        commands=tuple(state.commands),
        redirections=tuple(state.redirections),
        dynamic=state.dynamic,
        complete=complete,
        error=error,
    )


class _BashWalk:
    """Single pass over a bash AST collecting commands and file writes."""

    def __init__(self, source: bytes):
        self.source = source
        self.commands: list[ParsedCommand] = []
        self.redirections: list[ParsedRedirection] = []
        self.dynamic = False
        self.incomplete = False
        self.error: str | None = None

    # -- node helpers ------------------------------------------------------

    def text(self, node) -> str:
        return self.source[node.start_byte : node.end_byte].decode("utf-8", "replace")

    def static_text(self, node) -> tuple[str | None, bool]:
        """(literal text, dynamic) for an argument-like node."""
        kind = node.type
        if kind in {"word", "number"}:
            value = self.text(node)
            if any(marker in value for marker in _DYNAMIC_MARKERS):
                return value, True
            return value, False
        if kind in {"raw_string", "ansi_c_string"}:
            return _strip_quotes(self.text(node), kind), False
        if kind == "string":
            value = self.text(node)
            inner = _strip_quotes(value, "string")
            return inner, any(marker in inner for marker in _DYNAMIC_MARKERS)
        if kind in {"simple_expansion", "expansion", "command_substitution",
                    "process_substitution", "arithmetic_expansion", "concatenation",
                    "heredoc_body"}:
            return self.text(node), True
        # Anything else is an expression we do not model.
        return self.text(node), True

    # -- traversal ---------------------------------------------------------

    def visit(self, node) -> None:
        kind = node.type
        if kind == "command":
            self.visit_command(node)
            return
        if kind == "file_redirect":
            self.visit_file_redirect(node)
            return
        if kind == "heredoc_redirect":
            # The heredoc body is input data, never a command.
            return
        if kind in {"comment", "heredoc_start", "heredoc_body", "heredoc_end"}:
            return
        if kind == "function_definition":
            # The body is extracted, but nothing proves it is never called
            # with other arguments, so the command set is not complete.
            self.dynamic = True
            self.incomplete = True
            self.error = self.error or "function definition"
        if kind == "ERROR":
            self.incomplete = True
            self.error = self.error or "syntax error"
        if kind in {"for_statement", "c_style_for_statement", "while_statement",
                    "if_statement", "case_statement", "subshell"}:
            self.dynamic = True
        if kind in {"command_substitution", "process_substitution"}:
            self.dynamic = True
        for child in node.children:
            self.visit(child)

    def visit_command(self, node) -> None:
        name_node = node.child_by_field_name("name")
        name_span = (name_node.start_byte, name_node.end_byte) if name_node is not None else None
        args: list[str] = []
        literals: list[bool] = []
        command_dynamic = False
        executable: str | None = None
        for child in node.children:
            if name_span is not None and (child.start_byte, child.end_byte) == name_span:
                continue
            if child.type in {"file_redirect", "heredoc_redirect"}:
                self.visit(child)
                continue
            if not child.is_named:
                continue
            if child.type == "variable_assignment":
                # Leading `FOO=bar` environment prefix, not an argument.
                continue
            value, dynamic = self.static_text(child)
            command_dynamic = command_dynamic or dynamic
            args.append(value if value is not None else "")
            literals.append(not dynamic)
            # Nested commands (substitutions, subshell arguments) still run.
            self.visit(child)
        if name_node is not None:
            candidate = name_node
            if candidate.type == "command_name" and candidate.named_children:
                candidate = candidate.named_children[0]
            executable, dynamic = self.static_text(candidate)
            if dynamic:
                command_dynamic = True
                # A computed program name: what runs is unknowable.
                self.incomplete = True
                self.error = self.error or "dynamic command name"
        normalized = _normalize_command(
            executable or "", args, None, command_dynamic, literals
        )
        if normalized is None:
            self.incomplete = True
            self.error = self.error or "unparsed command"
            return
        self.commands.append(normalized)
        if command_dynamic:
            self.dynamic = True

    def visit_file_redirect(self, node) -> None:
        operator = ""
        destination = node.child_by_field_name("destination")
        destination_span = (
            (destination.start_byte, destination.end_byte) if destination is not None else None
        )
        for child in node.children:
            if destination_span is not None and (child.start_byte, child.end_byte) == destination_span:
                continue
            if not child.is_named:
                operator = self.text(child)
                break
        if destination is None:
            return
        target, dynamic = self.static_text(destination)
        if dynamic:
            self.dynamic = True
        if operator in _BASH_WRITE_REDIRECTS and target and not target.isdigit():
            # `2>&1` targets a stream number, not a file.
            self.redirections.append(
                ParsedRedirection(target, operator.endswith(">>"), literal=not dynamic)
            )


def _strip_quotes(value: str, kind: str) -> str:
    if kind == "raw_string" and len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1]
    if kind == "ansi_c_string" and value.startswith("$'") and value.endswith("'"):
        return value[2:-1]
    if kind == "string" and len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    return value


# ---------------------------------------------------------------------------
# Shared normalization
# ---------------------------------------------------------------------------

def _normalize_command(
    executable: str,
    args: list[str],
    wrappers: list[str] | None = None,
    dynamic: bool = False,
    arg_literals: list[bool] | None = None,
) -> ParsedCommand | None:
    """Strip env assignments and wrapper commands down to the real executable."""
    pending = list(args)
    literals = list(arg_literals or ()) or [True] * len(pending)
    if len(literals) < len(pending):
        literals.extend([True] * (len(pending) - len(literals)))
    seen: list[str] = list(wrappers or ())
    name = _executable_name(executable)
    while name and name in _WRAPPERS:
        seen.append(name)
        index = 0
        # `env KEY=VALUE` / `sudo -u user` / `nice -n 5` before the command.
        while index < len(pending) and _is_assignment(pending[index]):
            index += 1
        while index < len(pending) and pending[index].startswith("-"):
            flag = pending[index]
            index += 1
            if flag in _WRAPPER_FLAGS_WITH_VALUE and index < len(pending):
                index += 1
        if index >= len(pending):
            return None
        name = _executable_name(pending[index])
        pending = pending[index + 1 :]
        literals = literals[index + 1 :]
    if not name:
        return None
    return ParsedCommand(
        executable=name,
        args=tuple(pending),
        wrappers=tuple(seen),
        dynamic=dynamic,
        arg_literals=tuple(literals),
    )


def _is_assignment(token: str) -> bool:
    head, sep, _ = token.partition("=")
    if not sep or not head:
        return False
    return head.replace("_", "").replace("-", "").isalnum() and head[0].isalpha()


def _executable_name(token: str) -> str:
    name = str(token or "").strip().strip("\"'").replace("\\", "/").rsplit("/", 1)[-1]
    lowered = name.lower()
    lowered = lowered.removesuffix(".exe")
    return lowered.strip()
