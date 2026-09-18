"""Terminal-only shell safety policy.

Terminal profile treats the shell as a first-class way to work inside the task
workspace. This middleware only blocks commands that are clearly destructive or
outside the normal task workspace contract.
"""
from __future__ import annotations

import os
import re
import shlex

from .base import MAIN_AGENT_NAMES, AgentMiddleware

_SYSTEM_WRITE_PATH = r"(?:/etc/|/usr/|/bin/|/sbin/|/var/|~[/\\]|%USERPROFILE%|%WINDIR%|c:\\)"
_SYSTEM_PATH_WRITE_PATTERNS = (
    rf"(?:^|[^\d&])(?:>>?|&>)\s*['\"]?(?P<path>{_SYSTEM_WRITE_PATH}[^'\"\s;|&)]*)",
    rf"\bopen\s*\(\s*['\"](?P<path>{_SYSTEM_WRITE_PATH}[^'\"]*)",
    rf"\b(?:Path|PurePath)\s*\(\s*['\"](?P<path>{_SYSTEM_WRITE_PATH}[^'\"]*)['\"]\s*\)\.(?:write_text|write_bytes)\b",
    rf"\b(?:write_text|write_bytes)\s*\([^)]*['\"](?P<path>{_SYSTEM_WRITE_PATH}[^'\"]*)",
)
_COMMAND_START = r"(?:^|[;&|]\s*)"


class TerminalShellEditPolicyMiddleware(AgentMiddleware):
    def before_tool(
        self,
        tool_name: str,
        tool_args: dict,
        messages: list[dict],
        runtime_state=None,
        agent_name: str | None = None,
    ) -> str | None:
        if agent_name not in MAIN_AGENT_NAMES or tool_name != "run_bash":
            return None
        command = str(tool_args.get("command") or "")
        relax_container_paths = _allows_container_absolute_path_mutation(runtime_state)
        if _looks_dangerous_command(
            command,
            allow_container_absolute_path_mutation=relax_container_paths,
        ):
            return (
                "[blocked] Terminal profile allows shell edits inside the task workspace, but this "
                "command looks destructive or outside the workspace. Constrain the command to the "
                "task workspace and avoid destructive git/system operations."
            )
        return None


def _looks_dangerous_command(command: str, *, allow_container_absolute_path_mutation: bool = False) -> bool:
    return (
        _looks_like_system_path_write(
            command,
            allow_container_absolute_path_mutation=allow_container_absolute_path_mutation,
        )
        or _looks_like_destructive_rm(
            command,
            allow_container_absolute_path_mutation=allow_container_absolute_path_mutation,
        )
        or _looks_like_destructive_remove_item(
            command,
            allow_container_absolute_path_mutation=allow_container_absolute_path_mutation,
        )
        or _looks_like_destructive_del(
            command,
            allow_container_absolute_path_mutation=allow_container_absolute_path_mutation,
        )
        or _looks_like_destructive_git(command)
    )


def _looks_like_system_path_write(
    command: str,
    *,
    allow_container_absolute_path_mutation: bool,
) -> bool:
    for pattern in _SYSTEM_PATH_WRITE_PATTERNS:
        for match in re.finditer(pattern, command, flags=re.IGNORECASE):
            path = _strip_quotes(match.group("path"))
            if allow_container_absolute_path_mutation and _is_container_absolute_path(path):
                continue
            return True
    return False


def _allows_container_absolute_path_mutation(runtime_state) -> bool:
    metadata = getattr(runtime_state, "task_metadata", {}) or {}
    metadata = metadata if isinstance(metadata, dict) else {}
    permission_mode = (
        str(getattr(runtime_state, "permission_mode", "") or "")
        or str(metadata.get("permission_mode") or "")
        or os.environ.get("HARNESS_PERMISSION_MODE", "")
    )
    return permission_mode == "danger-full-access" and _is_terminal_eval_mode()


def _is_terminal_eval_mode() -> bool:
    return os.environ.get("HCA_TERMINAL_EVAL_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def _is_container_absolute_path(path: str) -> bool:
    normalized = _strip_quotes(path).replace("\\", "/").lower()
    return normalized.startswith("/")


def _looks_like_destructive_rm(
    command: str,
    *,
    allow_container_absolute_path_mutation: bool = False,
) -> bool:
    for args in _command_args(command, "rm"):
        tokens = _shell_words(args)
        if not _has_rm_recursive_force(tokens):
            continue
        if any(
            _is_dangerous_target(
                token,
                allow_container_absolute_path_mutation=allow_container_absolute_path_mutation,
            )
            for token in _positional_tokens(tokens)
        ):
            return True
    return False


def _looks_like_destructive_remove_item(
    command: str,
    *,
    allow_container_absolute_path_mutation: bool = False,
) -> bool:
    for args in _command_args(command, "remove-item"):
        tokens = _shell_words(args)
        has_recurse = any(_is_powershell_switch(token, ("recurse", "r")) for token in tokens)
        has_force = any(_is_powershell_switch(token, ("force", "f")) for token in tokens)
        if has_recurse and has_force and any(
            _is_dangerous_target(
                token,
                allow_container_absolute_path_mutation=allow_container_absolute_path_mutation,
            )
            for token in _positional_tokens(tokens)
        ):
            return True
    return False


def _looks_like_destructive_del(
    command: str,
    *,
    allow_container_absolute_path_mutation: bool = False,
) -> bool:
    for command_name in ("del", "erase"):
        for args in _command_args(command, command_name):
            tokens = _shell_words(args)
            has_recursive_delete = any(_strip_quotes(token).lower() == "/s" for token in tokens)
            if has_recursive_delete and any(
                _is_dangerous_target(
                    token,
                    allow_container_absolute_path_mutation=allow_container_absolute_path_mutation,
                )
                for token in _positional_tokens(tokens, skip_slash_options=True)
            ):
                return True
    return False


def _looks_like_destructive_git(command: str) -> bool:
    for args in _command_args(command, "git"):
        tokens = _shell_words(args)
        if not tokens:
            continue
        subcommand_index = _git_subcommand_index(tokens)
        if subcommand_index is None:
            continue
        subcommand = _strip_quotes(tokens[subcommand_index]).lower()
        subcommand_args = tokens[subcommand_index + 1 :]
        if subcommand == "reset" and any(_strip_quotes(token).lower() == "--hard" for token in subcommand_args):
            return True
        if subcommand == "checkout" and "--" in [_strip_quotes(token) for token in subcommand_args]:
            return True
        if subcommand == "restore" and _git_restore_uses_source(subcommand_args):
            return True
        if subcommand == "clean" and _git_clean_forces_directory_delete(subcommand_args):
            return True
    return False


def _command_args(command: str, command_name: str) -> list[str]:
    pattern = rf"{_COMMAND_START}{re.escape(command_name)}\b(?P<args>[^;&|]*)"
    return [match.group("args") for match in re.finditer(pattern, command, flags=re.IGNORECASE)]


def _shell_words(text: str) -> list[str]:
    try:
        return shlex.split(text, posix=False)
    except ValueError:
        try:
            return shlex.split(text, posix=True)
        except ValueError:
            return text.split()


def _strip_quotes(token: str) -> str:
    return token.strip().strip("'\"")


def _has_rm_recursive_force(tokens: list[str]) -> bool:
    recursive = False
    force = False
    for token in tokens:
        clean = _strip_quotes(token)
        if clean == "--":
            break
        if clean in {"--recursive", "--dir"}:
            recursive = True
        elif clean == "--force":
            force = True
        elif clean.startswith("-") and len(clean) > 1:
            flags = clean.lstrip("-")
            recursive = recursive or "r" in flags.lower()
            force = force or "f" in flags.lower()
    return recursive and force


def _is_powershell_switch(token: str, names: tuple[str, ...]) -> bool:
    clean = _strip_quotes(token).lower()
    if not clean.startswith("-"):
        return False
    return clean.lstrip("-") in names


def _positional_tokens(tokens: list[str], *, skip_slash_options: bool = False) -> list[str]:
    positional: list[str] = []
    force_positional = False
    skip_next = False
    options_with_values = {"-literalpath", "-path"}
    for token in tokens:
        clean = _strip_quotes(token)
        lower = clean.lower()
        if skip_next:
            positional.append(clean)
            skip_next = False
            continue
        if clean == "--":
            force_positional = True
            continue
        if not force_positional and lower in options_with_values:
            skip_next = True
            continue
        if not force_positional and clean.startswith("-"):
            continue
        if skip_slash_options and not force_positional and clean.startswith("/"):
            continue
        positional.append(clean)
    return positional


def _is_dangerous_target(
    token: str,
    *,
    allow_container_absolute_path_mutation: bool = False,
) -> bool:
    raw = _strip_quotes(token).lower()
    if raw in {"/", "/*", "~", "$home", "${home}", "%userprofile%", "%windir%"}:
        return True
    clean = raw.rstrip("/\\")
    if clean in {"", "."}:
        return False
    if clean in {"/etc", "/usr", "/bin", "/sbin", "/var"}:
        return True
    if clean.startswith(("/etc/", "/usr/", "/bin/", "/sbin/", "/var/")):
        return not allow_container_absolute_path_mutation
    if clean in {"~", "$home", "${home}", "%userprofile%", "%windir%"}:
        return True
    if clean.startswith(("~/", "~\\", "$home/", "$home\\", "${home}/", "${home}\\", "%userprofile%", "%windir%")):
        return True
    return bool(re.match(r"^[a-z]:", clean))


def _git_subcommand_index(tokens: list[str]) -> int | None:
    index = 0
    options_with_values = {"-c", "-C", "--git-dir", "--work-tree", "--namespace"}
    while index < len(tokens):
        token = _strip_quotes(tokens[index])
        if token in options_with_values:
            index += 2
            continue
        if any(token.startswith(option + "=") for option in options_with_values):
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        return index
    return None


def _git_restore_uses_source(tokens: list[str]) -> bool:
    stripped = [_strip_quotes(token).lower() for token in tokens]
    return any(token.startswith("--source=") or token == "--source" for token in stripped)


def _git_clean_forces_directory_delete(tokens: list[str]) -> bool:
    force = False
    directory = False
    for token in tokens:
        clean = _strip_quotes(token)
        if not clean.startswith("-"):
            continue
        flags = clean.lstrip("-").lower()
        force = force or "f" in flags
        directory = directory or "d" in flags
    return force and directory
