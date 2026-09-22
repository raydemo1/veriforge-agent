"""Runtime policy guard for deterministic, objectively bad tool calls.

Scope is deliberately narrow: this middleware only intercepts calls whose
shape is wrong independent of history — e.g. a bare ``rg`` that would block
on stdin, or a recursive repository-wide listing through shell where a
bounded built-in tool exists. It holds no per-turn counters and implements
no "Nth repetition is wasteful" heuristics; resource control belongs to
output limits, timeouts and explicit tool-call budgets.

It is a pure decision point: it returns an intercepted
:class:`~harness_code_agent.runtime.tool_result.ToolResult` and never
replays, sleeps, records actions or requests a fallback stop. Repetition of
blocked calls is tracked once, centrally, by
``ToolFailurePolicyMiddleware`` via the canonical ``ToolFailure`` model.
The intercepted results carry ``status_source="tool_policy"``, the
event-wire label for a policy interception (peer of permission/budget).
"""
from __future__ import annotations

import re
import shlex

from ..tool_result import ToolResult
from .base import AgentMiddleware

RG_OPTIONS_WITH_VALUE = {
    "-A",
    "-B",
    "-C",
    "-e",
    "-f",
    "-g",
    "-m",
    "-t",
    "-T",
    "--after-context",
    "--before-context",
    "--context",
    "--count-matches",
    "--encoding",
    "--engine",
    "--field-context-separator",
    "--field-match-separator",
    "--glob",
    "--iglob",
    "--max-count",
    "--max-depth",
    "--max-filesize",
    "--path-separator",
    "--pre",
    "--regexp",
    "--replace",
    "--sort",
    "--sort-files",
    "--type",
    "--type-add",
    "--type-clear",
}
GREP_OPTIONS_WITH_VALUE = {
    "-A",
    "-B",
    "-C",
    "-e",
    "-f",
    "-m",
    "--after-context",
    "--before-context",
    "--context",
    "--exclude",
    "--exclude-dir",
    "--include",
    "--max-count",
    "--regexp",
    "--file",
}
SHELL_CONTROL_TOKENS = {"|", "&&", "||", ";"}


class ToolGuardMiddleware(AgentMiddleware):
    """Blocks objectively unsafe/unbounded shell command shapes."""

    def before_tool(
        self,
        tool_name: str,
        tool_args: dict,
        messages: list[dict],
        runtime_state=None,
        agent_name: str | None = None,
        permission_decision=None,
    ) -> ToolResult | None:
        if tool_name != "run_bash":
            return None
        command = str((tool_args or {}).get("command") or "").strip()
        if not command:
            return None

        if _is_broad_recursive_shell_listing(command):
            return self._blocked(
                "run_bash",
                "Recursive repository listing/search through shell is blocked. "
                "Use list_files(depth=..., max_results=...) for file discovery or repo_search for text search.",
                category="repo_browse_shell",
            )

        if _looks_like_rg(command) and not _rg_has_explicit_path(command):
            return self._blocked(
                "run_bash",
                "Bare rg without an explicit search path is blocked because it can wait on stdin. "
                "Use repo_search(pattern=..., path=...) or provide an explicit bounded path.",
                category="bare_rg",
            )

        if _looks_like_shell_search_without_path(command):
            return self._blocked(
                "run_bash",
                "Repository search through shell is blocked for this command shape. Use repo_search(pattern=..., path=...).",
                category="repo_browse_shell",
            )

        return None

    def _blocked(self, tool_name: str, message: str, *, category: str) -> ToolResult:
        output = f"[blocked] {message}"
        return ToolResult(
            tool=tool_name,
            status="failed",
            output=output,
            error=message,
            metadata={
                "status_source": "tool_policy",
                "policy_violation": category,
            },
        )


def _collapse(value: str) -> str:
    return " ".join(value.strip().split())


def _tokens(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=False)
    except ValueError:
        return command.split()


def _looks_like_rg(command: str) -> bool:
    tokens = _tokens(command)
    if not tokens:
        return False
    executable = tokens[0].strip("\"'").lower()
    return executable in {"rg", "rg.exe"}


def _rg_has_explicit_path(command: str) -> bool:
    tokens = _tokens(command)
    if len(tokens) <= 1:
        return False
    positionals: list[str] = []
    index = 1
    used_regexp_option = False
    while index < len(tokens):
        token = tokens[index].strip()
        if token == "--":
            positionals.extend(tokens[index + 1:])
            break
        if token in {"-e", "--regexp"}:
            used_regexp_option = True
        if token in RG_OPTIONS_WITH_VALUE:
            index += 2
            continue
        if any(token.startswith(option + "=") for option in RG_OPTIONS_WITH_VALUE if option.startswith("--")):
            if token.startswith("--regexp="):
                used_regexp_option = True
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        positionals.append(token)
        index += 1
    needed_positionals = 1 if used_regexp_option else 2
    return len(positionals) >= needed_positionals


def _is_broad_recursive_shell_listing(command: str) -> bool:
    lowered = _collapse(command.lower())
    if re.search(r"\b(get-childitem|gci|ls)\b.*\s-recurse\b", lowered):
        # Only unbounded sweeps stay blocked; a concrete subdirectory is fine.
        return not _has_bounded_recursive_target(command)
    if any(
        re.search(pattern, lowered)
        for pattern in (
            r"\bdir\b.*\s/s\b",
            r"\bfindstr\b.*\s/s\b",
            r"\bfind\s+\.\s+.*-type\s+f\b",
            r"\btree\b.*\s/f\b",
        )
    ):
        return True
    return _is_broad_recursive_grep(command)


_PWSH_RECURSE_VALUE_OPTIONS = {"-path", "-literalpath", "-filter", "-include", "-exclude", "-depth"}


def _has_bounded_recursive_target(command: str) -> bool:
    """True when a recursive listing names a concrete subdirectory instead of a
    whole-drive or home-directory sweep (or relying on the current directory)."""
    tokens = _tokens(command)
    if len(tokens) <= 1:
        return False
    positionals: list[str] = []
    index = 1
    while index < len(tokens):
        token = str(tokens[index]).strip("\"'").lower()
        if token in _PWSH_RECURSE_VALUE_OPTIONS:
            if token in {"-path", "-literalpath"} and index + 1 < len(tokens):
                positionals.append(str(tokens[index + 1]))
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        positionals.append(str(tokens[index]))
        index += 1
    if not positionals:
        return False
    return _is_bounded_recursive_path(positionals[0])


def _is_bounded_recursive_path(target: str) -> bool:
    normalized = target.strip().strip("\"'").replace("/", "\\")
    lowered = normalized.lower().rstrip("\\")
    if not lowered:
        return False
    # Whole-drive roots, current/parent dirs, and the home directory itself
    # are unbounded; anything naming a concrete path below them is bounded.
    if lowered in {".", "..", "~", "$home", "$env:userprofile", "userprofile"}:
        return False
    if re.fullmatch(r"[a-z]:", lowered):
        return False
    return True


def _is_broad_recursive_grep(command: str) -> bool:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()

    for index, token in enumerate(tokens):
        executable = token.strip("\"'").lower()
        executable = executable.removesuffix(".exe")
        if executable != "grep":
            continue

        segment: list[str] = []
        for candidate in tokens[index + 1:]:
            if candidate in SHELL_CONTROL_TOKENS:
                break
            segment.append(candidate)
        if not _grep_segment_is_recursive(segment):
            continue
        targets = _grep_targets(segment)
        if targets and all(_is_bounded_file_target(target) for target in targets):
            continue
        if (
            targets
            and all(_is_explicit_absolute_target(target) for target in targets)
            and _has_bounded_search_output(command)
        ):
            continue
        return True
    return False


def _grep_segment_is_recursive(tokens: list[str]) -> bool:
    for token in tokens:
        if token in {"-r", "-R", "--recursive", "--dereference-recursive"}:
            return True
        if token.startswith("-") and not token.startswith("--"):
            flags = token[1:]
            if "r" in flags or "R" in flags:
                return True
    return False


def _grep_targets(tokens: list[str]) -> list[str]:
    positionals: list[str] = []
    pattern_from_option = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if re.match(r"^\d?>", token):
            index += 1
            continue
        if token in {"-e", "--regexp", "-f", "--file"}:
            pattern_from_option = True
        if token in GREP_OPTIONS_WITH_VALUE:
            index += 2
            continue
        if any(token.startswith(option + "=") for option in GREP_OPTIONS_WITH_VALUE if option.startswith("--")):
            if token.startswith(("--regexp=", "--file=")):
                pattern_from_option = True
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        positionals.append(token)
        index += 1
    if pattern_from_option:
        return positionals
    return positionals[1:] if positionals else []


def _is_bounded_file_target(target: str) -> bool:
    normalized = target.strip("\"'").replace("\\", "/").rstrip("/")
    if not normalized or normalized in {".", "..", "/", "*", "**", "./*", "./**"}:
        return False
    name = normalized.rsplit("/", 1)[-1]
    return bool(re.search(r"\.[A-Za-z0-9_+-]+$", name))


def _is_explicit_absolute_target(target: str) -> bool:
    normalized = target.strip("\"'").replace("\\", "/").rstrip("/")
    if normalized in {"", "/", "."} or any(char in normalized for char in "*?["):
        return False
    if normalized.startswith("/"):
        return True
    return bool(re.match(r"^[A-Za-z]:/", normalized))


def _has_bounded_search_output(command: str) -> bool:
    lowered = _collapse(command.lower())
    return bool(
        re.search(r"\|\s*head(?:\s+-n)?\s+-?\d+\b", lowered)
        or re.search(r"(?:^|\s)-m\s*\d+\b", lowered)
        or re.search(r"--max-count(?:=|\s+)\d+\b", lowered)
    )


def _looks_like_shell_search_without_path(command: str) -> bool:
    lowered = _collapse(command.lower())
    tokens = _tokens(command)
    if not tokens:
        return False
    executable = tokens[0].strip("\"'").lower()
    executable = executable.removesuffix(".exe")
    if executable == "grep":
        non_options = [token for token in tokens[1:] if not str(token).startswith("-")]
        return len(non_options) < 2
    if executable == "findstr":
        non_options = [token for token in tokens[1:] if not str(token).startswith("/")]
        return len(non_options) < 2
    if executable in {"select-string", "sls"}:
        if re.search(r"\s-(literal)?path\s+", lowered):
            return False
        non_options = [token for token in tokens[1:] if not str(token).startswith("-")]
        return len(non_options) < 2
    return False
