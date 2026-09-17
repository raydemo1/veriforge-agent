"""Runtime policy guard for unsafe or wasteful tool usage.

This middleware only *intercepts* calls (command-shape guards and per-turn
exploration budgets). It never counts failure streaks and never requests a
fallback stop: repeated blocked/policy failures are tracked once, centrally,
by ``ToolFailurePolicyMiddleware`` via the canonical ``ToolFailure`` model.
The intercepted results still carry ``status_source="tool_policy"``, which is
the event-wire label for a policy interception (peer of permission/budget).
"""
from __future__ import annotations

import re
import shlex

from ..tool_result import ToolResult
from .base import AgentMiddleware

SEARCH_TIMEOUT_SECONDS = 15
BROAD_REPO_SEARCH_BUDGET = 4
DEEP_ROOT_LIST_BUDGET = 2
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
SHELL_CONTROL_OPERATORS = ("|", "&&", "||", ";", ">", "<", "`")
SHELL_CONTROL_TOKENS = {"|", "&&", "||", ";"}


class ToolGuardMiddleware(AgentMiddleware):
    """Blocks repository browsing through shell and wasteful exploration."""

    def __init__(self):
        self._broad_repo_search_count = 0
        self._deep_root_list_count = 0

    def begin_turn(self, task: str, messages: list[dict], runtime_state=None,
                   agent_name: str | None = None) -> None:
        self._broad_repo_search_count = 0
        self._deep_root_list_count = 0

    def before_tool(
        self,
        tool_name: str,
        tool_args: dict,
        messages: list[dict],
        runtime_state=None,
        agent_name: str | None = None,
    ) -> ToolResult | None:
        if tool_name == "run_bash":
            return self._guard_shell(tool_args, runtime_state)
        if tool_name == "repo_search":
            return self._guard_repo_search(tool_args, runtime_state)
        if tool_name == "list_files":
            return self._guard_list_files(tool_args, runtime_state)
        if tool_name == "read_file":
            path = str((tool_args or {}).get("path") or "")
            if _is_observation_path(path):
                return self._blocked(
                    tool_name,
                    "read_file cannot read raw .harness/observations artifacts during normal runs. "
                    "Use summarized tool output, or set HARNESS_ALLOW_OBSERVATION_READ=1 for diagnosis.",
                    runtime_state,
                    category="internal_observation_read",
                    summary=f"read_file:{_shape_path(path)}",
                )
        return None

    def _guard_repo_search(self, tool_args: dict, runtime_state=None) -> ToolResult | None:
        path = str((tool_args or {}).get("path") or ".").strip() or "."
        if not _path_is_root(path):
            return None
        self._broad_repo_search_count += 1
        if self._broad_repo_search_count <= BROAD_REPO_SEARCH_BUDGET:
            return None
        return self._blocked(
            "repo_search",
            "Too many whole-repository searches in this turn. Narrow path/glob based on existing evidence before searching again.",
            runtime_state,
            category="exploration_budget",
            summary="repo_search:root",
        )

    def _guard_list_files(self, tool_args: dict, runtime_state=None) -> ToolResult | None:
        directory = str((tool_args or {}).get("directory") or ".").strip() or "."
        try:
            depth = int((tool_args or {}).get("depth") or 2)
        except (TypeError, ValueError):
            depth = 2
        if depth <= 2 or not _path_is_root(directory):
            return None
        self._deep_root_list_count += 1
        if self._deep_root_list_count <= DEEP_ROOT_LIST_BUDGET:
            return None
        return self._blocked(
            "list_files",
            "Too many deep root listings in this turn. Narrow directory or use repo_search with a specific path/glob.",
            runtime_state,
            category="exploration_budget",
            summary="list_files:deep_root",
        )

    def _guard_shell(self, tool_args: dict, runtime_state=None) -> ToolResult | None:
        command = str((tool_args or {}).get("command") or "").strip()
        lowered = _collapse(command.lower())
        if not lowered:
            return None

        if _is_broad_recursive_shell_listing(command):
            return self._blocked(
                "run_bash",
                "Recursive repository listing/search through shell is blocked. "
                "Use list_files(depth=..., max_results=...) for file discovery or repo_search for text search.",
                runtime_state,
                category="repo_browse_shell",
                summary=f"run_bash:{_command_family(lowered)}",
            )

        if _looks_like_rg(command):
            if _rg_has_explicit_path(command):
                return None
            if _is_simple_rg_search(command):
                tool_args["command"] = command.rstrip() + " ."
                current_timeout = tool_args.get("timeout")
                try:
                    timeout = int(current_timeout) if current_timeout is not None else SEARCH_TIMEOUT_SECONDS
                except (TypeError, ValueError):
                    timeout = SEARCH_TIMEOUT_SECONDS
                tool_args["timeout"] = min(timeout, SEARCH_TIMEOUT_SECONDS)
                return None
            return self._blocked(
                "run_bash",
                "Bare rg without an explicit search path is blocked because it can wait on stdin. "
                "Use repo_search(pattern=..., path=...) or provide an explicit bounded path.",
                runtime_state,
                category="bare_rg",
                summary="run_bash:rg_without_path",
            )

        if _looks_like_shell_search_without_path(command):
            return self._blocked(
                "run_bash",
                "Repository search through shell is blocked for this command shape. Use repo_search(pattern=..., path=...).",
                runtime_state,
                category="repo_browse_shell",
                summary=f"run_bash:{_command_family(lowered)}",
            )

        return None

    def _blocked(
        self,
        tool_name: str,
        message: str,
        runtime_state,
        *,
        category: str,
        summary: str,
    ) -> ToolResult:
        # Record the blocked action for fallback fingerprints; streak counting
        # and stop decisions live in ToolFailurePolicyMiddleware.
        fallback = getattr(runtime_state, "fallback", None)
        if fallback is not None:
            fallback.record_action(summary)
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


def _is_simple_rg_search(command: str) -> bool:
    if any(operator in command for operator in SHELL_CONTROL_OPERATORS):
        return False
    return _looks_like_rg(command)


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
    if any(
        re.search(pattern, lowered)
        for pattern in (
            r"\b(get-childitem|gci|ls)\b.*\s-recurse\b",
            r"\bdir\b.*\s/s\b",
            r"\bfindstr\b.*\s/s\b",
            r"\bfind\s+\.\s+.*-type\s+f\b",
            r"\btree\b.*\s/f\b",
        )
    ):
        return True
    return _is_broad_recursive_grep(command)


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


def _command_family(command: str) -> str:
    tokens = _tokens(command)
    if not tokens:
        return "empty"
    executable = tokens[0].strip("\"'").lower()
    executable = executable.removesuffix(".exe")
    return executable


def _shape_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    if not normalized or normalized == ".":
        return "."
    parts = [part for part in normalized.split("/") if part and part != "."]
    return "/".join(parts[:2]) if parts else "."


def _path_is_root(path: str) -> bool:
    normalized = path.replace("\\", "/").strip()
    return normalized in {"", ".", "./"}


def _is_observation_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return "/.harness/observations/" in f"/{normalized}/"
