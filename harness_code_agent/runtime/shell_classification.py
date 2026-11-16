"""Shell command analysis: facts only, never decisions.

The analyzer answers three questions about a shell command:

* ``effects``   — what kinds of things it does (read / write / delete / ...)
* ``traits``    — which shapes it has (recursive / destructive / dynamic / ...)
* ``targets``   — which paths it touches and in which scope

Syntax comes from :mod:`runtime.shell_parser` (PowerShell's native parser or
tree-sitter for bash).  This module never tokenizes shell text itself: it maps
the parser's facts onto the command vocabulary below.

It never returns allow/ask/deny and never rates risk. Permission decisions are
the sole responsibility of :mod:`runtime.permissions`.
"""
from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .shell_parser import (
    ParsedCommand,
    ShellParseResult,
    parse_shell_command,
)


class ShellEffect(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    NETWORK = "network"
    GIT_MUTATION = "git_mutation"


class ShellTrait(str, Enum):
    RECURSIVE = "recursive"
    DESTRUCTIVE = "destructive"
    #: A destructive operation whose own target is host-wide or a system path.
    #: Unlike ``DESTRUCTIVE``, this is bound to the command that owns the target;
    #: a workspace delete must not become catastrophic because another command
    #: in the same line merely reads a system file.
    CATASTROPHIC = "catastrophic"
    PRIVILEGED = "privileged"
    DYNAMIC = "dynamic"
    UNKNOWN_EFFECT = "unknown_effect"
    ANALYSIS_FAILED = "analysis_failed"
    #: A verification/build command from the known task vocabulary.  It still
    #: executes project code — it is an execution, not a read.
    VERIFY = "verify"
    #: The command sends data out: an upload, a POST/PUT body, a form.  Reading
    #: a URL is ordinary network use; moving local bytes out is a different
    #: grant and never a reviewer's call.
    NETWORK_UPLOAD = "network_upload"


class TargetScope(str, Enum):
    WORKSPACE = "workspace"
    EXTERNAL = "external"
    SYSTEM = "system"
    #: The shell sandbox's private, disposable temp directory. The job manager
    #: points TEMP/TMP/TMPDIR at it and grants write access there, so a write
    #: that provably targets it is neither a workspace nor an external write.
    TEMP = "temp"
    UNKNOWN = "unknown"


class TargetAccess(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"


@dataclass(frozen=True)
class ShellTarget:
    raw: str
    scope: TargetScope
    #: How the command touches this path. ``WRITE`` creates or modifies it,
    #: ``DELETE`` removes it, ``READ`` only consumes it.
    access: TargetAccess = TargetAccess.READ

    @property
    def write(self) -> bool:
        """Compatibility: a target the policy must treat as a mutation."""
        return self.access is not TargetAccess.READ

    @property
    def delete(self) -> bool:
        return self.access is TargetAccess.DELETE


@dataclass(frozen=True)
class ShellAnalysis:
    effects: frozenset[ShellEffect]
    traits: frozenset[ShellTrait]
    targets: tuple[ShellTarget, ...] = ()
    #: False when the parser could not enumerate every operation that may run.
    complete: bool = True

    @property
    def scopes(self) -> set[TargetScope]:
        return {target.scope for target in self.targets}

    @property
    def write_targets(self) -> tuple[ShellTarget, ...]:
        return tuple(target for target in self.targets if target.write)

    @property
    def delete_targets(self) -> tuple[ShellTarget, ...]:
        return tuple(target for target in self.targets if target.delete)

    @property
    def read_targets(self) -> tuple[ShellTarget, ...]:
        return tuple(
            target for target in self.targets if target.access is TargetAccess.READ
        )

    @property
    def uncertainty(self) -> str | None:
        """Why the analyzer cannot promise what this command does, if it cannot."""
        if ShellTrait.UNKNOWN_EFFECT in self.traits:
            return "存在无法静态确定副作用的执行"
        if ShellTrait.DYNAMIC in self.traits:
            return "命令包含运行时才能确定的值"
        return None


# ---------------------------------------------------------------------------
# Command vocabularies
# ---------------------------------------------------------------------------

_INTERPRETERS = {
    "python", "python3", "py", "node", "nodejs", "deno", "bun",
    "ruby", "perl", "php", "pwsh", "powershell", "bash", "sh", "zsh",
    "fish", "cmd", "dotnet",
}
_PYTHON_COMMANDS = {"python", "python3", "py"}
_INTERP_SCRIPT_SUFFIXES = (
    ".py", ".js", ".mjs", ".cjs", ".ts", ".ps1", ".sh", ".bash", ".zsh",
    ".bat", ".cmd", ".rb", ".pl", ".php",
)
_EVAL_COMMANDS = {"eval", "exec", "invoke-expression", "iex"}

#: Shells whose `-c`-style payload is shell text we can parse again.
_INLINE_SHELL = {"bash", "sh", "zsh", "dash", "ksh", "pwsh", "powershell"}
_INLINE_FLAGS = {"-c", "--command", "-command"}
#: `bash -c "bash -c ..."` chains are followed this far, then treated as opaque.
_MAX_INLINE_DEPTH = 2

_PRIVILEGED_WRAPPERS = {"sudo"}

_PACKAGE_MANAGERS = {
    "npm", "pnpm", "yarn", "bun", "pip", "pip3", "pipx", "poetry",
    "uv", "conda", "gem", "apt", "apt-get", "brew", "cargo", "go",
    "rpm", "dpkg", "winget", "choco", "installer",
}
_PACKAGE_MUTATE_SUB = {
    "install", "i", "add", "remove", "uninstall", "rm", "upgrade",
    "update", "link", "publish", "get", "ensurepip", "env", "create",
}

_DELETE_COMMANDS = {"rm", "del", "erase", "remove-item", "ri"}
_RECURSE_DIR_COMMANDS = {"rmdir", "rd"}
_FIND_COMMANDS = {"find", "find.exe"}
_FIND_EXEC_PREDICATES = {"-exec", "-execdir", "-ok", "-okdir"}
_FIND_WRITE_PREDICATES = {"-fprint", "-fprint0", "-fls"}
_FIND_PREDICATES = {
    "-name", "-iname", "-path", "-ipath", "-regex", "-iregex", "-type",
    "-newer", "-anewer", "-cnewer", "-size", "-mtime", "-atime", "-ctime",
    "-mmin", "-amin", "-cmin", "-perm", "-user", "-group", "-nouser",
    "-nogroup", "-maxdepth", "-mindepth", "-delete", "-exec", "-execdir",
    "-ok", "-okdir", "-print", "-print0", "-fprint", "-fprint0", "-fls",
    "-quit", "-depth", "-prune", "-empty", "-links", "-inum", "-samefile",
    "-readable", "-writable", "-executable", "-true", "-false", "-ls",
}
_PS_WRITE_COMMANDS = {
    "set-content", "add-content", "out-file", "tee-object", "tee",
    "copy-item", "move-item", "rename-item", "export-csv", "export-clixml",
    "new-item", "ni", "clear-content", "clear-item",
}
#: Cmdlets whose second positional argument is content, not another path.
_CONTENT_WRITE_COMMANDS = {
    "set-content", "add-content", "out-file", "tee-object", "tee",
    "export-csv", "export-clixml", "clear-content", "clear-item",
}
_COPY_COMMANDS = {"cp", "copy", "mv", "move", "ln", "xcopy", "robocopy"}
_MKDIR_COMMANDS = {"mkdir", "md", "new-item", "ni", "touch"}
_FORMATTERS = {"ruff", "black", "gofmt", "prettier", "isort", "autopep8", "yapf"}
_NETWORK_TOOLS = {"curl", "wget", "invoke-webrequest", "iwr", "invoke-restmethod", "irm"}
#: Flags that make a network tool send local data to the other side.  Short
#: curl flags are case-sensitive (`-F` form vs `-f` fail), so they are matched
#: exactly; long options and PowerShell parameters are matched lowercased.
_UPLOAD_SHORT_FLAGS = {
    "curl": {"-d", "-F", "-T"},
    "wget": set(),
    "ps": set(),
}
_UPLOAD_LONG_FLAGS = {
    "curl": {"--data", "--data-raw", "--data-binary", "--data-urlencode",
             "--form", "--upload-file", "--json"},
    "wget": {"--post-data", "--post-file", "--body-data", "--body-file", "--method"},
    "ps": {"-body", "-infile", "-form", "-method"},
}
_UPLOAD_METHOD_FLAGS = {
    "curl": {"-X", "--request"},
    "wget": {"--method"},
    "ps": {"-method"},
}
_UPLOAD_METHODS = {"post", "put", "patch"}
#: A redirection written with no operand in front of it (a POSIX habit) is
#: exposed by the PowerShell parser as a command named after the operator.
_REDIRECTION_COMMANDS = {">", ">>", "&>", "&>>", "*>", "2>", "2>>", "<", "1>", "1>>"}
_PROCESS_CONTROL = {
    "kill", "pkill", "taskkill", "stop-process", "shutdown", "reboot",
    "poweroff", "halt", "restart-computer", "chmod", "chown", "icacls",
    "takeown", "reg", "schtasks", "sc", "set-executionpolicy",
}
#: A directory change re-anchors every later relative path, so the workspace
#: scope of later targets can no longer be proven.
_DIRECTORY_CHANGE_COMMANDS = {
    "cd", "chdir", "set-location", "sl", "push-location", "pushd", "popd",
    "pop-location",
}
_READ_COMMANDS = {
    "cat", "type", "ls", "dir", "pwd", "whoami", "id", "uname", "grep",
    "rg", "head", "tail", "echo", "test", "diff", "wc", "md5sum",
    "sha1sum", "sha256sum", "shasum", "which", "where", "env", "printenv",
    "true", "false", "date", "hostname", "stat", "file", "cut", "uniq",
    "tr", "basename", "dirname", "realpath", "readlink", "jq", "sort",
    "get-content", "gc", "get-childitem", "gci", "get-item",
    "get-location", "get-command", "gcm", "get-process", "gps",
    "get-service", "test-path", "resolve-path", "select-string", "sls",
    "where-object", "select-object", "get-member", "gm", "sort-object",
    "measure-object", "measure", "format-table", "ft", "write-output",
    "out-string", "convertfrom-json", "convertto-json", "split-path",
    "join-path", "compare-object", "compare", "out-null",
}

_GIT_READ_SUB = {
    "status", "log", "diff", "show", "blame", "ls-files", "rev-parse",
    "describe", "cat-file", "symbolic-ref", "count-objects",
}
_GIT_NETWORK_SUB = {"push", "fetch", "pull", "clone", "ls-remote"}
_GIT_LOCAL_MUTATE_SUB = {
    "add", "apply", "am", "commit", "merge", "rebase", "cherry-pick",
    "stash", "revert", "init", "worktree", "branch", "tag", "reset",
    "restore", "checkout", "clean", "sparse-checkout", "switch",
}

# High-confidence host-destructive shapes. The analyzer only attaches the
# DESTRUCTIVE/CATASTROPHIC traits; the policy decides what that means. These
# are matched against the executable and its own arguments, never against the
# whole command line, so a quoted string in another command cannot trigger them.
_HOST_DESTRUCTIVE_EXECUTABLES = {
    "diskpart", "bcdedit", "remove-computer", "clear-disk", "initialize-disk",
}
_FORK_BOMB_PATTERN = r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"

_SYSTEM_PATH_FRAGMENTS = (
    "/etc", "/usr", "/bin", "/sbin", "/var", "/boot", "/sys", "/proc",
    "/lib", "/lib64", "/root", "/dev",
)
_WINDOWS_SYSTEM_FRAGMENTS = ("c:\\windows", "c:\\program files", "c:\\programdata")
_WINDIR_MARKERS = ("%windir%", "%systemroot%")

#: Characters whose presence means a path is only known at run time.
_DYNAMIC_PATH_MARKERS = ("$", "`", "%", "*", "?", "[")
#: A wildcard names a set of paths, so its target scope cannot be pinned.
_GLOB_MARKERS = ("*", "?", "[")

_PARALLEL_VERIFY_PREFIXES = (
    "python -m unittest",
    "python -m pytest",
    "pytest",
    "ruff check",
    "mypy",
    "npm test",
    "npm run test",
    "npm run build",
    "npm run lint",
    "npm run check",
    "npm run typecheck",
    "pnpm test",
    "pnpm run test",
    "pnpm run build",
    "pnpm run lint",
    "pnpm run check",
    "pnpm run typecheck",
    "yarn test",
    "yarn run test",
    "yarn run build",
    "yarn run lint",
    "yarn run check",
    "yarn run typecheck",
    "bun test",
    "bun run test",
    "bun run build",
    "bun run lint",
    "bun run check",
    "bun run typecheck",
    "go test",
    "cargo test",
    "tsc --noemit",
    "pdflatex",
    "latexmk",
    "make test",
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def analyze_shell_command(
    command: str,
    workspace_root: str | None = None,
) -> ShellAnalysis:
    """Return factual effects/traits/targets for a shell command.

    Deliberately not memoized: path scopes depend on the current filesystem
    (existence and symlink resolution), so a cached result could outlive the
    state it was derived from. Only the pure syntax parse is cached.
    """
    text = str(command or "").strip()
    if not text:
        return ShellAnalysis(frozenset(), frozenset())
    parsed = parse_shell_command(text)
    return analyze_parsed_command(parsed, text, workspace_root)


def analyze_parsed_command(
    parsed: ShellParseResult,
    command: str,
    workspace_root: str | None = None,
    *,
    depth: int = 0,
) -> ShellAnalysis:
    """Turn one parse result into facts (the single analysis shared by callers)."""
    root = _normalize_root(workspace_root)
    effects: set[ShellEffect] = set()
    traits: set[ShellTrait] = set()
    targets: list[ShellTarget] = []

    for invocation in parsed.commands:
        found_effects, found_traits, found_targets = _command_facts(invocation, root, depth)
        if _host_destructive(invocation):
            found_traits = found_traits | {ShellTrait.DESTRUCTIVE, ShellTrait.CATASTROPHIC}
            if TargetScope.SYSTEM not in {target.scope for target in found_targets}:
                found_targets = [*found_targets, ShellTarget("/", TargetScope.SYSTEM)]
        elif ShellTrait.DESTRUCTIVE in found_traits and any(
            target.scope is TargetScope.SYSTEM and target.write
            for target in found_targets
        ):
            found_traits = found_traits | {ShellTrait.CATASTROPHIC}
        effects |= found_effects
        traits |= found_traits
        targets.extend(found_targets)

    for redirection in parsed.redirections:
        if _discards_output(redirection.target):
            continue
        effects.add(ShellEffect.WRITE)
        targets.append(
            ShellTarget(
                redirection.target,
                _classify_path_scope(
                    redirection.target, root, literal=redirection.literal
                ),
                TargetAccess.WRITE,
            )
        )

    if parsed.dynamic:
        traits.add(ShellTrait.DYNAMIC)
    if parsed.analysis_failed:
        traits.add(ShellTrait.ANALYSIS_FAILED)
    if not parsed.complete:
        # The parser could not prove which operations run.
        effects.add(ShellEffect.EXECUTE)
        traits |= {ShellTrait.DYNAMIC, ShellTrait.UNKNOWN_EFFECT}

    temp_variables: set[str] = set()
    if _TEMP_ENV_ASSIGNMENT.search(command):
        # The command may point the variable anywhere before the write, so the
        # path is no longer provably the sandbox's private temp directory.
        targets = [
            ShellTarget(target.raw, TargetScope.UNKNOWN, target.access)
            if target.scope is TargetScope.TEMP
            else target
            for target in targets
        ]
    else:
        temp_variables = _temp_derived_variables(command)
    if temp_variables:
        targets = [
            ShellTarget(target.raw, TargetScope.TEMP, target.access)
            if target.scope is TargetScope.UNKNOWN
            and _temp_target_expression(target.raw, temp_variables)
            else target
            for target in targets
        ]

    if re.search(_FORK_BOMB_PATTERN, command, flags=re.IGNORECASE):
        traits |= {ShellTrait.DESTRUCTIVE, ShellTrait.CATASTROPHIC}
        # Host-wide destructive shapes have no single path target; mark the
        # whole host so the catastrophic guardrail can fire.
        if TargetScope.SYSTEM not in {target.scope for target in targets}:
            targets.append(ShellTarget("/", TargetScope.SYSTEM))

    if _TEMP_ENV_ASSIGNMENT.search(command):
        # After an assignment the variable may point anywhere, so the private
        # temp proof no longer holds; fall back to the unknown-scope gate.
        targets = [
            ShellTarget(target.raw, TargetScope.UNKNOWN, target.access)
            if target.scope is TargetScope.TEMP
            else target
            for target in targets
        ]

    return ShellAnalysis(
        frozenset(effects),
        frozenset(traits),
        tuple(_dedupe(targets)),
        complete=parsed.complete,
    )


# ---------------------------------------------------------------------------
# Per-command vocabulary
# ---------------------------------------------------------------------------

def _host_destructive(invocation: ParsedCommand) -> bool:
    """Whether one parsed command is a known host-wide destructive executable."""
    executable = Path(invocation.executable).name.lower().removesuffix(".exe")
    if executable.startswith("mkfs") or executable in _HOST_DESTRUCTIVE_EXECUTABLES:
        return True
    args = [_bare(arg).lower() for arg in invocation.args]
    if executable == "format":
        return any(re.fullmatch(r"[a-z]:", arg) for arg in args)
    if executable == "cipher":
        return "/w" in args
    if executable == "dd":
        return any(arg.startswith("of=/dev/") for arg in args)
    return False


def _command_facts(
    invocation: ParsedCommand, root: str | None, depth: int = 0
) -> tuple[set[ShellEffect], set[ShellTrait], list[ShellTarget]]:
    exe = invocation.executable
    args = invocation.args
    traits: set[ShellTrait] = set()
    if any(wrapper in _PRIVILEGED_WRAPPERS for wrapper in invocation.wrappers):
        traits.add(ShellTrait.PRIVILEGED)
    if invocation.dynamic:
        traits.add(ShellTrait.DYNAMIC)

    rendered = " ".join((exe, *args)).strip()
    if is_verify_shell_command(rendered):
        # Project verification runs project code: it is an execution with a
        # scoped side-effect budget, never a proven read-only command.
        return {ShellEffect.EXECUTE}, traits | {ShellTrait.VERIFY}, []

    handler = _DISPATCH.get(exe)
    if handler is not None:
        effects, found_traits, targets = handler(invocation, root, traits, depth)
        return effects, traits | found_traits, targets
    if exe in _REDIRECTION_COMMANDS:
        return _redirect_facts(args, root, exe, invocation.arg_literals)
    if _looks_like_inplace_formatter(exe, args):
        return _formatter_facts(exe, args, root, invocation.arg_literals)
    if exe in _DIRECTORY_CHANGE_COMMANDS:
        return (
            {ShellEffect.EXECUTE},
            traits | {ShellTrait.DYNAMIC, ShellTrait.UNKNOWN_EFFECT},
            [],
        )
    if exe in _PROCESS_CONTROL:
        return {ShellEffect.EXECUTE}, traits | {ShellTrait.UNKNOWN_EFFECT}, []
    # Unrecognized executable: an unknown program, not a dangerous one. The
    # sandbox bounds what it can do; the policy does not need to guess.
    return {ShellEffect.EXECUTE}, traits | {ShellTrait.UNKNOWN_EFFECT}, []


# ---------------------------------------------------------------------------
# Rule table: one handler per command family
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _CommandRule:
    names: frozenset[str]
    handler: Callable


def _delete_rule(invocation, root, traits, depth):
    return _delete_facts(
        invocation.args,
        root,
        literals=invocation.arg_literals,
        windows_switches=invocation.executable in {"del", "erase"},
    )


def _recursive_delete_rule(invocation, root, traits, depth):
    return _delete_facts(
        invocation.args,
        root,
        literals=invocation.arg_literals,
        always_recursive=True,
        windows_switches=True,
    )


def _find_rule(invocation, root, traits, depth):
    return _find_facts(invocation.args, root, invocation.arg_literals)


def _git_rule(invocation, root, traits, depth):
    return _git_facts(invocation.args, root, invocation.arg_literals)


def _package_manager_rule(invocation, root, traits, depth):
    return _package_manager_facts(invocation.executable, invocation.args, traits)


def _interpreter_rule(invocation, root, traits, depth):
    return _interpreter_facts(
        invocation.executable,
        invocation.args,
        root,
        traits,
        depth,
        invocation.arg_literals,
    )


def _network_rule(invocation, root, traits, depth):
    return _network_facts(
        invocation.args, root, traits, invocation.executable, invocation.arg_literals
    )


def _copy_move_rule(invocation, root, traits, depth):
    return _copy_move_facts(
        invocation.executable, invocation.args, root, invocation.arg_literals
    )


def _file_write_rule(invocation, root, traits, depth):
    return _file_write_facts(
        invocation.executable, invocation.args, root, invocation.arg_literals
    )


def _read_rule(invocation, root, traits, depth):
    return _read_facts(invocation.executable, invocation.args, root, invocation.arg_literals)


_COMMAND_RULES = (
    _CommandRule(frozenset(_DELETE_COMMANDS), _delete_rule),
    _CommandRule(frozenset(_RECURSE_DIR_COMMANDS), _recursive_delete_rule),
    _CommandRule(frozenset(_FIND_COMMANDS), _find_rule),
    _CommandRule(frozenset({"git"}), _git_rule),
    # Package managers before interpreters: `bun install` is a package
    # operation, not a generic script host.
    _CommandRule(frozenset(_PACKAGE_MANAGERS), _package_manager_rule),
    _CommandRule(frozenset(_INTERPRETERS), _interpreter_rule),
    _CommandRule(frozenset(_NETWORK_TOOLS), _network_rule),
    _CommandRule(frozenset(_COPY_COMMANDS), _copy_move_rule),
    _CommandRule(frozenset({"copy-item", "move-item"}), _copy_move_rule),
    _CommandRule(
        frozenset(_PS_WRITE_COMMANDS) | frozenset(_MKDIR_COMMANDS), _file_write_rule
    ),
    _CommandRule(frozenset(_READ_COMMANDS), _read_rule),
)
_DISPATCH = {}
for _rule in _COMMAND_RULES:
    for _name in _rule.names:
        # First rule wins: the copy/move handler must beat the generic
        # PowerShell write handler for Copy-Item/Move-Item.
        _DISPATCH.setdefault(_name, _rule.handler)


def _redirect_facts(args, root, operator: str, literals=()):
    """`&> out` written without a command: a pure redirection, nothing runs."""
    targets: list[ShellTarget] = []
    if args:
        target = _bare(args[0])
        literal = literals[0] if literals else True
        if target:
            access = TargetAccess.READ if operator.startswith("<") else TargetAccess.WRITE
            targets.append(_target(target, root, access, literal))
    effects = {ShellEffect.READ} if operator.startswith("<") else {ShellEffect.WRITE}
    return effects, set(), targets


def _delete_facts(
    args,
    root,
    *,
    literals=(),
    always_recursive: bool = False,
    windows_switches: bool = False,
):
    effects = {ShellEffect.DELETE}
    traits: set[ShellTrait] = set()
    targets: list[ShellTarget] = []
    recursive = always_recursive or _is_recursive_delete(args)
    positions = _positional_items(
        args,
        literals,
        skip_slash_options=windows_switches,
        powershell=not windows_switches,
    )
    for raw, literal in positions:
        targets.append(_target(raw, root, TargetAccess.DELETE, literal))
    if recursive and positions:
        traits.add(ShellTrait.RECURSIVE)
        traits.add(ShellTrait.DESTRUCTIVE)
    return effects, traits, targets


def _find_facts(args, root, literals=()):
    effects: set[ShellEffect] = set()
    traits: set[ShellTrait] = set()
    targets: list[ShellTarget] = []
    lowered = [_bare(token).lower() for token in args]
    deletes = "-delete" in lowered

    # Paths precede the expression: `find <path...> <predicates>`.  With
    # `-delete` they are the objects being removed, so they must count as write
    # targets — otherwise "delete the whole workspace" looks like a read.
    for index, token in enumerate(args):
        if _bare(token).lower() in _FIND_PREDICATES:
            break
        literal = literals[index] if index < len(literals) else True
        access = TargetAccess.DELETE if deletes else TargetAccess.READ
        targets.append(_target(_bare(token), root, access, literal))

    if deletes:
        effects.add(ShellEffect.DELETE)
        traits.add(ShellTrait.DESTRUCTIVE)
    for predicate in _FIND_EXEC_PREDICATES & set(lowered):
        # `find -exec <command>` runs a program we would have to parse again.
        effects.add(ShellEffect.EXECUTE)
        traits |= {ShellTrait.DYNAMIC, ShellTrait.UNKNOWN_EFFECT}
        break
    for index, token in enumerate(lowered):
        if token in _FIND_WRITE_PREDICATES and index + 1 < len(args):
            target = _bare(args[index + 1])
            literal = literals[index + 1] if index + 1 < len(literals) else True
            effects.add(ShellEffect.WRITE)
            targets.append(_target(target, root, TargetAccess.WRITE, literal))
    if not effects:
        effects.add(ShellEffect.READ)
    return effects, traits, targets


def _is_recursive_delete(args) -> bool:
    for token in args:
        bare = _bare(token).lower()
        if bare in {"-r", "-rf", "-fr", "--recursive", "/s"}:
            return True
        if bare.startswith("-") and not bare.startswith("--"):
            head = bare.lstrip("-")
            if head and set(head) <= {"r", "f", "v", "p", "i"} and "r" in head:
                return True
        if bare in {"-recurse", "-r:"}:
            return True
    return False


_GIT_GLOBAL_VALUE_OPTS = {
    "--git-dir", "--work-tree", "--namespace", "--exec-path", "--super-prefix",
    "--config-env",
}
_GIT_GLOBAL_REPO_OPTS = {"--git-dir", "--work-tree"}


def _git_facts(args, root, literals=()):
    effects: set[ShellEffect] = set()
    traits: set[ShellTrait] = set()
    targets: list[ShellTarget] = []
    sub_index, repos = _git_parse_globals(args, literals)
    if sub_index is None:
        return effects, traits, targets

    sub = _bare(args[sub_index]).lower()
    sub_args = args[sub_index + 1 :]
    flags = {_bare(a).lower() for a in sub_args}

    if sub in _GIT_NETWORK_SUB:
        effects |= {ShellEffect.NETWORK, ShellEffect.GIT_MUTATION}
        if any(f in {"-f", "--force", "--force-with-lease"} for f in flags):
            traits.add(ShellTrait.DESTRUCTIVE)
        targets.extend(_git_repo_targets(repos, root, TargetAccess.WRITE))
        if sub == "clone":
            clone_target = _git_clone_target(sub_args)
            targets.append(_target(clone_target, root, TargetAccess.WRITE))
        return effects, traits, targets

    if sub in _GIT_READ_SUB:
        effects.add(ShellEffect.READ)
        targets.extend(_git_repo_targets(repos, root, TargetAccess.READ))
        return effects, traits, targets
    if sub == "remote" and (
        not sub_args or sub_args[0] in {"-v", "--verbose", "show", "get-url"}
    ):
        effects.add(ShellEffect.READ)
        targets.extend(_git_repo_targets(repos, root, TargetAccess.READ))
        return effects, traits, targets

    if sub in _GIT_LOCAL_MUTATE_SUB:
        effects.add(ShellEffect.GIT_MUTATION)
        # No -C/--git-dir means the repository is the workspace itself.
        repo_targets = repos or ((".", True),)
        targets.extend(_git_repo_targets(repo_targets, root, TargetAccess.WRITE))
    if sub == "reset" and "--hard" in flags:
        traits.add(ShellTrait.DESTRUCTIVE)
    if sub == "clean" and _git_clean_forces_directory_delete(sub_args):
        traits.add(ShellTrait.DESTRUCTIVE)
    if sub == "restore" and _git_restore_uses_source(sub_args):
        traits.add(ShellTrait.DESTRUCTIVE)
    if sub == "checkout" and "--" in flags:
        traits.add(ShellTrait.DESTRUCTIVE)
    if sub == "branch" and any(_bare(a).lower() == "-d" for a in sub_args):
        traits.add(ShellTrait.DESTRUCTIVE)
    return effects, traits, targets


def _git_parse_globals(args, literals):
    """Split git's global options from the subcommand.

    ``-C`` and ``--git-dir``/``--work-tree`` select the repository, including
    repositories outside the workspace; their values therefore become targets
    instead of being mistaken for the subcommand.
    """
    repos: list[tuple[str, bool]] = []
    index = 0
    while index < len(args):
        raw = _bare(args[index])
        literal = literals[index] if index < len(literals) else True
        low = raw.lower()
        if raw in {"-C", "-c"}:
            if index + 1 >= len(args):
                return None, repos
            if raw == "-C":
                repos.append(
                    (
                        _bare(args[index + 1]),
                        literals[index + 1] if index + 1 < len(literals) else True,
                    )
                )
            index += 2
            continue
        if low in _GIT_GLOBAL_VALUE_OPTS:
            if index + 1 >= len(args):
                return None, repos
            if low in _GIT_GLOBAL_REPO_OPTS:
                repos.append(
                    (
                        _bare(args[index + 1]),
                        literals[index + 1] if index + 1 < len(literals) else True,
                    )
                )
            index += 2
            continue
        if low.startswith(tuple(f"{option}=" for option in _GIT_GLOBAL_VALUE_OPTS)):
            key, _, value = raw.partition("=")
            if key.lower() in _GIT_GLOBAL_REPO_OPTS:
                repos.append((value, literal))
            index += 1
            continue
        if raw.startswith("-"):
            index += 1
            continue
        return index, repos
    return None, repos


def _git_repo_targets(repos, root, access):
    return [_target(raw, root, access, literal) for raw, literal in repos]


def _git_clone_target(sub_args) -> str:
    for token in sub_args:
        if token.startswith("-"):
            continue
        if "://" in token or token.endswith(".git") or ":" in token:
            continue
        return token
    return "."


def _interpreter_facts(
    exe, args, root, traits: set[ShellTrait], depth: int = 0, literals=()
):
    if args and _bare(args[0]).lower() in {"--version", "-v", "-version"}:
        return {ShellEffect.READ}, traits, []

    # `bash -c '<script>'` / `pwsh -Command '<script>'` carry shell text we can
    # parse instead of guessing: a literal payload is analyzed as the commands
    # it will actually run, so an inner `rm -rf /` still hits the hard denials.
    inline = _inline_payload(exe, args, depth)
    if inline is not None:
        payload, shell = inline
        nested = analyze_parsed_command(
            parse_shell_command(payload, shell), payload, root, depth=depth + 1
        )
        return (
            nested.effects | {ShellEffect.EXECUTE},
            nested.traits | traits,
            list(nested.targets),
        )

    # An interpreter runs code we have not read, whatever its arguments are.
    found: set[ShellTrait] = traits | {ShellTrait.DYNAMIC, ShellTrait.UNKNOWN_EFFECT}
    targets: list[ShellTarget] = []
    if args:
        first = _bare(args[0])
        if first and not first.startswith("-"):
            low = first.lower()
            if low.endswith(_INTERP_SCRIPT_SUFFIXES) or "." not in low.rsplit("/", 1)[-1]:
                targets.append(
                    _target(
                        first,
                        root,
                        TargetAccess.READ,
                        literals[0] if literals else True,
                    )
                )
    return {ShellEffect.EXECUTE}, found, targets


def _inline_payload(exe: str, args, depth: int) -> tuple[str, str] | None:
    """The literal script text an interpreter flag will execute, if visible."""
    if exe not in _INLINE_SHELL or depth >= _MAX_INLINE_DEPTH:
        return None
    for index, token in enumerate(args):
        if _bare(token).lower() not in _INLINE_FLAGS or index + 1 >= len(args):
            continue
        payload = _bare(args[index + 1])
        if not payload:
            return None
        # A value the outer shell still has to expand is not a visible script.
        if any(marker in payload for marker in ("$(", "`")):
            return None
        shell = "powershell" if exe in {"pwsh", "powershell"} else "bash"
        return payload, shell
    return None


def _network_facts(args, root, traits: set[ShellTrait], exe: str = "", literals=()):
    effects = {ShellEffect.NETWORK, ShellEffect.READ}
    targets: list[ShellTarget] = []
    if _network_sends_data(exe, args):
        traits = traits | {ShellTrait.NETWORK_UPLOAD}
    output_path = _option_value(args, ("-o", "--output", "-outfile"))
    if output_path:
        effects.add(ShellEffect.WRITE)
        targets.append(
            _target(
                output_path,
                root,
                TargetAccess.WRITE,
                _literal_for(args, literals, output_path),
            )
        )
    if any(_bare(a) in {"-O", "--remote-name"} for a in args):
        effects.add(ShellEffect.WRITE)
    return effects, traits, targets


def _network_sends_data(exe: str, args) -> bool:
    """Whether this network call uploads or posts local content."""
    kind = "ps" if exe in {"invoke-webrequest", "iwr", "invoke-restmethod", "irm"} else exe
    short = _UPLOAD_SHORT_FLAGS.get(kind, set())
    long_flags = _UPLOAD_LONG_FLAGS.get(kind, set())
    method_flags = _UPLOAD_METHOD_FLAGS.get(kind, set())
    tokens = [_bare(token) for token in args]

    def value_at(index: int, inline: str) -> str:
        if inline:
            return inline
        return tokens[index + 1] if index + 1 < len(tokens) else ""

    for index, token in enumerate(tokens):
        if token in short:
            return True
        if kind == "curl" and token.startswith("-X") and len(token) > 2:
            # Attached form: `curl -XPOST url`.
            return token[2:].lower() in _UPLOAD_METHODS
        head, separator, inline = token.partition("=")
        head_low = head.lower()
        if token in method_flags or head_low in method_flags:
            return value_at(index, inline if separator else "").lower() in _UPLOAD_METHODS
        if head_low.startswith("--"):
            if head_low not in long_flags and not any(
                head_low.startswith(flag + "-") for flag in long_flags
            ):
                continue
        elif head_low not in long_flags:
            continue
        if head_low.lstrip("-").endswith("method"):
            return value_at(index, inline if separator else "").lower() in _UPLOAD_METHODS
        return True
    return False


_PM_RUN_SUB = {
    "run", "run-script", "start", "test", "build", "lint", "check",
    "typecheck", "exec", "dlx", "x", "watch",
}
_PM_QUERY_SUB = {
    "ls", "list", "view", "show", "info", "why", "outdated", "search",
    "help", "version", "root", "prefix", "bin", "doctor", "freeze", "get-url",
}
_PM_CONFIG_MUTATE_SUB = {"set", "delete", "edit", "unset"}
_PM_VALUE_OPTIONS = {"--prefix", "--cwd", "-C", "--registry", "--workspace", "-w"}


def _package_manager_facts(exe: str, args, traits: set[ShellTrait]):
    sub = _first_subcommand(args)
    if sub is None:
        return {ShellEffect.READ}, traits, []
    low = _bare(sub).lower()
    if exe == "go" and low in {"get", "install"}:
        return (
            {ShellEffect.EXECUTE, ShellEffect.NETWORK},
            traits | {ShellTrait.UNKNOWN_EFFECT},
            [],
        )
    if low == "config":
        second = _first_subcommand(args, after=sub)
        if second is not None and _bare(second).lower() in _PM_CONFIG_MUTATE_SUB:
            return {ShellEffect.EXECUTE}, traits | {ShellTrait.UNKNOWN_EFFECT}, []
        return {ShellEffect.READ}, traits, []
    if low in _PACKAGE_MUTATE_SUB:
        return (
            {ShellEffect.EXECUTE, ShellEffect.NETWORK},
            traits | {ShellTrait.UNKNOWN_EFFECT},
            [],
        )
    if low in _PM_RUN_SUB:
        # A project-defined script: it runs code we have not read, but it is
        # not inherently dangerous, and the sandbox bounds its writes.
        return {ShellEffect.EXECUTE}, traits | {ShellTrait.UNKNOWN_EFFECT}, []
    if low in _PM_QUERY_SUB:
        return {ShellEffect.READ}, traits, []
    # An unknown package-manager subcommand runs package or project code.
    return {ShellEffect.EXECUTE}, traits | {ShellTrait.UNKNOWN_EFFECT}, []


_COPY_VALUE_OPTIONS = {
    "--suffix", "-S", "--mode", "-m", "--owner", "-o", "--group", "-g",
    "--chown", "--chmod", "--backup",
}


def _copy_move_facts(exe: str, args, root, literals=()):
    effects: set[ShellEffect] = set()
    targets: list[ShellTarget] = []
    if exe in {"copy-item", "move-item"}:
        sources, destination = _ps_copy_operands(args, literals)
    else:
        operands, target_dir = _path_operands(
            args,
            literals,
            value_options=_COPY_VALUE_OPTIONS,
            target_option="-t" if exe in {"cp", "copy"} else None,
            skip_slash_options=exe in {"xcopy", "robocopy"},
        )
        if target_dir is not None:
            sources, destination = operands, target_dir
        elif len(operands) >= 2:
            sources, destination = operands[:-1], operands[-1]
        else:
            sources, destination = operands, None

    moving = exe in {"mv", "move", "move-item"}
    access = TargetAccess.DELETE if moving else TargetAccess.READ
    for raw, literal in sources:
        targets.append(_target(raw, root, access, literal))
    if destination is not None:
        targets.append(
            _target(destination[0], root, TargetAccess.WRITE, destination[1])
        )
    effects.add(ShellEffect.READ)
    if moving:
        effects.add(ShellEffect.DELETE)
    if destination is not None:
        effects.add(ShellEffect.WRITE)
    return effects, set(), targets


def _file_write_facts(exe: str, args, root, literals=()):
    targets: list[ShellTarget] = []
    if exe in {"new-item", "ni"}:
        values = _ps_path_operands(args, literals)
    elif exe in {"mkdir", "md", "touch"}:
        values = _positional_items(
            args, literals, skip_slash_options=True, powershell=True
        )
    elif exe == "rename-item":
        values = _rename_operands(args, literals)
    else:
        values = _ps_path_operands(args, literals)
        if exe in _CONTENT_WRITE_COMMANDS:
            # The content/value argument is not another path.
            values = values[:1]
    for raw, literal in values:
        targets.append(_target(raw, root, TargetAccess.WRITE, literal))
    return {ShellEffect.WRITE}, set(), targets


_READ_PATH_COMMANDS = {
    "cat", "type", "head", "tail", "stat", "file", "wc",
    "get-content", "gc", "get-item", "get-childitem", "gci", "ls", "dir",
}


def _read_facts(exe: str, args, root, literals=()):
    targets: list[ShellTarget] = []
    if exe in _READ_PATH_COMMANDS:
        if exe.startswith("get-") or exe in {"gc", "gci"}:
            values = _ps_path_operands(args, literals)
        else:
            values = _positional_items(args, literals, skip_slash_options=True)
        for raw, literal in values:
            if raw.isdigit():
                continue
            targets.append(_target(raw, root, TargetAccess.READ, literal))
    return {ShellEffect.READ}, set(), targets


def _first_subcommand(args, *, after=None):
    start = 0
    if after is not None:
        start = list(args).index(after) + 1 if after in args else 0
    index = start
    while index < len(args):
        raw = _bare(args[index])
        low = raw.lower()
        if raw in _PM_VALUE_OPTIONS or low in _PM_VALUE_OPTIONS:
            index += 2
            continue
        if raw.startswith("-"):
            index += 1
            continue
        return raw
    return None


# ---------------------------------------------------------------------------
# Argument tokens
# ---------------------------------------------------------------------------

#: PowerShell parameters that consume the following argument as a path/value.
_PS_PATH_PARAMETERS = {
    "-path", "-literalpath", "-file", "-filepath", "-destination", "-target",
    "-source", "-outfile", "-output",
}
_PS_VALUE_PARAMETERS = {
    "-filter", "-include", "-exclude", "-depth", "-stream", "-credential",
    "-encoding", "-name", "-value", "-itemtype", "-newname", "-displayname",
}


def _positional_items(
    args, literals=(), *, skip_slash_options: bool = False, powershell: bool = False
) -> list[tuple[str, bool]]:
    """Positional path arguments with their literal/dynamic flag.

    PowerShell parameters are role-aware: ``-Path out.txt`` yields a path,
    while ``-ItemType Directory`` and ``-Value hi`` consume their operand as a
    value.  The AST exposes the parameter and its operand as separate elements,
    so the role is carried until the next token is consumed.
    """
    positions: list[tuple[str, bool]] = []
    pending: str | None = None
    for index, token in enumerate(args):
        bare = _bare(token)
        literal = literals[index] if index < len(literals) else True
        if pending is not None:
            if pending == "path":
                positions.append((bare, literal))
            pending = None
            continue
        if powershell and bare.startswith("-"):
            head, separator, inline = bare.partition(":")
            head_low = head.lower()
            if head_low in _PS_PATH_PARAMETERS:
                if separator and inline:
                    positions.append((inline, literal))
                else:
                    pending = "path"
                continue
            if head_low in _PS_VALUE_PARAMETERS:
                if not separator:
                    pending = "value"
                continue
            continue
        if bare.startswith("-"):
            continue
        if skip_slash_options and bare.startswith("/") and _looks_like_windows_switch(bare):
            continue
        positions.append((bare, literal))
    return positions


def _ps_path_operands(args, literals=()):
    return _positional_items(args, literals, powershell=True)


def _ps_copy_operands(args, literals=()):
    sources: list[tuple[str, bool]] = []
    destination: tuple[str, bool] | None = None
    pending: str | None = None
    for index, token in enumerate(args):
        bare = _bare(token)
        literal = literals[index] if index < len(literals) else True
        if pending == "source":
            sources.append((bare, literal))
            pending = None
            continue
        if pending == "destination":
            destination = (bare, literal)
            pending = None
            continue
        head, separator, inline = bare.partition(":")
        low = head.lower()
        if low in {"-path", "-literalpath"}:
            if separator and inline:
                sources.append((inline, literal))
            else:
                pending = "source"
            continue
        if low in {"-destination", "-target"}:
            if separator and inline:
                destination = (inline, literal)
            else:
                pending = "destination"
            continue
        if low in _PS_VALUE_PARAMETERS:
            continue
        if low.startswith("-"):
            continue
        sources.append((bare, literal))
    if destination is None and len(sources) > 1:
        destination = sources.pop()
    return sources, destination


def _path_operands(
    args,
    literals=(),
    *,
    value_options=(),
    target_option: str | None = None,
    skip_slash_options: bool = False,
):
    operands: list[tuple[str, bool]] = []
    target: tuple[str, bool] | None = None
    index = 0
    while index < len(args):
        raw = _bare(args[index])
        literal = literals[index] if index < len(literals) else True
        low = raw.lower()
        if target_option is not None and raw == target_option:
            if index + 1 < len(args):
                target = (
                    _bare(args[index + 1]),
                    literals[index + 1] if index + 1 < len(literals) else True,
                )
            index += 2
            continue
        if target_option is not None and low.startswith(target_option + "="):
            target = (raw.split("=", 1)[1], literal)
            index += 1
            continue
        if raw in value_options:
            index += 2
            continue
        if raw.startswith("-"):
            index += 1
            continue
        if skip_slash_options and _looks_like_windows_switch(raw):
            index += 1
            continue
        operands.append((raw, literal))
        index += 1
    return operands, target


def _rename_operands(args, literals=()):
    values: list[tuple[str, bool]] = []
    pending: str | None = None
    for index, token in enumerate(args):
        bare = _bare(token)
        literal = literals[index] if index < len(literals) else True
        if pending is not None:
            values.append((bare, literal))
            pending = None
            continue
        head, separator, inline = bare.partition(":")
        low = head.lower()
        if low in {"-path", "-literalpath", "-newname"}:
            if separator and inline:
                values.append((inline, literal))
            else:
                pending = "path"
            continue
        if low in _PS_VALUE_PARAMETERS:
            continue
        if bare.startswith("-"):
            continue
        values.append((bare, literal))
    return values


def _literal_for(args, literals, value: str) -> bool:
    for index, token in enumerate(args):
        if _bare(token) == value and index < len(literals):
            return literals[index]
    return True


def _option_value(args, names: tuple[str, ...]) -> str | None:
    """Value of an option, matched case-sensitively for short flags.

    ``-o`` and ``-O`` are different curl flags; only the long form may be
    matched loosely and with ``=value``.
    """
    for index, token in enumerate(args):
        bare = _bare(token)
        if bare in names and index + 1 < len(args):
            return _bare(args[index + 1])
        for name in names:
            if name.startswith("--") and bare.lower().startswith(name + "="):
                return bare.split("=", 1)[1]
    return None


def _looks_like_windows_switch(token: str) -> bool:
    return bool(re.fullmatch(r"/[a-zA-Z]+", token))


def _looks_like_inplace_formatter(exe: str, args) -> bool:
    if exe in _FORMATTERS:
        if exe in {"gofmt", "prettier"}:
            return "-w" in args or "--write" in args
        if exe == "ruff":
            return "check" in args and "--fix" in args
        return True
    if exe == "sed":
        return any(a == "-i" or a.startswith("-i") for a in args)
    return False


def _formatter_facts(exe, args, root, literals=()):
    if exe == "sed":
        items = _sed_file_items(args, literals)
    else:
        filtered = list(args)
        filtered_literals = list(literals)
        if exe == "ruff" and filtered and filtered[0] == "check":
            filtered = filtered[1:]
            filtered_literals = filtered_literals[1:]
        items = _positional_items(
            filtered, filtered_literals, skip_slash_options=True
        )
    return (
        {ShellEffect.WRITE},
        set(),
        [_target(raw, root, TargetAccess.WRITE, literal) for raw, literal in items],
    )


def _sed_file_items(args, literals=()):
    """Drop the sed script expression; keep the files it edits in place."""
    items = _positional_items(args, literals, skip_slash_options=True)
    script_from_option = any(
        _bare(token).lower() in {"-e", "-f", "--expression", "--file"}
        or _bare(token).lower().startswith(("--expression=", "--file="))
        for token in args
    )
    if script_from_option:
        return items
    return items[1:] if items else []


def _target(raw: str, root, access: TargetAccess, literal: bool = True) -> ShellTarget:
    return ShellTarget(
        _bare(raw),
        _classify_path_scope(raw, root, literal=literal),
        access,
    )


def _git_restore_uses_source(args) -> bool:
    for index, token in enumerate(args):
        if token == "--source" and index + 1 < len(args):
            return True
        if token.startswith("--source="):
            return True
    return False


def _git_clean_forces_directory_delete(args) -> bool:
    joined = "".join(_bare(a).lstrip("-") for a in args if a.startswith("-"))
    return "f" in joined and "d" in joined


# ---------------------------------------------------------------------------
# Path scope classification
# ---------------------------------------------------------------------------

#: Environment variables the job manager repoints at the per-call private temp
#: directory (see :class:`ShellJobManager`). Only the explicit forms are
#: recognised: a plain ``$TEMP`` is an ordinary shell variable whose value the
#: command can change, so it is not proof of the sandbox temp directory.
_PRIVATE_TEMP_PATH = re.compile(
    r"^(?:\$env:(?:temp|tmp|tmpdir)|\$\{env:(?:temp|tmp|tmpdir)\}"
    r"|%(?:temp|tmp|tmpdir)%)(?=$|[\\/])",
    re.IGNORECASE,
)

#: Any assignment to a temp variable invalidates the static proof above: the
#: analyzer cannot tell which value the variable holds afterwards.
_TEMP_ENV_ASSIGNMENT = re.compile(
    r"(?:\$env:(?:temp|tmp|tmpdir)\s*=|\$\{env:(?:temp|tmp|tmpdir)\}\s*="
    r"|(?:^|[\s;&|(`\"'])(?:temp|tmp|tmpdir)\s*="
    r"|\bset\s+(?:temp|tmp|tmpdir)\s*=\s*|\bsetx\s+(?:temp|tmp|tmpdir)\b"
    r"|\bset-item\b[^\n;|&]*\benv:(?:temp|tmp|tmpdir)\b"
    r"|\bsetenvironmentvariable\s*\(\s*['\"](?:temp|tmp|tmpdir)['\"])",
    re.IGNORECASE,
)

#: ``$name = <temp expression>`` as one statement.  PowerShell executes
#: top-to-bottom, so a later assignment replaces the earlier value.
_POWERSHELL_ASSIGNMENT = re.compile(
    r"(?:^|[;\n&|])[ \t]*\$(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<rhs>[^\r\n;]+)",
    re.MULTILINE,
)
_JOIN_TEMP_PATH = re.compile(
    r"^join-path\s+\$env:(?:temp|tmp|tmpdir)\s+(?P<rest>.+)$", re.IGNORECASE
)


def _safe_temp_tail(value: str) -> bool:
    """A literal suffix that cannot leave the private temp directory."""
    tail = value.strip().strip("'\"").strip()
    return not (
        "$" in tail
        or "%" in tail
        or ".." in tail
        or any(marker in tail for marker in "*?")
    )


def _temp_derived_variables(command: str) -> set[str]:
    """PowerShell variables proven to hold a path inside the private temp.

    Only the two shapes the traces actually use are tracked: a direct
    ``$env:TEMP\\...`` string and ``Join-Path $env:TEMP <literal>``.  Anything
    dynamic, traversing or wildcarded stays unproven and therefore unknown.
    """
    variables: set[str] = set()
    for match in _POWERSHELL_ASSIGNMENT.finditer(command):
        name = match.group("var").lower()
        rhs = match.group("rhs").strip()
        rvalue = rhs.strip("'\"").strip()
        direct = _PRIVATE_TEMP_PATH.match(rvalue.lower())
        join = _JOIN_TEMP_PATH.match(rhs)
        if direct is not None:
            proven = _safe_temp_tail(rvalue[direct.end():])
        elif join is not None:
            proven = _safe_temp_tail(join.group("rest"))
        else:
            proven = False
        if proven:
            variables.add(name)
        else:
            variables.discard(name)
    return variables


def _split_target_list(text: str) -> list[str]:
    """Split a PowerShell comma-separated argument while respecting nesting."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote = ""
    for char in text:
        if quote:
            current.append(char)
            if char == quote:
                quote = ""
        elif char in "'\"":
            quote = char
            current.append(char)
        elif char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return parts


def _temp_element(text: str, variables: set[str]) -> bool:
    value = text.strip()
    while value.startswith("(") and value.endswith(")"):
        value = value[1:-1].strip()
    lowered = value.strip("'\"").strip().lower()
    if lowered in {f"${name}" for name in variables} | {
        f"${{{name}}}" for name in variables
    }:
        return True
    direct = _PRIVATE_TEMP_PATH.match(lowered)
    if direct is not None:
        return _safe_temp_tail(lowered[direct.end():])
    join = _JOIN_TEMP_PATH.match(value)
    return join is not None and _safe_temp_tail(join.group("rest"))


def _temp_target_expression(raw: str, variables: set[str]) -> bool:
    """A whole target argument proven to name only private-temp paths."""
    parts = [part for part in _split_target_list(_bare(raw)) if part.strip()]
    return bool(parts) and all(_temp_element(part, variables) for part in parts)


def _classify_path_scope(raw_path: str, root: str | None, *, literal: bool = True) -> TargetScope:
    token = _bare(raw_path).strip()
    if not token:
        return TargetScope.WORKSPACE
    if token in {"/", "/*", "\\"}:
        return TargetScope.SYSTEM
    lowered = token.lower()
    if lowered in {".", "./", ".\\"}:
        return TargetScope.WORKSPACE
    temp_match = _PRIVATE_TEMP_PATH.match(lowered)
    if temp_match is not None:
        # The variable is proven to name the sandbox temp root, but a dynamic
        # suffix, traversal or wildcard could still leave it.
        if not _safe_temp_tail(lowered[temp_match.end():]):
            return TargetScope.UNKNOWN
        return TargetScope.TEMP

    # Home / Windows env markers are always user-level system targets, even
    # when the parser reports them as a runtime expression.
    if lowered in {"~", "$home", "${home}", "%userprofile%", "%homepath%", "$env:userprofile"}:
        return TargetScope.SYSTEM
    for marker in _WINDIR_MARKERS:
        if marker in lowered:
            return TargetScope.SYSTEM
    if not literal:
        # The parser already told us this value is resolved at run time.
        return TargetScope.UNKNOWN

    normalized_low = lowered.replace("\\", "/").rstrip("/")
    # POSIX system paths.
    if normalized_low in _SYSTEM_PATH_FRAGMENTS or any(
        normalized_low.startswith(fragment + "/") for fragment in _SYSTEM_PATH_FRAGMENTS
    ):
        return TargetScope.SYSTEM

    # Windows drive roots / system directories.
    if re.match(r"^[a-z]:(?:/|$)", normalized_low):
        for fragment in (f.replace("\\", "/") for f in _WINDOWS_SYSTEM_FRAGMENTS):
            if normalized_low.startswith(fragment):
                return TargetScope.SYSTEM
        if normalized_low.endswith(":"):
            return TargetScope.SYSTEM

    # A wildcard names a set of paths we do not know statically.
    if any(marker in token for marker in _GLOB_MARKERS):
        return TargetScope.UNKNOWN

    if root:
        return _scope_against_root(token, root)
    return TargetScope.EXTERNAL if _is_absolute_path(token) else TargetScope.WORKSPACE


def _is_absolute_path(token: str) -> bool:
    if token.startswith(("/", "\\")):
        return True
    if re.match(r"^[a-zA-Z]:", token):
        return True
    return Path(token).expanduser().is_absolute()


def _scope_against_root(token: str, root: str) -> TargetScope:
    """Resolve a relative path against the real workspace and contain it."""
    try:
        candidate = Path(token).expanduser()
        if not candidate.is_absolute():
            candidate = Path(root) / candidate
        normalized = os.path.normcase(os.path.normpath(str(candidate)))
        if not _is_same_or_child(normalized, root):
            return TargetScope.EXTERNAL
        if candidate.exists():
            # A symlink inside the workspace can still point outside it.
            resolved = os.path.normcase(os.path.normpath(str(candidate.resolve())))
            if not _is_same_or_child(resolved, root):
                return TargetScope.EXTERNAL
        return TargetScope.WORKSPACE
    except (OSError, ValueError):
        return TargetScope.UNKNOWN


def _is_same_or_child(value: str, parent: str) -> bool:
    try:
        return os.path.commonpath((value, parent)) == parent
    except ValueError:
        return False


def _normalize_root(root: str | None) -> str | None:
    if not root:
        return None
    try:
        return os.path.normcase(os.path.normpath(str(Path(root).resolve())))
    except OSError:
        return os.path.normcase(os.path.normpath(str(root)))


def _discards_output(target: str) -> bool:
    lowered = _bare(target).lower()
    return lowered in {"/dev/null", "$null", "nul", "$none"}


def _bare(token: object) -> str:
    return str(token).strip().strip("\"'")


def _dedupe(items: list) -> list:
    return list(dict.fromkeys(items))


# ---------------------------------------------------------------------------
# Persistent approval prefixes
# ---------------------------------------------------------------------------

#: Commands whose first non-flag argument selects a sub-command: a prefix must
#: keep it, otherwise "allow npm test" would allow every npm invocation.
_SUBCOMMAND_TOOLS = {
    "git", "npm", "pnpm", "yarn", "bun", "cargo", "go", "uv", "pip", "pip3",
    "poetry", "dotnet", "docker", "make", "ruff",
}


def derive_persistent_stage_prefixes(command: str, *, allow_scoped: bool = False) -> list[list[str]] | None:
    """One prefix per command, or None when no stable rule can be persisted.

    ``allow_scoped`` lets a user persist a command whose write/delete scope is
    recorded on the rule itself. Uploads, privileged wrappers, unknown
    executables and redirected commands still never become reusable grants.
    """
    parsed = parse_shell_command(command)
    if not parsed.complete or not parsed.commands:
        return None
    if _persistence_blocked(parsed, allow_scoped=allow_scoped):
        return None
    prefixes: list[list[str]] = []
    for invocation in parsed.commands:
        prefix = _stable_prefix(invocation)
        if prefix is None:
            return None
        prefixes.append(prefix)
    return prefixes or None


def command_matches_stage_prefixes(
    command: str, prefixes: list[list[str]], *, allow_scoped: bool = False
) -> bool:
    """Whether ``command`` is still the operation an allowlist rule approved."""
    parsed = parse_shell_command(command)
    if not parsed.complete or not parsed.commands:
        return False
    if _persistence_blocked(parsed, allow_scoped=allow_scoped):
        return False
    if len(parsed.commands) != len(prefixes):
        return False
    for invocation, prefix in zip(parsed.commands, prefixes):
        stable = _stable_prefix(invocation)
        if stable is None:
            return False
        wanted = [str(token).strip().strip("\"'").lower() for token in prefix]
        if not wanted:
            return False
        # The rule must be at least as specific as the command it now matches:
        # a legacy one-token rule cannot cover a sub-command invocation.
        if len(wanted) < len(stable):
            return False
        if stable != wanted[: len(stable)]:
            return False
    return True


def command_matches_prefix(command: str, prefix: list[str]) -> bool:
    """Legacy single-prefix rule: one prefix for every command in the line."""
    return command_matches_stage_prefixes(command, [prefix])


def _persistence_blocked(parsed: ShellParseResult, *, allow_scoped: bool = False) -> bool:
    """Shapes that must never become a reusable "allow" rule."""
    if parsed.redirections and any(
        not _discards_output(redirection.target) for redirection in parsed.redirections
    ):
        return True
    for invocation in parsed.commands:
        effects, traits, _ = _command_facts(invocation, None)
        if _stable_prefix(invocation) is None:
            return True
        if allow_scoped:
            # A user-chosen scoped rule may cover writes, deletes, git and
            # network reads. Uploads, privilege escalation and unknown
            # executables remain per-approval.
            if ShellTrait.NETWORK_UPLOAD in traits:
                return True
            if traits & {ShellTrait.PRIVILEGED, ShellTrait.UNKNOWN_EFFECT}:
                return True
            continue
        if effects & {
            ShellEffect.DELETE,
            ShellEffect.GIT_MUTATION,
            # A reusable rule records the command head, not where it writes or
            # what it sends: writes and network access stay per-approval so a
            # rule cannot silently widen into an external or exfiltrating run.
            ShellEffect.WRITE,
            ShellEffect.NETWORK,
        }:
            return True
        if traits & {ShellTrait.DESTRUCTIVE, ShellTrait.PRIVILEGED, ShellTrait.RECURSIVE}:
            return True
        # An interpreter runs code whose contents can change after the rule was
        # written. Even a named script path is therefore not a stable grant.
        if ShellTrait.UNKNOWN_EFFECT in traits:
            return True
    return False


def _stable_prefix(invocation: ParsedCommand) -> list[str] | None:
    """The narrowest stable head for an approval rule, or None if too broad."""
    exe = invocation.executable
    args = invocation.args
    if any(wrapper in _PRIVILEGED_WRAPPERS for wrapper in invocation.wrappers):
        return None
    if exe in _EVAL_COMMANDS:
        return None
    if exe in _PYTHON_COMMANDS:
        # A named module or a concrete script path is narrow enough to persist;
        # bare `python` or `python -c` would allow any program.
        if len(args) >= 2 and args[0] == "-m":
            return [exe, "-m", args[1].lower()]
        if args and _is_script_path(_bare(args[0])):
            return [exe, _bare(args[0]).replace("\\", "/")]
        return None
    if exe in _INTERPRETERS:
        return None
    if exe in _SUBCOMMAND_TOOLS:
        for token in args:
            bare = _bare(token)
            if bare.startswith("-"):
                continue
            return [exe, bare.lower()]
        return [exe]
    return [exe]


def _is_script_path(token: str) -> bool:
    if not token or token.startswith("-") or any(
        marker in token for marker in _DYNAMIC_PATH_MARKERS
    ):
        return False
    return "." in token.replace("\\", "/").rsplit("/", 1)[-1].lstrip(".")


# ---------------------------------------------------------------------------
# Verify / long-running hints (scheduling, not permission state)
# ---------------------------------------------------------------------------

def _normalize_command(command: str) -> str:
    return " ".join(str(command or "").strip().lower().split())


def is_verify_shell_command(command: str) -> bool:
    normalized = _normalize_command(command)
    return any(
        normalized == prefix or normalized.startswith(prefix + " ")
        for prefix in _PARALLEL_VERIFY_PREFIXES
    )


def is_long_running_shell_command(command: str) -> bool:
    lowered = _normalize_command(command)
    if not lowered:
        return False
    if _is_cd_prefixed_long_running_command(lowered):
        return True
    if contains_stateful_shell_operation(lowered):
        return False
    return _is_direct_long_running_command(lowered)


def contains_stateful_shell_operation(command: str) -> bool:
    patterns = (
        r"(?:^|[;&|]\s*)cd(?:\s|$)",
        r"\bset-location\b",
        r"(?:^|[;&|]\s*)export\s+",
        r"(?:^|[;&|]\s*)source\s+",
        r"(?:^|[;&|]\s*)set\s+",
        r"(?:^|[;&|]\s*)alias\s+",
        r"\bactivate\b",
        r"\bconda\s+activate\b",
    )
    return any(re.search(pattern, command) for pattern in patterns)


def _is_cd_prefixed_long_running_command(command: str) -> bool:
    match = re.match(r"^cd\s+[^;&|]+&&\s*(?P<inner>.+)$", command)
    return bool(match and _is_direct_long_running_command(match.group("inner").strip()))


def _is_direct_long_running_command(command: str) -> bool:
    if _is_obviously_not_long_running(command):
        return False
    patterns = (
        r"^npm\s+run\s+(dev|start)(?:\s|$)",
        r"^npm\s+start(?:\s|$)",
        r"^(pnpm|yarn|bun)\s+(dev|start)(?:\s|$)",
        r"^(pnpm|yarn|bun)\s+run\s+(dev|start)(?:\s|$)",
        r"^vite(?:\s|$)",
        r"^npx\s+vite(?:\s|$)",
        r"^next\s+(dev|start)(?:\s|$)",
        r"^npx\s+next\s+(dev|start)(?:\s|$)",
        r"^webpack\s+serve(?:\s|$)",
        r"^npx\s+webpack\s+serve(?:\s|$)",
        r"^python\s+manage\.py\s+runserver(?:\s|$)",
        r"^python3\s+manage\.py\s+runserver(?:\s|$)",
        r"^flask\s+run(?:\s|$)",
        r"^python\s+-m\s+flask\s+run(?:\s|$)",
        r"^uvicorn\s+[\w.: -]+",
        r"^python\s+-m\s+uvicorn\s+[\w.: -]+",
        r"^fastapi\s+(dev|run)(?:\s|$)",
        r"^python\s+-m\s+http\.server(?:\s|$)",
        r"^python3\s+-m\s+http\.server(?:\s|$)",
        r"^tsc\b.*\s--watch(?:\s|$)",
        r"^cargo\s+watch(?:\s|$)",
    )
    return any(re.search(pattern, command) for pattern in patterns)


def _is_obviously_not_long_running(command: str) -> bool:
    blocked_fragments = (
        "npm test", "npm run test", "pnpm test", "pnpm run test",
        "yarn test", "yarn run test", "bun test", "pytest",
        "python -m pytest", "python -m unittest", "go test", "cargo test",
        "npm install",
    )
    return any(fragment in command for fragment in blocked_fragments)
