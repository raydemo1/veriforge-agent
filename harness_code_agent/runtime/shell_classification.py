"""Shell command analysis: facts only, never decisions.

The analyzer answers three questions about a shell command:

* ``effects``   — what kinds of things it does (read / write / delete / ...)
* ``traits``    — which shapes it has (recursive / destructive / dynamic / ...)
* ``targets``   — which paths it touches and in which scope

It never returns allow/ask/deny and never rates risk. Permission decisions are
the sole responsibility of :mod:`runtime.permissions`.
"""
from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path


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
    PRIVILEGED = "privileged"
    DYNAMIC = "dynamic"
    UNKNOWN_EFFECT = "unknown_effect"


class TargetScope(str, Enum):
    WORKSPACE = "workspace"
    EXTERNAL = "external"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class SandboxMode(str, Enum):
    HOST = "host"
    DOCKER = "docker"


@dataclass(frozen=True)
class ShellTarget:
    raw: str
    scope: TargetScope


@dataclass(frozen=True)
class ShellInvocation:
    wrappers: tuple[str, ...]
    executable: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class ShellAnalysis:
    effects: frozenset[ShellEffect]
    traits: frozenset[ShellTrait]
    targets: tuple[ShellTarget, ...] = ()


# ---------------------------------------------------------------------------
# Command vocabularies
# ---------------------------------------------------------------------------

_INTERPRETERS = {
    "python", "python3", "py", "node", "nodejs", "deno", "bun",
    "ruby", "perl", "php", "pwsh", "powershell", "bash", "sh", "zsh",
    "fish", "cmd", "dotnet",
}
_PYTHON_COMMANDS = {"python", "python3", "py"}
_INTERP_EXEC_FLAGS = {
    "-c", "-e", "--eval", "/c", "/k", "-command", "--command",
    "-encodedcommand", "-enc",
}
_INTERP_SCRIPT_SUFFIXES = (
    ".py", ".js", ".mjs", ".cjs", ".ts", ".ps1", ".sh", ".bash", ".zsh",
    ".bat", ".cmd", ".rb", ".pl", ".php",
)
_EVAL_COMMANDS = {"eval", "exec", "invoke-expression", "iex"}

_WRAPPERS = {"sudo", "command", "nohup", "time", "nice", "stdbuf", "ionice"}
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

_DELETE_COMMANDS = {"rm", "rm.exe", "del", "erase", "remove-item", "ri"}
_RECURSE_DIR_COMMANDS = {"rmdir", "rd"}
_PS_DELETE_ALIASES = {
    "set-content", "add-content", "out-file", "tee-object", "tee",
    "copy-item", "move-item", "rename-item", "export-csv", "export-clixml",
    "new-item", "ni", "clear-content", "clear-item",
}
_COPY_COMMANDS = {"cp", "copy", "mv", "move", "rename", "ren", "ln", "xcopy", "robocopy"}
_MKDIR_COMMANDS = {"mkdir", "md", "new-item", "ni", "touch"}
_FORMATTERS = {"ruff", "black", "gofmt", "prettier", "isort", "autopep8", "yapf"}
_NETWORK_TOOLS = {"curl", "curl.exe", "wget", "wget.exe", "Invoke-WebRequest", "iwr", "invoke-restmethod", "irm"}
_PROCESS_CONTROL = {
    "kill", "pkill", "taskkill", "stop-process", "shutdown", "reboot",
    "poweroff", "halt", "restart-computer", "chmod", "chown", "icacls",
    "takeown", "reg", "schtasks", "sc", "set-executionpolicy",
}
_READ_COMMANDS = {
    "cat", "type", "ls", "dir", "pwd", "whoami", "id", "uname", "grep",
    "rg", "head", "tail", "echo", "test", "diff", "wc", "md5sum",
    "sha1sum", "sha256sum", "shasum", "which", "where", "env", "printenv",
    "true", "false", "date", "hostname", "stat", "file", "cut", "uniq",
    "tr", "basename", "dirname", "realpath", "readlink", "jq", "sort",
    "find", "get-content", "gc", "get-childitem", "gci", "get-item",
    "get-location", "get-command", "gcm", "get-process", "gps",
    "get-service", "test-path", "resolve-path", "select-string", "sls",
    "where-object", "select-object", "get-member", "gm", "sort-object",
    "measure-object", "measure", "format-table", "ft", "write-output",
    "out-string", "convertfrom-json", "convertto-json", "split-path",
    "join-path", "compare-object", "compare",
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
# DESTRUCTIVE trait (and a synthetic system target when the shape itself is
# host-wide); the policy decides what that means.
_HOST_DESTRUCTIVE_PATTERNS = (
    r"\bmkfs(?:\.\w+)?\b",
    r"\bdiskpart\b",
    r"\bformat\s+[a-z]:",
    r"\bdd\b[^\n]*\bof=/dev/",
    r"\bcipher\s+/w\b",
    r"\bbcdedit\b",
    r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
    r"\bRemove-Computer\b",
    r"\bClear-Disk\b",
    r"\bInitialize-Disk\b",
)

_SYSTEM_PATH_FRAGMENTS = (
    "/etc", "/usr", "/bin", "/sbin", "/var", "/boot", "/sys", "/proc",
    "/lib", "/lib64", "/root", "/dev",
)
_WINDOWS_SYSTEM_FRAGMENTS = ("c:\\windows", "c:\\program files", "c:\\programdata")
_HOME_MARKERS = ("~", "$home", "${home}", "%userprofile%", "%homepath%")
_WINDIR_MARKERS = ("%windir%", "%systemroot%")

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

@lru_cache(maxsize=1024)
def analyze_shell_command(
    command: str,
    workspace_root: str | None = None,
    sandbox_mode: str = SandboxMode.HOST.value,
) -> ShellAnalysis:
    """Return factual effects/traits/targets for a (possibly compound) command."""
    text = str(command or "").strip()
    if not text:
        return ShellAnalysis(frozenset(), frozenset())

    root = _normalize_root(workspace_root)
    try:
        sandbox = SandboxMode(sandbox_mode)
    except ValueError:
        sandbox = SandboxMode.HOST

    effects: set[ShellEffect] = set()
    traits: set[ShellTrait] = set()
    targets: list[ShellTarget] = []

    if _has_command_substitution(text):
        effects.add(ShellEffect.EXECUTE)
        traits |= {ShellTrait.DYNAMIC, ShellTrait.UNKNOWN_EFFECT}

    for segment in _split_segments(text):
        seg_effects, seg_traits, seg_targets = _analyze_segment(segment, root, sandbox)
        effects |= seg_effects
        traits |= seg_traits
        targets.extend(seg_targets)

    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _HOST_DESTRUCTIVE_PATTERNS):
        traits.add(ShellTrait.DESTRUCTIVE)
        # Host-wide destructive shapes have no single path target; mark the
        # whole host so the catastrophic guardrail can fire.
        if not any(t.scope is TargetScope.SYSTEM for t in targets):
            targets.append(ShellTarget("/", TargetScope.SYSTEM))

    return ShellAnalysis(
        frozenset(effects),
        frozenset(traits),
        tuple(_dedupe(targets)),
    )


# ---------------------------------------------------------------------------
# Segment analysis
# ---------------------------------------------------------------------------

def _analyze_segment(
    segment: str, root: str | None, sandbox: SandboxMode
) -> tuple[set[ShellEffect], set[ShellTrait], list[ShellTarget]]:
    effects: set[ShellEffect] = set()
    traits: set[ShellTrait] = set()
    targets: list[ShellTarget] = []

    invocation = _parse_invocation(segment)
    if invocation is None:
        redir_targets = _redirection_targets(segment, root, sandbox)
        if redir_targets:
            effects.add(ShellEffect.WRITE)
            targets.extend(redir_targets)
        return effects, traits, targets

    exe = invocation.executable
    args = invocation.args

    for wrapper in invocation.wrappers:
        if wrapper in _PRIVILEGED_WRAPPERS:
            traits.add(ShellTrait.PRIVILEGED)

    # Redirection targets apply regardless of the executable.
    redir_targets = _redirection_targets(segment, root, sandbox)
    if redir_targets:
        effects.add(ShellEffect.WRITE)
        targets.extend(redir_targets)

    # Vocabulary-based classification.
    if exe in _DELETE_COMMANDS:
        windows_switches = exe in {"del", "erase"}
        e, tr, tg = _delete_facts(
            args, root, sandbox, force_flag=True, windows_switches=windows_switches
        )
        effects |= e
        traits |= tr
        targets.extend(tg)
    elif exe in _RECURSE_DIR_COMMANDS:
        e, tr, tg = _delete_facts(
            args, root, sandbox, force_flag=False,
            always_recursive=True, windows_switches=True,
        )
        effects |= e
        traits |= tr
        targets.extend(tg)
    elif exe == "git":
        e, tr, tg = _git_facts(args, root, sandbox)
        effects |= e
        traits |= tr
        targets.extend(tg)
    elif exe in _INTERPRETERS:
        e, tr, tg = _interpreter_facts(exe, args, root, sandbox)
        effects |= e
        traits |= tr
        targets.extend(tg)
    elif exe in _NETWORK_TOOLS:
        e, tg = _network_facts(exe, args, root, sandbox)
        effects |= e
        targets.extend(tg)
    elif exe in _MKDIR_COMMANDS:
        effects.add(ShellEffect.WRITE)
        targets.extend(_positional_targets(args, root, sandbox))
    elif exe in _PS_DELETE_ALIASES or exe in _COPY_COMMANDS:
        effects.add(ShellEffect.WRITE)
        targets.extend(_positional_targets(args, root, sandbox, powershell_path_options=True))
    elif _looks_like_inplace_formatter(exe, args):
        effects.add(ShellEffect.WRITE)
        targets.extend(_formatter_targets(exe, args, root, sandbox))
    elif exe in _PROCESS_CONTROL:
        effects.add(ShellEffect.EXECUTE)
        traits.add(ShellTrait.UNKNOWN_EFFECT)
    elif exe in _PACKAGE_MANAGERS:
        if _package_mutates(exe, args):
            effects |= {ShellEffect.EXECUTE, ShellEffect.NETWORK}
            traits.add(ShellTrait.UNKNOWN_EFFECT)
        else:
            effects.add(ShellEffect.READ)
    elif is_verify_shell_command(segment):
        effects.add(ShellEffect.READ)
    elif exe in _READ_COMMANDS:
        effects.add(ShellEffect.READ)
    else:
        # Unrecognized executable: we cannot prove what it does.
        effects.add(ShellEffect.EXECUTE)
        traits.add(ShellTrait.UNKNOWN_EFFECT)

    return effects, traits, targets


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------

def _delete_facts(args, root, sandbox, *, force_flag: bool,
                  always_recursive: bool = False, windows_switches: bool = False):
    effects = {ShellEffect.DELETE}
    traits: set[ShellTrait] = set()
    targets: list[ShellTarget] = []
    recursive = always_recursive or _is_recursive_delete(args, force_flag=force_flag)
    positions = _positional_tokens(
        args,
        skip_slash_options=windows_switches,
        powershell_path_options=not windows_switches,
    )
    for raw in positions:
        scope = _classify_path_scope(raw, root, sandbox)
        targets.append(ShellTarget(raw, scope))
    if recursive and positions:
        traits.add(ShellTrait.RECURSIVE)
        traits.add(ShellTrait.DESTRUCTIVE)
    return effects, traits, targets


def _is_recursive_delete(args, *, force_flag: bool) -> bool:
    for token in args:
        bare = _bare(token).lower()
        if bare in {"-r", "-rf", "-fr", "-r --force", "--recursive", "/s"}:
            return True
        if bare.startswith("-") and "r" in bare.strip("-") and not any(
            ch in bare for ch in ("d", "v")
        ):
            # -rf / -fr / -R etc., but not pure option words like --remove
            head = bare.lstrip("-")
            if head and set(head) <= {"r", "f", "v", "p", "i"}:
                return True
        if force_flag and _is_powershell_switch(token, ("recurse", "r")):
            return True
    return False


def _git_facts(args, root, sandbox):
    effects: set[ShellEffect] = set()
    traits: set[ShellTrait] = set()
    targets: list[ShellTarget] = []
    if not args:
        return effects, traits, targets

    sub_index = _git_subcommand_index(args)
    if sub_index is None:
        return effects, traits, targets
    sub = args[sub_index]
    sub_args = args[sub_index + 1 :]
    flags = {_bare(a).lower() for a in sub_args}

    if sub in _GIT_NETWORK_SUB:
        effects |= {ShellEffect.NETWORK, ShellEffect.GIT_MUTATION}
        if any(f in {"-f", "--force", "--force-with-lease"} for f in flags):
            traits.add(ShellTrait.DESTRUCTIVE)
        if sub == "clone":
            targets.append(ShellTarget(_git_clone_target(sub_args), TargetScope.WORKSPACE))
        return effects, traits, targets

    if sub in _GIT_READ_SUB:
        effects.add(ShellEffect.READ)
        return effects, traits, targets
    if sub == "remote" and (not sub_args or sub_args[0] in {"-v", "--verbose", "show", "get-url"}):
        effects.add(ShellEffect.READ)
        return effects, traits, targets

    if sub in _GIT_LOCAL_MUTATE_SUB:
        effects.add(ShellEffect.GIT_MUTATION)
        targets.append(ShellTarget(".", TargetScope.WORKSPACE))
    if sub == "reset" and "--hard" in flags:
        traits.add(ShellTrait.DESTRUCTIVE)
    if sub == "clean" and _git_clean_forces_directory_delete(sub_args):
        traits.add(ShellTrait.DESTRUCTIVE)
    if sub == "restore" and _git_restore_uses_source(sub_args):
        traits.add(ShellTrait.DESTRUCTIVE)
    if sub == "checkout" and "--" in flags:
        traits.add(ShellTrait.DESTRUCTIVE)
    if sub == "branch" and any(_bare(a) == "-D" for a in sub_args):
        traits.add(ShellTrait.DESTRUCTIVE)
    return effects, traits, targets


def _git_clone_target(sub_args) -> str:
    for token in sub_args:
        if token.startswith("-"):
            continue
        if "://" in token or token.endswith(".git") or ":" in token:
            continue
        return token
    return "."


def _interpreter_facts(exe, args, root, sandbox):
    # Test runners and builds are treated as reads (derived artifacts only).
    if is_verify_shell_command(" ".join((exe, *args))):
        return {ShellEffect.READ}, set(), []
    if args and _bare(args[0]).lower() in {"--version", "-v", "-v", "-V"}:
        return {ShellEffect.READ}, set(), []

    traits = {ShellTrait.DYNAMIC, ShellTrait.UNKNOWN_EFFECT}
    effects = {ShellEffect.EXECUTE}
    targets: list[ShellTarget] = []

    script_path: str | None = None
    if args:
        first = _bare(args[0])
        if first in _INTERP_EXEC_FLAGS or first in _EVAL_COMMANDS:
            return effects, traits, targets
        if not first.startswith("-"):
            script_path = first
        elif exe in _PYTHON_COMMANDS and first == "-m":
            # `python -m pytest` is covered by verify above; other modules run.
            return effects, traits, targets
    if script_path is None:
        return effects, traits, targets
    low = script_path.lower()
    if low.endswith(_INTERP_SCRIPT_SUFFIXES) or "." not in low.rsplit("/", 1)[-1]:
        targets.append(
            ShellTarget(script_path, _classify_path_scope(script_path, root, sandbox))
        )
    return effects, traits, targets


def _network_facts(exe, args, root, sandbox):
    effects = {ShellEffect.NETWORK, ShellEffect.READ}
    targets: list[ShellTarget] = []
    output_path = _option_value(args, ("-o", "--output"))
    if output_path:
        effects.add(ShellEffect.WRITE)
        targets.append(
            ShellTarget(output_path, _classify_path_scope(output_path, root, sandbox))
        )
    if any(a in {"-O", "--remote-name"} for a in args):
        effects.add(ShellEffect.WRITE)
    return effects, targets


def _package_mutates(exe: str, args) -> bool:
    if exe in {"go"} and args and args[0] in {"get", "install"}:
        return True
    if not args:
        return False
    return args[0] in _PACKAGE_MUTATE_SUB


# ---------------------------------------------------------------------------
# Invocation parsing / wrapper normalization
# ---------------------------------------------------------------------------

_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*=")


def _parse_invocation(segment: str) -> ShellInvocation | None:
    tokens = _shell_words(segment)
    if not tokens:
        return None
    wrappers: list[str] = []
    index = 0
    # Leading env assignments: FOO=bar command ...
    while index < len(tokens) and _ASSIGNMENT_RE.match(tokens[index]):
        index += 1
    # Wrapper commands (sudo, env, nohup, ...).
    while index < len(tokens):
        name = _executable_name(tokens[index]).lower()
        if name in _WRAPPERS:
            wrappers.append(name)
            index += 1
            # `env KEY=VALUE cmd` — drop assignments between env and the command.
            if name == "env":
                while index < len(tokens) and _ASSIGNMENT_RE.match(tokens[index]):
                    index += 1
            # sudo flags (-E, -u user, ...) — skip flag tokens and their values.
            while index < len(tokens) and tokens[index].startswith("-"):
                token = tokens[index]
                index += 1
                if token in {"-u", "-g", "--user", "--group"} and index < len(tokens):
                    index += 1
            continue
        break
    if index >= len(tokens):
        return None
    executable = _executable_name(tokens[index]).lower()
    return ShellInvocation(tuple(wrappers), executable, tuple(tokens[index + 1 :]))


def _executable_name(token: str) -> str:
    return token.replace("\\", "/").rsplit("/", 1)[-1].strip("'\"")


def _shell_words(text: str) -> list[str]:
    # posix=False preserves Windows backslashes in paths (C:\Windows).
    try:
        return [_bare(token) for token in shlex.split(text, posix=False)]
    except ValueError:
        return text.split()


def _bare(token: object) -> str:
    return str(token).strip().strip("\"'")


# ---------------------------------------------------------------------------
# Token / git helpers
# ---------------------------------------------------------------------------

def _positional_tokens(
    args,
    *,
    skip_slash_options: bool = False,
    powershell_path_options: bool = False,
) -> list[str]:
    positions: list[str] = []
    skip_next = False
    path_opts = {"-path", "-literalpath", "--output", "-o"} if powershell_path_options else set()
    for token in args:
        bare = _bare(token)
        if skip_next:
            skip_next = False
            continue
        if bare in path_opts:
            skip_next = True
            continue
        if bare.startswith("-"):
            continue
        if skip_slash_options and bare.startswith("/") and _looks_like_windows_switch(bare):
            continue
        positions.append(bare)
    return positions


def _positional_targets(args, root, sandbox, *, powershell_path_options=False):
    return [
        ShellTarget(raw, _classify_path_scope(raw, root, sandbox))
        for raw in _positional_tokens(
            args,
            skip_slash_options=True,
            powershell_path_options=powershell_path_options,
        )
    ]


def _option_value(args, names: tuple[str, ...]) -> str | None:
    for index, token in enumerate(args):
        bare = _bare(token)
        if bare in names and index + 1 < len(args):
            return _bare(args[index + 1])
        for name in names:
            if bare.startswith(name + "="):
                return bare.split("=", 1)[1]
    return None


def _looks_like_windows_switch(token: str) -> bool:
    return bool(re.fullmatch(r"/[a-zA-Z]+", token))


def _is_powershell_switch(token: str, names: tuple[str, ...]) -> bool:
    bare = _bare(token).lower().lstrip("-")
    return bare in names


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


def _formatter_targets(exe, args, root, sandbox):
    filtered = list(args)
    if exe == "ruff" and filtered and filtered[0] == "check":
        filtered = filtered[1:]
    return _positional_targets(filtered, root, sandbox)


def _git_subcommand_index(tokens) -> int | None:
    for index, token in enumerate(tokens):
        bare = _bare(token)
        if bare.startswith("-"):
            continue
        return index
    return None


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

def _classify_path_scope(
    raw_path: str,
    root: str | None,
    sandbox: SandboxMode,
) -> TargetScope:
    token = _bare(raw_path).strip()
    if not token:
        return TargetScope.WORKSPACE
    low = token.replace("\\", "/").lower().rstrip("/").rstrip("*").rstrip("/")
    if token in {"/", "/*"}:
        return TargetScope.SYSTEM
    if low in {"", "."}:
        return TargetScope.WORKSPACE

    # Home / Windows env markers are always user-level system targets.
    if token.lower() in {"~", "$home", "${home}", "%userprofile%"}:
        return TargetScope.SYSTEM
    for marker in _HOME_MARKERS:
        lowered_marker = marker.lower()
        if low == lowered_marker or low.startswith(lowered_marker.rstrip("%") + "/"):
            return TargetScope.SYSTEM
    for marker in _WINDIR_MARKERS:
        if marker.lower() in low:
            return TargetScope.SYSTEM

    # POSIX system paths.
    if low in _SYSTEM_PATH_FRAGMENTS or any(
        low.startswith(fragment + "/") for fragment in _SYSTEM_PATH_FRAGMENTS
    ):
        return TargetScope.SYSTEM

    # Windows drive roots / system directories.
    if re.match(r"^[a-z]:(?:/|$)", low):
        for fragment in (f.replace("\\", "/") for f in _WINDOWS_SYSTEM_FRAGMENTS):
            if low.startswith(fragment):
                return TargetScope.SYSTEM
        if low.endswith(":"):
            return TargetScope.SYSTEM
        if root:
            return _scope_against_root(token, root, sandbox)
        return TargetScope.EXTERNAL

    # POSIX-style absolute path.
    if token.startswith("/"):
        if root:
            return _scope_against_root(token, root, sandbox)
        return (
            TargetScope.WORKSPACE
            if sandbox is SandboxMode.DOCKER
            else TargetScope.EXTERNAL
        )

    expanded = Path(token).expanduser()
    if expanded.is_absolute():
        if root:
            return _scope_against_root(token, root, sandbox)
        return (
            TargetScope.WORKSPACE
            if sandbox is SandboxMode.DOCKER
            else TargetScope.EXTERNAL
        )

    return TargetScope.WORKSPACE


def _scope_against_root(token: str, root: str, sandbox: SandboxMode) -> TargetScope:
    try:
        candidate = Path(token).expanduser()
        if not candidate.is_absolute():
            candidate = Path(root) / candidate
        resolved = os.path.normcase(os.path.normpath(str(candidate.resolve())))
        if _is_same_or_child(resolved, root):
            return TargetScope.WORKSPACE
        if sandbox is SandboxMode.DOCKER and not re.match(r"^[a-z]:[\\/]", token.lower()):
            return TargetScope.WORKSPACE
    except (OSError, ValueError):
        pass
    return TargetScope.EXTERNAL


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


# ---------------------------------------------------------------------------
# Redirections / substitution
# ---------------------------------------------------------------------------

def _redirection_targets(segment: str, root, sandbox) -> list[ShellTarget]:
    targets: list[ShellTarget] = []
    for token in _write_redirection_targets(segment):
        targets.append(ShellTarget(token, _classify_path_scope(token, root, sandbox)))
    return targets


def _write_redirection_targets(command: str) -> list[str]:
    found: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(command):
        char = command[index]
        if quote:
            if char == quote:
                quote = None
            elif char == "\\" and quote == '"' and index + 1 < len(command):
                index += 1
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == ">" and not _redirection_at_is_safe(command, index):
            rest = command[index + 1 :].lstrip()
            if rest.startswith(">"):
                rest = rest[1:].lstrip()
            match = re.match(r"['\"]?([^\s;&|]+)", rest)
            if match:
                found.append(match.group(1).strip("'\""))
        index += 1
    return found


def _redirection_at_is_safe(command: str, index: int) -> bool:
    # 2>&1 / >/dev/null / > $null / numeric stream merges.
    rest = command[index + 1 :].lstrip()
    if rest.startswith("&"):
        return True
    match = re.match(r"['\"]?([^\s;&|]+)", rest)
    if match:
        target = match.group(1).strip("'\"").lower()
        if target in {"/dev/null", "$null", "nul"}:
            return True
    prefix = command[:index].rstrip()
    return bool(re.search(r"\d$", prefix)) and rest.startswith("&")


def _has_command_substitution(command: str) -> bool:
    return "$(" in command or "`" in command


# ---------------------------------------------------------------------------
# Segment splitting
# ---------------------------------------------------------------------------

def _split_segments(command: str) -> list[str]:
    """Split on ;, &&, ||, | outside quotes into independently analyzed parts."""
    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    index = 0
    text = command.strip()
    while index < len(text):
        char = text[index]
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            elif char == "\\" and quote == '"' and index + 1 < len(text):
                index += 1
                current.append(text[index])
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
        elif char == ";":
            _push_segment(segments, "".join(current))
            current = []
        elif char in {"&", "|"}:
            double = index + 1 < len(text) and text[index + 1] == char
            # `&>`/`&>>` merged redirection and `2>&1`-style fd duplication
            # are not command separators.
            redirected = char == "&" and (
                (index + 1 < len(text) and text[index + 1] == ">")
                or bool(current) and current[-1] == ">"
            )
            if redirected:
                current.append(char)
            elif double:
                _push_segment(segments, "".join(current))
                current = []
                index += 1
            else:
                _push_segment(segments, "".join(current))
                current = []
        else:
            current.append(char)
        index += 1
    _push_segment(segments, "".join(current))
    return segments


def _push_segment(segments: list[str], raw: str) -> None:
    piece = raw.strip()
    if piece:
        segments.append(piece)


def _dedupe(items: list) -> list:
    return list(dict.fromkeys(items))


# ---------------------------------------------------------------------------
# Approval prefixes (persisted allowlist rules)
# ---------------------------------------------------------------------------

def derive_persistent_prefix(command: str) -> list[str] | None:
    """Return one stable approval prefix for a simple command or compound."""
    segments = _split_simple_compound(command)
    if len(segments) > 1:
        prefixes = [
            prefix
            for segment in segments
            if not _is_literal_expression(segment)
            for prefix in [_derive_single_prefix(segment)]
        ]
        if prefixes and all(prefix == prefixes[0] for prefix in prefixes):
            return prefixes[0]
        return None
    return _derive_single_prefix(command)


def command_matches_prefix(command: str, prefix: list[str]) -> bool:
    """Match every meaningful segment against one normalized approval prefix."""
    segments = _split_simple_compound(command)
    if not segments:
        return False
    matched = False
    for segment in segments:
        if _is_literal_expression(segment):
            continue
        tokens = [
            _normalize_token(token) for token in _tokenize_approval_command(segment)
        ]
        if not tokens or tokens[: len(prefix)] != prefix:
            return False
        matched = True
    return matched


def _derive_single_prefix(command: str) -> list[str] | None:
    tokens = _tokenize_approval_command(command)
    if len(tokens) < 2:
        return None
    normalized = [_normalize_token(token) for token in tokens]
    python_index = next(
        (index for index, token in enumerate(normalized) if token in _PYTHON_COMMANDS),
        None,
    )
    if python_index is not None:
        if len(normalized) <= python_index + 1:
            return None
        launcher = normalized[python_index + 1]
        if launcher in {"-", "-c", "-i"}:
            return None
        if launcher == "-m":
            if len(normalized) <= python_index + 2:
                return None
            return normalized[: python_index + 3]
        if launcher.startswith("-"):
            return None
        return normalized[: python_index + 2]
    return normalized[: min(3, len(normalized))]


def _split_simple_compound(command: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quote = ""
    text = command.strip()
    index = 0
    while index < len(text):
        char = text[index]
        if quote:
            current.append(char)
            if char == quote:
                quote = ""
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
        elif char == ";" or (
            char == "&" and index + 1 < len(text) and text[index + 1] == "&"
        ):
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            if char == "&":
                index += 1
        elif char in "|&<>":
            return []
        else:
            current.append(char)
        index += 1
    if quote:
        return []
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _tokenize_approval_command(command: str) -> list[str]:
    if not command.strip() or re.search(r"[|;&<>]", command):
        return []
    try:
        return shlex.split(command, posix=False)
    except ValueError:
        return []


def _is_literal_expression(command: str) -> bool:
    text = command.strip()
    return len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}


def _normalize_token(token: object) -> str:
    return str(token).strip().strip("\"'").lower()


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


def is_workspace_write_shell_command(command: str, workspace_root: str | None = None) -> bool:
    """Boundary used by read-only review mode."""
    analysis = analyze_shell_command(command, workspace_root)
    if ShellEffect.GIT_MUTATION in analysis.effects and ShellEffect.NETWORK not in analysis.effects:
        return True
    if not (analysis.effects & {ShellEffect.WRITE, ShellEffect.DELETE}):
        return False
    return any(target.scope is TargetScope.WORKSPACE for target in analysis.targets)


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
