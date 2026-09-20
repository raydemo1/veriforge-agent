"""Composition root for built-in tool registration."""

from __future__ import annotations

from ..execution_planner import CallEffect, ResourceClaim, workspace_claim
from ..permissions import (
    TOOL_PERMISSION_CONTROL,
    TOOL_PERMISSION_EDIT,
    TOOL_PERMISSION_NETWORK_READ,
    TOOL_PERMISSION_READ,
    TOOL_PERMISSION_SHELL,
)
from ..shell_classification import (
    ShellEffect,
    ShellTrait,
    TargetScope,
    analyze_shell_command,
    is_long_running_shell_command,
    is_verify_shell_command,
)
from ..tool_call_validation import strict_tool_schema
from ..tool_registry import (
    TOOL_CAPABILITY_MAIN,
    TOOL_CAPABILITY_READONLY_AGENT,
    TOOL_CAPABILITY_WORKER_AGENT,
    ToolRegistry,
)
from .agents import (
    apply_agent_changes,
    close_agent,
    followup_agent,
    interrupt_agent,
    list_agents,
    read_agent_changes,
    read_agent_conflicts,
    resolve_agent_conflicts,
    send_agent_message,
    spawn_agent,
    wait_agents,
)
from .browser import browser_test, stop_dev_server
from .discovery import tool_search
from .filesystem import (
    apply_patch,
    list_files,
    read_file,
    read_skill_file,
    repo_search,
    write_file,
)
from .interaction import ask_user
from .memory_tools import (
    memory_forget,
    memory_read,
    memory_search,
    memory_validate,
    memory_write,
)
from .schemas import BROWSER_TOOL_SCHEMAS, CORE_TOOL_SCHEMAS
from .shell import list_shell_jobs, read_shell_output, run_bash, stop_shell_job
from .todo import update_todo
from .web import web_fetch, web_search


def _build_builtin_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    schemas = {
        schema["function"]["name"]: schema
        for schema in CORE_TOOL_SCHEMAS + BROWSER_TOOL_SCHEMAS
    }
    main = {TOOL_CAPABILITY_MAIN}
    all_agents = main | {TOOL_CAPABILITY_READONLY_AGENT, TOOL_CAPABILITY_WORKER_AGENT}
    worker_agents = main | {TOOL_CAPABILITY_WORKER_AGENT}

    def add(name, handler, permission, effect=None, *, capabilities=main):
        registry.register(
            schemas[name],
            handler,
            permission=permission,
            effect=effect,
            capabilities=capabilities,
        )

    add(
        "read_file",
        read_file,
        TOOL_PERMISSION_READ,
        _path_effect("path", access="read"),
        capabilities=all_agents,
    )
    add(
        "read_skill_file",
        read_skill_file,
        TOOL_PERMISSION_READ,
        capabilities=all_agents,
    )
    add(
        "repo_search",
        repo_search,
        TOOL_PERMISSION_READ,
        _path_effect("path", access="read", scope="subtree", default="."),
        capabilities=all_agents,
    )
    add(
        "list_files",
        list_files,
        TOOL_PERMISSION_READ,
        _path_effect("directory", access="read", scope="subtree", default="."),
        capabilities=all_agents,
    )
    add("memory_search", memory_search, TOOL_PERMISSION_READ, capabilities=all_agents)
    add("memory_read", memory_read, TOOL_PERMISSION_READ, capabilities=all_agents)
    add(
        "web_search",
        web_search,
        TOOL_PERMISSION_NETWORK_READ,
        CallEffect(
            (ResourceClaim("network", "*", "global", "read"),),
            concurrency_key="network",
        ),
        capabilities=all_agents,
    )
    add(
        "web_fetch",
        web_fetch,
        TOOL_PERMISSION_NETWORK_READ,
        CallEffect(
            (ResourceClaim("network", "*", "global", "read"),),
            concurrency_key="network",
        ),
        capabilities=all_agents,
    )
    add(
        "run_bash",
        run_bash,
        TOOL_PERMISSION_SHELL,
        _shell_effect,
        capabilities=all_agents,
    )
    add("browser_test", browser_test, TOOL_PERMISSION_SHELL, capabilities=all_agents)
    add(
        "write_file",
        write_file,
        TOOL_PERMISSION_EDIT,
        _path_effect("path", access="write"),
        capabilities=worker_agents,
    )
    add(
        "apply_patch",
        apply_patch,
        TOOL_PERMISSION_EDIT,
        _path_effect("path", access="write"),
        capabilities=worker_agents,
    )

    add(
        "tool_search",
        tool_search,
        TOOL_PERMISSION_READ,
        CallEffect.global_exclusive(kind="control"),
    )
    add(
        "spawn_agent", spawn_agent, TOOL_PERMISSION_READ, CallEffect(kind="agent_spawn")
    )
    add(
        "send_agent_message",
        send_agent_message,
        TOOL_PERMISSION_READ,
        _agent_effect("agent_id", access="write"),
    )
    add(
        "followup_agent",
        followup_agent,
        TOOL_PERMISSION_READ,
        _agent_effect("agent_id", access="write"),
    )
    add(
        "wait_agents",
        wait_agents,
        TOOL_PERMISSION_READ,
        CallEffect(
            (ResourceClaim("agent-session", "*", "global", "read"),), kind="agent_wait"
        ),
    )
    add(
        "list_agents",
        list_agents,
        TOOL_PERMISSION_READ,
        CallEffect(
            (ResourceClaim("agent-session", "*", "global", "read"),), kind="agent_list"
        ),
    )
    add(
        "interrupt_agent",
        interrupt_agent,
        TOOL_PERMISSION_CONTROL,
        _agent_effect("agent_id", access="write"),
    )
    add(
        "read_agent_changes",
        read_agent_changes,
        TOOL_PERMISSION_READ,
        _agent_effect("proposal_id", access="read", domain="agent-proposal"),
    )
    add(
        "apply_agent_changes",
        apply_agent_changes,
        TOOL_PERMISSION_EDIT,
        _proposal_effect,
    )
    add(
        "read_agent_conflicts",
        read_agent_conflicts,
        TOOL_PERMISSION_READ,
        _agent_effect("conflict_id", access="read", domain="agent-conflict"),
    )
    add(
        "resolve_agent_conflicts",
        resolve_agent_conflicts,
        TOOL_PERMISSION_EDIT,
        _conflict_effect,
    )
    add(
        "close_agent",
        close_agent,
        TOOL_PERMISSION_CONTROL,
        _agent_effect("agent_id", access="write"),
    )
    add(
        "ask_user",
        ask_user,
        TOOL_PERMISSION_READ,
        CallEffect.global_exclusive(kind="interaction"),
    )
    add("memory_write", memory_write, TOOL_PERMISSION_EDIT)
    add("memory_validate", memory_validate, TOOL_PERMISSION_EDIT)
    add("memory_forget", memory_forget, TOOL_PERMISSION_EDIT)
    add(
        "update_todo",
        update_todo,
        TOOL_PERMISSION_CONTROL,
        CallEffect.global_exclusive(kind="control"),
    )
    add(
        "list_shell_jobs",
        list_shell_jobs,
        TOOL_PERMISSION_READ,
        CallEffect.global_exclusive(kind="control"),
    )
    add(
        "read_shell_output",
        read_shell_output,
        TOOL_PERMISSION_READ,
        CallEffect.global_exclusive(kind="control"),
    )
    add(
        "stop_shell_job",
        stop_shell_job,
        TOOL_PERMISSION_CONTROL,
        CallEffect.global_exclusive(kind="control"),
    )
    add("stop_dev_server", stop_dev_server, TOOL_PERMISSION_SHELL)
    return registry


def _root(context) -> str:
    return str(context.workspace.root) if context is not None else "."


def _path_effect(
    argument: str, *, access: str, scope: str = "exact", default: str = ""
):
    def resolve(args, context):
        return CallEffect(
            (
                workspace_claim(
                    _root(context),
                    args.get(argument) or default,
                    scope=scope,
                    access=access,
                ),
            )
        )

    return resolve


def _workspace_global(access: str):
    def resolve(_args, context):
        return CallEffect(
            (workspace_claim(_root(context), ".", scope="global", access=access),)
        )

    return resolve


def _agent_effect(argument: str, *, access: str, domain: str = "agent"):
    def resolve(args, _context):
        return CallEffect(
            (ResourceClaim(domain, str(args.get(argument) or "*"), "exact", access),)
        )

    return resolve


def _proposal_effect(args, context):
    coordinator = (
        getattr(context, "agent_coordinator", None) if context is not None else None
    )
    if coordinator is None:
        return CallEffect.global_exclusive(kind="agent_proposal")
    try:
        paths = coordinator.proposal_paths(str(args.get("proposal_id") or ""))
    except (KeyError, ValueError):
        return CallEffect.global_exclusive(kind="agent_proposal")
    return CallEffect(
        tuple(
            workspace_claim(_root(context), path, scope="exact", access="write")
            for path in paths
        ),
        kind="agent_proposal",
    )


def _conflict_effect(args, context):
    coordinator = (
        getattr(context, "agent_coordinator", None) if context is not None else None
    )
    if coordinator is None:
        return CallEffect.global_exclusive(kind="agent_conflict")
    try:
        paths = coordinator.conflict_paths(str(args.get("conflict_id") or ""))
    except (KeyError, ValueError):
        return CallEffect.global_exclusive(kind="agent_conflict")
    return CallEffect(
        tuple(
            workspace_claim(_root(context), path, scope="exact", access="write")
            for path in paths
        ),
        kind="agent_conflict",
    )


def _shell_effect(args, context):
    command = str(args.get("command") or "")
    root = _root(context)
    sandbox = getattr(
        getattr(context, "permission_policy", None), "sandbox_mode", None
    )
    analysis = analyze_shell_command(command, root, sandbox)
    effects = analysis.effects
    traits = analysis.traits
    scopes = {t.scope for t in analysis.targets}

    workspace_read = workspace_claim(root, ".", scope="global", access="read")

    if is_long_running_shell_command(command):
        return CallEffect.global_exclusive(kind="long_running")
    non_workspace = scopes & {
        TargetScope.SYSTEM,
        TargetScope.EXTERNAL,
        TargetScope.UNKNOWN,
    }
    if ShellTrait.DESTRUCTIVE in traits and non_workspace:
        return CallEffect.global_exclusive(kind="destructive")
    if ShellTrait.RECURSIVE in traits:
        return CallEffect.global_exclusive(kind="recursive_delete")
    if ShellTrait.UNKNOWN_EFFECT in traits:
        # Unknown programs may touch anything: run alone as a batch barrier.
        return CallEffect.global_exclusive(kind="unknown_execution")
    if ShellEffect.GIT_MUTATION in effects:
        # Git index lock + whole-repo state: serialize against other mutations.
        return CallEffect.global_exclusive(
            kind="git_remote_mutation" if ShellEffect.NETWORK in effects else "git_mutation"
        )
    if effects & {ShellEffect.WRITE, ShellEffect.DELETE}:
        if TargetScope.SYSTEM in scopes:
            return CallEffect.global_exclusive(kind="system_mutation")
        if TargetScope.EXTERNAL in scopes or TargetScope.UNKNOWN in scopes:
            return CallEffect.global_exclusive(kind="external_mutation")
        claims = _shell_target_write_claims(analysis, root)
        if not claims:
            claims = (workspace_claim(root, ".", scope="global", access="write"),)
        return CallEffect(claims, kind="workspace_mutation")
    if is_verify_shell_command(command):
        return CallEffect(
            (workspace_read, ResourceClaim("workspace:derived", "*", "global", "write")),
            kind="verify",
        )
    if ShellEffect.NETWORK in effects:
        claims = [ResourceClaim("network", "*", "global", "read"), workspace_read]
        return CallEffect(tuple(claims), kind="inspect")
    return CallEffect((workspace_read,), kind="inspect")


def _shell_target_write_claims(analysis, root) -> tuple[ResourceClaim, ...]:
    claims: list[ResourceClaim] = []
    for target in analysis.targets:
        if target.scope is not TargetScope.WORKSPACE or not target.raw:
            continue
        raw = target.raw.rstrip("/\\")
        claims.append(
            workspace_claim(
                root,
                raw or ".",
                scope="subtree" if _looks_like_directory_target(raw) else "exact",
                access="write",
            )
        )
    return tuple(claims)


def _looks_like_directory_target(raw: str) -> bool:
    tail = raw.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return "." not in tail


BUILTIN_TOOL_REGISTRY = _build_builtin_tool_registry()
TOOL_SCHEMAS = [strict_tool_schema(schema) for schema in CORE_TOOL_SCHEMAS]
