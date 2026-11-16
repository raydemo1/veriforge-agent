"""Composition root for built-in tool registration."""

from __future__ import annotations

import os

from ..execution_planner import (
    CallEffect,
    ResourceClaim,
    analyzed_shell_write_claims,
    workspace_claim,
)
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
)
from ..tool_call_validation import strict_tool_schema
from ..tool_registry import (
    BUILTIN_AGENTS_SCOPE,
    TOOL_CAPABILITY_MAIN,
    TOOL_CAPABILITY_READONLY_AGENT,
    TOOL_CAPABILITY_WORKER_AGENT,
    ToolRegistry,
)
from .agents import (
    close_agent,
    followup_agent,
    interrupt_agent,
    send_agent_message,
    send_parent_message,
    spawn_agent,
    wait_agents,
)
from .code_intelligence import code_intelligence
from .discovery import search_tools
from .filesystem import (
    apply_patch,
    read_file,
    read_skill_file,
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
from .runtime_status import get_runtime_status
from .schemas import CORE_TOOL_SCHEMAS
from .shell import list_shell_jobs, read_shell_output, run_bash, stop_shell_job
from .todo import update_todo
from .web import web_fetch, web_search


def _build_builtin_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    schemas = {schema["function"]["name"]: schema for schema in CORE_TOOL_SCHEMAS}
    main = {TOOL_CAPABILITY_MAIN}
    all_agents = main | {TOOL_CAPABILITY_READONLY_AGENT, TOOL_CAPABILITY_WORKER_AGENT}
    worker_agents = main | {TOOL_CAPABILITY_WORKER_AGENT}
    # Child-only: the main agent has no parent, so the upward channel is
    # absent from its registry. This is the fixed Main <-> Child topology.
    child_agents = {TOOL_CAPABILITY_READONLY_AGENT, TOOL_CAPABILITY_WORKER_AGENT}

    def add(name, handler, permission, effect=None, *, capabilities=main, disclosure="core", discovery_scope=None):
        registry.register(
            schemas[name],
            handler,
            permission=permission,
            effect=effect,
            capabilities=capabilities,
            disclosure=disclosure,
            discovery_scope=discovery_scope,
        )

    add("code_intelligence", code_intelligence, TOOL_PERMISSION_READ,
        lambda _args, context: CallEffect((workspace_claim(_root(context), ".", scope="global", access="read"),)),
        capabilities=all_agents)
    add("get_runtime_status", get_runtime_status, TOOL_PERMISSION_READ,
        CallEffect(barrier=True, kind="control"), capabilities=all_agents)

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
        "bash",
        run_bash,
        TOOL_PERMISSION_SHELL,
        _shell_effect,
        capabilities=all_agents,
    )
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
        "search_tools",
        search_tools,
        TOOL_PERMISSION_READ,
        CallEffect(barrier=True, kind="control"),
        capabilities=all_agents,
    )
    # Agent coordination is a demand-loaded surface: the main agent pays for
    # these schemas only after search_tools mounts them. send_parent_message is
    # the exception -- children always need their one upward channel.
    add(
        "spawn_agent", spawn_agent, TOOL_PERMISSION_READ, CallEffect(kind="agent_spawn"),
        disclosure="deferred",
        discovery_scope=BUILTIN_AGENTS_SCOPE,
    )
    add(
        "send_agent_message",
        send_agent_message,
        TOOL_PERMISSION_READ,
        _agent_effect("agent_id", access="write"),
        disclosure="deferred",
        discovery_scope=BUILTIN_AGENTS_SCOPE,
    )
    add(
        "send_parent_message",
        send_parent_message,
        TOOL_PERMISSION_READ,
        CallEffect((ResourceClaim("agent", "main_agent", "exact", "write"),)),
        capabilities=child_agents,
        discovery_scope=BUILTIN_AGENTS_SCOPE,
    )
    add(
        "followup_agent",
        followup_agent,
        TOOL_PERMISSION_READ,
        _agent_effect("agent_id", access="write"),
        disclosure="deferred",
        discovery_scope=BUILTIN_AGENTS_SCOPE,
    )
    add(
        "wait_agents",
        wait_agents,
        TOOL_PERMISSION_READ,
        CallEffect(
            (ResourceClaim("agent-session", "*", "global", "read"),), kind="agent_wait"
        ),
        disclosure="deferred",
        discovery_scope=BUILTIN_AGENTS_SCOPE,
    )
    add(
        "interrupt_agent",
        interrupt_agent,
        TOOL_PERMISSION_CONTROL,
        _agent_effect("agent_id", access="write"),
        disclosure="deferred",
        discovery_scope=BUILTIN_AGENTS_SCOPE,
    )
    add(
        "close_agent",
        close_agent,
        TOOL_PERMISSION_CONTROL,
        _agent_effect("agent_id", access="write"),
        disclosure="deferred",
        discovery_scope=BUILTIN_AGENTS_SCOPE,
    )
    add(
        "ask_user",
        ask_user,
        TOOL_PERMISSION_READ,
        CallEffect(barrier=True, kind="interaction"),
    )
    add("memory_write", memory_write, TOOL_PERMISSION_EDIT)
    add("memory_validate", memory_validate, TOOL_PERMISSION_EDIT)
    add("memory_forget", memory_forget, TOOL_PERMISSION_EDIT)
    add(
        "update_todo",
        update_todo,
        TOOL_PERMISSION_CONTROL,
        CallEffect(barrier=True, kind="control"),
    )
    add(
        "list_shell_jobs",
        list_shell_jobs,
        TOOL_PERMISSION_READ,
        CallEffect(barrier=True, kind="control"),
    )
    add(
        "read_shell_output",
        read_shell_output,
        TOOL_PERMISSION_READ,
        CallEffect(barrier=True, kind="control"),
    )
    add(
        "stop_shell_job",
        stop_shell_job,
        TOOL_PERMISSION_CONTROL,
        CallEffect(barrier=True, kind="control"),
    )
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


def _agent_effect(argument: str, *, access: str, domain: str = "agent"):
    def resolve(args, _context):
        return CallEffect(
            (ResourceClaim(domain, str(args.get(argument) or "*"), "exact", access),)
        )

    return resolve


def shell_effect(args, context, *, shell_analysis=None, background=None):
    root = _root(context)
    effect = _analyzed_shell_effect(
        args, root, shell_analysis=shell_analysis, background=background
    )
    if os.name == "nt":
        # The shell-job manager owns the concrete write lease on Windows; this
        # effect only orders calls inside one batch and must not make the
        # executor claim a project scope the manager did not.
        return CallEffect(
            effect.resources,
            barrier=effect.barrier,
            concurrency_key=effect.concurrency_key,
            kind="native_shell",
        )
    return effect


def _shell_effect(args, context):
    return shell_effect(args, context)


def _analyzed_shell_effect(args, root, *, shell_analysis=None, background=None):
    command = str(args.get("command") or "")
    # Reuse the preparation-time analysis when the executor supplied one.
    analysis = shell_analysis or analyze_shell_command(command, root)
    effects = analysis.effects
    traits = analysis.traits
    scopes = analysis.scopes

    workspace_read = workspace_claim(root, ".", scope="global", access="read")
    workspace_write = workspace_claim(root, ".", scope="global", access="write")

    background = (
        is_long_running_shell_command(command) if background is None else bool(background)
    )
    if background:
        # A background job may write anywhere in the workspace but must not be
        # an unconditional batch barrier: unrelated tools can still run.
        return CallEffect((workspace_write,), kind="long_running")
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
        # Unknown programs may touch anything: keep them mutually serial, but
        # only against workspace claims instead of every unrelated tool.
        return CallEffect((workspace_write,), kind="unknown_execution")
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
        claims = analyzed_shell_write_claims(root, command, analysis=analysis)
        if not claims:
            if analysis.write_targets and all(
                target.scope is TargetScope.TEMP for target in analysis.write_targets
            ):
                # The private per-call temp directory is disposable and not part
                # of the workspace, so no workspace lease is needed.
                return CallEffect((workspace_read,), kind="temp_mutation")
            claims = (workspace_claim(root, ".", scope="global", access="write"),)
        return CallEffect(claims, kind="workspace_mutation")
    if ShellTrait.VERIFY in traits:
        return CallEffect(
            (workspace_read, ResourceClaim("workspace:derived", "*", "global", "write")),
            kind="verify",
        )
    if ShellEffect.NETWORK in effects:
        claims = [ResourceClaim("network", "*", "global", "read"), workspace_read]
        return CallEffect(tuple(claims), kind="inspect")
    return CallEffect((workspace_read,), kind="inspect")


BUILTIN_TOOL_REGISTRY = _build_builtin_tool_registry()
TOOL_SCHEMAS = [strict_tool_schema(schema) for schema in CORE_TOOL_SCHEMAS]
