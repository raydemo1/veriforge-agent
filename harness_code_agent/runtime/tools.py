# ruff: noqa: F401  -- this module is a deliberate re-export facade
"""Compatibility facade for the legacy runtime.tools module.

Production code should import narrower modules such as runtime.tool_registry,
runtime.tool_runner, or runtime.builtins.*. This module keeps the historical
public surface available for tests, users, and older integrations.
"""
from __future__ import annotations

import os

from .. import config
from .builtins.agents import (
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
from .builtins.browser import browser_test, stop_dev_server
from .builtins.discovery import tool_search
from .builtins.filesystem import (
    READ_FILE_MAX_LINES,
    READ_FILE_MAX_OUTPUT_TOKENS,
    _resolve,
    apply_patch,
    list_files,
    read_file,
    read_skill_file,
    repo_search,
    write_file,
)
from .builtins.interaction import ask_user
from .builtins.memory_tools import (
    memory_forget,
    memory_read,
    memory_search,
    memory_validate,
    memory_write,
)
from .builtins.registry import BUILTIN_TOOL_REGISTRY, TOOL_SCHEMAS
from .builtins.schemas import BROWSER_TOOL_SCHEMAS, CORE_TOOL_SCHEMAS
from .builtins.shell import (
    _build_shell_output,
    list_shell_jobs,
    read_shell_output,
    run_bash,
    stop_shell_job,
)
from .builtins.todo import update_todo
from .builtins.web import web_fetch, web_search
from .execution_planner import (
    CallEffect,
    ExecutionPlanner,
    ResourceClaim,
    ResourceCoordinator,
)
from .tool_call_validation import (
    ToolCall,
    ToolCallValidationResult,
    ToolError,
    strict_tool_schema,
    validate_tool_arguments,
    validate_tool_call,
)
from .tool_registry import (
    ToolRegistry,
    ToolSpec,
    tool_schemas_for_profile,
)
from .tool_result import ToolResult
from .tool_runner import (
    _registry_for_context,
    execute_tool,
    execute_tool_result,
    finalize_executed_tool_result,
    finalize_intercepted_tool_result,
)
