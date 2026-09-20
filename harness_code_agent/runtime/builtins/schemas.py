"""OpenAI function-calling schemas for built-in tools."""
from __future__ import annotations

import os

from ... import config
from .filesystem import READ_FILE_MAX_LINES

CORE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a workspace file. Prefer bounded reads with start_line and max_lines unless the file is known small. "
                f"For sequential scans of large files (>1000 lines), use a larger window up to {READ_FILE_MAX_LINES} lines "
                "to reduce round trips. Use narrower windows when following search hits, inspecting local context, "
                "or avoiding the per-call token cap on dense files."
            ),
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside workspace"},
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "1-based starting line for a bounded read.",
                    },
                    "max_lines": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": READ_FILE_MAX_LINES,
                        "description": f"Maximum lines to return for a bounded read. Must be <= {READ_FILE_MAX_LINES}. "
                        "For sequential scans of large files, prefer a larger window; use smaller windows for targeted reads "
                        "or token-dense content. "
                        "The per-call output is also capped by a token limit (whichever is smaller).",
                    },
                    "include_line_numbers": {
                        "type": "boolean",
                        "description": "Prefix returned lines with 1-based line numbers.",
                        "default": False,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill_file",
            "description": "Read a skill file from the packaged skills catalog. Use this to load a skill's SKILL.md or any sub-files referenced within it. Path should be like 'catalog/frontend-design/SKILL.md'.",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string", "description": "Relative path to skill file (e.g. 'catalog/frontend-design/SKILL.md')"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_search",
            "description": "Search hidden deferred tools available to the current profile. Matching tools are revealed for future tool calls in this conversation.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language description of the tool capability needed.",
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 8,
                        "description": "Maximum number of matching hidden tools to reveal.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_agent",
            "description": "Start one focused background agent and return its id immediately. Emit multiple independent spawn_agent calls in one response to run them concurrently.",
            "parameters": {
                "type": "object",
                "required": ["name", "role", "task"],
                "properties": {
                    "name": {"type": "string", "description": "Short unique agent name."},
                    "role": {"type": "string", "enum": ["explorer", "test_designer", "reviewer", "verifier", "worker"]},
                    "task": {"type": "string", "description": "Bounded delegated task."},
                    "expected_output": {"type": "string"},
                    "allowed_paths": {"type": "array", "items": {"type": "string"}, "default": [], "description": "Required writable ownership paths for worker; optional for read-only roles."},
                    "fork_turns": {"oneOf": [{"type": "string", "enum": ["none", "all"]}, {"type": "integer", "minimum": 1, "maximum": 5}], "default": "none"},
                    "model_intensity": {"type": "string", "enum": ["fast", "normal", "hard", "max"]},
                    "max_turns": {"type": "integer", "minimum": 1, "maximum": 20, "default": 6},
                    "max_seconds": {"type": "integer", "minimum": 30, "maximum": 1800, "default": 300},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file in the workspace.",
            "parameters": {
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside workspace"},
                    "content": {"type": "string", "description": "File content to write"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a safe text patch to one file. The search text must match exactly once, or the patch fails without modifying the file.",
            "parameters": {
                "type": "object",
                "required": ["path", "search", "replace"],
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside workspace"},
                    "search": {"type": "string", "description": "Existing text to replace. Must match exactly once."},
                    "replace": {"type": "string", "description": "Replacement text"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_todo",
            "description": (
                "Create or update the current execution todo list. The call replaces the "
                "whole list, so resend all items each time. Use this for concrete remaining "
                "work on non-trivial multi-step tasks, not for reasoning notes or high-level "
                "strategy. Skip this tool entirely for simple tasks. Update it at meaningful "
                "milestones rather than after every tool call."
            ),
            "parameters": {
                "type": "object",
                "required": ["items"],
                "properties": {
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 20,
                        "description": "Complete todo list in execution order.",
                        "items": {
                            "type": "object",
                            "required": ["text", "status"],
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": (
                                        "Stable id you choose (short kebab/snake string). "
                                        "Keep the same id across updates so an item can change "
                                        "status. Omit it on first creation to get an assigned id."
                                    ),
                                },
                                "text": {"type": "string", "maxLength": 500},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed", "cancelled"],
                                },
                            },
                        },
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repo_search",
            "description": (
                "Search workspace code/text using bounded ripgrep. Use this for repository exploration instead of run_bash rg/grep/find commands. "
                "The harness always supplies an explicit path, short timeout, default generated-directory excludes, and max result limits."
            ),
            "parameters": {
                "type": "object",
                "required": ["pattern"],
                "properties": {
                    "pattern": {"type": "string", "description": "Text or regex pattern to search for."},
                    "path": {
                        "type": "string",
                        "description": "Relative file or directory path to search. Defaults to workspace root.",
                        "default": ".",
                    },
                    "glob": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional ripgrep glob filters, e.g. ['*.py'] or ['!*.lock'].",
                        "default": [],
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether search should be case-sensitive.",
                        "default": False,
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                        "description": "Maximum result lines to return.",
                        "default": 100,
                    },
                    "context_lines": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 5,
                        "description": "Context lines around each match.",
                        "default": 0,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List workspace files and directories with bounded depth. Defaults to depth=2. "
                "Use higher depth only when you also provide bounded max_results and exclusions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Relative directory path (default: root)",
                        "default": ".",
                    },
                    "depth": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Maximum listing depth. depth=1 lists only direct children; default depth=2 includes one nested level.",
                        "default": 2,
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "description": "Maximum entries to return.",
                        "default": 200,
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Include hidden dot paths except protected internal/generated defaults.",
                        "default": False,
                    },
                    "exclude": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Additional directory names to exclude.",
                        "default": [],
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": (
                "Ask the user one focused multiple-choice question. "
                "Use when the model needs a user decision before continuing. "
                "The UI always includes an Other/其他 option with free-text input, "
                "and returns the selected option as structured JSON."
            ),
            "parameters": {
                "type": "object",
                "required": ["question", "options"],
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to show the user.",
                    },
                    "options": {
                        "type": "array",
                        "description": (
                            "Choices to show before the automatic Other choice. "
                            "Pass concise labels with optional values/descriptions. "
                            "3-4 options is ideal; maximum 9 (only the first 9 get number-key shortcuts)."
                        ),
                        "items": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "required": ["label"],
                                    "properties": {
                                        "label": {"type": "string", "description": "Short visible option label."},
                                        "value": {"type": "string", "description": "Value returned to the model if selected."},
                                        "description": {"type": "string", "description": "Optional one-line explanation."},
                                    },
                                },
                            ],
                        },
                    },
                    "other_label": {
                        "type": "string",
                        "description": "Label for the automatic free-text option. Defaults to 其他.",
                        "default": "其他",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": (
                "Search project and user long-term memory. Results include status, version, "
                "applicability, sources, and a concise body preview."
            ),
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                    "scope": {"type": "string", "enum": ["project", "user", "both"], "default": "both"},
                    "paths": {"type": "array", "items": {"type": "string"}, "default": []},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": (
                "Create or edit a durable Markdown memory immediately. Use supersedes plus "
                "expected_version for a correction that must preserve the old memory as audit history."
            ),
            "parameters": {
                "type": "object",
                "required": ["topic", "body"],
                "properties": {
                    "topic": {"type": "string"},
                    "body": {"type": "string"},
                    "scope": {"type": "string", "enum": ["project", "user"], "default": "project"},
                    "applicability": {"type": "string", "default": ""},
                    "source_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "memory_id": {"type": "string", "default": ""},
                    "expected_version": {"type": ["integer", "null"], "minimum": 1, "default": None},
                    "supersedes": {"type": "string", "default": ""},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": "Read one memory by stable id, including its metadata and full body.",
            "parameters": {
                "type": "object",
                "required": ["memory_id"],
                "properties": {
                    "memory_id": {"type": "string"},
                    "scope": {"type": "string", "enum": ["project", "user"], "default": "project"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_validate",
            "description": "Confirm a memory after checking its current evidence and refresh its fingerprints.",
            "parameters": {
                "type": "object", "required": ["memory_id", "expected_version"],
                "properties": {
                    "memory_id": {"type": "string"},
                    "expected_version": {"type": "integer", "minimum": 1},
                    "scope": {"type": "string", "enum": ["project", "user"], "default": "project"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_forget",
            "description": "Permanently remove a memory body after an explicit user request to forget it.",
            "parameters": {
                "type": "object", "required": ["memory_id", "expected_version"],
                "properties": {
                    "memory_id": {"type": "string"},
                    "expected_version": {"type": "integer", "minimum": 1},
                    "scope": {"type": "string", "enum": ["project", "user"], "default": "project"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "Execute a shell command in the workspace directory. "
                + (
                    "When HARNESS_SANDBOX_MODE=docker this runs inside a Linux Bash Docker sandbox with the workspace mounted at /workspace. "
                    if (config.SANDBOX_MODE or "host").strip().lower() == "docker"
                    else ""
                )
                + (
                    f"On Windows this uses the explicitly configured HARNESS_WINDOWS_SHELL={config.WINDOWS_SHELL} backend with no fallback. "
                    + (
                        "Use Bash syntax; commands run inside WSL and start in the workspace. "
                        if (config.WINDOWS_SHELL or "pwsh").strip().lower() == "wsl"
                        else "Use PowerShell 7 syntax; commands start in the Windows workspace. "
                    )
                    if os.name == "nt"
                    else "On POSIX this runs a shell suitable for standard Bash-style commands. "
                )
                + "Use for installing deps, running builds, starting servers, running tests, etc. "
                "Keep each call to one logical verification whenever practical. "
                "A command that runs to completion is reported as success even when it exits non-zero (a failing test run or compiler error is a normal result for you to read and act on); the exit code is appended as [exit_code: N]. Only timeouts or tool/runtime errors are reported as failed. "
                "Do not use shell for repository search or file listing; use repo_search/list_files/read_file. "
                "Repository-browsing shell commands such as bare rg without a path, recursive grep/findstr, Get-ChildItem -Recurse, or dir /s may be blocked. "
                "For long-running verification commands (compilation, training), increase the timeout parameter. "
                "For dev servers, watch mode, and runserver commands, this returns a background shell job id; use read_shell_output, list_shell_jobs, and stop_shell_job to manage it. "
                "Prefer bounded inspection commands such as rg, head/tail, sed -n, Select-Object -First/-Last, or line counts instead of dumping whole files or logs. "
                "Stderr is preserved separately in output for easier debugging."
            ),
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run; keep inspection commands bounded and avoid combining unrelated checks."},
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 300). Increase for long builds/training.",
                        "default": 300,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_shell_jobs",
            "description": "List background shell jobs started by long-running run_bash commands in the current session.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_shell_output",
            "description": "Read recent stdout/stderr output from a background shell job.",
            "parameters": {
                "type": "object",
                "required": ["job_id"],
                "properties": {
                    "job_id": {"type": "string", "description": "Background shell job id returned by run_bash."},
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum recent output characters to return (default 12000, capped at 100000).",
                        "default": 12000,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_shell_job",
            "description": "Stop a background shell job and its child process tree.",
            "parameters": {
                "type": "object",
                "required": ["job_id"],
                "properties": {
                    "job_id": {"type": "string", "description": "Background shell job id returned by run_bash."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_agent_message",
            "description": "Steer a running agent at its next safe model boundary without waiting for its whole turn to finish.",
            "parameters": {"type": "object", "required": ["agent_id", "message"], "properties": {"agent_id": {"type": "string"}, "message": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "followup_agent",
            "description": "Give an agent another task. Running agents receive it at the next safe boundary; idle agents start a new turn.",
            "parameters": {"type": "object", "required": ["agent_id", "task"], "properties": {"agent_id": {"type": "string"}, "task": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait_agents",
            "description": "Wait until all selected background agents finish, then return their summaries.",
            "parameters": {"type": "object", "properties": {"agent_ids": {"type": "array", "items": {"type": "string"}}, "timeout_seconds": {"type": "number", "minimum": 0, "maximum": 300, "default": 30}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_agents",
            "description": "List open background agent threads and their status.",
            "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["", "queued", "running", "completed", "blocked", "failed", "interrupted"]}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "interrupt_agent",
            "description": "Interrupt only the selected agent's current turn while preserving its thread and completed artifacts.",
            "parameters": {"type": "object", "required": ["agent_id"], "properties": {"agent_id": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_agent_changes",
            "description": "Read a complete isolated change proposal with pagination.",
            "parameters": {"type": "object", "required": ["proposal_id"], "properties": {"proposal_id": {"type": "string"}, "path": {"type": "string"}, "offset": {"type": "integer", "minimum": 0, "default": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 50000, "default": 12000}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_agent_changes",
            "description": "Three-way merge and atomically apply a worker proposal. Real overlapping edits return a conflict id without changing the workspace.",
            "parameters": {"type": "object", "required": ["proposal_id"], "properties": {"proposal_id": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_agent_conflicts",
            "description": "Read base, main, worker, and marked merge content for a proposal conflict.",
            "parameters": {"type": "object", "required": ["conflict_id"], "properties": {"conflict_id": {"type": "string"}, "path": {"type": "string"}, "offset": {"type": "integer", "minimum": 0, "default": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 50000, "default": 12000}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_agent_conflicts",
            "description": "Submit final content for every conflicted file and atomically commit the complete merge after revalidation.",
            "parameters": {"type": "object", "required": ["conflict_id", "resolutions"], "properties": {"conflict_id": {"type": "string"}, "resolutions": {"type": "object", "additionalProperties": {"type": "string"}}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_agent",
            "description": "Close an idle agent thread and release its isolated workspace. Unapplied changes require explicit discard_changes.",
            "parameters": {"type": "object", "required": ["agent_id"], "properties": {"agent_id": {"type": "string"}, "discard_changes": {"type": "boolean", "default": False}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web with the standard web search provider. Use this as an independent web search tool when Exa search is not available or when this provider is specifically needed. Returns titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Max results to return (default 5)",
                        "default": 5,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch and read the text content of a web page. Use after web_search to read a specific page in detail.",
            "parameters": {
                "type": "object",
                "required": ["url"],
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                },
            },
        },
    },
]


# --- Evaluator-only tools (browser testing) ---

BROWSER_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "browser_test",
            "description": (
                "Launch a headless Chromium browser to test the running application. "
                "Navigates to a URL, performs UI actions (click, fill, scroll, evaluate JS), "
                "captures console errors, and takes a screenshot. "
                "Optionally starts a dev server first via start_command."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to (e.g. http://localhost:5173)",
                    },
                    "actions": {
                        "type": "array",
                        "description": "List of browser actions to perform sequentially",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["click", "fill", "wait", "evaluate", "scroll"],
                                    "description": "Action type",
                                },
                                "selector": {
                                    "type": "string",
                                    "description": "CSS selector (for click/fill)",
                                },
                                "value": {
                                    "type": "string",
                                    "description": "Text for fill, JS code for evaluate, pixels for scroll",
                                },
                                "delay": {
                                    "type": "integer",
                                    "description": "Milliseconds to wait (for wait action)",
                                },
                            },
                        },
                    },
                    "screenshot": {
                        "type": "boolean",
                        "description": "Take a screenshot after actions (default: true)",
                        "default": True,
                    },
                    "start_command": {
                        "type": "string",
                        "description": "Shell command to start the dev server (e.g. 'npm run dev'). Only needed on first call.",
                    },
                    "port": {
                        "type": "integer",
                        "description": "Port the dev server runs on (default: 5173)",
                        "default": 5173,
                    },
                    "startup_wait": {
                        "type": "integer",
                        "description": "Seconds to wait for dev server to start (default: 8)",
                        "default": 8,
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_dev_server",
            "description": "Stop the background dev server started by browser_test.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
