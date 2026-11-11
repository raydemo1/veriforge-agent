"""OpenAI function-calling schemas for built-in tools."""
from __future__ import annotations

import os

from ..agent_roles import AGENT_ROLES
from .filesystem import READ_FILE_MAX_LINES

CORE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_runtime_status",
            "description": 'Read current task, budget, verification, environment or agent state on demand.',
            "parameters": {
                "type": "object", "required": ["section"],
                "properties": {"section": {"type": "string", "enum": ["task", "budget", "verification", "environment", "agents"]}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_intelligence",
            "description": 'Read semantic definitions, references, document symbols or diagnostics from a system-installed language server. Workspace-local servers are never executed. Returns unavailable when the server or operation is unavailable. Diagnostics describe code understanding, not task verification.',
            "parameters": {
                "type": "object",
                "required": ["operation", "path"],
                "properties": {
                    "operation": {"type": "string", "enum": ["definition", "references", "symbols", "diagnostics"]},
                    "path": {"type": "string", "description": "File path inside the workspace"},
                    "line": {"type": "integer", "minimum": 1, "description": "1-based line, required for definition/references"},
                    "column": {"type": "integer", "minimum": 1, "description": "1-based Unicode character column, required for definition/references"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": 'Read a workspace file in token-bounded pages. Follow next_read exactly when present to continue from where the last page ended.',
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
                    "start_column": {
                        "type": "integer", "minimum": 1, "default": 1,
                        "description": "1-based Unicode character position within start_line; use next_read for continuation.",
                    },
                    "max_lines": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": READ_FILE_MAX_LINES,
                        "description": 'Maximum lines to return for a bounded read. Must be <= 2000. The per-call output is also capped by a token limit (whichever is smaller).',
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
            "description": "Read a file from the packaged skills catalog. Paths look like 'catalog/frontend-design/SKILL.md'.",
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
            "name": "search_tools",
            "description": "Find deferred built-in or MCP tools using short English keywords. Use a discovery scope to narrow the search. Without one, search registered tools only. Matching usable tools are mounted as native functions; unavailable MCP tools may appear as history.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short English keywords naming the capability needed.",
                    },
                    "server": {
                        "type": "string",
                        "description": "Discovery scope from the system prompt, such as builtin:agents or mcp:github. Only an MCP scope connects to a server. A bare id also selects MCP.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                        "description": (
                            "Maximum number of ranked search matches. A matched "
                            "workflow group mounts all of its tools, so the number "
                            "mounted can exceed this limit."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_agent",
            "description": 'Start a focused background agent and return its id immediately.',
            "parameters": {
                "type": "object",
                "required": ["name", "role", "task"],
                "properties": {
                    "name": {"type": "string", "description": "Short unique agent name."},
                    "role": {"type": "string", "enum": list(AGENT_ROLES)},
                    "task": {"type": "string", "description": "Bounded delegated task."},
                    "expected_output": {"type": "string"},
                    "allowed_paths": {"type": "array", "items": {"type": "string"}, "default": [], "description": "Required writable ownership paths for worker; optional for read-only roles."},
                    "fork_turns": {"oneOf": [{"type": "string", "enum": ["none", "all"]}, {"type": "integer", "minimum": 1, "maximum": 5}], "default": "none"},
                    "model_intensity": {"type": "string", "enum": ["fast", "normal", "hard", "max"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": 'Create or overwrite a file in the workspace.',
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
            "description": 'Apply a text patch to one file. The search text must match exactly once, or the patch fails without modifying the file.',
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
            "description": 'Replace the current execution todo list with the given items in execution order; every call resends the whole list. Use it for concrete remaining work in multi-step tasks.',
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
            "description": 'Search workspace code and text with bounded ripgrep. The harness supplies an explicit path, a short timeout, generated-directory excludes and a result cap.',
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
            "description": 'List workspace files and directories with bounded depth. depth=1 lists direct children; the default depth=2 adds one nested level.',
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
            "description": 'Ask the user one multiple-choice question. The UI always adds an Other/free-text option, and the answer returns as structured JSON with the selected value.',
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
                        "description": "Choices to show before the automatic Other choice. Pass concise labels with optional values/descriptions. Maximum 9 (only the first 9 get number-key shortcuts).",
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
    {'type': 'function',
     'function': {'name': 'memory_search',
                  'description': 'Search current memory with a focused query; page by scope and filter by paths or kind.',
                  'parameters': {'type': 'object',
                                 'required': ['query'],
                                 'properties': {'query': {'type': 'string'},
                                                'scope': {'type': 'string',
                                                          'enum': ['project', 'user', 'both'],
                                                          'default': 'both'},
                                                'paths': {'type': 'array',
                                                          'items': {'type': 'string'},
                                                          'default': []},
                                                'limit': {'type': 'integer',
                                                          'minimum': 1,
                                                          'maximum': 50,
                                                          'default': 8},
                                                'offset': {'type': 'integer', 'minimum': 0, 'default': 0},
                                                'max_tokens': {'type': 'integer',
                                                               'minimum': 512,
                                                               'maximum': 8000,
                                                               'default': 2000},
                                                'kind': {'type': ['string', 'null'],
                                                         'enum': ['preference',
                                                                  'project_fact',
                                                                  'convention',
                                                                  'decision',
                                                                  'lesson',
                                                                  'workflow',
                                                                  None],
                                                         'default': None}}}}},
    {'type': 'function',
     'function': {'name': 'memory_write',
                  'description': 'Save a source-backed fact. Quote the current user verbatim. Use the existing '
                                 'id and expected_version to correct a fact; a conflicting create is held for '
                                 'review.',
                  'parameters': {'type': 'object',
                                 'required': ['topic', 'fact_key', 'kind', 'body', 'source_quote'],
                                 'properties': {                                                'topic': {'type': 'string',
                                                         'description': 'Topic name, or an existing topic\'s alias.'},
                                                'fact_key': {'type': 'string',
                                                             'description': 'Stable qualified property key, e.g. package.manager. Reuse it for the same setting across scopes.'},
                                                'kind': {'type': 'string',
                                                         'description': 'Nature of one fact, independent of its topic.',
                                                         'enum': ['preference',
                                                                  'project_fact',
                                                                  'convention',
                                                                  'decision',
                                                                  'lesson',
                                                                  'workflow']},
                                                'body': {'type': 'string'},
                                                'source_quote': {'type': 'string'},
                                                'scope': {'type': 'string',
                                                          'enum': ['project', 'user'],
                                                          'description': 'Use project for repository-specific facts or preferences; user only for explicit cross-project personal preferences.',
                                                          'default': 'project'},
                                                'applicability': {'type': 'string', 'default': '',
                                                                  'description': 'Empty for unconditional defaults, otherwise the applicable condition. path: conditions are project scope.'},
                                                'source_paths': {'type': 'array',
                                                                 'items': {'type': 'string'},
                                                                 'default': []},
                                                'memory_id': {'type': 'string', 'default': ''},
                                 'restore_forgotten': {'type': 'boolean', 'default': False},
                                                'expected_version': {'type': ['integer', 'null'],
                                                                     'default': None,
                                                                     'minimum': 1}}}}},
    {'type': 'function',
     'function': {'name': 'memory_read',
                  'description': 'Read a fact, or set history=true for previous revisions. Pass topic or '
                                 'topic_id to expand a topic, or omit identifiers to list topics.',
                  'parameters': {'type': 'object',
                                 'required': ['scope'],
                                 'properties': {'scope': {'type': 'string',
                                                          'enum': ['project', 'user'],
                                                          'default': 'project'},
                                                'memory_id': {'type': 'string', 'default': ''},
                                                'topic_id': {'type': 'string', 'default': ''},
                                                'topic': {'type': 'string', 'default': '',
                                                          'description': 'Exact topic name or known alias in the selected scope.'},
                                                'history': {'type': 'boolean', 'default': False},
                                                'offset': {'type': 'integer', 'minimum': 0, 'default': 0},
                                                'limit': {'type': 'integer',
                                                          'minimum': 1,
                                                          'maximum': 20,
                                                          'default': 10}}}}},
    {'type': 'function',
     'function': {'name': 'memory_validate',
                  'description': 'Confirm a fact with a fresh verbatim user source and refresh file evidence.',
                  'parameters': {'type': 'object',
                                 'required': ['memory_id', 'expected_version', 'source_quote'],
                                 'properties': {'memory_id': {'type': 'string'},
                                                'expected_version': {'type': 'integer', 'minimum': 1},
                                                'source_quote': {'type': 'string'},
                                                'scope': {'type': 'string',
                                                          'enum': ['project', 'user'],
                                                          'default': 'project'}}}}},
    {'type': 'function',
     'function': {'name': 'memory_forget',
                  'description': 'Permanently forget a memory only on an explicit user request, quoted verbatim.',
                  'parameters': {'type': 'object',
                                 'required': ['memory_id', 'expected_version', 'source_quote'],
                                 'properties': {'memory_id': {'type': 'string'},
                                                'expected_version': {'type': 'integer', 'minimum': 1},
                                                'source_quote': {'type': 'string'},
                                                'scope': {'type': 'string',
                                                          'enum': ['project', 'user'],
                                                          'default': 'project'}}}}},
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": 'Execute a shell command in the workspace directory. On Windows use PowerShell 7 syntax; processes run under the native directory write sandbox: the project is read-only unless write_paths declares writable directories (private TEMP is always writable), and a background job cannot claim the whole shared workspace. A command that runs to completion is reported as success even when it exits non-zero - the code is appended as [exit_code: N]; only timeouts and runtime errors are failures. Dev servers and watch mode return a background job id for the shell-job tools. Unbounded repository-wide listing or searching may be blocked. Never create a git worktree on your own.',
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."},
                    **({"write_paths": {
                        "type": "array", "items": {"type": "string"}, "default": [],
                        "description": "Workspace directories this process may write, relative to the workspace. Omit for a read-only project. Permits last until the process tree exits."
                    }} if os.name == "nt" else {}),
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 300). Increase for long builds and training.",
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
            "description": 'List background shell jobs started by long-running run_bash commands in the current session.',
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_shell_output",
            "description": 'Read consecutive stdout/stderr output from a background shell job. Continue with next_offset; offset counts Unicode characters.',
            "parameters": {
                "type": "object",
                "required": ["job_id"],
                "properties": {
                    "job_id": {"type": "string", "description": "Background shell job id returned by run_bash."},
                    "offset": {"type": "integer", "minimum": 0, "default": 0,
                               "description": "0-based Unicode character offset, returned as next_offset."},
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum consecutive output characters to return from offset (default 12000, capped at 100000).",
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
            "description": 'Stop a background shell job and its child process tree.',
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
            "description": 'Steer a running agent at its next safe model boundary without waiting for its whole turn to finish.',
            "parameters": {"type": "object", "required": ["agent_id", "message"], "properties": {"agent_id": {"type": "string"}, "message": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_parent_message",
            "description": "Report to the parent agent one of: a finding that changes the parent's plan, a missing decision or constraint, or a high-value risk. The parent sees it at the next safe boundary. Routine progress is not a message; the task result returns through completion.",
            "parameters": {"type": "object", "required": ["message"], "properties": {"message": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "followup_agent",
            "description": 'Give an agent another task. A running agent receives it at its next safe boundary; an idle agent starts a new turn.',
            "parameters": {"type": "object", "required": ["agent_id", "task"], "properties": {"agent_id": {"type": "string"}, "task": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait_agents",
            "description": 'Wait until all selected background agents finish, then return their summaries.',
            "parameters": {"type": "object", "properties": {"agent_ids": {"type": "array", "items": {"type": "string"}}, "timeout_seconds": {"type": "number", "minimum": 0, "maximum": 300, "default": 30}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "interrupt_agent",
            "description": "Interrupt the selected agent's current turn while preserving its thread and completed artifacts.",
            "parameters": {"type": "object", "required": ["agent_id"], "properties": {"agent_id": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_agent_changes",
            "description": 'Read an isolated change proposal with pagination. Pass conflict_id instead to read base, main, worker and marked merge content for that conflict.',
            "parameters": {
                "type": "object",
                "anyOf": [{"required": ["proposal_id"]}, {"required": ["conflict_id"]}],
                "properties": {
                    "proposal_id": {"type": "string", "description": "Proposal to read. Optional when conflict_id is given."},
                    "conflict_id": {"type": "string", "default": "", "description": "Conflict id returned by apply_agent_changes; reads the conflicted files."},
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50000, "default": 12000},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_agent_changes",
            "description": 'Three-way merge and atomically apply a worker proposal. Overlapping edits return a conflict id without changing the workspace. Pass conflict_id with resolutions to commit the final content and complete the merge after revalidation.',
            "parameters": {
                "type": "object",
                "anyOf": [{"required": ["proposal_id"]}, {"required": ["conflict_id"]}],
                "properties": {
                    "proposal_id": {"type": "string", "description": "Proposal to apply. Optional when resolving with conflict_id; when both are given they must name the same proposal."},
                    "conflict_id": {"type": "string", "default": "", "description": "Conflict id returned by a previous apply. Requires resolutions."},
                    "resolutions": {"type": "object", "additionalProperties": {"type": "string"}},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_agent",
            "description": 'Close an idle agent thread and release its isolated workspace. Unapplied changes require explicit discard_changes.',
            "parameters": {"type": "object", "required": ["agent_id"], "properties": {"agent_id": {"type": "string"}, "discard_changes": {"type": "boolean", "default": False}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": 'Search the web. Returns titles, URLs, and snippets.',
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
            "description": 'Fetch and read the text content of a web page.',
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
