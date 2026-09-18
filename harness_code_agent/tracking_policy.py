"""Shared todo policy for runtime profiles and skill routing."""

TASK_TRACKING_POLICY = """\
## Todo Management
Use the todo tool (update_todo) for non-trivial multi-step execution.

Create a todo list when the task has several concrete work items, multiple
components, or a distinct verification stage. Skip todo management for simple
tasks that need only one or two actions.

Judge the nature of the task instead of counting files, tools, commands, or
tests. Reading many files to answer one question may need no todo at all; a
multi-part change may warrant one before any tool is called.

Keep the todo list aligned with actual work: update it when a meaningful item
is completed, added, abandoned, or materially changed. Do not update it after
every tool call.
"""

TASK_TRACKING_CATALOG_POLICY = (
    "Use the built-in update_todo tool to track concrete multi-step execution: "
    "create a short checklist for work with several items or a distinct "
    "verification stage, and skip it for simple one- or two-action tasks. "
    "Todo state is session state written through update_todo; formal plan.md "
    "files and approval belong to interactive planning flows, not todo items."
)
