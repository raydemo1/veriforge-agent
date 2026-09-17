"""Middleware base primitives."""
from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from ..tool_result import ToolResult

if TYPE_CHECKING:
    from ..tool_failures import FailureAction, ToolFailure

MAIN_AGENT_NAMES = {"main_agent"}

# status_source values emitted when a call is intercepted before execution.
_BLOCKED_SOURCES = {
    "permission",
    "approval",
    "budget",
    "tool_policy",
    "delegate_policy",
    "user_question",
}


def tool_blocked(result: ToolResult) -> bool:
    """True when the call was intercepted by policy/permissions before running."""
    source = str((result.metadata or {}).get("status_source") or "")
    return result.status == "failed" and source in _BLOCKED_SOURCES


def tool_failed(result: ToolResult) -> bool:
    """True when the tool ran and reported a failure (not an interception)."""
    return result.status == "failed" and not tool_blocked(result)


def result_text(result: ToolResult) -> str:
    """Human-facing text of a result (error first, full output as fallback)."""
    return result.error or result.output or ""


class AgentMiddleware(ABC):
    """Base class for agent middlewares."""

    def on_conversation_start(self, messages: list[dict], runtime_state=None,
                              agent_name: str | None = None) -> list[dict]:
        """Called after a live conversation is initialized. Return messages to append."""
        return []

    def on_conversation_close(self, messages: list[dict], runtime_state=None,
                              agent_name: str | None = None) -> None:
        """Called when a live conversation is closing."""
        return

    def on_context_compacted(self, messages: list[dict], runtime_state=None,
                             agent_name: str | None = None,
                             phase: str | None = None) -> list[dict]:
        """Called after conversation context is compacted. Return messages to inject."""
        return []

    def augment_user_prompt(
        self,
        user_prompt: str,
        messages: list[dict],
        runtime_state=None,
        agent_name: str | None = None,
        mention_paths: list[str] | None = None,
    ) -> str | None:
        """Called after user prompt mentions are resolved and before the turn is formatted."""
        return None

    def before_tool(
        self,
        tool_name: str,
        tool_args: dict,
        messages: list[dict],
        runtime_state=None,
        agent_name: str | None = None,
    ) -> str | None:
        """Called before each tool execution. Return a blocking message, or None."""
        return None

    def on_tool_allowed(
        self,
        tool_name: str,
        tool_args: dict,
        messages: list[dict],
        runtime_state=None,
        agent_name: str | None = None,
    ) -> None:
        """Called after every before_tool gate allows the tool, just before it is queued."""
        return

    def post_tool(self, tool_name: str, tool_args: dict, result: ToolResult,
                  messages: list[dict], runtime_state=None,
                  agent_name: str | None = None) -> str | None:
        """Called after each tool execution. Return a message to inject, or None."""
        return None

    def on_tool_failure(
        self,
        failure: "ToolFailure",
        messages: list[dict],
        runtime_state=None,
        agent_name: str | None = None,
    ) -> "FailureAction | None":
        """Called exactly once per failed call, including intercepted calls.

        Unlike post_tool, this hook also fires for validation/permission/budget
        failures that never reached tool execution. The middleware is a pure
        decision point: return a FailureAction (or None); the ToolExecutor is
        the sole executor of side effects such as stop or injected guidance.
        """
        return None

    def pre_exit(self, messages: list[dict], runtime_state=None,
                 agent_name: str | None = None) -> str | None:
        """Called when the agent wants to stop. Return a message to force continuation, or None."""
        return None

    def per_iteration(self, iteration: int, messages: list[dict], runtime_state=None,
                      agent_name: str | None = None) -> str | None:
        """Called at the start of each iteration. Return a message to inject, or None."""
        return None

    def begin_turn(self, task: str, messages: list[dict], runtime_state=None,
                   agent_name: str | None = None) -> None:
        """Called before a new user turn is appended in a live conversation."""
        return
