"""Unified post-failure decision middleware.

Boundary (see ``runtime/tool_failures.py``):

* The middleware is a **pure decision point** — it classifies an already
  normalized :class:`~harness_code_agent.runtime.tool_failures.ToolFailure`
  and returns a :class:`FailureAction`. It never replays a tool, never sleeps,
  never calls ``request_stop`` itself and never mutates runtime state.
* ``ToolExecutor`` performs the side effect implied by the action (inject a
  correction, or request a fallback stop).
* Protocol-level failures (the tool never ran) get a *bounded* correction
  loop: at most ``schema_streak_stop - 1`` regenerations, with the allowed
  argument schema injected on the second failure.
* Task-level failures (permission denied, non-zero exit, test failure, ...)
  stay visible to the agent unchanged. The runtime never hides or replays
  them; only the turn-level failure budget can force a stop as a global
  circuit breaker.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..tool_failures import (
    FailureAction,
    FailureCategory,
    FailureKind,
    FailureVisibility,
)
from .base import MAIN_AGENT_NAMES, AgentMiddleware

if TYPE_CHECKING:
    from ..tool_failures import ToolFailure

#: Inject an explicit schema correction from this consecutive attempt on.
DEFAULT_SCHEMA_STREAK_GUIDANCE = 2
#: Stop after this many consecutive protocol failures for one tool.
DEFAULT_SCHEMA_STREAK_STOP = 3
#: Stop after this many consecutive blocking policy failures for one tool.
DEFAULT_POLICY_STREAK_STOP = 2
#: Turn-level circuit breaker for task-visible failures.
DEFAULT_TURN_FAILURE_BUDGET = 10


class ToolFailurePolicyMiddleware(AgentMiddleware):
    """Decide what happens after a (normalized) tool call failure."""

    def __init__(
        self,
        *,
        tool_registry: Any | None = None,
        schema_streak_guidance: int = DEFAULT_SCHEMA_STREAK_GUIDANCE,
        schema_streak_stop: int = DEFAULT_SCHEMA_STREAK_STOP,
        policy_streak_stop: int = DEFAULT_POLICY_STREAK_STOP,
        turn_failure_budget: int = DEFAULT_TURN_FAILURE_BUDGET,
    ) -> None:
        self.tool_registry = tool_registry
        self.schema_streak_guidance = schema_streak_guidance
        self.schema_streak_stop = schema_streak_stop
        self.policy_streak_stop = policy_streak_stop
        self.turn_failure_budget = turn_failure_budget

    def begin_turn(self, task: str, messages: list[dict], runtime_state=None,
                   agent_name: str | None = None) -> None:
        """Streaks and the turn budget are scoped to one user turn."""
        if agent_name in MAIN_AGENT_NAMES and runtime_state is not None:
            runtime_state.failures.reset()

    def on_tool_failure(
        self,
        failure: "ToolFailure",
        messages: list[dict],
        runtime_state=None,
        agent_name: str | None = None,
    ) -> FailureAction | None:
        if agent_name not in MAIN_AGENT_NAMES or runtime_state is None:
            return None

        # Global circuit breaker: too many task-visible failures this turn.
        # Protocol failures do not consume this budget.
        if (
            failure.visibility == FailureVisibility.TASK
            and failure.counted_in_tracker
            and runtime_state.failures.turn_failure_count >= self.turn_failure_budget
        ):
            return FailureAction.stop(
                reason="turn_failure_budget_exhausted",
                message=(
                    f"[stop] {runtime_state.failures.turn_failure_count} tool failures "
                    f"in this turn (budget {self.turn_failure_budget}). Replan the task "
                    "from the last successful state instead of issuing more failing calls."
                ),
                limit_type="turn_failures",
                limit=self.turn_failure_budget,
            )

        # Protocol-level: the tool never executed; bounded self-correction.
        if failure.visibility == FailureVisibility.PROTOCOL:
            if not failure.retryable:
                return FailureAction.return_to_agent()
            if failure.attempt >= self.schema_streak_stop:
                return FailureAction.stop(
                    reason="retry_budget_exhausted",
                    message=(
                        f"[stop] {failure.attempt} consecutive validation failures for "
                        f"`{failure.tool_name}`. Do not repeat the same call; use a "
                        "different argument set or a different tool."
                    ),
                    limit_type="schema_failures",
                    limit=self.schema_streak_stop,
                )
            if failure.attempt >= self.schema_streak_guidance:
                return FailureAction.auto_retry(message=self._schema_correction(failure))
            return FailureAction.auto_retry()

        # invalid_call kinds that require reasoning rather than format repair.
        if failure.kind == FailureKind.TOOL_SCHEMA_ERROR:
            return FailureAction.stop(
                reason="invalid_tool_definition",
                message=(
                    f"[stop] tool `{failure.tool_name}` has an invalid schema; it cannot "
                    "be called. Choose a different tool."
                ),
                limit_type="schema_failures",
                limit=1,
            )
        if failure.kind == FailureKind.UNKNOWN_TOOL:
            if failure.attempt >= 2:
                return FailureAction.stop(
                    reason="retry_budget_exhausted",
                    message=(
                        f"[stop] repeated calls to unknown tool `{failure.tool_name}`. "
                        "Use only tools from the provided tool list."
                    ),
                    limit_type="unknown_tool",
                    limit=2,
                )
            return FailureAction.return_to_agent()

        # Human / external gates: never auto-retry, never count toward stop.
        if (
            failure.kind == FailureKind.APPROVAL_DENIED
            or failure.kind == FailureKind.USER_ATTENTION_REQUIRED
            or failure.kind == FailureKind.BUDGET_EXCEEDED
        ):
            return FailureAction.return_to_agent()

        # Blocking policy failures: visible once, stop on repetition.
        if failure.category == FailureCategory.POLICY:
            if failure.attempt >= self.policy_streak_stop:
                return FailureAction.stop(
                    reason="repeated_policy_failure",
                    message=(
                        f"[stop] {failure.attempt} consecutive blocked attempts for "
                        f"`{failure.tool_name}`. This operation is not permitted; choose "
                        "a different approach instead of repeating it."
                    ),
                    limit_type="policy_failures",
                    limit=self.policy_streak_stop,
                )
            return FailureAction.return_to_agent()

        # Execution / resource / verification failures: the raw result is the
        # information the agent needs; never hide or replay it.
        return FailureAction.return_to_agent()

    # ------------------------------------------------------------------
    # Correction text
    # ------------------------------------------------------------------

    def _schema_correction(self, failure: "ToolFailure") -> str:
        lines = [
            f"Previous {failure.attempt} calls to `{failure.tool_name}` failed validation.",
        ]
        if failure.message:
            lines.append(f"Last error: {failure.message}")
        parameters = self._parameters_schema(failure.tool_name)
        properties = parameters.get("properties") if isinstance(parameters, dict) else None
        if isinstance(properties, dict) and properties:
            required = set(parameters.get("required") or [])
            lines.append("")
            lines.append("Allowed arguments are exactly:")
            for name in sorted(properties):
                spec = properties.get(name)
                type_name = _type_name(spec)
                marker = " (required)" if name in required else ""
                lines.append(f"- {name}: {type_name}{marker}")
            lines.append("")
            lines.append(
                "Do not invent additional fields. Fix the arguments and call "
                f"`{failure.tool_name}` again."
            )
        else:
            lines.append(
                "Check the tool's declared parameters, remove any extra fields, and "
                f"call `{failure.tool_name}` again."
            )
        return "\n".join(lines)

    def _parameters_schema(self, tool_name: str) -> dict:
        if self.tool_registry is None:
            return {}
        schema = self.tool_registry.schema_for(tool_name)
        if not isinstance(schema, dict):
            return {}
        # OpenAI function wrapper: {"type": "function", "function": {...}}
        function = schema.get("function")
        if isinstance(function, dict) and isinstance(function.get("parameters"), dict):
            return function["parameters"]
        if isinstance(schema.get("parameters"), dict):
            return schema["parameters"]
        # Tolerate a raw JSON-schema shape.
        return schema if "properties" in schema else {}


def _type_name(spec: Any) -> str:
    if not isinstance(spec, dict):
        return "any"
    if isinstance(spec.get("enum"), list) and spec["enum"]:
        values = ", ".join(str(value) for value in spec["enum"])
        return f"enum[{values}]"
    type_value = spec.get("type")
    if isinstance(type_value, list):
        return " | ".join(str(value) for value in type_value)
    if isinstance(type_value, str):
        return type_value
    if "$ref" in spec:
        return str(spec["$ref"]).rsplit("/", 1)[-1]
    return "any"
