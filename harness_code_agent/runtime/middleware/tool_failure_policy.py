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
* A tool whose own schema is broken is a plain infrastructure error: the
  result is exposed with a note that the *definition* is wrong (not the
  arguments), so retrying unchanged cannot help. The turn is not stopped,
  no replacement tool is suggested (capabilities are not interchangeable),
  and no ``unavailable_tools`` state is kept.
* Policy blocks (permission denied, blocked command shape, ...) stay
  visible: the first block is returned as-is; repeating the *same* blocked
  operation (same fingerprint) gets a guidance reminder phrased at the
  *operation* level — comply with the policy or obtain permission. It never
  says "use another tool", which would invite bypassing the policy through
  a different tool.
* Execution / resource / verification failures are always returned with
  their raw result. The streak only advances while the objective
  fingerprint (tool + kind + args + error signature) stays identical — a
  changed command or a changed error is progress, and repeating one
  identical failure only earns a guidance nudge, never a turn kill.
* There is deliberately **no** global "N failures per turn" circuit
  breaker: productive debugging produces many *different* failures.
  Runaway protection belongs to the resource layer (tool timeout, LLM
  request timeout, iteration/tool-call budgets, subagent depth, host
  cancellation).
* A non-zero process exit code (a failing test suite, a compiler error, ...)
  is **not** a tool failure: ``run_bash`` started and completed normally, so
  the result stays ``success`` with the raw output and exit code for the
  model to interpret. Only transport/runtime failures (shell exception,
  timeout, missing job manager) become ToolFailures.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..tool_failures import (
    FINGERPRINTED_CATEGORIES,
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
#: Attach a "do not repeat this blocked operation" nudge from this attempt on.
DEFAULT_POLICY_STREAK_GUIDANCE = 2
#: First strong nudge for one identical execution failure repeating.
DEFAULT_EXECUTION_STREAK_GUIDANCE = 3


def _schema_definition_error(tool_name: str) -> str:
    return (
        f"[tool error] `{tool_name}` has an invalid runtime schema and cannot "
        "be invoked as currently defined.\n"
        "This is a tool-definition error, not an argument error. Retrying the "
        "same call unchanged will not fix it.\n"
        "Continue if the task can be completed without this capability; "
        "otherwise report the blocker."
    )


def _policy_repetition_guidance(attempt: int) -> str:
    return (
        f"[SYSTEM] This exact operation has just been blocked by policy "
        f"({attempt} identical attempts). Retrying it unchanged will not succeed. "
        "Change the requested operation so it complies with the policy, or "
        "obtain the required permission if applicable."
    )


class ToolFailurePolicyMiddleware(AgentMiddleware):
    """Decide what happens after a (normalized) tool call failure."""

    def __init__(
        self,
        *,
        tool_registry: Any | None = None,
        schema_streak_guidance: int = DEFAULT_SCHEMA_STREAK_GUIDANCE,
        schema_streak_stop: int = DEFAULT_SCHEMA_STREAK_STOP,
        policy_streak_guidance: int = DEFAULT_POLICY_STREAK_GUIDANCE,
        execution_streak_guidance: int = DEFAULT_EXECUTION_STREAK_GUIDANCE,
    ) -> None:
        self.tool_registry = tool_registry
        self.schema_streak_guidance = schema_streak_guidance
        self.schema_streak_stop = schema_streak_stop
        self.policy_streak_guidance = policy_streak_guidance
        self.execution_streak_guidance = execution_streak_guidance

    def begin_turn(self, task: str, messages: list[dict], runtime_state=None,
                   agent_name: str | None = None) -> None:
        """Streaks are scoped to one user turn."""
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

        # Protocol-level: the tool never executed; bounded self-correction.
        if failure.visibility == FailureVisibility.PROTOCOL:
            if not failure.retryable:
                return FailureAction.return_to_agent()
            if failure.attempt >= self.schema_streak_stop:
                return FailureAction.stop(
                    reason="retry_budget_exhausted",
                    message=(
                        f"[stop] {failure.attempt} consecutive validation failures for "
                        f"`{failure.tool_name}`. Do not repeat the same call; fix the "
                        "arguments according to the allowed schema."
                    ),
                    limit_type="schema_failures",
                    limit=self.schema_streak_stop,
                )
            if failure.attempt >= self.schema_streak_guidance:
                return FailureAction.request_regeneration(message=self._schema_correction(failure))
            return FailureAction.request_regeneration()

        # Broken tool *definition*: infrastructure/config error, not a model
        # mistake. Expose it with the explanation every time; no retry, no
        # stop, no suggestion of a substitute capability.
        if failure.kind == FailureKind.TOOL_SCHEMA_ERROR:
            return FailureAction.return_to_agent(
                message=_schema_definition_error(failure.tool_name)
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

        # A human denial ends the turn: the user said no, so the agent must
        # not keep trying alternative paths this turn. The next user message
        # re-evaluates from scratch. Other gates stay recoverable.
        if failure.kind == FailureKind.APPROVAL_DENIED:
            return FailureAction.stop(
                reason="approval_denied",
                message="[stop] 用户已拒绝该操作，本轮任务终止。",
                limit_type="approval_denied",
                limit=1,
            )
        if (
            failure.kind == FailureKind.USER_ATTENTION_REQUIRED
            or failure.kind == FailureKind.BUDGET_EXCEEDED
        ):
            return FailureAction.return_to_agent()

        # Blocking policy failures: the block itself already prevented the
        # unsafe operation, so the turn is never killed for repeating one.
        # The first identical block is returned as-is; repeating the very
        # same blocked operation gets an operation-level nudge. A changed
        # operation (different fingerprint) is new information and resets
        # the streak. Guidance never names another tool: that would invite
        # performing the same forbidden action through a different tool.
        if failure.category == FailureCategory.POLICY:
            if failure.attempt >= self.policy_streak_guidance:
                return FailureAction.return_to_agent(
                    message=_policy_repetition_guidance(failure.attempt)
                )
            return FailureAction.return_to_agent()

        # Execution / resource / verification failures: the raw result is the
        # information the agent needs; never hide or replay it. The streak only
        # advances while the objective fingerprint (tool + kind + args + error
        # signature) stays identical — changing commands or errors is progress,
        # and a long productive debugging chain of *different* failures is
        # fine. Repetition earns guidance, never a stop; runaway protection is
        # owned by the resource layer (timeouts, iteration/call budgets).
        if (
            failure.category in FINGERPRINTED_CATEGORIES
            and failure.attempt >= self.execution_streak_guidance
        ):
            return FailureAction.return_to_agent(
                message=(
                    f"[SYSTEM] The same `{failure.tool_name}` failure has now "
                    f"occurred {failure.attempt} times in a row with the same "
                    "arguments and error. Do not repeat this call unchanged: "
                    "the error output shows what is wrong, so change the "
                    "arguments or the underlying state before trying again."
                )
            )
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
