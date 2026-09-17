"""Unified tool failure model, normalization, and failure accounting.

Every failed tool call — whether rejected at the validation boundary,
intercepted by a policy/permission middleware, or failed after actually
running — is normalized into one :class:`ToolFailure` shape.

Responsibility split:

* :class:`ToolFailure` / :class:`FailureTracker` only describe and count
  failures ("what failed" and "how many times in a row").
* :class:`~harness_code_agent.runtime.middleware.tool_failure_policy.ToolFailurePolicyMiddleware`
  decides what should happen next (a :class:`FailureAction`), with no side
  effects of its own.
* ``ToolExecutor`` executes the action.  Nothing in this module retries or
  replays a tool call.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .tool_result import ToolResult


# ---------------------------------------------------------------------------
# Taxonomy constants
# ---------------------------------------------------------------------------


class FailureCategory:
    INVALID_CALL = "invalid_call"
    POLICY = "policy"
    RESOURCE = "resource"
    EXECUTION = "execution"
    VERIFICATION = "verification"


class FailurePhase:
    PARSE = "parse"
    SCHEMA_VALIDATION = "schema_validation"
    SEMANTIC_VALIDATION = "semantic_validation"
    POLICY = "policy"
    RESOURCE = "resource"
    EXECUTION = "execution"
    VERIFICATION = "verification"


class FailureVisibility:
    #: Protocol-level failure: the tool never ran; the model can regenerate
    #: the call through a bounded correction loop without task-level cost.
    PROTOCOL = "protocol"
    #: Task-level failure: the result must stay visible to the agent.
    TASK = "task"


class FailureKind:
    # invalid_call
    INVALID_JSON = "invalid_json"
    INVALID_ARGUMENTS = "invalid_arguments"
    INVALID_TOOL_CALL = "invalid_tool_call"
    UNKNOWN_TOOL = "unknown_tool"
    TOOL_SCHEMA_ERROR = "tool_schema_error"
    # policy
    WORKSPACE_ESCAPE = "workspace_escape"
    BLOCKED_COMMAND = "blocked_command"
    PERMISSION_DENIED = "permission_denied"
    APPROVAL_DENIED = "approval_denied"
    POLICY_VIOLATION = "policy_violation"
    PROFILE_BLOCKED = "profile_blocked"
    USER_ATTENTION_REQUIRED = "user_attention_required"
    # resource
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMEOUT = "timeout"
    # execution
    COMMAND_NOT_FOUND = "command_not_found"
    FILE_NOT_FOUND = "file_not_found"
    PROCESS_FAILED = "process_failed"
    TOOL_INTERNAL_ERROR = "tool_internal_error"
    EXECUTION_FAILED = "execution_failed"
    # verification
    VERIFICATION_FAILED = "verification_failed"


class FailureMode:
    #: Feed a correction back and let the model regenerate the call. The
    #: tool was never executed; no runtime replay happens.
    AUTO_RETRY = "auto_retry"
    #: Keep the failure result visible; the agent decides what to do.
    RETURN_TO_AGENT = "return_to_agent"
    #: Budget exhausted or unrecoverable: stop the turn.
    STOP = "stop"
    #: Approval is requested before execution. Post-failure policy never
    #: produces this (the approval flow runs pre-execution); it exists to
    #: document the control vocabulary.
    REQUEST_APPROVAL = "request_approval"


#: invalid_call kinds that require reasoning/config changes rather than a
#: pure protocol repair, hence task-visible.
_TASK_ONLY_INVALID_KINDS = frozenset(
    {FailureKind.UNKNOWN_TOOL, FailureKind.TOOL_SCHEMA_ERROR}
)

#: Kinds that never participate in retry/streak accounting.
_NO_ACCOUNTING_KINDS = frozenset(
    {
        FailureKind.APPROVAL_DENIED,
        FailureKind.USER_ATTENTION_REQUIRED,
        FailureKind.BUDGET_EXCEEDED,
    }
)

# kind -> (category, default phase or None, retryable, replan, user_action)
_KIND_SPECS: dict[str, tuple[str, str | None, bool, bool, bool]] = {
    FailureKind.INVALID_JSON: (FailureCategory.INVALID_CALL, FailurePhase.PARSE, True, False, False),
    FailureKind.INVALID_ARGUMENTS: (FailureCategory.INVALID_CALL, None, True, False, False),
    FailureKind.INVALID_TOOL_CALL: (
        FailureCategory.INVALID_CALL,
        FailurePhase.SCHEMA_VALIDATION,
        True,
        False,
        False,
    ),
    FailureKind.UNKNOWN_TOOL: (
        FailureCategory.INVALID_CALL,
        FailurePhase.SCHEMA_VALIDATION,
        False,
        True,
        False,
    ),
    FailureKind.TOOL_SCHEMA_ERROR: (
        FailureCategory.INVALID_CALL,
        FailurePhase.SCHEMA_VALIDATION,
        False,
        True,
        False,
    ),
    FailureKind.WORKSPACE_ESCAPE: (
        FailureCategory.POLICY,
        FailurePhase.SEMANTIC_VALIDATION,
        True,
        True,
        False,
    ),
    FailureKind.BLOCKED_COMMAND: (
        FailureCategory.POLICY,
        FailurePhase.SEMANTIC_VALIDATION,
        False,
        True,
        False,
    ),
    FailureKind.PERMISSION_DENIED: (
        FailureCategory.POLICY,
        FailurePhase.POLICY,
        False,
        True,
        False,
    ),
    FailureKind.APPROVAL_DENIED: (
        FailureCategory.POLICY,
        FailurePhase.POLICY,
        False,
        False,
        True,
    ),
    FailureKind.POLICY_VIOLATION: (
        FailureCategory.POLICY,
        FailurePhase.POLICY,
        False,
        True,
        False,
    ),
    FailureKind.PROFILE_BLOCKED: (
        FailureCategory.POLICY,
        FailurePhase.POLICY,
        False,
        True,
        False,
    ),
    FailureKind.USER_ATTENTION_REQUIRED: (
        FailureCategory.POLICY,
        FailurePhase.POLICY,
        False,
        False,
        True,
    ),
    FailureKind.BUDGET_EXCEEDED: (
        FailureCategory.RESOURCE,
        FailurePhase.RESOURCE,
        False,
        False,
        False,
    ),
    FailureKind.TIMEOUT: (FailureCategory.RESOURCE, FailurePhase.RESOURCE, True, False, False),
    FailureKind.COMMAND_NOT_FOUND: (
        FailureCategory.EXECUTION,
        FailurePhase.EXECUTION,
        True,
        False,
        False,
    ),
    FailureKind.FILE_NOT_FOUND: (
        FailureCategory.EXECUTION,
        FailurePhase.EXECUTION,
        True,
        False,
        False,
    ),
    FailureKind.PROCESS_FAILED: (
        FailureCategory.EXECUTION,
        FailurePhase.EXECUTION,
        True,
        False,
        False,
    ),
    FailureKind.TOOL_INTERNAL_ERROR: (
        FailureCategory.EXECUTION,
        FailurePhase.EXECUTION,
        True,
        False,
        False,
    ),
    FailureKind.EXECUTION_FAILED: (
        FailureCategory.EXECUTION,
        FailurePhase.EXECUTION,
        True,
        False,
        False,
    ),
    FailureKind.VERIFICATION_FAILED: (
        FailureCategory.VERIFICATION,
        FailurePhase.VERIFICATION,
        True,
        True,
        False,
    ),
}

#: Policy kinds that block execution and should never be auto-retried
#: (approval/user-action kinds excluded).
BLOCKING_POLICY_KINDS = frozenset(
    {
        FailureKind.WORKSPACE_ESCAPE,
        FailureKind.BLOCKED_COMMAND,
        FailureKind.PERMISSION_DENIED,
        FailureKind.POLICY_VIOLATION,
        FailureKind.PROFILE_BLOCKED,
    }
)

_LEGACY_VALIDATION_PHASE = {
    "syntactic": FailurePhase.PARSE,
    "structural": FailurePhase.SCHEMA_VALIDATION,
    "semantic": FailurePhase.SEMANTIC_VALIDATION,
    "parse": FailurePhase.PARSE,
    "schema_validation": FailurePhase.SCHEMA_VALIDATION,
    "semantic_validation": FailurePhase.SEMANTIC_VALIDATION,
}

_SOURCE_DEFAULT_KIND = {
    "permission": FailureKind.PERMISSION_DENIED,
    "approval": FailureKind.APPROVAL_DENIED,
    "tool_policy": FailureKind.POLICY_VIOLATION,
    "delegate_policy": FailureKind.PROFILE_BLOCKED,
    "agent_policy": FailureKind.PROFILE_BLOCKED,
    "user_question": FailureKind.USER_ATTENTION_REQUIRED,
    "budget": FailureKind.BUDGET_EXCEEDED,
    "fallback": FailureKind.BUDGET_EXCEEDED,
    "timeout": FailureKind.TIMEOUT,
    "exception": FailureKind.TOOL_INTERNAL_ERROR,
    "registry": FailureKind.UNKNOWN_TOOL,
    "runtime": FailureKind.EXECUTION_FAILED,
    "native": FailureKind.EXECUTION_FAILED,
    "shell": FailureKind.PROCESS_FAILED,
    "shell_job": FailureKind.EXECUTION_FAILED,
    "mcp": FailureKind.EXECUTION_FAILED,
    "mcporter": FailureKind.EXECUTION_FAILED,
    "browser": FailureKind.EXECUTION_FAILED,
    "unstructured": FailureKind.EXECUTION_FAILED,
    "": FailureKind.EXECUTION_FAILED,
}

#: Sources where output text may refine the generic execution kind.
_TEXT_REFINABLE_SOURCES = frozenset(
    {
        "",
        "native",
        "runtime",
        "shell",
        "shell_job",
        "mcp",
        "mcporter",
        "browser",
        "unstructured",
    }
)


# ---------------------------------------------------------------------------
# Failure model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolFailure:
    """Normalized failure for one tool call."""

    tool_call_id: str
    tool_name: str
    phase: str
    kind: str
    category: str
    message: str
    retryable: bool
    replan_required: bool = False
    user_action_required: bool = False
    attempt: int = 1
    intercepted: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def signature(self) -> str:
        return f"{self.tool_name}|{self.category}|{self.kind}"

    @property
    def visibility(self) -> str:
        if (
            self.category == FailureCategory.INVALID_CALL
            and self.kind not in _TASK_ONLY_INVALID_KINDS
        ):
            return FailureVisibility.PROTOCOL
        return FailureVisibility.TASK

    @property
    def counted_in_tracker(self) -> bool:
        """False for approvals/user attention/budget stops (no retry cost)."""
        return self.kind not in _NO_ACCOUNTING_KINDS

    def stamp_metadata(self, result: ToolResult) -> ToolResult:
        """Return ``result`` with canonical ``failure_*`` metadata added."""
        metadata = dict(result.metadata or {})
        metadata.update(
            {
                "failure_category": self.category,
                "failure_kind": self.kind,
                "failure_phase": self.phase,
                "failure_visibility": self.visibility,
                "failure_retryable": self.retryable,
                "failure_replan_required": self.replan_required,
                "failure_user_action": self.user_action_required,
                "failure_attempt": self.attempt,
                "failure_intercepted": self.intercepted,
            }
        )
        return replace(result, metadata=metadata)

    @classmethod
    def from_tool_error(
        cls,
        *,
        tool_call_id: str,
        tool_name: str,
        error: Any,
        intercepted: bool = True,
    ) -> ToolFailure:
        """Build a failure from a validation ``ToolError`` (structural use)."""
        kind = str(getattr(error, "kind", "") or FailureKind.INVALID_ARGUMENTS)
        category, phase, default_retryable, replan, user_action = _spec_for_kind(kind)
        legacy_phase = str(getattr(error, "phase", "") or "")
        if phase is None:
            phase = _LEGACY_VALIDATION_PHASE.get(
                legacy_phase, FailurePhase.SCHEMA_VALIDATION
            )
        retryable = bool(getattr(error, "retryable", default_retryable))
        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            phase=phase,
            kind=kind,
            category=category,
            message=str(getattr(error, "message", "") or ""),
            retryable=retryable,
            replan_required=replan,
            user_action_required=user_action,
            intercepted=intercepted,
        )

    @classmethod
    def from_result(
        cls,
        *,
        tool_call_id: str,
        tool_name: str,
        result: ToolResult,
        intercepted: bool,
    ) -> ToolFailure:
        """Normalize any failed ToolResult (validation, policy, execution...)."""
        metadata = dict(getattr(result, "metadata", None) or {})

        # Already normalized (e.g. produced via ToolError.to_result).
        canonical_kind = metadata.get("failure_kind")
        if isinstance(canonical_kind, str) and canonical_kind:
            category, _phase, default_retryable, replan, user_action = _spec_for_kind(
                canonical_kind
            )
            phase = str(metadata.get("failure_phase") or _phase or FailurePhase.EXECUTION)
            retry_value = metadata.get("failure_retryable", default_retryable)
            return cls(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                phase=phase,
                kind=canonical_kind,
                category=str(metadata.get("failure_category") or category),
                message=_result_message(result),
                retryable=bool(retry_value),
                replan_required=bool(metadata.get("failure_replan_required", replan)),
                user_action_required=bool(
                    metadata.get("failure_user_action", user_action)
                ),
                attempt=int(metadata.get("failure_attempt", 1) or 1),
                intercepted=bool(
                    metadata.get("failure_intercepted", intercepted)
                ),
            )

        source = str(metadata.get("status_source", "") or "").strip().lower()
        text = _result_message(result)
        lowered = text.lower()

        if source == "validation":
            kind = str(metadata.get("error_kind", "") or FailureKind.INVALID_ARGUMENTS)
            category, phase, default_retryable, replan, user_action = _spec_for_kind(kind)
            legacy_phase = str(metadata.get("validation_phase", "") or "")
            if phase is None:
                phase = _LEGACY_VALIDATION_PHASE.get(
                    legacy_phase, FailurePhase.SEMANTIC_VALIDATION
                )
            retryable = _optional_bool(metadata.get("retryable"), default_retryable)
            return cls(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                phase=phase,
                kind=kind,
                category=category,
                message=text,
                retryable=retryable,
                replan_required=replan,
                user_action_required=user_action,
                intercepted=intercepted,
            )

        if not source and "[approval_denied]" in lowered:
            kind = FailureKind.APPROVAL_DENIED
        elif metadata.get("timed_out"):
            kind = FailureKind.TIMEOUT
        else:
            kind = _SOURCE_DEFAULT_KIND.get(source, FailureKind.EXECUTION_FAILED)

        category, phase, retryable, replan, user_action = _spec_for_kind(kind)

        if kind in {FailureKind.EXECUTION_FAILED, FailureKind.PROCESS_FAILED} and (
            source in _TEXT_REFINABLE_SOURCES
        ):
            kind, category, phase = _refine_by_text(lowered, kind, category, phase)

        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            phase=phase,
            kind=kind,
            category=category,
            message=text,
            retryable=retryable,
            replan_required=replan,
            user_action_required=user_action,
            intercepted=intercepted,
        )


# ---------------------------------------------------------------------------
# FailureAction — the only thing a failure policy returns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureAction:
    mode: str
    message: str | None = None
    stop_reason: str = ""
    stop_limit_type: str = ""
    stop_limit: int | None = None

    @classmethod
    def auto_retry(cls, message: str | None = None) -> FailureAction:
        return cls(mode=FailureMode.AUTO_RETRY, message=message)

    @classmethod
    def return_to_agent(cls) -> FailureAction:
        return cls(mode=FailureMode.RETURN_TO_AGENT)

    @classmethod
    def stop(
        cls,
        *,
        reason: str,
        message: str,
        limit_type: str,
        limit: int,
    ) -> FailureAction:
        return cls(
            mode=FailureMode.STOP,
            message=message,
            stop_reason=reason,
            stop_limit_type=limit_type,
            stop_limit=limit,
        )

    @classmethod
    def request_approval(cls) -> FailureAction:
        return cls(mode=FailureMode.REQUEST_APPROVAL)


# ---------------------------------------------------------------------------
# FailureTracker — per-tool attempts plus turn-level task failure cost
# ---------------------------------------------------------------------------


@dataclass
class FailureStreak:
    tool_name: str
    category: str
    kind: str
    count: int
    last_batch_key: str = ""


@dataclass
class FailureTracker:
    """Per-tool consecutive failure streaks and a turn task-failure budget.

    Updated only on the ToolExecutor main thread, so no locking is needed.
    """

    _streaks: dict[str, FailureStreak] = field(default_factory=dict)
    turn_failure_count: int = 0

    def observe(self, failure: ToolFailure, *, batch_key: str = "") -> ToolFailure:
        """Record a failure; return the failure with its streak attempt set.

        Approvals / user-attention / budget stops do not consume retry
        budget and are returned untouched. Protocol failures drive the
        per-tool streak only; task failures also increment the turn cost.

        ``batch_key`` identifies one assistant tool-call batch. Parallel
        sibling calls in the same batch are a single decision, so identical
        failures inside one batch do not advance the "consecutive attempts"
        streak (they still each consume the turn cost).
        """
        if not failure.counted_in_tracker:
            return failure

        previous = self._streaks.get(failure.tool_name)
        same_batch = bool(batch_key) and previous is not None and previous.last_batch_key == batch_key
        if (
            previous is not None
            and previous.category == failure.category
            and previous.kind == failure.kind
        ):
            count = previous.count if same_batch else previous.count + 1
            streak = replace(previous, count=count, last_batch_key=batch_key)
        else:
            count = 1
            streak = FailureStreak(
                tool_name=failure.tool_name,
                category=failure.category,
                kind=failure.kind,
                count=count,
                last_batch_key=batch_key,
            )
        self._streaks[failure.tool_name] = streak

        if failure.visibility == FailureVisibility.TASK:
            self.turn_failure_count += 1

        return replace(failure, attempt=count)

    def observe_success(self, tool_name: str) -> None:
        self._streaks.pop(tool_name, None)

    def current(self, tool_name: str) -> FailureStreak | None:
        return self._streaks.get(tool_name)

    def reset(self) -> None:
        self._streaks.clear()
        self.turn_failure_count = 0


def failure_batch_key(messages: list[dict] | None) -> str:
    """Stable key for the latest assistant tool-call batch.

    Parallel calls emitted in one assistant message share the same key, so
    identical sibling failures are not counted as consecutive attempts.
    """
    for message in reversed(messages or []):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        ids = [
            str(call.get("id") or "")
            for call in tool_calls
            if isinstance(call, dict) and call.get("id")
        ]
        if ids:
            return "|".join(ids)
        return ""
    return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec_for_kind(
    kind: str,
) -> tuple[str, str | None, bool, bool, bool]:
    spec = _KIND_SPECS.get(kind)
    if spec is None:
        return (
            FailureCategory.EXECUTION,
            FailurePhase.EXECUTION,
            True,
            False,
            False,
        )
    return spec


def _result_message(result: ToolResult) -> str:
    return str(result.error or result.output or "")


def _optional_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _refine_by_text(
    lowered: str,
    kind: str,
    category: str,
    phase: str,
) -> tuple[str, str, str]:
    if "timed out" in lowered or "timeout" in lowered:
        return FailureKind.TIMEOUT, FailureCategory.RESOURCE, FailurePhase.RESOURCE
    if "command not found" in lowered:
        return FailureKind.COMMAND_NOT_FOUND, category, phase
    if "no such file or directory" in lowered or "file not found" in lowered:
        return FailureKind.FILE_NOT_FOUND, category, phase
    if "traceback" in lowered or "modulenotfounderror" in lowered:
        return FailureKind.TOOL_INTERNAL_ERROR, category, phase
    return kind, category, phase
