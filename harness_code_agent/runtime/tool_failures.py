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

import hashlib
import json
import re
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
    #: Workspace write quota or free-disk floor reached.
    QUOTA_EXCEEDED = "quota_exceeded"
    #: Subagent concurrency/open-agent capacity is full.
    SUBAGENT_CAPACITY = "subagent_capacity"
    # execution
    #: A backend-specific timeout was hit (run_bash, HTTP, MCP transport...).
    TIMEOUT = "timeout"
    COMMAND_NOT_FOUND = "command_not_found"
    FILE_NOT_FOUND = "file_not_found"
    PROCESS_FAILED = "process_failed"
    TOOL_INTERNAL_ERROR = "tool_internal_error"
    EXECUTION_FAILED = "execution_failed"
    # verification
    VERIFICATION_FAILED = "verification_failed"


class FailureMode:
    #: The call never ran; feed a correction back to the model and let it
    #: *regenerate* the call. The runtime never replays a tool itself.
    REQUEST_REGENERATION = "request_regeneration"
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

# kind -> (category, default phase or None, retryable, user_action)
_KIND_SPECS: dict[str, tuple[str, str | None, bool, bool]] = {
    FailureKind.INVALID_JSON: (FailureCategory.INVALID_CALL, FailurePhase.PARSE, True, False),
    FailureKind.INVALID_ARGUMENTS: (FailureCategory.INVALID_CALL, None, True, False),
    FailureKind.INVALID_TOOL_CALL: (
        FailureCategory.INVALID_CALL,
        FailurePhase.SCHEMA_VALIDATION,
        True,
        False,
    ),
    FailureKind.UNKNOWN_TOOL: (
        FailureCategory.INVALID_CALL,
        FailurePhase.SCHEMA_VALIDATION,
        False,
        False,
    ),
    FailureKind.TOOL_SCHEMA_ERROR: (
        FailureCategory.INVALID_CALL,
        FailurePhase.SCHEMA_VALIDATION,
        False,
        False,
    ),
    FailureKind.WORKSPACE_ESCAPE: (
        FailureCategory.POLICY,
        FailurePhase.SEMANTIC_VALIDATION,
        True,
        False,
    ),
    FailureKind.BLOCKED_COMMAND: (
        FailureCategory.POLICY,
        FailurePhase.SEMANTIC_VALIDATION,
        False,
        False,
    ),
    FailureKind.PERMISSION_DENIED: (
        FailureCategory.POLICY,
        FailurePhase.POLICY,
        False,
        False,
    ),
    FailureKind.APPROVAL_DENIED: (
        FailureCategory.POLICY,
        FailurePhase.POLICY,
        False,
        True,
    ),
    FailureKind.POLICY_VIOLATION: (
        FailureCategory.POLICY,
        FailurePhase.POLICY,
        False,
        False,
    ),
    FailureKind.PROFILE_BLOCKED: (
        FailureCategory.POLICY,
        FailurePhase.POLICY,
        False,
        False,
    ),
    FailureKind.USER_ATTENTION_REQUIRED: (
        FailureCategory.POLICY,
        FailurePhase.POLICY,
        False,
        True,
    ),
    FailureKind.BUDGET_EXCEEDED: (
        FailureCategory.RESOURCE,
        FailurePhase.RESOURCE,
        False,
        False,
    ),
    FailureKind.QUOTA_EXCEEDED: (
        FailureCategory.RESOURCE,
        FailurePhase.RESOURCE,
        False,
        False,
    ),
    FailureKind.SUBAGENT_CAPACITY: (
        FailureCategory.RESOURCE,
        FailurePhase.RESOURCE,
        True,
        False,
    ),
    # Timeout is a backend execution outcome (the backend killed its own
    # work), not a resource-capacity classification.
    FailureKind.TIMEOUT: (FailureCategory.EXECUTION, FailurePhase.EXECUTION, True, False),
    FailureKind.COMMAND_NOT_FOUND: (
        FailureCategory.EXECUTION,
        FailurePhase.EXECUTION,
        True,
        False,
    ),
    FailureKind.FILE_NOT_FOUND: (
        FailureCategory.EXECUTION,
        FailurePhase.EXECUTION,
        True,
        False,
    ),
    FailureKind.PROCESS_FAILED: (
        FailureCategory.EXECUTION,
        FailurePhase.EXECUTION,
        True,
        False,
    ),
    FailureKind.TOOL_INTERNAL_ERROR: (
        FailureCategory.EXECUTION,
        FailurePhase.EXECUTION,
        True,
        False,
    ),
    FailureKind.EXECUTION_FAILED: (
        FailureCategory.EXECUTION,
        FailurePhase.EXECUTION,
        True,
        False,
    ),
    FailureKind.VERIFICATION_FAILED: (
        FailureCategory.VERIFICATION,
        FailurePhase.VERIFICATION,
        True,
        False,
    ),
}

#: Policy kinds that block execution and should never trigger a regeneration
#: request (approval/user-action kinds excluded).
BLOCKING_POLICY_KINDS = frozenset(
    {
        FailureKind.WORKSPACE_ESCAPE,
        FailureKind.BLOCKED_COMMAND,
        FailureKind.PERMISSION_DENIED,
        FailureKind.POLICY_VIOLATION,
        FailureKind.PROFILE_BLOCKED,
    }
)

#: Categories where the streak must reflect *what exactly* keeps failing,
#: not just the failure kind. A different error signature or different
#: arguments means the agent changed something, so the consecutive streak
#: resets ("new information"). Protocol streaks stay coarse.
FINGERPRINTED_CATEGORIES = frozenset(
    {
        FailureCategory.POLICY,
        FailureCategory.EXECUTION,
        FailureCategory.RESOURCE,
        FailureCategory.VERIFICATION,
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
    "quota": FailureKind.QUOTA_EXCEEDED,
    "resource": FailureKind.SUBAGENT_CAPACITY,
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

#: metadata.resource_kind values for status_source="resource"/"quota".
_RESOURCE_KIND_MAP = {
    "subagent_capacity": FailureKind.SUBAGENT_CAPACITY,
    "workspace_quota": FailureKind.QUOTA_EXCEEDED,
    "free_disk": FailureKind.QUOTA_EXCEEDED,
    "quota": FailureKind.QUOTA_EXCEEDED,
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
    user_action_required: bool = False
    attempt: int = 1
    intercepted: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    #: Call arguments, used to build the fine-grained repetition fingerprint.
    tool_args: dict[str, Any] | None = None
    #: Fine fingerprint (tool + kind + normalized args + error signature);
    #: empty for failures constructed without repetition accounting.
    fingerprint: str = ""

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
                "failure_user_action": self.user_action_required,
                "failure_attempt": self.attempt,
                "failure_intercepted": self.intercepted,
                "failure_fingerprint": self.fingerprint,
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
        tool_args: dict[str, Any] | None = None,
    ) -> ToolFailure:
        """Build a failure from a validation ``ToolError`` (structural use)."""
        kind = str(getattr(error, "kind", "") or FailureKind.INVALID_ARGUMENTS)
        category, phase, default_retryable, user_action = _spec_for_kind(kind)
        legacy_phase = str(getattr(error, "phase", "") or "")
        if phase is None:
            phase = _LEGACY_VALIDATION_PHASE.get(
                legacy_phase, FailurePhase.SCHEMA_VALIDATION
            )
        retryable = bool(getattr(error, "retryable", default_retryable))
        message = str(getattr(error, "message", "") or "")
        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            phase=phase,
            kind=kind,
            category=category,
            message=message,
            retryable=retryable,
            user_action_required=user_action,
            intercepted=intercepted,
            tool_args=tool_args,
            fingerprint=failure_fingerprint(tool_name, kind, tool_args, message),
        )

    @classmethod
    def from_result(
        cls,
        *,
        tool_call_id: str,
        tool_name: str,
        result: ToolResult,
        intercepted: bool,
        tool_args: dict[str, Any] | None = None,
    ) -> ToolFailure:
        """Normalize any failed ToolResult (validation, policy, execution...)."""
        metadata = dict(getattr(result, "metadata", None) or {})

        # Already normalized (e.g. produced via ToolError.to_result).
        canonical_kind = metadata.get("failure_kind")
        if isinstance(canonical_kind, str) and canonical_kind:
            category, _phase, default_retryable, user_action = _spec_for_kind(
                canonical_kind
            )
            phase = str(metadata.get("failure_phase") or _phase or FailurePhase.EXECUTION)
            retry_value = metadata.get("failure_retryable", default_retryable)
            message = _result_message(result)
            return cls(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                phase=phase,
                kind=canonical_kind,
                category=str(metadata.get("failure_category") or category),
                message=message,
                retryable=bool(retry_value),
                user_action_required=bool(
                    metadata.get("failure_user_action", user_action)
                ),
                attempt=int(metadata.get("failure_attempt", 1) or 1),
                intercepted=bool(
                    metadata.get("failure_intercepted", intercepted)
                ),
                tool_args=tool_args,
                fingerprint=failure_fingerprint(tool_name, canonical_kind, tool_args, message),
            )

        source = str(metadata.get("status_source", "") or "").strip().lower()
        text = _result_message(result)
        lowered = text.lower()

        if source == "validation":
            kind = str(metadata.get("error_kind", "") or FailureKind.INVALID_ARGUMENTS)
            category, phase, default_retryable, user_action = _spec_for_kind(kind)
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
                user_action_required=user_action,
                intercepted=intercepted,
                tool_args=tool_args,
                fingerprint=failure_fingerprint(tool_name, kind, tool_args, text),
            )

        if not source and "[approval_denied]" in lowered:
            kind = FailureKind.APPROVAL_DENIED
        elif metadata.get("timed_out"):
            kind = FailureKind.TIMEOUT
        elif source in {"resource", "quota"}:
            resource_kind = str(metadata.get("resource_kind", "") or "").strip().lower()
            kind = _RESOURCE_KIND_MAP.get(
                resource_kind,
                _SOURCE_DEFAULT_KIND.get(source, FailureKind.EXECUTION_FAILED),
            )
        else:
            kind = _SOURCE_DEFAULT_KIND.get(source, FailureKind.EXECUTION_FAILED)

        category, phase, retryable, user_action = _spec_for_kind(kind)

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
            user_action_required=user_action,
            intercepted=intercepted,
            tool_args=tool_args,
            fingerprint=failure_fingerprint(tool_name, kind, tool_args, text),
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
    def request_regeneration(cls, message: str | None = None) -> FailureAction:
        return cls(mode=FailureMode.REQUEST_REGENERATION, message=message)

    @classmethod
    def return_to_agent(cls, message: str | None = None) -> FailureAction:
        return cls(mode=FailureMode.RETURN_TO_AGENT, message=message)

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
    #: Fine fingerprint for execution/resource/verification streaks; the
    #: count only advances while this stays identical.
    fingerprint: str = ""


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

        fingerprint = failure.fingerprint or failure_fingerprint(
            failure.tool_name,
            failure.kind,
            failure.tool_args,
            failure.message,
        )
        previous = self._streaks.get(failure.tool_name)
        same_batch = bool(batch_key) and previous is not None and previous.last_batch_key == batch_key
        same_kind = (
            previous is not None
            and previous.category == failure.category
            and previous.kind == failure.kind
        )
        # For real execution outcomes a changed fingerprint is new
        # information: the agent changed the command or got a different
        # error, so the "same failure repeated" streak restarts at 1.
        fingerprint_resets = (
            same_kind
            and failure.category in FINGERPRINTED_CATEGORIES
            and previous.fingerprint != fingerprint
        )
        if same_kind and not fingerprint_resets:
            count = previous.count if same_batch else previous.count + 1
            streak = replace(
                previous,
                count=count,
                last_batch_key=batch_key,
                fingerprint=fingerprint,
            )
        else:
            count = 1
            streak = FailureStreak(
                tool_name=failure.tool_name,
                category=failure.category,
                kind=failure.kind,
                count=count,
                last_batch_key=batch_key,
                fingerprint=fingerprint,
            )
        self._streaks[failure.tool_name] = streak

        if failure.visibility == FailureVisibility.TASK:
            self.turn_failure_count += 1

        return replace(failure, attempt=count, fingerprint=fingerprint)

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
) -> tuple[str, str | None, bool, bool]:
    spec = _KIND_SPECS.get(kind)
    if spec is None:
        return (
            FailureCategory.EXECUTION,
            FailurePhase.EXECUTION,
            True,
            False,
        )
    return spec


def _result_message(result: ToolResult) -> str:
    return str(result.error or result.output or "")


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_WHITESPACE_RE = re.compile(r"\s+")

#: Keep the fingerprint material compact and focused on the discriminating
#: part of the error; pytest/traceback headers are usually much longer.
_FAILURE_TEXT_LIMIT = 800


def _normalize_failure_args(args: Any) -> str:
    if not isinstance(args, dict):
        return ""
    try:
        return json.dumps(
            args,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return str(args)


def _normalize_failure_text(text: str) -> str:
    normalized = _ANSI_ESCAPE_RE.sub("", str(text or "")).lower()
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized[:_FAILURE_TEXT_LIMIT]


def failure_fingerprint(
    tool_name: str,
    kind: str,
    args: Any,
    message: str,
) -> str:
    """Hash the objective identity of a repeated failure.

    Material: tool name + failure kind + canonical arguments JSON + the
    normalized error text. Different commands or different error output
    produce different fingerprints, so genuine progress across attempts is
    not mistaken for repetition.
    """
    material = "\x1f".join(
        (
            str(tool_name),
            str(kind),
            _normalize_failure_args(args),
            _normalize_failure_text(message),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


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
        return FailureKind.TIMEOUT, FailureCategory.EXECUTION, FailurePhase.EXECUTION
    if "command not found" in lowered:
        return FailureKind.COMMAND_NOT_FOUND, category, phase
    if "no such file or directory" in lowered or "file not found" in lowered:
        return FailureKind.FILE_NOT_FOUND, category, phase
    if "traceback" in lowered or "modulenotfounderror" in lowered:
        return FailureKind.TOOL_INTERNAL_ERROR, category, phase
    return kind, category, phase
