import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _install_fake_openai_module() -> None:
    openai = types.ModuleType("openai")

    class OpenAI:
        def __init__(self, *args, **kwargs):
            pass

    openai.OpenAI = OpenAI
    sys.modules["openai"] = openai


_install_fake_openai_module()

from harness_code_agent.agent.conversation import Agent, AgentConversation
from harness_code_agent.agent.runtime_state import AgentRuntimeState
from harness_code_agent.runtime import tools
from harness_code_agent.runtime.middlewares import (
    AgentMiddleware,
    ToolFailurePolicyMiddleware,
    ToolGuardMiddleware,
)
from harness_code_agent.runtime.permissions import PermissionPolicy
from harness_code_agent.runtime.tool_call_validation import ToolError
from harness_code_agent.runtime.tool_context import ToolContext
from harness_code_agent.runtime.tool_failures import (
    FailureCategory,
    FailureKind,
    FailureMode,
    FailureTracker,
    ToolFailure,
)
from harness_code_agent.runtime.tool_result import ToolResult
from harness_code_agent.sessions.events import EventBus, classify_tool_failure
from harness_code_agent.workspace.service import WorkspaceService

_READ_EFFECT = tools.CallEffect((tools.ResourceClaim("workspace", "*", "global", "read"),))

_PROBE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "probe",
        "description": "probe",
        "parameters": {
            "type": "object",
            "required": ["value"],
            "properties": {
                "value": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "additionalProperties": False,
        },
    },
}


def _tool_call(call_id: str, name: str, args=None):
    arguments = args if isinstance(args, str) else json.dumps(args or {})
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


_RUN_BASH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_bash",
        "description": "run a shell command",
        "parameters": {
            "type": "object",
            "required": ["command"],
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer"},
            },
            "additionalProperties": False,
        },
    },
}
_RECURSIVE_LIST_ARGS = {"command": "Get-ChildItem -Recurse"}


def _shell_guard_registry():
    registry = tools.ToolRegistry()

    def _must_not_run(**kwargs):
        raise AssertionError(f"guarded command must not execute: {kwargs}")

    registry.register(_RUN_BASH_SCHEMA, _must_not_run, permission="shell")
    return registry


class _RepeatCompletions:
    """Returns the same tool calls for the first ``repeat`` requests, then text.

    Regenerated calls get fresh ids (``<id>__r<n>``), matching the real API
    contract where every assistant tool-call batch has unique call ids.
    """

    def __init__(self, tool_calls, repeat=1):
        self.calls = 0
        self._tool_calls = tool_calls
        self._repeat = repeat

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self._repeat:
            calls = self._tool_calls
            if self.calls > 1:
                calls = [
                    SimpleNamespace(
                        id=f"{call.id}__r{self.calls}",
                        type=call.type,
                        function=call.function,
                    )
                    for call in calls
                ]
            message = SimpleNamespace(content=None, tool_calls=calls)
            finish_reason = "tool_calls"
        else:
            message = SimpleNamespace(content="done", tool_calls=None)
            finish_reason = "stop"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
            usage=None,
        )


def _conversation(root: Path, registry: tools.ToolRegistry, tool_calls, middlewares, repeat=1):
    context = ToolContext(
        workspace=WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots"),
        permission_policy=PermissionPolicy(mode="danger-full-access"),
        event_bus=EventBus(),
        tool_registry=registry,
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_RepeatCompletions(tool_calls, repeat=repeat))
    )
    with patch("harness_code_agent.agent.conversation.get_client", return_value=fake_client):
        conversation = AgentConversation(
            Agent(
                "main_agent",
                "system",
                use_tools=True,
                tool_schemas=registry.schemas(),
                middlewares=list(middlewares),
                tool_context=context,
            )
        )
    return conversation, context


class _FakeRegistry:
    def __init__(self, schema):
        self._schema = schema

    def schema_for(self, name):
        return self._schema if name == "probe" else None


class FailureSpyMiddleware(AgentMiddleware):
    def __init__(self):
        self.calls = []

    def on_tool_failure(self, failure, messages, runtime_state=None, agent_name=None):
        self.calls.append(failure)
        return None


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class NormalizationTests(unittest.TestCase):
    def test_tool_error_to_result_carries_canonical_and_legacy_keys(self):
        result = ToolError(
            kind="invalid_arguments",
            message="field `offset` must be integer",
            retryable=True,
            phase="structural",
        ).to_result("probe")

        self.assertEqual(result.metadata["status_source"], "validation")
        # Legacy keys remain untouched.
        self.assertEqual(result.metadata["error_kind"], "invalid_arguments")
        self.assertTrue(result.metadata["retryable"])
        self.assertEqual(result.metadata["validation_phase"], "structural")
        # Canonical keys are added.
        self.assertEqual(result.metadata["failure_category"], FailureCategory.INVALID_CALL)
        self.assertEqual(result.metadata["failure_kind"], FailureKind.INVALID_ARGUMENTS)
        self.assertEqual(result.metadata["failure_phase"], "schema_validation")
        self.assertEqual(result.metadata["failure_visibility"], "protocol")
        self.assertTrue(result.metadata["failure_retryable"])
        self.assertEqual(result.metadata["failure_attempt"], 1)
        self.assertTrue(result.metadata["failure_intercepted"])
        self.assertEqual(classify_tool_failure(result), "validation_error")

    def test_from_result_maps_permission_approval_budget_and_exception(self):
        cases = [
            ({"status_source": "permission"}, FailureKind.PERMISSION_DENIED, "task", "tool_error"),
            ({"status_source": "approval"}, FailureKind.APPROVAL_DENIED, "task", "user_cancelled"),
            ({"status_source": "budget"}, FailureKind.BUDGET_EXCEEDED, "task", "runtime_error"),
            ({"status_source": "exception"}, FailureKind.TOOL_INTERNAL_ERROR, "task", "runtime_error"),
            ({"status_source": "shell"}, FailureKind.PROCESS_FAILED, "task", "runtime_error"),
        ]
        for metadata, kind, visibility, event_category in cases:
            with self.subTest(metadata=metadata):
                result = ToolResult(
                    tool="probe",
                    status="failed",
                    output="boom",
                    error="boom",
                    metadata=dict(metadata),
                )
                failure = ToolFailure.from_result(
                    tool_call_id="tc", tool_name="probe", result=result, intercepted=False
                )
                self.assertEqual(failure.kind, kind)
                self.assertEqual(failure.visibility, visibility)
                self.assertEqual(classify_tool_failure(failure.stamp_metadata(result)), event_category)

    def test_from_result_never_infers_approval_from_text(self):
        # Regression: a legacy plain-text block containing the old marker must
        # not be classified as an approval denial without status_source metadata.
        result = ToolResult(
            tool="probe",
            status="failed",
            output="[approval_denied] legacy middleware said no",
            error="[approval_denied] legacy middleware said no",
            metadata={},
        )
        failure = ToolFailure.from_result(
            tool_call_id="tc", tool_name="probe", result=result, intercepted=True
        )
        self.assertNotEqual(failure.kind, FailureKind.APPROVAL_DENIED)

    def test_from_result_classify_event_category_after_stamp(self):
        result = ToolResult(
            tool="probe",
            status="failed",
            output="boom",
            error="boom",
            metadata={"status_source": "permission"},
        )
        failure = ToolFailure.from_result(
            tool_call_id="tc", tool_name="probe", result=result, intercepted=True
        )
        stamped = failure.stamp_metadata(result)
        self.assertEqual(classify_tool_failure(stamped), "tool_error")

    def test_text_refines_generic_execution_failure(self):
        cases = [
            ("The operation timed out", FailureKind.TIMEOUT, FailureCategory.EXECUTION),
            ("bash: pytest: command not found", FailureKind.COMMAND_NOT_FOUND, FailureCategory.EXECUTION),
            ("cat: missing.txt: No such file or directory", FailureKind.FILE_NOT_FOUND, FailureCategory.EXECUTION),
            ("Traceback: ModuleNotFoundError: x", FailureKind.TOOL_INTERNAL_ERROR, FailureCategory.EXECUTION),
        ]
        for text, kind, category in cases:
            with self.subTest(text=text):
                result = ToolResult(
                    tool="run_bash",
                    status="failed",
                    output=text,
                    error=text,
                    metadata={"status_source": "shell", "return_code": 1},
                )
                failure = ToolFailure.from_result(
                    tool_call_id="tc", tool_name="run_bash", result=result, intercepted=False
                )
                self.assertEqual(failure.kind, kind)
                self.assertEqual(failure.category, category)

    def test_unknown_source_defaults_to_execution_failed(self):
        result = ToolResult(
            tool="mcp__x",
            status="failed",
            output="weird",
            error="weird",
            metadata={"status_source": "mystery"},
        )
        failure = ToolFailure.from_result(
            tool_call_id="tc", tool_name="mcp__x", result=result, intercepted=False
        )
        self.assertEqual(failure.kind, FailureKind.EXECUTION_FAILED)
        self.assertEqual(failure.category, FailureCategory.EXECUTION)
        self.assertEqual(failure.visibility, "task")

    def test_resource_status_source_kind_taxonomy(self):
        cases = [
            (
                {"status_source": "resource", "resource_kind": "subagent_capacity"},
                FailureKind.SUBAGENT_CAPACITY,
                FailureCategory.RESOURCE,
                True,
            ),
            (
                {"status_source": "resource", "resource_kind": "workspace_quota"},
                FailureKind.QUOTA_EXCEEDED,
                FailureCategory.RESOURCE,
                False,
            ),
            (
                {"status_source": "resource", "resource_kind": "free_disk"},
                FailureKind.QUOTA_EXCEEDED,
                FailureCategory.RESOURCE,
                False,
            ),
            # Legacy emitter of the pre-taxonomy source keeps mapping.
            (
                {"status_source": "quota"},
                FailureKind.QUOTA_EXCEEDED,
                FailureCategory.RESOURCE,
                False,
            ),
        ]
        for metadata, kind, category, retryable in cases:
            with self.subTest(metadata=metadata):
                result = ToolResult(
                    tool="spawn_agent",
                    status="failed",
                    output="busy",
                    error="busy",
                    metadata=metadata,
                )
                failure = ToolFailure.from_result(
                    tool_call_id="tc", tool_name="spawn_agent", result=result,
                    intercepted=False,
                )
                self.assertEqual(failure.kind, kind)
                self.assertEqual(failure.category, category)
                self.assertIs(failure.retryable, retryable)
                self.assertTrue(failure.counted_in_tracker)

    def test_backend_timeout_is_execution_not_resource(self):
        result = ToolResult(
            tool="run_bash",
            status="failed",
            output="timed out",
            error="timed out",
            metadata={"status_source": "shell", "timed_out": True},
        )
        failure = ToolFailure.from_result(
            tool_call_id="tc", tool_name="run_bash", result=result, intercepted=False
        )
        self.assertEqual(failure.kind, FailureKind.TIMEOUT)
        self.assertEqual(failure.category, FailureCategory.EXECUTION)
        self.assertEqual(failure.phase, "execution")


class FailureTrackerTests(unittest.TestCase):
    def _observe(self, tracker, *, kind=FailureKind.INVALID_ARGUMENTS, tool="probe",
                 message="bad", args=None):
        result = ToolResult(
            tool=tool,
            status="failed",
            output=message,
            error=message,
            metadata={"failure_kind": kind},
        )
        failure = ToolFailure.from_result(
            tool_call_id="tc", tool_name=tool, result=result,
            intercepted=True, tool_args=args,
        )
        return tracker.observe(failure)

    def test_same_tool_same_kind_accumulates_other_kind_restarts(self):
        tracker = FailureTracker()
        first = self._observe(tracker)
        second = self._observe(tracker)
        self.assertEqual((first.attempt, second.attempt), (1, 2))
        switched = self._observe(tracker, kind=FailureKind.INVALID_JSON)
        self.assertEqual(switched.attempt, 1)
        self.assertEqual(tracker.current("probe").kind, FailureKind.INVALID_JSON)

    def test_protocol_failures_do_not_consume_turn_budget_task_failures_do(self):
        tracker = FailureTracker()
        self._observe(tracker, kind=FailureKind.INVALID_ARGUMENTS)
        self._observe(tracker, kind=FailureKind.INVALID_ARGUMENTS)
        self.assertEqual(tracker.turn_failure_count, 0)
        self._observe(tracker, kind=FailureKind.PROCESS_FAILED)
        self.assertEqual(tracker.turn_failure_count, 1)

    def test_approval_and_budget_are_not_counted(self):
        tracker = FailureTracker()
        self._observe(tracker, kind=FailureKind.APPROVAL_DENIED)
        self._observe(tracker, kind=FailureKind.BUDGET_EXCEEDED)
        self.assertIsNone(tracker.current("probe"))
        self.assertEqual(tracker.turn_failure_count, 0)

    def test_same_batch_identical_failures_do_not_advance_streak(self):
        failure = self._observe(FailureTracker(), kind=FailureKind.POLICY_VIOLATION)
        tracker = FailureTracker()

        f1 = tracker.observe(failure, batch_key="batch-1")
        f2 = tracker.observe(failure, batch_key="batch-1")
        f3 = tracker.observe(failure, batch_key="batch-2")

        # Two sibling calls share one decision; the next batch is attempt 2.
        self.assertEqual([f.attempt for f in (f1, f2, f3)], [1, 1, 2])
        # Every task-visible failure still pays the turn cost individually.
        self.assertEqual(tracker.turn_failure_count, 3)

    def test_observes_without_batch_key_keep_consecutive_semantics(self):
        failure = self._observe(FailureTracker(), kind=FailureKind.POLICY_VIOLATION)
        tracker = FailureTracker()

        attempts = [tracker.observe(failure).attempt for _ in range(3)]

        self.assertEqual(attempts, [1, 2, 3])

    def test_identical_execution_failure_accumulates_by_fingerprint(self):
        tracker = FailureTracker()

        attempts = [
            self._observe(
                tracker,
                kind=FailureKind.PROCESS_FAILED,
                message="tests/test_x.py::test_a assert 1 == 2",
                args={"command": "pytest tests/test_x.py"},
            ).attempt
            for _ in range(4)
        ]

        self.assertEqual(attempts, [1, 2, 3, 4])

    def test_changed_error_signature_resets_execution_streak(self):
        tracker = FailureTracker()

        a1 = self._observe(
            tracker, kind=FailureKind.PROCESS_FAILED,
            message="ModuleNotFoundError: No module named 'numpy'",
            args={"command": "pytest"},
        )
        a2 = self._observe(
            tracker, kind=FailureKind.PROCESS_FAILED,
            message="ModuleNotFoundError: No module named 'numpy'",
            args={"command": "pytest"},
        )
        a3 = self._observe(
            tracker, kind=FailureKind.PROCESS_FAILED,
            message="AssertionError: expected 2 got 3",
            args={"command": "pytest"},
        )

        self.assertEqual([a1.attempt, a2.attempt, a3.attempt], [1, 2, 1])
        # Progressing failures are still task-visible and keep paying turn cost.
        self.assertEqual(tracker.turn_failure_count, 3)

    def test_changed_args_reset_execution_streak(self):
        tracker = FailureTracker()

        a1 = self._observe(
            tracker, kind=FailureKind.PROCESS_FAILED, message="1 failed",
            args={"command": "pytest tests/test_a.py"},
        )
        a2 = self._observe(
            tracker, kind=FailureKind.PROCESS_FAILED, message="1 failed",
            args={"command": "pytest tests/test_b.py"},
        )

        self.assertEqual([a1.attempt, a2.attempt], [1, 1])
        self.assertNotEqual(a1.fingerprint, a2.fingerprint)

    def test_protocol_streak_stays_coarse_across_messages(self):
        tracker = FailureTracker()

        attempts = [
            self._observe(tracker, message=f"invalid value {i}").attempt
            for i in range(3)
        ]

        self.assertEqual(attempts, [1, 2, 3])
        self.assertEqual(tracker.turn_failure_count, 0)

    def test_success_clears_streak_and_reset_clears_everything(self):
        tracker = FailureTracker()
        self._observe(tracker)
        self._observe(tracker, kind=FailureKind.PROCESS_FAILED)
        tracker.observe_success("probe")
        self.assertIsNone(tracker.current("probe"))
        self.assertEqual(tracker.turn_failure_count, 1)
        tracker.reset()
        self.assertEqual(tracker.turn_failure_count, 0)


# ---------------------------------------------------------------------------
# Policy decision table (pure decision, no side effects)
# ---------------------------------------------------------------------------


class PolicyDecisionTests(unittest.TestCase):
    def setUp(self):
        self.middleware = ToolFailurePolicyMiddleware(tool_registry=_FakeRegistry(_PROBE_SCHEMA))
        self.state = AgentRuntimeState()

    def _observe(self, kind=FailureKind.INVALID_ARGUMENTS, *, tool="probe",
                 message="bad", args=None):
        result = ToolResult(
            tool=tool,
            status="failed",
            output=message,
            error=message,
            metadata={"failure_kind": kind},
        )
        failure = ToolFailure.from_result(
            tool_call_id="tc", tool_name=tool, result=result,
            intercepted=True, tool_args=args,
        )
        return self.state.failures.observe(failure)

    def _decide(self, failure):
        return self.middleware.on_tool_failure(
            failure, [], runtime_state=self.state, agent_name="main_agent"
        )

    def test_schema_streak_regenerate_then_correction_then_stop(self):
        first = self._decide(self._observe())
        self.assertEqual(first.mode, FailureMode.REQUEST_REGENERATION)
        self.assertIsNone(first.message)

        second = self._decide(self._observe())
        self.assertEqual(second.mode, FailureMode.REQUEST_REGENERATION)
        self.assertIn("Allowed arguments are exactly", second.message)
        self.assertIn("value: string (required)", second.message)
        self.assertIn("limit: integer", second.message)
        # Middleware must not perform the stop itself.
        self.assertFalse(self.state.fallback.stop_requested)

        third = self._decide(self._observe())
        self.assertEqual(third.mode, FailureMode.STOP)
        self.assertEqual(third.stop_reason, "retry_budget_exhausted")
        self.assertEqual(third.stop_limit_type, "schema_failures")
        self.assertFalse(self.state.fallback.stop_requested)
        # Protocol failures never consume the turn budget.
        self.assertEqual(self.state.failures.turn_failure_count, 0)

    def test_unknown_tool_returns_once_then_stops(self):
        first = self._decide(self._observe(FailureKind.UNKNOWN_TOOL, tool="nope"))
        self.assertEqual(first.mode, FailureMode.RETURN_TO_AGENT)
        second = self._decide(self._observe(FailureKind.UNKNOWN_TOOL, tool="nope"))
        self.assertEqual(second.mode, FailureMode.STOP)
        self.assertEqual(second.stop_reason, "retry_budget_exhausted")

    def test_tool_schema_error_is_exposed_as_infrastructure_error(self):
        first = self._decide(self._observe(FailureKind.TOOL_SCHEMA_ERROR))
        self.assertEqual(first.mode, FailureMode.RETURN_TO_AGENT)
        # The explanation is attached on the first occurrence: it is a
        # tool-definition error, not an argument error the model could repair.
        self.assertIn("invalid runtime schema", first.message)
        self.assertIn("tool-definition error, not an argument error", first.message)
        self.assertIn("Retrying the same call unchanged will not fix it", first.message)
        self.assertIn("report the blocker", first.message)
        # No substitute capability is ever suggested.
        self.assertNotIn("another tool", first.message)
        self.assertNotIn("different tool", first.message)
        self.assertFalse(self.state.fallback.stop_requested)

        # Repeating the broken tool keeps exposing the same infra error.
        # No regeneration (the schema cannot change mid-turn), no stop.
        actions = [
            self._decide(self._observe(FailureKind.TOOL_SCHEMA_ERROR))
            for _ in range(3)
        ]
        for action in actions:
            self.assertEqual(action.mode, FailureMode.RETURN_TO_AGENT)
            self.assertIn("tool-definition error", action.message)
        self.assertFalse(self.state.fallback.stop_requested)
        # Failures are still counted for observability, but nothing acts on
        # the count as a circuit breaker.
        self.assertEqual(self.state.failures.turn_failure_count, 4)

    def test_approval_denied_never_stops_or_counts(self):
        for _ in range(5):
            action = self._decide(self._observe(FailureKind.APPROVAL_DENIED))
            self.assertEqual(action.mode, FailureMode.RETURN_TO_AGENT)
        self.assertFalse(self.state.fallback.stop_requested)
        self.assertEqual(self.state.failures.turn_failure_count, 0)

    def test_blocking_policy_exposes_once_then_guides_but_never_stops(self):
        args = {"command": "grep -r secret ."}

        first = self._decide(self._observe(FailureKind.POLICY_VIOLATION, args=args))
        self.assertEqual(first.mode, FailureMode.RETURN_TO_AGENT)
        self.assertIsNone(first.message)

        second = self._decide(self._observe(FailureKind.POLICY_VIOLATION, args=args))
        self.assertEqual(second.mode, FailureMode.RETURN_TO_AGENT)
        self.assertIn("blocked by policy", second.message)
        self.assertIn("2 identical attempts", second.message)
        # Operation-level wording: comply or get permission, never "switch
        # tool" (which would invite performing the same forbidden action via
        # another tool).
        self.assertNotIn("another tool", second.message)
        self.assertNotIn("different tool", second.message)
        self.assertIn("complies with the policy", second.message)

        third = self._decide(self._observe(FailureKind.POLICY_VIOLATION, args=args))
        self.assertEqual(third.mode, FailureMode.RETURN_TO_AGENT)
        self.assertIn("3 identical attempts", third.message)

        # The block itself already prevented the action: no turn kill.
        self.assertFalse(self.state.fallback.stop_requested)
        self.assertEqual(self.state.failures.turn_failure_count, 3)

    def test_changed_blocked_call_resets_policy_streak(self):
        first = self._decide(
            self._observe(FailureKind.POLICY_VIOLATION, args={"command": "grep -r a ."})
        )
        self.assertIsNone(first.message)
        guided = self._decide(
            self._observe(FailureKind.POLICY_VIOLATION, args={"command": "grep -r a ."})
        )
        self.assertIsNotNone(guided.message)

        # Different command shape is new information: streak restarts, so the
        # next block is a plain exposure again.
        changed = self._decide(
            self._observe(FailureKind.POLICY_VIOLATION, args={"command": "grep -r b src"})
        )
        self.assertEqual(changed.mode, FailureMode.RETURN_TO_AGENT)
        self.assertIsNone(changed.message)
        self.assertFalse(self.state.fallback.stop_requested)
        # The counter remains as observability; nothing stops on it.
        self.assertEqual(self.state.failures.turn_failure_count, 3)

    def test_many_policy_blocks_never_stop_without_a_global_budget(self):
        args = {"command": "grep -r secret ."}
        for _ in range(12):
            action = self._decide(self._observe(FailureKind.POLICY_VIOLATION, args=args))
            self.assertEqual(action.mode, FailureMode.RETURN_TO_AGENT)
        self.assertFalse(self.state.fallback.stop_requested)

    def test_execution_failures_are_always_returned_without_message(self):
        action = self._decide(self._observe(FailureKind.PROCESS_FAILED, message="exit 1"))
        self.assertEqual(action.mode, FailureMode.RETURN_TO_AGENT)
        self.assertIsNone(action.message)

    def test_identical_execution_failure_guides_third_and_keeps_guiding(self):
        args = {"command": "pytest tests/test_x.py"}
        message = "tests/test_x.py::test_a assert 1 == 2"

        first = self._decide(self._observe(FailureKind.PROCESS_FAILED, message=message, args=args))
        second = self._decide(self._observe(FailureKind.PROCESS_FAILED, message=message, args=args))
        third = self._decide(self._observe(FailureKind.PROCESS_FAILED, message=message, args=args))

        self.assertEqual(first.mode, FailureMode.RETURN_TO_AGENT)
        self.assertIsNone(first.message)
        self.assertEqual(second.mode, FailureMode.RETURN_TO_AGENT)
        self.assertIsNone(second.message)
        # Third identical failure stays return_to_agent (never auto-replays),
        # but carries an explicit do-not-repeat reminder.
        self.assertEqual(third.mode, FailureMode.RETURN_TO_AGENT)
        self.assertIsNotNone(third.message)
        self.assertIn("3 times in a row", third.message)
        self.assertIn("probe", third.message)

        # Further identical repeats keep guiding; they never kill the turn.
        fourth = self._decide(self._observe(FailureKind.PROCESS_FAILED, message=message, args=args))
        self.assertEqual(fourth.mode, FailureMode.RETURN_TO_AGENT)
        self.assertIsNotNone(fourth.message)
        self.assertEqual(fourth.stop_reason, "")
        self.assertFalse(self.state.fallback.stop_requested)

    def test_progressing_execution_errors_never_trigger_streak_action(self):
        messages = [
            "ModuleNotFoundError: No module named 'numpy'",
            "ModuleNotFoundError: No module named 'pandas'",
            "SyntaxError: invalid syntax",
            "AssertionError: expected 2 got 3",
        ]

        actions = [
            self._decide(
                self._observe(
                    FailureKind.PROCESS_FAILED,
                    message=msg,
                    args={"command": "pytest"},
                )
            )
            for msg in messages
        ]

        for action in actions:
            self.assertEqual(action.mode, FailureMode.RETURN_TO_AGENT)
            self.assertIsNone(action.message)
        self.assertFalse(self.state.fallback.stop_requested)
        self.assertEqual(self.state.failures.turn_failure_count, 4)

    def test_progressing_args_never_trigger_streak_action(self):
        actions = [
            self._decide(
                self._observe(
                    FailureKind.PROCESS_FAILED,
                    message="1 failed",
                    args={"command": f"pytest tests/test_{i}.py"},
                )
            )
            for i in range(4)
        ]

        for action in actions:
            self.assertEqual(action.mode, FailureMode.RETURN_TO_AGENT)
            self.assertIsNone(action.message)
        self.assertFalse(self.state.fallback.stop_requested)

    def test_high_failure_count_does_not_stop_a_productively_changing_chain(self):
        # A normal debugging chain produces many *different* failures: the
        # global failure count is telemetry only, never a circuit breaker.
        for index in range(12):
            action = self._decide(
                self._observe(
                    FailureKind.PROCESS_FAILED,
                    message=f"distinct error {index}",
                    args={"command": f"pytest tests/test_{index}.py"},
                )
            )
            self.assertEqual(action.mode, FailureMode.RETURN_TO_AGENT)
            self.assertIsNone(action.message)
        self.assertFalse(self.state.fallback.stop_requested)

    def test_high_failure_count_alone_never_stops_even_identical_failures(self):
        self.state.failures.turn_failure_count = 99
        action = self._decide(self._observe(FailureKind.PROCESS_FAILED))
        self.assertEqual(action.mode, FailureMode.RETURN_TO_AGENT)
        self.assertEqual(action.stop_reason, "")

    def test_protocol_failure_unaffected_by_high_turn_count(self):
        self.state.failures.turn_failure_count = 10
        action = self._decide(self._observe(FailureKind.INVALID_ARGUMENTS))
        self.assertEqual(action.mode, FailureMode.REQUEST_REGENERATION)

    def test_subagent_is_not_policy_governed(self):
        action = self.middleware.on_tool_failure(
            self._observe(), [], runtime_state=self.state, agent_name="research_agent"
        )
        self.assertIsNone(action)

    def test_begin_turn_resets_tracker(self):
        self._decide(self._observe(FailureKind.PROCESS_FAILED))
        self.assertEqual(self.state.failures.turn_failure_count, 1)
        self.middleware.begin_turn("task", [], runtime_state=self.state, agent_name="main_agent")
        self.assertEqual(self.state.failures.turn_failure_count, 0)
        self.assertIsNone(self.state.failures.current("probe"))


# ---------------------------------------------------------------------------
# Executor integration
# ---------------------------------------------------------------------------


class ExecutorFailureIntegrationTests(unittest.TestCase):
    def _registry(self):
        registry = tools.ToolRegistry()
        registry.register(
            _PROBE_SCHEMA,
            lambda value: ToolResult(tool="probe", status="success", output=value),
            permission="read",
        )

        def reader(label):
            return ToolResult(tool="reader", status="success", output=f"ok {label}")

        registry.register(
            {
                "type": "function",
                "function": {
                    "name": "reader",
                    "description": "reader",
                    "parameters": {
                        "type": "object",
                        "properties": {"label": {"type": "string"}},
                    },
                },
            },
            reader,
            permission="read",
            effect=_READ_EFFECT,
        )
        return registry

    def test_intercepted_validation_failure_reaches_on_tool_failure_and_event(self):
        registry = self._registry()
        spy = FailureSpyMiddleware()
        with tempfile.TemporaryDirectory() as tmp:
            conversation, context = _conversation(
                Path(tmp),
                registry,
                [_tool_call("tc_bad", "probe", {"value": 1})],
                [spy, ToolFailurePolicyMiddleware(tool_registry=registry)],
            )
            with (
                patch("harness_code_agent.agent.conversation.config.MAX_AGENT_ITERATIONS", 2),
                patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=1),
            ):
                conversation.run_until_idle()

        self.assertEqual(len(spy.calls), 1)
        failure = spy.calls[0]
        self.assertEqual(failure.tool_call_id, "tc_bad")
        self.assertEqual(failure.kind, FailureKind.INVALID_ARGUMENTS)
        self.assertTrue(failure.intercepted)
        event = next(event for event in context.event_bus.events if event.type == "failure")
        self.assertEqual(event.payload["metadata"]["failure_kind"], "invalid_arguments")
        self.assertEqual(event.payload["metadata"]["failure_visibility"], "protocol")
        self.assertEqual(event.payload["category"], "validation_error")

    def test_third_consecutive_schema_failure_requests_stop_with_correction(self):
        registry = self._registry()
        spy = FailureSpyMiddleware()
        with tempfile.TemporaryDirectory() as tmp:
            conversation, _context = _conversation(
                Path(tmp),
                registry,
                [_tool_call("tc_bad", "probe", {"value": 1})],
                [spy, ToolFailurePolicyMiddleware(tool_registry=registry)],
                repeat=5,
            )
            with (
                patch("harness_code_agent.agent.conversation.config.MAX_AGENT_ITERATIONS", 6),
                patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=1),
            ):
                conversation.run_until_idle()
            state = conversation.runtime_state

        self.assertEqual([failure.attempt for failure in spy.calls], [1, 2, 3])
        self.assertTrue(state.fallback.stop_requested)
        self.assertEqual(state.fallback.stop_reason, "retry_budget_exhausted")
        self.assertEqual(state.fallback.stop_limit_type, "schema_failures")
        injected = [
            str(message.get("content", ""))
            for message in conversation.messages
            if message.get("role") == "user"
        ]
        self.assertTrue(any("Allowed arguments are exactly" in text for text in injected))
        self.assertTrue(any("3 consecutive validation failures" in text for text in injected))

    def test_legacy_string_interception_is_normalized_as_tool_policy(self):
        registry = self._registry()
        spy = FailureSpyMiddleware()

        class BlockMiddleware(AgentMiddleware):
            # Legacy external middleware contract: a plain string block.
            def before_tool(self, tool_name, tool_args, messages, runtime_state=None, agent_name=None):
                return "[blocked] not allowed"

        with tempfile.TemporaryDirectory() as tmp:
            conversation, context = _conversation(
                Path(tmp),
                registry,
                [_tool_call("tc_p", "reader", {"label": "a"})],
                [BlockMiddleware(), spy, ToolFailurePolicyMiddleware(tool_registry=registry)],
            )
            with (
                patch("harness_code_agent.agent.conversation.config.MAX_AGENT_ITERATIONS", 2),
                patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=1),
            ):
                conversation.run_until_idle()

        self.assertEqual(len(spy.calls), 1)
        self.assertEqual(spy.calls[0].kind, FailureKind.POLICY_VIOLATION)
        self.assertTrue(spy.calls[0].intercepted)
        event = next(event for event in context.event_bus.events if event.type == "failure")
        self.assertEqual(event.payload["metadata"]["failure_category"], "policy")
        self.assertEqual(event.payload["category"], "tool_error")
        # Single policy failure stays visible, no stop.
        self.assertFalse(conversation.runtime_state.fallback.stop_requested)

    def test_mixed_parallel_batch_counts_failures_per_call(self):
        registry = self._registry()
        spy = FailureSpyMiddleware()
        calls = [
            _tool_call("tc_bad", "probe", {"value": 1}),
            _tool_call("tc_good", "reader", {"label": "a"}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            conversation, _context = _conversation(
                Path(tmp),
                registry,
                calls,
                [spy, ToolFailurePolicyMiddleware(tool_registry=registry)],
            )
            with (
                patch("harness_code_agent.agent.conversation.config.MAX_AGENT_ITERATIONS", 2),
                patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=1),
            ):
                conversation.run_until_idle()
            state = conversation.runtime_state
            tool_messages = [msg for msg in conversation.messages if msg.get("role") == "tool"]

        self.assertEqual([failure.tool_call_id for failure in spy.calls], ["tc_bad"])
        self.assertEqual([msg["tool_call_id"] for msg in tool_messages], ["tc_bad", "tc_good"])
        self.assertIn("ok a", tool_messages[1]["content"])
        # Protocol failure does not consume the turn budget ...
        self.assertEqual(state.failures.turn_failure_count, 0)
        # ... and the successful tool has no streak recorded.
        self.assertIsNone(state.failures.current("reader"))


def _run_schema_streak(root: Path):
    """Drive the bounded schema correction loop to its third-attempt stop."""
    registry = tools.ToolRegistry()
    registry.register(
        _PROBE_SCHEMA,
        lambda value: ToolResult(tool="probe", status="success", output=value),
        permission="read",
    )
    spy = FailureSpyMiddleware()
    conversation, context = _conversation(
        root,
        registry,
        [_tool_call("tc_bad", "probe", {"value": 1})],
        [spy, ToolFailurePolicyMiddleware(tool_registry=registry)],
        repeat=5,
    )
    with (
        patch("harness_code_agent.agent.conversation.config.MAX_AGENT_ITERATIONS", 6),
        patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=1),
    ):
        conversation.run_until_idle()
    return conversation, context, registry


class FailureObservabilityTests(unittest.TestCase):
    def test_decision_event_emitted_for_every_policy_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            _conversation, context, _registry = _run_schema_streak(Path(tmp))

        decisions = [
            event.payload for event in context.event_bus.events
            if event.type == "tool_failure_decision"
        ]
        self.assertEqual([item["attempt"] for item in decisions], [1, 2, 3])
        self.assertEqual(
            [item["mode"] for item in decisions],
            [FailureMode.REQUEST_REGENERATION, FailureMode.REQUEST_REGENERATION, FailureMode.STOP],
        )
        for item in decisions:
            self.assertEqual(item["tool"], "probe")
            self.assertTrue(item["tool_call_id"].startswith("tc_bad"))
            self.assertEqual(item["kind"], FailureKind.INVALID_ARGUMENTS)
            self.assertEqual(item["category"], FailureCategory.INVALID_CALL)
            self.assertEqual(item["phase"], "schema_validation")
            self.assertEqual(item["visibility"], "protocol")
            self.assertTrue(item["intercepted"])
            self.assertEqual(item["source"], "ToolFailurePolicyMiddleware")
            self.assertEqual(item["turn_failure_count"], 0)
        self.assertEqual(decisions[0]["stop_reason"], "")
        self.assertEqual(decisions[1]["stop_reason"], "")
        self.assertEqual(decisions[2]["stop_reason"], "retry_budget_exhausted")

    def test_middleware_activity_outcomes_track_regenerated_guided_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            _conversation, context, _registry = _run_schema_streak(Path(tmp))

        activities = [
            event.payload for event in context.event_bus.events
            if event.type == "middleware_activity"
        ]
        self.assertEqual(
            [item["outcome"] for item in activities],
            ["regenerated", "guided", "stopped"],
        )
        for item in activities:
            self.assertIn("ToolFailurePolicyMiddleware", item["sources"])
        # The deciding middleware is counted in the hooks it saw.
        self.assertGreaterEqual(activities[0]["hooks"], 2)

    def test_trace_jsonl_records_failure_policy_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _run_schema_streak(root)
            trace_path = root / ".harness" / "traces" / "trace_main_agent.jsonl"
            lines = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]

        decisions = [line for line in lines if line["event"] == "tool_failure_policy"]
        self.assertEqual(len(decisions), 3)
        self.assertEqual(
            [(line["attempt"], line["mode"]) for line in decisions],
            [(1, FailureMode.REQUEST_REGENERATION), (2, FailureMode.REQUEST_REGENERATION), (3, FailureMode.STOP)],
        )
        final = decisions[-1]
        self.assertEqual(final["kind"], "invalid_arguments")
        self.assertEqual(final["visibility"], "protocol")
        self.assertTrue(final["intercepted"])
        self.assertEqual(final["stop_reason"], "retry_budget_exhausted")
        # The correction text remains on the dedicated middleware channel.
        injections = [
            line for line in lines
            if line["event"] == "middleware" and line["hook"] == "on_tool_failure"
        ]
        self.assertTrue(any("Allowed arguments are exactly" in line["message"] for line in injections))

    def test_task_level_decision_emits_return_to_agent_and_counts_turn(self):
        registry = tools.ToolRegistry()
        registry.register(
            {
                "type": "function",
                "function": {
                    "name": "reader",
                    "description": "reader",
                    "parameters": {"type": "object", "properties": {"label": {"type": "string"}}},
                },
            },
            lambda label: ToolResult(tool="reader", status="success", output=label),
            permission="read",
            effect=_READ_EFFECT,
        )

        class BlockMiddleware(AgentMiddleware):
            def before_tool(self, tool_name, tool_args, messages, runtime_state=None, agent_name=None):
                return "[blocked] not allowed"

        with tempfile.TemporaryDirectory() as tmp:
            conversation, context = _conversation(
                Path(tmp),
                registry,
                [_tool_call("tc_p", "reader", {"label": "a"})],
                [BlockMiddleware(), ToolFailurePolicyMiddleware(tool_registry=registry)],
            )
            with (
                patch("harness_code_agent.agent.conversation.config.MAX_AGENT_ITERATIONS", 2),
                patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=1),
            ):
                conversation.run_until_idle()
            turn_count = conversation.runtime_state.failures.turn_failure_count

        decision = next(
            event.payload for event in context.event_bus.events
            if event.type == "tool_failure_decision"
        )
        self.assertEqual(decision["mode"], FailureMode.RETURN_TO_AGENT)
        self.assertEqual(decision["kind"], FailureKind.POLICY_VIOLATION)
        self.assertEqual(decision["visibility"], "task")
        self.assertEqual(decision["turn_failure_count"], 1)
        self.assertEqual(turn_count, 1)

    def test_no_decision_event_when_no_middleware_governs_the_failure(self):
        registry = tools.ToolRegistry()
        registry.register(
            {
                "type": "function",
                "function": {
                    "name": "reader",
                    "description": "reader",
                    "parameters": {"type": "object", "properties": {"label": {"type": "string"}}},
                },
            },
            lambda label: ToolResult(tool="reader", status="success", output=label),
            permission="read",
            effect=_READ_EFFECT,
        )

        class BlockMiddleware(AgentMiddleware):
            def before_tool(self, tool_name, tool_args, messages, runtime_state=None, agent_name=None):
                return "[blocked] not allowed"

        # Only a no-op observing middleware: a failure exists, but nobody decides.
        with tempfile.TemporaryDirectory() as tmp:
            conversation, context = _conversation(
                Path(tmp),
                registry,
                [_tool_call("tc_p", "reader", {"label": "a"})],
                [BlockMiddleware(), FailureSpyMiddleware()],
            )
            with (
                patch("harness_code_agent.agent.conversation.config.MAX_AGENT_ITERATIONS", 2),
                patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=1),
            ):
                conversation.run_until_idle()

        self.assertFalse(
            [event for event in context.event_bus.events if event.type == "tool_failure_decision"]
        )

    def test_guard_blocks_are_exposed_and_guided_but_never_stop(self):
        spy = FailureSpyMiddleware()
        with tempfile.TemporaryDirectory() as tmp:
            conversation, context = _conversation(
                Path(tmp),
                _shell_guard_registry(),
                [_tool_call("tc_b1", "run_bash", _RECURSIVE_LIST_ARGS)],
                [ToolGuardMiddleware(), spy, ToolFailurePolicyMiddleware()],
                repeat=3,
            )
            with (
                patch("harness_code_agent.agent.conversation.config.MAX_AGENT_ITERATIONS", 4),
                patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=1),
            ):
                conversation.run_until_idle()
            state = conversation.runtime_state

        self.assertEqual([failure.attempt for failure in spy.calls], [1, 2, 3])
        # A blocked call is already prevented; repetition only adds guidance,
        # never a stop.
        self.assertFalse(state.fallback.stop_requested)
        decision = [
            event.payload for event in context.event_bus.events
            if event.type == "tool_failure_decision"
        ]
        self.assertEqual(
            [item["mode"] for item in decision],
            [
                FailureMode.RETURN_TO_AGENT,
                FailureMode.RETURN_TO_AGENT,
                FailureMode.RETURN_TO_AGENT,
            ],
        )
        self.assertEqual(decision[-1]["kind"], FailureKind.POLICY_VIOLATION)
        # Guidance is injected as user-role messages: none on the first
        # identical block, one for each repeated block.
        guidance = [
            msg["content"]
            for msg in conversation.messages
            if msg.get("role") == "user" and "blocked by policy" in str(msg.get("content"))
        ]
        self.assertEqual(len(guidance), 2)

    def test_parallel_guard_blocks_in_one_batch_do_not_stop(self):
        spy = FailureSpyMiddleware()
        calls = [
            _tool_call("tc_a", "run_bash", _RECURSIVE_LIST_ARGS),
            _tool_call("tc_b", "run_bash", _RECURSIVE_LIST_ARGS),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            conversation, _context = _conversation(
                Path(tmp),
                _shell_guard_registry(),
                calls,
                [ToolGuardMiddleware(), spy, ToolFailurePolicyMiddleware()],
            )
            with (
                patch("harness_code_agent.agent.conversation.config.MAX_AGENT_ITERATIONS", 2),
                patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=1),
            ):
                conversation.run_until_idle()
            state = conversation.runtime_state

        self.assertEqual([failure.tool_call_id for failure in spy.calls], ["tc_a", "tc_b"])
        self.assertEqual([failure.attempt for failure in spy.calls], [1, 1])
        self.assertFalse(state.fallback.stop_requested)
        # Sibling failures are one streak attempt but two real turn failures.
        self.assertEqual(state.failures.turn_failure_count, 2)
        self.assertEqual(state.failures.current("run_bash").count, 1)


if __name__ == "__main__":
    unittest.main()
