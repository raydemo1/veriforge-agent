from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harness_code_agent.runtime.builtins.registry import BUILTIN_TOOL_REGISTRY
from harness_code_agent.runtime.permissions import PermissionPolicy
from harness_code_agent.runtime.tool_call_validation import (
    ToolCall,
    strict_tool_schema,
    validate_tool_arguments,
    validate_tool_call,
)
from harness_code_agent.runtime.tool_context import ToolContext
from harness_code_agent.runtime.tool_runner import execute_tool_result
from harness_code_agent.sessions.events import EventBus
from harness_code_agent.workspace.service import WorkspaceService


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "required": ["mode", "count"],
                "properties": {
                    "mode": {"type": "string", "enum": ["safe", "fast"]},
                    "count": {"type": "integer", "minimum": 1, "maximum": 3},
                },
            },
        },
    }


class ToolCallValidationTests(unittest.TestCase):
    def test_registry_closes_object_argument_maps_by_default(self):
        strict = strict_tool_schema(_schema("probe"))

        parameters = strict["function"]["parameters"]
        self.assertFalse(parameters["additionalProperties"])

    def test_validation_rejects_non_object_root(self):
        result = validate_tool_call(
            ToolCall(0, "tc", "read_file", "[]"),
            BUILTIN_TOOL_REGISTRY,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.error.kind, "invalid_arguments")
        self.assertEqual(result.error.phase, "structural")
        self.assertIn("root must be an object", result.error.message)

    def test_validation_enforces_schema_and_closed_properties(self):
        from harness_code_agent.runtime.tool_registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(_schema("probe"), lambda **_: None, permission="read")

        extra = validate_tool_arguments(
            "probe",
            {"mode": "safe", "count": 1, "unexpected": True},
            registry,
        )
        wrong_type = validate_tool_arguments(
            "probe",
            {"mode": "safe", "count": "1"},
            registry,
        )

        self.assertEqual(extra.error.kind, "invalid_arguments")
        self.assertIn("additional properties are not allowed", extra.error.message)
        self.assertEqual(wrong_type.error.kind, "invalid_arguments")
        self.assertIn("count must be an integer", wrong_type.error.message)

    def test_semantic_validation_rejects_workspace_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = ToolContext(
                workspace=WorkspaceService(root=Path(tmp)),
                permission_policy=PermissionPolicy(mode="workspace-write"),
                event_bus=EventBus(),
            )
            result = validate_tool_arguments(
                "write_file",
                {"path": "../../outside.txt", "content": "x"},
                BUILTIN_TOOL_REGISTRY,
                context,
            )

        self.assertEqual(result.error.kind, "workspace_escape")
        self.assertEqual(result.error.phase, "semantic")
        self.assertTrue(result.error.retryable)

    def test_invalid_json_is_retryable_without_repairing_the_payload(self):
        result = validate_tool_call(
            ToolCall(0, "tc", "read_file", '{"path": "a.txt",}'),
            BUILTIN_TOOL_REGISTRY,
        )

        self.assertEqual(result.error.kind, "invalid_json")
        self.assertTrue(result.error.retryable)
        self.assertEqual(result.arguments, {})

    def test_missing_and_blank_model_arguments_require_an_explicit_object(self):
        for raw in (None, "", "  "):
            with self.subTest(raw=raw):
                result = validate_tool_call(ToolCall(0, "tc", "list_shell_jobs", raw), BUILTIN_TOOL_REGISTRY)
                self.assertEqual(result.error.kind, "invalid_json")

        self.assertTrue(validate_tool_call(ToolCall(0, "tc", "list_shell_jobs", "{}"), BUILTIN_TOOL_REGISTRY).valid)
        self.assertTrue(validate_tool_arguments("list_shell_jobs", None, BUILTIN_TOOL_REGISTRY).valid)

    def test_invalid_values_fail_during_decoding_before_schema_validation(self):
        payloads = (
            '{"path":"first.py","path":"second.py"}',
            '{"nested":{"key":1,"key":2}}',
            '{"timeout":NaN}',
            '{"timeout":Infinity}',
            '{"timeout":-Infinity}',
            '{"timeout":1e1000}',
            r'{"path":"\ud800"}',
            '{"nested":' + '[' * 2000 + '0' + ']' * 2000 + '}',
        )
        for raw in payloads:
            with self.subTest(raw=raw[:80]):
                result = validate_tool_call(ToolCall(0, "tc", "read_file", raw), BUILTIN_TOOL_REGISTRY)
                self.assertEqual(result.error.kind, "invalid_json")
                self.assertEqual(result.error.phase, "syntactic")
                self.assertEqual(result.arguments, {})

    def test_direct_invalid_values_are_rejected_before_writing_call_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolContext(
                workspace=WorkspaceService(root=root),
                permission_policy=PermissionPolicy(mode="workspace-write"),
                event_bus=EventBus(root / "events.jsonl"),
            )
            result = execute_tool_result(
                "run_bash", {"command": "echo unused", "timeout": float("inf")}, tool_context=context,
            )

        self.assertEqual(result.metadata["error_kind"], "invalid_arguments")
        self.assertEqual(result.metadata["validation_phase"], "syntactic")
        self.assertEqual(context.event_bus.events[0].payload["args"], {})


if __name__ == "__main__":
    unittest.main()
