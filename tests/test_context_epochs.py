import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from harness_code_agent import config
from harness_code_agent.agent.context_manager import SUMMARY_MARKER, ContextManager
from harness_code_agent.agent.conversation import Agent
from harness_code_agent.agent.history import ContextHistory, tool_pairs_closed
from harness_code_agent.agent.providers import ProviderAdapter
from harness_code_agent.agent.request_diagnostics import RequestDiagnostics
from harness_code_agent.agent.runtime_state import AgentRuntimeState
from harness_code_agent.agent.tool_executor import ToolExecutor
from harness_code_agent.runtime.builtins.registry import BUILTIN_TOOL_REGISTRY
from harness_code_agent.runtime.builtins.runtime_status import get_runtime_status
from harness_code_agent.runtime.permissions import PermissionPolicy
from harness_code_agent.runtime.tool_context import ToolContext
from harness_code_agent.sessions.events import EventBus
from harness_code_agent.sessions.journal import SessionJournal
from harness_code_agent.workspace.service import WorkspaceService


def test_frozen_history_and_failed_persistence(tmp_path):
    history = ContextHistory()
    journal = SessionJournal(tmp_path / "journal.jsonl")
    message = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "secret",
        "tool_calls": [{"id": "one", "function": {"name": "read", "arguments": "{}"}}],
    }
    history.append(message, journal=journal)
    message["tool_calls"][0]["function"]["name"] = "changed"
    snapshot = history.snapshot()
    snapshot[0]["reasoning_content"] = "changed"
    assert history.snapshot()[0]["tool_calls"][0]["function"]["name"] == "read"
    assert history.snapshot()[0]["reasoning_content"] == "secret"
    before = journal.sequence
    with (
        patch.object(Path, "open", side_effect=OSError("disk full")),
        pytest.raises(OSError),
    ):
        history.append({"role": "user", "content": "lost"}, journal=journal)
    assert len(history.snapshot()) == 1
    assert journal.sequence == before


def test_epoch_recovery_keeps_runtime_system_and_original_log(tmp_path):
    history = ContextHistory()
    journal = SessionJournal(tmp_path / "journal.jsonl")
    history.append({"role": "system", "content": "old"}, journal=journal)
    history.start_epoch("initial", history.snapshot(), {}, journal=journal)
    event = {"role": "system", "content": "[HARNESS_RUNTIME_EVENT v1]\nwarning"}
    history.append(event, journal=journal, metadata={"kind": "runtime_event"})
    history.start_epoch(
        "profile_changed",
        [{"role": "system", "content": "new"}, event],
        {},
        journal=journal,
    )
    assert journal.recovery_messages("current") == [
        {"role": "system", "content": "current"},
        event,
    ]
    assert journal.read()[0].payload["message"]["content"] == "old"


@pytest.mark.parametrize("stream", [False, True])
def test_reasoning_survives_multiple_user_turns(stream):
    adapter = ProviderAdapter("deepseek")
    history = [{"role": "user", "content": "first"}]
    if stream:
        response = adapter.assistant_message_from_stream(
            [
                {
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": "first thought",
                                "content": "answer",
                            }
                        }
                    ]
                }
            ]
        ).assistant_message
    else:
        response = adapter.assistant_message_from_response(
            {"content": "answer", "reasoning_content": "first thought"}
        )
    history.extend([response, {"role": "user", "content": "second"}])
    for thinking in (True, False):
        kwargs = adapter.chat_kwargs(
            profile=config.ModelProfile(
                provider="deepseek", model="deepseek-v4-flash", thinking=thinking
            ),
            messages=history,
            max_tokens=100,
            tools=[],
        )
        assert kwargs["messages"][1]["reasoning_content"] == "first thought"
    assert (
        "reasoning_content"
        not in ProviderAdapter("openai").chat_kwargs(
            model="example", messages=history, max_tokens=100
        )["messages"][1]
    )


def test_diagnostics_detect_reasoning_and_tool_order_without_content():
    diagnostics = RequestDiagnostics()
    request = {
        "model": "m",
        "messages": [{"role": "assistant", "reasoning_content": "sensitive"}],
        "tools": [{"name": "one"}, {"name": "two"}],
    }
    diagnostics.begin(request, provider="deepseek")
    request["messages"][0]["reasoning_content"] = "another secret"
    record = diagnostics.begin(request, provider="deepseek")
    assert record["first_difference"] == "messages[0].reasoning_content"
    assert "secret" not in json.dumps(record)
    request["tools"].reverse()
    assert diagnostics.begin(request, provider="deepseek")[
        "first_difference"
    ].startswith("tools[0]")
    assert diagnostics.finish(record)["cache_hit_tokens"] is None


def test_long_turn_compaction_preserves_complete_tail_and_reasoning():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "task"},
    ]
    for index in range(30):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "reason " * 1000,
                    "tool_calls": [
                        {
                            "id": str(index),
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": str(index),
                    "content": "output " * 500,
                },
            ]
        )
    result = ContextManager().compact(
        messages,
        lambda _: "## Task goal\nContinue",
        current_turn_start_index=1,
        state={"current_user_task": "task", "current_todo": "current"},
    )
    assert result is not None
    assert tool_pairs_closed(result.messages)
    assert result.messages[-2]["reasoning_content"] == messages[-2]["reasoning_content"]
    assert "[CURRENT RUNTIME FACTS]" in result.messages[1]["content"]
    assert messages[1]["content"] == "task"


def test_existing_summary_alone_is_not_compacted_again():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": SUMMARY_MARKER + "\nold summary " * 1000},
        {"role": "user", "content": "current task"},
    ]
    with patch("harness_code_agent.agent.conversation.llm_call_simple") as summarize:
        result = ContextManager().compact(
            messages, summarize, current_turn_start_index=2, state={}, force=True,
        )
    assert result is None
    summarize.assert_not_called()


def test_summary_input_keeps_long_tool_results_and_arguments_without_reasoning():
    arguments = json.dumps({"path": "x" * 600, "important_option": "ARGUMENT_TAIL"})
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "old task"},
        {
            "role": "assistant", "content": None, "reasoning_content": "PRIVATE_REASONING",
            "tool_calls": [{"id": "read", "function": {"name": "read", "arguments": arguments}}],
        },
        {"role": "tool", "tool_call_id": "read", "content": "log " * 1000 + "IMPORTANT_RESULT_TAIL"},
        {"role": "user", "content": "next task"},
    ]
    captured = []

    def summarize(request):
        captured.append(request[-1]["content"])
        return "## Task goal\nContinue"

    assert ContextManager().compact(
        messages, summarize, current_turn_start_index=4, state={}, force=True,
    ) is not None
    assert "IMPORTANT_RESULT_TAIL" in captured[0]
    assert "ARGUMENT_TAIL" in captured[0]
    assert "PRIVATE_REASONING" not in captured[0]


def test_oversize_summary_request_does_not_call_provider():
    from harness_code_agent.agent.llm_channel import llm_call_simple

    with (
        patch.object(config, "CONTEXT_WINDOW_TOKENS", 200000),
        patch("harness_code_agent.agent.llm_channel.context.count_request_tokens", return_value=190001),
        patch("harness_code_agent.agent.llm_channel.client_scope") as client,
    ):
        assert llm_call_simple([{"role": "user", "content": "history"}]) == ""
    client.assert_not_called()


@pytest.fixture
def runtime_conversation(tmp_path):
    registry = BUILTIN_TOOL_REGISTRY.copy()
    context = ToolContext(
        workspace=WorkspaceService(
            root=tmp_path, snapshots_dir=tmp_path / ".harness" / "snapshots"
        ),
        permission_policy=PermissionPolicy(mode="danger-full-access"),
        event_bus=EventBus(),
        tool_registry=registry,
        allowed_tool_permissions={"read", "control"},
    )
    with patch(
        "harness_code_agent.agent.conversation.get_client",
        return_value=SimpleNamespace(close=lambda: None),
    ):
        conversation = Agent(
            "test",
            "rules",
            tool_schemas=[registry.schema_for("get_runtime_status")],
            tool_context=context,
        ).start_conversation("task")
    yield conversation
    conversation.close()


def execute(conversation, name, args):
    call = {
        "id": str(len(conversation.messages)),
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }
    conversation._append_message(
        {"role": "assistant", "content": None, "tool_calls": [call]}
    )
    ToolExecutor(conversation).execute([call])
    return conversation.messages[-1]["content"]


def test_budget_warning_is_appended_once_at_safe_boundary(runtime_conversation):
    conversation = runtime_conversation
    with patch("harness_code_agent.config.AGENT_BUDGET_WARN_FRACTION", 0.5):
        conversation._maybe_emit_budget_warning("tool_calls", 5, 10)
    conversation._flush_runtime_events()
    first = conversation.messages
    conversation._flush_runtime_events()
    assert conversation.messages == first
    assert first[-1]["role"] == "system"
    result = get_runtime_status("task", runtime_state=conversation.runtime_state)
    assert json.loads(result.output)["task"] == "task"


def test_unknown_verification_does_not_infer_success():
    result = get_runtime_status("verification", runtime_state=AgentRuntimeState())
    assert json.loads(result.output)["status"] == "not_recorded"


def test_epoch_failure_preserves_active_view_and_original_log(tmp_path):
    history = ContextHistory()
    journal = SessionJournal(tmp_path / "journal.jsonl")
    history.append({"role": "system", "content": "original"}, journal=journal)
    original = history.start_epoch("initial", history.snapshot(), {}, journal=journal)
    with (
        patch.object(journal, "append", side_effect=OSError("disk full")),
        pytest.raises(OSError),
    ):
        history.start_epoch(
            "profile_changed",
            [{"role": "system", "content": "new"}],
            {},
            journal=journal,
        )
    assert history.epoch == original
    assert history.raw_snapshot() == history.snapshot()


def test_pending_tools_defer_warning_and_compaction(runtime_conversation):
    conversation = runtime_conversation
    pending = {
        "id": "pending",
        "type": "function",
        "function": {"name": "get_runtime_status", "arguments": '{"section":"task"}'},
    }
    conversation._append_message(
        {"role": "assistant", "content": None, "tool_calls": [pending]}
    )
    conversation.runtime_state.pending_budget_warnings.add("tool_calls")
    before = conversation.messages
    conversation._flush_runtime_events()
    assert conversation.messages == before
    with pytest.raises(ValueError):
        conversation.start_epoch("compaction", conversation.messages)
    conversation._append_message(
        {"role": "tool", "tool_call_id": "pending", "content": "done"}
    )
    conversation._flush_runtime_events()
    assert conversation.messages[-1]["content"].startswith("[HARNESS_RUNTIME_EVENT")


def test_status_query_counts_against_tool_budget_and_todo_is_not_reinjected(
    runtime_conversation,
):
    conversation = runtime_conversation
    from harness_code_agent.agent.runtime_state import TodoItem, TodoList

    conversation.runtime_state.todo = TodoList(items=[TodoItem("one", "task")])
    assert conversation._build_prompt() == conversation.messages
    execute(conversation, "get_runtime_status", {"section": "budget"})
    assert conversation.runtime_state.fallback.tool_call_count == 1
    assert '"used": 1' in conversation.messages[-1]["content"]


def test_verification_detects_modified_evidence_without_rerunning_checks(tmp_path):
    import hashlib

    path = tmp_path / "module.py"
    path.write_text("x = 1", encoding="utf-8")
    state = AgentRuntimeState(
        verification={
            "checks": [{"status": "passed"}],
            "sequence": 1,
            "fingerprints": {
                "module.py": hashlib.sha256(path.read_bytes()).hexdigest()
            },
        }
    )
    context = SimpleNamespace(workspace=SimpleNamespace(root=tmp_path))
    assert not json.loads(
        get_runtime_status(
            "verification", runtime_state=state, tool_context=context
        ).output
    )["stale"]
    path.write_text("invalid code", encoding="utf-8")
    assert json.loads(
        get_runtime_status(
            "verification", runtime_state=state, tool_context=context
        ).output
    )["stale"]


def test_hard_budget_does_not_make_a_notification_request(runtime_conversation):
    conversation = runtime_conversation
    conversation.runtime_state.fallback.tool_call_count = 1
    with (
        patch("harness_code_agent.config.MAX_AGENT_TOOL_CALLS", 1),
        patch.object(conversation.llm, "request_assistant_message") as request,
    ):
        conversation.run_until_idle()
    request.assert_not_called()
    assert (
        conversation.runtime_state.fallback.stop_reason == "tool_call_budget_exceeded"
    )


def test_failed_context_refresh_keeps_original_view():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "old work " * 500},
        {"role": "user", "content": "current task"},
    ]

    def fail(_):
        raise RuntimeError("refresh failed")

    result = ContextManager().compact(
        messages,
        lambda _: "## Task goal\nContinue",
        current_turn_start_index=2,
        state={},
        refresh=fail,
    )
    assert result is None
    assert messages[1]["content"] == "old work " * 500


def test_identical_retained_messages_keep_their_recent_source_identity(tmp_path):
    journal = SessionJournal(tmp_path / "journal.jsonl")
    history = ContextHistory()
    message = {"role": "user", "content": "continue"}
    history.append(message, journal=journal)
    latest = history.append(message, journal=journal)
    history.start_epoch("compaction", [message], {}, journal=journal)
    assert history.sequences == (latest,)
    assert journal.read()[-1].payload["source_sequences"] == [latest]
