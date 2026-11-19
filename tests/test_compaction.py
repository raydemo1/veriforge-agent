"""Tests for context compaction strategy (PLAN.md)."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# CompactionGate — controls when compaction is allowed
# ---------------------------------------------------------------------------

class CompactionGateTests(unittest.TestCase):
    """CompactionGate tracks active tool calls."""

    def _make_gate(self):
        from harness_code_agent.agent.compaction import CompactionGate
        return CompactionGate()

    def test_compaction_waits_until_all_tool_calls_finish(self):
        gate = self._make_gate()
        self.assertTrue(gate.can_compact())
        gate.begin_tool_call()
        gate.begin_tool_call()
        self.assertFalse(gate.can_compact())
        gate.end_tool_call()
        self.assertFalse(gate.can_compact())
        gate.end_tool_call()
        self.assertTrue(gate.can_compact())



# ---------------------------------------------------------------------------
# Threshold behavior — token usage triggers different actions
# ---------------------------------------------------------------------------

class CompactionThresholdTests(unittest.TestCase):
    """Token usage thresholds drive compaction strategy."""

    def test_threshold_follows_window_and_reserves_output_space(self):
        from harness_code_agent import config
        from harness_code_agent.agent.compaction import (
            compaction_action,
            get_thresholds,
        )

        self.assertEqual(get_thresholds(100000).compact, 80000)
        self.assertEqual(get_thresholds(64000, max_output_tokens=32768).compact, 31232)
        with patch.object(config, "CONTEXT_WINDOW_TOKENS", 200000), patch.object(config, "COMPRESS_THRESHOLD", 190000):
            self.assertEqual(get_thresholds(max_output_tokens=32768).compact, 167232)
        thresholds = get_thresholds(128000)
        self.assertEqual(compaction_action(thresholds.compact - 1, thresholds), "none")
        self.assertEqual(compaction_action(thresholds.compact, thresholds), "auto_compact")

class ContextAnxietyTests(unittest.TestCase):
    def test_detect_anxiety_is_a_structured_soft_signal_in_english_and_chinese(self):
        from harness_code_agent.agent.context import detect_anxiety

        english = detect_anxiety([
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "Due to the context limit, let me wrap up here."},
        ])
        self.assertTrue(english and english.detected)
        self.assertGreaterEqual(english.score, 2)
        self.assertEqual(english.source, "assistant_recent_messages")
        self.assertTrue(any("context limit" in reason.lower() for reason in english.reasons))

        chinese = detect_anxiety([{"role": "assistant", "content": "我快没上下文空间了，先收尾一下。上下文快满了，我只覆盖关键部分。"}])
        self.assertGreaterEqual(chinese.score, 2)
        self.assertTrue(any("上下文" in reason for reason in chinese.reasons))

        calm = detect_anxiety([{"role": "assistant", "content": "I will continue with the next verification step."}])
        self.assertFalse(calm)
        self.assertEqual((calm.detected, calm.score, calm.reasons), (False, 0, []))

# ---------------------------------------------------------------------------
# /compact slash command
# ---------------------------------------------------------------------------

@pytest.mark.session
class CompactCommandTests(unittest.TestCase):
    """Tests for /compact and its subcommands."""

    def _make_session(self, tmpdir):
        """Create a minimal mock session for command testing."""
        session = MagicMock()
        session.cwd = Path(tmpdir)
        session.conversation = MagicMock()
        session.conversation.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "done"},
        ]
        return session

    def test_compact_command_is_a_plain_manual_trigger(self):
        from harness_code_agent.tui.commands import default_command_registry

        registry = default_command_registry()
        self.assertEqual(registry._by_name["/compact"].usage, "/compact")
        self.assertEqual(registry.execute("/compact", self._make_session(".")).action, "compact")
        for args in ("status", "history"):
            self.assertIn("用法：/compact", registry.execute(f"/compact {args}", MagicMock()).text)

    def test_manual_compact_runs_without_waiting_for_threshold(self):
        from harness_code_agent.agent.conversation import Agent

        conv = Agent("test_agent", "sys", use_tools=False).start_conversation()
        conv.add_user_turn("older task" * 200)
        conv._append_message({"role": "assistant", "content": "older answer"})
        conv.add_user_turn("current task")
        with patch(
            "harness_code_agent.agent.conversation.llm_call_simple",
            return_value="## Task goal\ncurrent task\n## Next action\ncontinue",
        ) as summarize:
            result = conv.compact_now()

        self.assertIn("已压缩对话", result)
        self.assertIn("structured working state", conv.messages[1]["content"])
        self.assertEqual(conv.messages[-1]["content"], "current task")
        summarize.assert_called_once()


@pytest.mark.session
class AgentConversationCompactionLifecycleTests(unittest.TestCase):
    def test_compacts_before_rejecting_a_request_in_the_old_threshold_gap(self):
        from harness_code_agent import config
        from harness_code_agent.agent.context_manager import CompactionResult
        from harness_code_agent.agent.conversation import Agent

        agent = Agent("test_agent", "sys", use_tools=False)
        conv = agent.start_conversation("task")
        original = conv.messages
        candidate = [original[0], {"role": "user", "content": "recovered task"}]

        with (
            patch.object(config, "CONTEXT_WINDOW_TOKENS", 200000),
            patch.object(config, "COMPRESS_THRESHOLD", 190000),
            patch(
                "harness_code_agent.support.tokens.count_request_tokens",
                side_effect=[168000, 1000, 1000],
            ),
            patch.object(conv.context_manager, "compact", return_value=CompactionResult(candidate, "summary", 1)) as compact,
            patch.object(
                conv.llm, "request_assistant_message",
                return_value=({"role": "assistant", "content": "done"}, "stop"),
            ) as request,
        ):
            conv.run_until_idle()

        compact.assert_called_once()
        request.assert_called_once()
        self.assertNotIn("max_tokens", request.call_args.args[0])
        self.assertEqual(conv.history.epoch.reason, "compaction")

    def test_failed_compaction_preserves_history_and_stops_an_oversize_request(self):
        from harness_code_agent import config
        from harness_code_agent.agent.conversation import Agent

        agent = Agent("test_agent", "sys", use_tools=False)
        conv = agent.start_conversation("task")
        original = conv.messages

        with (
            patch.object(config, "CONTEXT_WINDOW_TOKENS", 200000),
            patch.object(config, "COMPRESS_THRESHOLD", 190000),
            patch("harness_code_agent.support.tokens.count_request_tokens", return_value=168000),
            patch.object(conv.context_manager, "compact", return_value=None) as compact,
            patch.object(conv.llm, "request_assistant_message") as request,
        ):
            conv.run_until_idle()

        compact.assert_called_once()
        request.assert_not_called()
        self.assertEqual(conv.messages, original)
        self.assertEqual(conv.runtime_state.fallback.stop_reason, "context_budget_exceeded")

    def test_context_compacted_hook_replaces_dynamic_context_block(self):
        from harness_code_agent.agent.conversation import Agent
        from harness_code_agent.runtime.middleware import AgentMiddleware

        class RefreshMiddleware(AgentMiddleware):
            def on_conversation_start(self, messages, runtime_state=None, agent_name=None):
                return [{"role": "system", "content": "[HARNESS_DYNAMIC_CONTEXT:test]\nold"}]

            def on_context_compacted(self, messages, runtime_state=None, agent_name=None, phase=None):
                return [{"role": "system", "content": f"[HARNESS_DYNAMIC_CONTEXT:test]\nnew:{phase}"}]

        conv = Agent(
            "test_agent",
            "sys",
            use_tools=False,
            middlewares=[RefreshMiddleware()],
        ).start_conversation("task")

        messages = conv._refreshed_context(conv.messages, phase="summarizing_history")

        dynamic = [
            message for message in messages
            if str(message.get("content") or "").startswith("[HARNESS_DYNAMIC_CONTEXT:test]")
        ]
        self.assertEqual(len(dynamic), 1)
        self.assertIn("new:summarizing_history", dynamic[0]["content"])
        self.assertEqual(messages[1], dynamic[0])

    def test_auto_compaction_summarizes_and_returns_below_threshold_after_summary(self):
        from harness_code_agent.agent.compaction import get_thresholds
        from harness_code_agent.agent.context_manager import CompactionResult
        from harness_code_agent.agent.conversation import Agent

        thresholds = get_thresholds()
        agent = Agent("test_agent", "sys", use_tools=False)
        conv = agent.start_conversation("task")
        summarized_messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "[COMPACTED CONTEXT]\nsummary"},
            {"role": "user", "content": "task"},
        ]

        with (
            patch(
                "harness_code_agent.support.tokens.count_tokens",
                side_effect=[
                    thresholds.compact,
                    thresholds.compact - 1,
                    thresholds.compact - 1,
                ],
            ),
            patch("harness_code_agent.agent.conversation.context.detect_anxiety", return_value=False),
            patch.object(
                conv.context_manager,
                "compact",
                return_value=CompactionResult(summarized_messages, "summary", 1),
            ) as summarize,
            patch.object(
                conv.llm,
                "request_assistant_message",
                return_value=({"role": "assistant", "content": "done"}, "stop"),
            ),
        ):
            conv.run_until_idle()

        summarize.assert_called_once()

    def test_auto_compaction_persists_latest_summary(self):
        from harness_code_agent.agent.compaction import get_thresholds
        from harness_code_agent.agent.context_manager import CompactionResult
        from harness_code_agent.agent.conversation import Agent
        from harness_code_agent.runtime.permissions import PermissionPolicy
        from harness_code_agent.runtime.tool_context import ToolContext
        from harness_code_agent.sessions.store import SessionStore
        from harness_code_agent.workspace.service import WorkspaceService
        from harness_code_agent.workspace.state_paths import ensure_workspace_state

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = SessionStore(ensure_workspace_state(root))
            session = store.create(
                cwd=root,
                model="test-model",
                permission_mode="workspace-write",
            )
            tool_context = ToolContext(
                workspace=WorkspaceService(root=root, snapshots_dir=session.snapshots_dir),
                permission_policy=PermissionPolicy(mode="workspace-write"),
                event_bus=store.event_bus(session),
                session_id=session.id,
            )
            thresholds = get_thresholds()
            agent = Agent("test_agent", "sys", use_tools=False, tool_context=tool_context)
            conv = agent.start_conversation("task")
            summarized_messages = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "[COMPACTED CONTEXT]\nPersist me"},
                {"role": "user", "content": "task"},
            ]

            with (
                patch(
                    "harness_code_agent.support.tokens.count_tokens",
                    side_effect=[thresholds.compact] + [thresholds.compact - 1] * 10,
                ),
                patch("harness_code_agent.agent.conversation.context.detect_anxiety", return_value=False),
                patch.object(
                    conv.context_manager,
                    "compact",
                    return_value=CompactionResult(summarized_messages, "Persist me", 1),
                ),
                patch.object(
                    conv.llm,
                    "request_assistant_message",
                    return_value=({"role": "assistant", "content": "done"}, "stop"),
                ),
            ):
                conv.run_until_idle()

            latest = session.compacted_dir / "latest.md"
            self.assertTrue(latest.exists())
            self.assertIn("Persist me", latest.read_text(encoding="utf-8"))

    def test_auto_compaction_suspends_for_turn_when_summary_still_over_threshold(self):
        from harness_code_agent.agent.compaction import get_thresholds
        from harness_code_agent.agent.conversation import Agent

        thresholds = get_thresholds()
        agent = Agent("test_agent", "sys", use_tools=False)
        conv = agent.start_conversation("task")

        with (
            patch(
                "harness_code_agent.support.tokens.count_tokens",
                side_effect=[
                    thresholds.compact,
                    thresholds.compact,
                    thresholds.compact,
                ],
            ),
            patch("harness_code_agent.agent.conversation.context.detect_anxiety", return_value=False),
            patch.object(conv.context_manager, "compact", return_value=None) as summarize,
            patch.object(
                conv.llm,
                "request_assistant_message",
                side_effect=[
                    ({"role": "assistant", "content": "continue"}, None),
                    ({"role": "assistant", "content": "done"}, "stop"),
                ],
            ),
        ):
            conv.run_until_idle()

        self.assertEqual(summarize.call_count, 1)
        self.assertTrue(conv.runtime_state.auto_compaction_suspended)

    def test_repeated_refill_never_discards_context_after_failed_summary(self):
        from harness_code_agent.agent.compaction import get_thresholds
        from harness_code_agent.agent.conversation import Agent

        thresholds = get_thresholds()
        agent = Agent("test_agent", "sys", use_tools=False)
        conv = agent.start_conversation("first")
        conv.runtime_state.fallback.request_stop(reason="loop_detected")

        with (
            patch(
                "harness_code_agent.support.tokens.count_tokens",
                side_effect=[
                    thresholds.compact,       # main loop → trigger
                    thresholds.compact,       # after summarize → still over → suspend this turn
                    thresholds.compact,       # outbound request size
                ],
            ),
            patch("harness_code_agent.agent.conversation.context.detect_anxiety", return_value=False),
            patch.object(conv.context_manager, "compact", return_value=None),
            patch.object(
                conv.llm,
                "request_assistant_message",
                return_value=({"role": "assistant", "content": "done"}, "stop"),
            ),
        ):
            conv.run_until_idle()

        self.assertEqual(conv.messages[1]["content"], "first")
        self.assertTrue(conv.runtime_state.auto_compaction_suspended)

    def test_context_anxiety_below_threshold_only_records_soft_signal(self):
        from harness_code_agent.agent.compaction import get_thresholds
        from harness_code_agent.agent.context import ContextAnxietySignal
        from harness_code_agent.agent.conversation import Agent

        thresholds = get_thresholds()
        agent = Agent("test_agent", "sys", use_tools=False)
        conv = agent.start_conversation("task")
        observed: list[dict] = []

        class Bus:
            def emit_event(self, event):
                observed.append(event.to_event().to_dict())

        conv.event_bus = Bus()
        conv.emitter.event_bus = Bus()
        signal = ContextAnxietySignal(
            detected=True,
            score=2,
            reasons=["due to context limit", "let me wrap up"],
        )

        with (
            patch("harness_code_agent.support.tokens.count_tokens", return_value=thresholds.compact - 1),
            patch("harness_code_agent.agent.conversation.context.detect_anxiety", return_value=signal),
            patch.object(conv.context_manager, "compact") as summarize,
            patch.object(
                conv.llm,
                "request_assistant_message",
                return_value=({"role": "assistant", "content": "done"}, "stop"),
            ),
        ):
            conv.run_until_idle()

        summarize.assert_not_called()
        anxiety_events = [event for event in observed if event["type"] == "context_anxiety_observed"]
        self.assertEqual(len(anxiety_events), 1)
        self.assertEqual(anxiety_events[0]["payload"]["score"], 2)
        self.assertEqual(anxiety_events[0]["payload"]["reasons"], ["due to context limit", "let me wrap up"])

    def test_context_anxiety_soft_signal_emits_once_per_turn(self):
        from harness_code_agent.agent.compaction import get_thresholds
        from harness_code_agent.agent.context import ContextAnxietySignal
        from harness_code_agent.agent.conversation import Agent

        thresholds = get_thresholds()
        agent = Agent("test_agent", "sys", use_tools=False)
        conv = agent.start_conversation("task")
        observed: list[dict] = []

        class Bus:
            def emit_event(self, event):
                observed.append(event.to_event().to_dict())

        conv.event_bus = Bus()
        conv.emitter.event_bus = Bus()
        signal = ContextAnxietySignal(
            detected=True,
            score=2,
            reasons=["due to context limit", "let me wrap up"],
        )

        with (
            patch("harness_code_agent.support.tokens.count_tokens", return_value=thresholds.compact - 1),
            patch("harness_code_agent.agent.conversation.context.detect_anxiety", return_value=signal),
            patch.object(
                conv.llm,
                "request_assistant_message",
                side_effect=[
                    ({"role": "assistant", "content": "continue"}, None),
                    ({"role": "assistant", "content": "done"}, "stop"),
                ],
            ),
        ):
            conv.run_until_idle()

        anxiety_events = [event for event in observed if event["type"] == "context_anxiety_observed"]
        self.assertEqual(len(anxiety_events), 1)


# ---------------------------------------------------------------------------
# Session events for compaction
# ---------------------------------------------------------------------------

class CompactionEventTests(unittest.TestCase):
    """New session event types for compaction lifecycle."""

    def test_compaction_and_anxiety_events_serialize_their_payloads(self):
        from harness_code_agent.sessions.events import (
            ContextAnxietyObservedEvent,
            ContextCompactionCommittedEvent,
            ContextCompactionStartedEvent,
        )

        started = ContextCompactionStartedEvent(token_count=100000, threshold=108800, forced=False).to_event()
        self.assertEqual((started.type, started.payload["token_count"]), ("context_compaction_started", 100000))
        committed = ContextCompactionCommittedEvent(
            summary_chars=500, messages_before=20, messages_after=5, tokens_saved=40000).to_event()
        self.assertEqual((committed.type, committed.payload["tokens_saved"]), ("context_compaction_committed", 40000))
        reasons = ["due to context limit", "let me wrap up"]
        anxiety = ContextAnxietyObservedEvent(token_count=120000, threshold=170000, score=2, reasons=reasons).to_event()
        self.assertEqual(anxiety.type, "context_anxiety_observed")
        self.assertEqual((anxiety.payload["score"], anxiety.payload["threshold"], anxiety.payload["reasons"]),
                         (2, 170000, reasons))

# ---------------------------------------------------------------------------
# TUI state — compaction events update status bar
# ---------------------------------------------------------------------------

class TuiStateCompactionTests(unittest.TestCase):
    """TUI state.py should handle compaction events."""

    def _make_state(self):
        from harness_code_agent.tui.state import SessionStatusSnapshot, TuiState
        snapshot = SessionStatusSnapshot(
            model="test",
            provider="test",
            permission_mode="workspace-write",
            session_id="test-123",
            cwd=Path("/tmp"),
        )
        return TuiState(snapshot=snapshot)

    def test_compaction_progress_lives_in_the_status_bar_only(self):
        def apply(state, event_type, payload, agent=None):
            event = MagicMock()
            event.to_dict.return_value = {"type": event_type, "payload": payload, **({"agent": agent} if agent else {})}
            return state.apply_event(event)

        state = self._make_state()
        self.assertIsNone(apply(state, "context_compaction_started", {"token_count": 100000, "forced": False}))
        self.assertIn("compact", state.snapshot.status.lower())
        self.assertIsNotNone(apply(state, "context_compaction_committed", {"tokens_saved": 40000}))
        # Anxiety detection is internal and never reaches the transcript.
        self.assertIsNone(apply(state, "context_anxiety_observed",
                                {"token_count": 120000, "threshold": 170000, "score": 2, "reasons": ["limit"]}))

        child = self._make_state()
        child.snapshot.status = "idle"
        self.assertIsNone(apply(child, "context_compaction_started", {"token_count": 100000, "forced": False}, "worker-1"))
        self.assertEqual(child.snapshot.status, "idle")

# ---------------------------------------------------------------------------
# Config — context window tokens
# ---------------------------------------------------------------------------

class ContextWindowConfigTests(unittest.TestCase):
    """HARNESS_CONTEXT_WINDOW_TOKENS config."""

    def test_context_window_defaults_to_200k_and_threshold_follows_it(self):
        import importlib

        from harness_code_agent import config

        try:
            for window, expected in ((None, (200000, 160000)), ("64000", (64000, 51200))):
                env = {"HARNESS_CONTEXT_WINDOW_TOKENS": window} if window else {}
                with patch.dict(os.environ, env):
                    if not window:
                        os.environ.pop("HARNESS_CONTEXT_WINDOW_TOKENS", None)
                    os.environ.pop("COMPRESS_THRESHOLD", None)
                    importlib.reload(config)
                    self.assertEqual((config.CONTEXT_WINDOW_TOKENS, config.COMPRESS_THRESHOLD), expected)
        finally:
            importlib.reload(config)

if __name__ == "__main__":
    unittest.main()
