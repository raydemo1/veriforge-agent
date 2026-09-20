"""Tests for context compaction strategy (PLAN.md)."""
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _install_fake_openai_module() -> None:
    if "openai" in sys.modules:
        return
    openai = types.ModuleType("openai")

    class OpenAI:
        def __init__(self, *args, **kwargs):
            pass

    openai.OpenAI = OpenAI
    sys.modules["openai"] = openai


_install_fake_openai_module()



# ---------------------------------------------------------------------------
# CompactionGate — controls when compaction is allowed
# ---------------------------------------------------------------------------

class CompactionGateTests(unittest.TestCase):
    """CompactionGate tracks active tool calls, dirty state, and message revision."""

    def _make_gate(self):
        from harness_code_agent.agent.compaction import CompactionGate
        return CompactionGate()

    def test_initial_state_allows_compaction(self):
        gate = self._make_gate()
        self.assertTrue(gate.can_compact())

    def test_active_tool_calls_block_compaction(self):
        gate = self._make_gate()
        gate.begin_tool_call()
        gate.begin_tool_call()
        self.assertFalse(gate.can_compact())

    def test_tool_result_allows_compaction(self):
        gate = self._make_gate()
        gate.begin_tool_call()
        gate.begin_tool_call()
        gate.end_tool_call()
        gate.end_tool_call()
        self.assertTrue(gate.can_compact())

    def test_partial_tool_calls_still_block(self):
        gate = self._make_gate()
        gate.begin_tool_call()
        gate.begin_tool_call()
        gate.end_tool_call()
        self.assertFalse(gate.can_compact())

    def test_message_revision_increments(self):
        gate = self._make_gate()
        r0 = gate.revision
        gate.bump_revision()
        r1 = gate.revision
        self.assertEqual(r1, r0 + 1)

    def test_coalescing_window_prevents_rapid_compaction(self):
        gate = self._make_gate()
        gate.mark_compacted()
        # Within coalescing window, should not compact again
        self.assertFalse(gate.can_compact(coalesce_seconds=30))

    def test_coalescing_window_expires(self):
        import time
        gate = self._make_gate()
        gate._last_compact_time = time.time() - 60
        self.assertTrue(gate.can_compact(coalesce_seconds=30))

    def test_dirty_flag_tracks_context_changes(self):
        gate = self._make_gate()
        self.assertFalse(gate.dirty)
        gate.mark_dirty()
        self.assertTrue(gate.dirty)
        gate.mark_compacted()
        self.assertFalse(gate.dirty)


# ---------------------------------------------------------------------------
# Threshold behavior — token usage triggers different actions
# ---------------------------------------------------------------------------

class CompactionThresholdTests(unittest.TestCase):
    """Token usage thresholds drive compaction strategy."""

    def test_thresholds_derived_from_context_window(self):
        """Thresholds should be percentages of HARNESS_CONTEXT_WINDOW_TOKENS."""
        with patch.dict(os.environ, {"HARNESS_CONTEXT_WINDOW_TOKENS": "100000"}):
            from harness_code_agent.agent import compaction
            thresholds = compaction.get_thresholds(100000)
            self.assertEqual(thresholds.compact, 85000)          # 85%

    def test_below_compact_threshold_no_action(self):
        from harness_code_agent.agent.compaction import (
            compaction_action,
            get_thresholds,
        )
        thresholds = get_thresholds(128000)
        action = compaction_action(thresholds.compact - 1, thresholds)
        self.assertEqual(action, "none")

    def test_at_compact_threshold_triggers_auto_compact(self):
        from harness_code_agent.agent.compaction import (
            compaction_action,
            get_thresholds,
        )
        thresholds = get_thresholds(128000)
        action = compaction_action(thresholds.compact, thresholds)
        self.assertEqual(action, "auto_compact")


class ContextAnxietyTests(unittest.TestCase):
    def test_detect_anxiety_returns_structured_soft_signal(self):
        from harness_code_agent.agent.context import detect_anxiety

        signal = detect_anxiety([
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "Due to the context limit, let me wrap up here."},
        ])

        self.assertTrue(signal.detected)
        self.assertTrue(signal)
        self.assertGreaterEqual(signal.score, 2)
        self.assertEqual(signal.source, "assistant_recent_messages")
        self.assertTrue(any("context limit" in reason.lower() for reason in signal.reasons))

    def test_detect_anxiety_absent_signal_is_falsey(self):
        from harness_code_agent.agent.context import detect_anxiety

        signal = detect_anxiety([
            {"role": "assistant", "content": "I will continue with the next verification step."},
        ])

        self.assertFalse(signal)
        self.assertFalse(signal.detected)
        self.assertEqual(signal.score, 0)
        self.assertEqual(signal.reasons, [])

    def test_detect_anxiety_recognizes_chinese_context_limit_language(self):
        from harness_code_agent.agent.context import detect_anxiety

        signal = detect_anxiety([
            {"role": "assistant", "content": "我快没上下文空间了，先收尾一下。上下文快满了，我只覆盖关键部分。"},
        ])

        self.assertTrue(signal.detected)
        self.assertGreaterEqual(signal.score, 2)
        self.assertTrue(any("上下文" in reason for reason in signal.reasons))


# ---------------------------------------------------------------------------
# /compact slash command
# ---------------------------------------------------------------------------

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

    def test_compact_command_is_registered_for_manual_compaction(self):
        from harness_code_agent.tui.commands import default_command_registry
        registry = default_command_registry()
        spec = registry._by_name.get("/compact")
        self.assertIsNotNone(spec, "/compact command should be registered")
        self.assertEqual(spec.usage, "/compact")
        self.assertEqual(registry.execute("/compact", self._make_session(".")).action, "compact")

    def test_compact_requires_a_plain_command(self):
        from harness_code_agent.tui.commands import default_command_registry
        registry = default_command_registry()
        session = MagicMock()

        self.assertIn("用法：/compact", registry.execute("/compact status", session).text)
        self.assertIn("用法：/compact", registry.execute("/compact history", session).text)

    def test_manual_compact_runs_without_waiting_for_threshold(self):
        from harness_code_agent.agent.conversation import Agent

        conv = Agent("test_agent", "sys", use_tools=False).start_conversation()
        conv.add_user_turn("older task")
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


class AgentConversationCompactionLifecycleTests(unittest.TestCase):
    def test_message_revision_tracks_user_assistant_tool_and_middleware_appends(self):
        from harness_code_agent.agent.conversation import Agent

        agent = Agent("test_agent", "sys", use_tools=False)
        conv = agent.start_conversation()
        initial = conv.compaction_gate.revision

        conv.add_user_turn("task")

        self.assertGreater(conv.compaction_gate.revision, initial)

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

        conv._refresh_dynamic_context_after_compaction(phase="summarizing_history")

        dynamic = [
            message for message in conv.messages
            if str(message.get("content") or "").startswith("[HARNESS_DYNAMIC_CONTEXT:test]")
        ]
        self.assertEqual(len(dynamic), 1)
        self.assertIn("new:summarizing_history", dynamic[0]["content"])
        self.assertEqual(conv.messages[1], dynamic[0])

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
                "harness_code_agent.agent.conversation.context.count_tokens",
                side_effect=[
                    thresholds.compact,
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

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = SessionStore(root / ".harness")
            session = store.create(
                profile="coding-agent",
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
                    "harness_code_agent.agent.conversation.context.count_tokens",
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
                "harness_code_agent.agent.conversation.context.count_tokens",
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
        conv.runtime_state.context_refill_streak = 1
        conv.runtime_state.fallback.request_stop(reason="loop_detected")

        with (
            patch(
                "harness_code_agent.agent.conversation.context.count_tokens",
                side_effect=[
                    thresholds.compact,       # main loop → trigger
                    thresholds.compact,       # after summarize → still over → suspend this turn
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
        self.assertGreaterEqual(conv.runtime_state.context_refill_streak, 2)
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
            patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=thresholds.compact - 1),
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
            patch("harness_code_agent.agent.conversation.context.count_tokens", return_value=thresholds.compact - 1),
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
# Compacted storage — persistence in .harness/sessions/<id>/compacted/
# ---------------------------------------------------------------------------

class CompactedStorageTests(unittest.TestCase):
    """Compacted directory structure and persistence."""

    def test_session_dataclass_has_compacted_path(self):
        """Session dataclass should include compacted_dir."""
        from harness_code_agent.sessions.store import Session
        s = Session(
            id="test",
            root=Path("/tmp/test"),
            metadata_path=Path("/tmp/test/session.json"),
            events_path=Path("/tmp/test/events.jsonl"),
            snapshots_dir=Path("/tmp/test/snapshots"),
            summary_path=Path("/tmp/test/summary.md"),
        )
        # compacted_dir should exist as an attribute
        self.assertTrue(hasattr(s, "compacted_dir"))

    def test_session_store_creates_compacted_dir(self):
        """SessionStore.create() should create the compacted/ subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from harness_code_agent.sessions.store import SessionStore
            store = SessionStore(Path(tmpdir))
            session = store.create(
                profile="coding-agent",
                cwd=tmpdir,
                model="test",
                permission_mode="workspace-write",
            )
            compacted = session.root / "compacted"
            self.assertTrue(compacted.exists())
            self.assertTrue((compacted / "history").exists())


# ---------------------------------------------------------------------------
# Session events for compaction
# ---------------------------------------------------------------------------

class CompactionEventTests(unittest.TestCase):
    """New session event types for compaction lifecycle."""

    def test_compaction_started_event(self):
        from harness_code_agent.sessions.events import ContextCompactionStartedEvent
        event = ContextCompactionStartedEvent(
            token_count=100000,
            threshold=108800,
            forced=False,
        )
        structured = event.to_event()
        self.assertEqual(structured.type, "context_compaction_started")
        self.assertEqual(structured.payload["token_count"], 100000)

    def test_compaction_committed_event(self):
        from harness_code_agent.sessions.events import ContextCompactionCommittedEvent
        event = ContextCompactionCommittedEvent(
            summary_chars=500,
            messages_before=20,
            messages_after=5,
            tokens_saved=40000,
        )
        structured = event.to_event()
        self.assertEqual(structured.type, "context_compaction_committed")
        self.assertIn("tokens_saved", structured.payload)

    def test_context_anxiety_observed_event(self):
        from harness_code_agent.sessions.events import ContextAnxietyObservedEvent
        event = ContextAnxietyObservedEvent(
            token_count=120000,
            threshold=170000,
            score=2,
            reasons=["due to context limit", "let me wrap up"],
        )
        structured = event.to_event()
        self.assertEqual(structured.type, "context_anxiety_observed")
        self.assertEqual(structured.payload["score"], 2)
        self.assertEqual(structured.payload["threshold"], 170000)
        self.assertEqual(structured.payload["reasons"], ["due to context limit", "let me wrap up"])

# ---------------------------------------------------------------------------
# TUI state — compaction events update status bar
# ---------------------------------------------------------------------------

class TuiStateCompactionTests(unittest.TestCase):
    """TUI state.py should handle compaction events."""

    def _make_state(self):
        from harness_code_agent.tui.state import SessionStatusSnapshot, TuiState
        snapshot = SessionStatusSnapshot(
            profile="coding-agent",
            model="test",
            provider="test",
            permission_mode="workspace-write",
            session_id="test-123",
            cwd=Path("/tmp"),
        )
        return TuiState(snapshot=snapshot)

    def test_compaction_started_updates_status(self):
        state = self._make_state()
        event = MagicMock()
        event.to_dict.return_value = {
            "type": "context_compaction_started",
            "payload": {"token_count": 100000, "forced": False},
        }
        block = state.apply_event(event)
        # Compaction progress lives in the status bar only, not the transcript.
        self.assertIsNone(block)
        self.assertIn("compact", state.snapshot.status.lower())

    def test_compaction_committed_restores_idle(self):
        state = self._make_state()
        event = MagicMock()
        event.to_dict.return_value = {
            "type": "context_compaction_committed",
            "payload": {"tokens_saved": 40000},
        }
        block = state.apply_event(event)
        self.assertIsNotNone(block)

    def test_handoff_reset_shows_notice(self):
        state = self._make_state()
        event = MagicMock()
        event.to_dict.return_value = {
            "type": "context_compaction_started",
            "payload": {"token_count": 180000, "phase": "handoff_reset"},
        }
        block = state.apply_event(event)
        self.assertIsNone(block)
        self.assertIn("compact", state.snapshot.status.lower())

    def test_context_anxiety_observed_shows_soft_notice(self):
        state = self._make_state()
        event = MagicMock()
        event.to_dict.return_value = {
            "type": "context_anxiety_observed",
            "payload": {"token_count": 120000, "threshold": 170000, "score": 2, "reasons": ["due to context limit"]},
        }
        block = state.apply_event(event)
        # Internal anxiety detection stays out of the transcript.
        self.assertIsNone(block)


# ---------------------------------------------------------------------------
# Config — context window tokens
# ---------------------------------------------------------------------------

class ContextWindowConfigTests(unittest.TestCase):
    """HARNESS_CONTEXT_WINDOW_TOKENS config."""

    def test_context_window_defaults_to_200k(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HARNESS_CONTEXT_WINDOW_TOKENS", None)
            os.environ.pop("COMPRESS_THRESHOLD", None)
            import importlib

            from harness_code_agent import config
            importlib.reload(config)
            self.assertEqual(config.CONTEXT_WINDOW_TOKENS, 200000)
            self.assertEqual(config.COMPRESS_THRESHOLD, 170000)
            # Restore default for later tests in this process.
            importlib.reload(config)

    def test_context_window_configurable(self):
        with patch.dict(os.environ, {"HARNESS_CONTEXT_WINDOW_TOKENS": "64000"}):
            os.environ.pop("COMPRESS_THRESHOLD", None)
            # Need to reload to pick up env change
            import importlib

            from harness_code_agent import config
            importlib.reload(config)
            self.assertEqual(config.CONTEXT_WINDOW_TOKENS, 64000)
            self.assertEqual(config.COMPRESS_THRESHOLD, 54400)
            # Restore default
            importlib.reload(config)


if __name__ == "__main__":
    unittest.main()
