from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from harness_code_agent.runtime.approvals import ApprovalRequest
from harness_code_agent.sessions.events import SessionEvent
from harness_code_agent.tui.approval import (
    ApprovalAllowlist,
    _persistent_prefix_for_request,
)
from harness_code_agent.tui.commands import default_command_registry
from harness_code_agent.tui.completion import (
    current_mention_query,
    mention_candidates,
    replace_mention_fragment,
)
from harness_code_agent.tui.state import (
    SessionStatusSnapshot,
    TranscriptBlock,
    TuiState,
)


class TerminalUiTests(unittest.TestCase):
    def test_command_registry_exposes_structured_panel_actions(self):
        registry = default_command_registry(skill_registry=SimpleNamespace(user_commands=[]))
        self.assertEqual(registry.execute("/observe", SimpleNamespace()).action, "observe")
        self.assertIn("/checkpoint", registry.command_names())
        self.assertNotIn("/profile", registry.command_names())

    def test_mentions_search_files_and_replace_only_active_fragment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README file.md").write_text("demo", encoding="utf-8")
            store = SimpleNamespace(list_sessions=list)
            candidates = mention_candidates(root, "read", store)
        self.assertEqual(candidates[0].insert_text, 'file:"README file.md"')
        self.assertEqual(current_mention_query("inspect @rea"), ("rea", -4))
        self.assertEqual(replace_mention_fragment("inspect @rea", candidates[0].insert_text), 'inspect @file:"README file.md" ')

    def test_transcript_state_tracks_tool_and_file_events(self):
        state = TuiState(SessionStatusSnapshot("general", "model", "provider", "workspace-write", "session", Path.cwd()))
        call = state.apply_event(SessionEvent(1, 0, "tool_call", "main", {"tool": "read_file", "args": {"path": "README.md"}}))
        change = state.apply_event(SessionEvent(2, 0, "file_change", "main", {"operation": "edit", "path": "README.md", "diff": "+new"}))
        self.assertEqual(call.kind, "tool")
        self.assertIn("README.md", call.title)
        self.assertEqual(change.kind, "file")
        self.assertEqual(state.snapshot.dirty_count, 1)

    def test_transcript_shows_child_parent_report(self):
        state = TuiState(SessionStatusSnapshot("general", "model", "provider", "workspace-write", "session", Path.cwd()))
        block = state.apply_event(SessionEvent(
            1, 0, "agent_parent_message", "reviewer",
            {"name": "reviewer", "message": "evidence contradicts the plan"},
        ))

        self.assertEqual(block.kind, "agent")
        self.assertIn("reviewer", block.title)
        self.assertEqual(block.body, "evidence contradicts the plan")

    def test_transcript_hides_internal_route_fallbacks_and_only_shows_real_switches(self):
        state = TuiState(SessionStatusSnapshot("general", "model", "provider", "workspace-write", "session", Path.cwd()))
        stayed = state.apply_event(SessionEvent(
            1,
            0,
            "profile_route_decision",
            "main",
            {
                "profile": "general",
                "action": "stay",
                "switched": False,
                "fallback_used": True,
                "fallback_reason": "low local route confidence",
            },
        ))
        switched = state.apply_event(SessionEvent(
            2,
            0,
            "profile_route_decision",
            "main",
            {
                "profile": "coding-agent",
                "action": "switch_profile",
                "switched": True,
                "fallback_used": False,
            },
        ))

        self.assertIsNone(stayed)
        self.assertEqual(switched.title, "工作模式")
        self.assertEqual(switched.body, "已切换到 coding-agent")
        self.assertNotIn("兜底", switched.body)

    def test_project_allowlist_reuses_persisted_command_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            allowlist = ApprovalAllowlist(tmp)
            request = ApprovalRequest("run_bash", {"command": "python scripts/check.py --fast"}, "shell_risky", "confirm")
            prefix = _persistent_prefix_for_request(request)
            self.assertIsNotNone(prefix)
            allowlist.add_prefix_rule(prefix, command=request.args["command"])
            self.assertTrue(allowlist.matches("python scripts/check.py --all"))
            self.assertFalse(allowlist.matches("python scripts/delete.py --all"))

    def test_fallback_block_ids_are_unique_and_replay_stable(self):
        def build_ids() -> list[str]:
            state = TuiState(SessionStatusSnapshot("general", "model", "provider", "workspace-write", "session", Path.cwd()))
            state.add_block(TranscriptBlock("notice", "same", "first", turn=2))
            state.add_block(TranscriptBlock("notice", "same", "second", turn=2))
            return [block.id for block in state.blocks]

        live_ids = build_ids()
        replay_ids = build_ids()

        self.assertEqual(live_ids, replay_ids)
        self.assertEqual(len(set(live_ids)), 2)


if __name__ == "__main__":
    unittest.main()
