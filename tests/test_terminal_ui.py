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
    MentionIndex,
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

    def test_project_allowlist_supports_pipeline_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            allowlist = ApprovalAllowlist(tmp)
            command = (
                "Get-Item calc.html | Select-Object Name,Length; "
                "(Get-Content calc.html | Measure-Object -Line).Lines"
            )
            request = ApprovalRequest("run_bash", {"command": command}, "shell_risky", "confirm")
            prefixes = _persistent_prefix_for_request(request)
            self.assertIsNotNone(prefixes)
            self.assertEqual(len(prefixes), 4)  # Get-Item | Select-Object | Get-Content | Measure-Object
            allowlist.add_prefix_rule(prefixes, command=command)
            # Same stage heads with different arguments still match.
            self.assertTrue(allowlist.matches(
                "Get-Item other.html | Select-Object Length; (Get-Content other.html | Measure-Object -Word).Lines"
            ))
            # A different command head must never match.
            self.assertFalse(allowlist.matches(
                "Remove-Item calc.html | Select-Object Name; (Get-Content calc.html | Measure-Object -Line).Lines"
            ))

    def test_pipeline_allowlist_survives_redirections(self):
        with tempfile.TemporaryDirectory() as tmp:
            allowlist = ApprovalAllowlist(tmp)
            command = "git log --oneline -5 2>$null; git status --short 2>&1"
            request = ApprovalRequest("run_bash", {"command": command}, "shell_risky", "confirm")
            prefixes = _persistent_prefix_for_request(request)
            self.assertIsNotNone(prefixes)
            allowlist.add_prefix_rule(prefixes, command=command)
            self.assertTrue(allowlist.matches("git log --oneline -3 2>$null; git status --short 2>&1"))
            self.assertFalse(allowlist.matches("git push origin main 2>$null"))

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

    def test_agent_status_dedup_and_completed_wording(self):
        state = TuiState(SessionStatusSnapshot("general", "model", "provider", "workspace-write", "session", Path.cwd()))
        running = state.apply_event(SessionEvent(1, 0, "agent_status", "main", {"name": "worker", "status": "running"}))
        closed = state.apply_event(SessionEvent(2, 0, "agent_status", "main", {"name": "worker", "status": "closed"}))
        completed = state.apply_event(SessionEvent(3, 0, "agent_status", "main", {
            "name": "worker",
            "status": "completed",
            "proposal_id": "proposal_abc123",
        }))
        failed = state.apply_event(SessionEvent(4, 0, "agent_status", "main", {
            "name": "verifier",
            "status": "failed",
            "error": "pytest failed",
        }))

        self.assertIsNone(running)
        self.assertIsNone(closed)
        self.assertEqual(completed.title, "worker 已完成")
        self.assertEqual(completed.body, "有改动待应用")
        self.assertNotIn("proposal_abc123", completed.body)
        self.assertEqual(failed.title, "verifier 失败")
        self.assertEqual(failed.body, "pytest failed")

    def test_llm_usage_updates_context_tokens(self):
        from harness_code_agent.config import CONTEXT_WINDOW_TOKENS

        state = TuiState(SessionStatusSnapshot("general", "model", "provider", "workspace-write", "session", Path.cwd()))
        result = state.apply_event(SessionEvent(1, 0, "llm_usage", "main", {"prompt_tokens": 16000}))

        self.assertIsNone(result)
        self.assertEqual(state.snapshot.context_tokens, 16000)
        self.assertEqual(state.snapshot.context_window_tokens, int(CONTEXT_WINDOW_TOKENS))

    def test_run_bash_failure_keeps_raw_output(self):
        state = TuiState(SessionStatusSnapshot("general", "model", "provider", "workspace-write", "session", Path.cwd()))
        call = state.apply_event(SessionEvent(1, 0, "tool_call", "main", {
            "tool": "run_bash",
            "args": {"command": "pytest tests/test_thing.py"},
        }))
        result = state.apply_event(SessionEvent(2, 0, "tool_result", "main", {
            "tool": "run_bash",
            "status": "success",
            "return_code": 1,
            "output": "3 failed\nModuleNotFoundError: No module named 'xxx'",
        }))

        self.assertIs(result, call)
        self.assertEqual(result.status, "failed")
        self.assertIn("命令执行失败", result.title)
        self.assertIn("ModuleNotFoundError", result.body)
        self.assertIn("退出码 1", result.body)

    def test_approval_and_policy_rejections_collapse_to_a_short_line(self):
        state = TuiState(SessionStatusSnapshot("general", "model", "provider", "workspace-write", "session", Path.cwd()))
        state.apply_event(SessionEvent(1, 0, "tool_call", "main", {
            "tool": "run_bash",
            "args": {"command": "Remove-Item x"},
        }))
        denied = state.apply_event(SessionEvent(2, 0, "tool_result", "main", {
            "tool": "run_bash",
            "status": "failed",
            "error": "[approval_denied] 用户已拒绝该操作。",
            "metadata": {"status_source": "approval"},
        }))
        self.assertEqual(denied.status, "failed")
        self.assertEqual(denied.body, "操作已拒绝")

        state.apply_event(SessionEvent(3, 0, "tool_call", "main", {
            "tool": "run_bash",
            "args": {"command": "Get-ChildItem . -Recurse"},
        }))
        # A guard block is an intermediate self-correction step: the user
        # never sees it, and the running block is cleaned up.
        blocked = state.apply_event(SessionEvent(4, 0, "tool_result", "main", {
            "tool": "run_bash",
            "status": "failed",
            "error": "[blocked] Recursive repository listing...",
            "metadata": {"status_source": "tool_policy"},
        }))
        self.assertIsNone(blocked)

    def test_tool_failures_render_once_and_denial_stop_stays_quiet(self):
        state = TuiState(SessionStatusSnapshot("general", "model", "provider", "workspace-write", "session", Path.cwd()))
        state.apply_event(SessionEvent(1, 0, "tool_call", "main", {
            "tool": "run_bash",
            "args": {"command": "pytest"},
        }))
        state.apply_event(SessionEvent(2, 0, "tool_result", "main", {
            "tool": "run_bash",
            "status": "failed",
            "error": "[approval_denied] 用户已拒绝该操作。",
            "metadata": {"status_source": "approval"},
        }))
        # The tool_result already showed "操作已拒绝": a failure for the same
        # tool would duplicate it, and the approval_denied stop must not add
        # a "代理已停止" block on top of a decision the user made.
        failure = state.apply_event(SessionEvent(3, 0, "failure", "main", {
            "tool": "run_bash",
            "category": "approval_denied",
            "message": "用户已拒绝该操作。",
        }))
        self.assertIsNone(failure)
        fallback = state.apply_event(SessionEvent(4, 0, "agent_fallback", "main", {
            "reason": "approval_denied",
            "last_tool": "run_bash",
        }))
        self.assertIsNone(fallback)
        # System-level failures without a tool still render.
        system_failure = state.apply_event(SessionEvent(5, 0, "failure", "main", {
            "category": "runtime",
            "message": "后台线程崩溃",
        }))
        self.assertIsNotNone(system_failure)
        self.assertIn("后台线程崩溃", system_failure.body)

    def test_mention_index_caches_file_and_session_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "readme.md").write_text("x", encoding="utf-8")

            class Store:
                def list_sessions(self):
                    return [{"id": "s1"}]

                def read_events(self, session_id):
                    return [{"type": "user_input", "payload": {"text": "hello review"}}]

            index = MentionIndex(root, Store())
            first = index.candidates("review")
            (root / "review-new.md").write_text("x", encoding="utf-8")
            cached = index.candidates("review")

            self.assertEqual([c.display for c in cached], [c.display for c in first])

            fresh = MentionIndex(root, Store(), ttl_seconds=0).candidates("review")
            self.assertIn("review-new.md", [c.display for c in fresh])


if __name__ == "__main__":
    unittest.main()
