from __future__ import annotations

import os
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
from harness_code_agent.tui.completion import MentionIndex
from harness_code_agent.tui.state import (
    SessionStatusSnapshot,
    TranscriptBlock,
    TuiState,
)


class TerminalUiTests(unittest.TestCase):
    def test_command_registry_exposes_structured_panel_actions(self):
        registry = default_command_registry(skill_registry=SimpleNamespace(skills=[]))
        self.assertEqual(registry.execute("/observe", SimpleNamespace()).action, "observe")
        self.assertNotIn("/checkpoint", registry.command_names())
        self.assertNotIn("/profile", registry.command_names())

    def test_mentions_quote_file_paths_with_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README file.md").write_text("demo", encoding="utf-8")
            store = SimpleNamespace(list_sessions=list)
            candidates = MentionIndex(root, store).candidates("read")
        self.assertEqual(candidates[0].insert_text, 'file:"README file.md"')

    def test_transcript_state_tracks_tool_and_file_events(self):
        state = TuiState(SessionStatusSnapshot("model", "provider", "workspace-write", "session", Path.cwd()))
        call = state.apply_event(SessionEvent(1, 0, "tool_call", "main", {"tool": "read_file", "args": {"path": "README.md"}}))
        change = state.apply_event(SessionEvent(2, 0, "file_change", "main", {"operation": "edit", "path": "README.md", "diff": "+new"}))
        self.assertEqual(call.kind, "tool")
        self.assertIn("README.md", call.title)
        self.assertEqual(change.kind, "file")
        self.assertEqual(state.snapshot.dirty_count, 1)

    def test_transcript_shows_child_parent_report(self):
        state = TuiState(SessionStatusSnapshot("model", "provider", "workspace-write", "session", Path.cwd()))
        block = state.apply_event(SessionEvent(
            1, 0, "agent_parent_message", "reviewer",
            {"name": "reviewer", "message": "evidence contradicts the plan"},
        ))

        agent = state.agent_blocks["reviewer"]
        self.assertEqual(agent.role, "group")
        self.assertEqual(agent.title, "reviewer")
        self.assertEqual(block.kind, "agent")
        self.assertEqual(block.parent_id, agent.id)
        self.assertEqual(block.direction, "in")
        self.assertEqual(block.body, "evidence contradicts the plan")

    def test_child_internal_events_stay_inside_the_agent_timeline(self):
        state = TuiState(SessionStatusSnapshot("model", "provider", "workspace-write", "session", Path.cwd()))
        state.apply_event(SessionEvent(1, 0, "llm_usage", "main_agent", {"prompt_tokens": 1200}))
        state.apply_event(SessionEvent(2, 0, "turn_started", "main_agent", {"turn": 1}))

        spawned = state.apply_event(SessionEvent(3, 0, "agent_spawned", "worker-1", {
            "agent_id": "agent_1", "name": "worker-1", "role": "worker",
            "task": "build the parser", "status": "queued",
        }))
        self.assertEqual(spawned.role, "group")
        self.assertEqual(spawned.body, "build the parser")

        self.assertIsNone(state.apply_event(SessionEvent(
            4, 0, "llm_usage", "worker-1", {"agent_id": "agent_1", "prompt_tokens": 90000},
        )))
        self.assertEqual(state.snapshot.context_tokens, 1200)
        call = state.apply_event(SessionEvent(
            5, 0, "tool_call", "worker-1",
            {"agent_id": "agent_1", "tool": "bash", "args": {"command": "pytest -q"}, "tool_call_id": "c1"},
        ))
        self.assertEqual(call.parent_id, spawned.id)
        self.assertEqual(state.snapshot.status, "thinking")
        self.assertIsNone(state.active_tool_blocks.get(("main", "bash")))

        result = state.apply_event(SessionEvent(
            6, 0, "tool_result", "worker-1",
            {"agent_id": "agent_1", "tool": "bash", "status": "success", "return_code": 0,
             "output": "ok", "tool_call_id": "c1"},
        ))
        self.assertIs(result, call)
        self.assertEqual(result.parent_id, spawned.id)

        failure = state.apply_event(SessionEvent(
            7, 0, "failure", "worker-1",
            {"agent_id": "agent_1", "message": "worker failed"},
        ))
        self.assertEqual(failure.parent_id, spawned.id)
        self.assertEqual(state.snapshot.status, "thinking")

        # Shared-workspace file changes join the agent timeline and still
        # update the global dirty count.
        change = state.apply_event(SessionEvent(
            8, 0, "file_change", "worker-1",
            {"agent_id": "agent_1", "operation": "modify", "path": "shared.py", "diff": "+new"},
        ))
        self.assertEqual(change.kind, "file")
        self.assertEqual(change.parent_id, spawned.id)
        self.assertEqual(state.snapshot.dirty_count, 1)

        completed = state.apply_event(SessionEvent(
            9, 0, "agent_status", "worker-1",
            {"agent_id": "agent_1", "name": "worker-1", "status": "completed", "summary": "parser ready"},
        ))
        self.assertIs(completed, spawned)
        self.assertEqual(completed.status, "success")
        self.assertEqual(completed.body, "parser ready")
        self.assertEqual(list(state.agent_blocks), ["agent_1"])

    def test_parallel_subagents_pair_same_named_tools_by_call_id(self):
        state = TuiState(SessionStatusSnapshot("model", "provider", "workspace-write", "session", Path.cwd()))
        agents = {}
        for index, name in enumerate(("alpha", "beta", "gamma"), 1):
            agents[name] = state.apply_event(SessionEvent(index, 0, "agent_spawned", name, {
                "agent_id": f"agent_{index}", "name": name, "task": f"task {name}", "status": "queued",
            }))
        calls = {}
        sequence = 10
        for index, name in enumerate(agents, 1):
            calls[name] = state.apply_event(SessionEvent(sequence, 0, "tool_call", name, {
                "agent_id": f"agent_{index}", "tool": "bash",
                "args": {"command": f"echo {name}"}, "tool_call_id": f"call-{name}",
            }))
            sequence += 1
        for index, name in reversed(list(enumerate(agents, 1))):
            result = state.apply_event(SessionEvent(sequence, 0, "tool_result", name, {
                "agent_id": f"agent_{index}", "tool": "bash", "status": "success",
                "return_code": 0, "output": name, "tool_call_id": f"call-{name}",
            }))
            sequence += 1
            self.assertIs(result, calls[name])
            self.assertEqual(result.parent_id, agents[name].id)

        self.assertEqual(state.active_tool_calls, {})
        self.assertEqual(state.active_tool_blocks, {})

    def test_child_tool_results_never_consume_the_main_agents_call(self):
        state = TuiState(SessionStatusSnapshot("model", "provider", "workspace-write", "session", Path.cwd()))
        main_call = state.apply_event(SessionEvent(1, 0, "tool_call", "main_agent", {
            "tool": "bash", "args": {"command": "pytest -q"}, "tool_call_id": "main-call",
        }))
        agent = state.apply_event(SessionEvent(2, 0, "agent_spawned", "worker", {
            "agent_id": "agent_1", "name": "worker", "task": "run checks", "status": "queued",
        }))
        child_call = state.apply_event(SessionEvent(3, 0, "tool_call", "worker", {
            "agent_id": "agent_1", "tool": "bash", "args": {"command": "pytest -q"},
            "tool_call_id": "child-call",
        }))

        child_result = state.apply_event(SessionEvent(4, 0, "tool_result", "worker", {
            "agent_id": "agent_1", "tool": "bash", "status": "success", "return_code": 0,
            "output": "child", "tool_call_id": "child-call",
        }))
        self.assertIs(child_result, child_call)
        self.assertEqual(child_result.parent_id, agent.id)

        main_result = state.apply_event(SessionEvent(5, 0, "tool_result", "main_agent", {
            "tool": "bash", "status": "success", "return_code": 0, "output": "main",
            "tool_call_id": "main-call",
        }))
        self.assertIs(main_result, main_call)
        self.assertIsNone(main_result.parent_id)

    def test_agent_followup_reuses_the_same_inline_record(self):
        state = TuiState(SessionStatusSnapshot("model", "provider", "workspace-write", "session", Path.cwd()))
        spawned = state.apply_event(SessionEvent(1, 0, "agent_spawned", "worker", {
            "agent_id": "agent_1", "name": "worker", "task": "first", "status": "queued",
        }))
        state.apply_event(SessionEvent(2, 0, "agent_status", "worker", {
            "agent_id": "agent_1", "name": "worker", "status": "completed", "summary": "first done",
        }))
        message = state.apply_event(SessionEvent(3, 0, "agent_message", "worker", {
            "agent_id": "agent_1", "name": "worker", "status": "queued",
            "message": "try again", "mode": "followup",
        }))
        self.assertEqual(message.parent_id, spawned.id)
        self.assertEqual(spawned.status, "pending")
        self.assertEqual(spawned.body, "继续任务")

        resumed = state.apply_event(SessionEvent(4, 0, "agent_status", "worker", {
            "agent_id": "agent_1", "name": "worker", "status": "running",
        }))
        self.assertIs(resumed, spawned)
        self.assertEqual(resumed.status, "running")

        completed = state.apply_event(SessionEvent(5, 0, "agent_status", "worker", {
            "agent_id": "agent_1", "name": "worker", "status": "completed", "summary": "second done",
        }))
        self.assertIs(completed, spawned)
        self.assertEqual(completed.body, "second done")
        self.assertEqual(len(state.agent_blocks), 1)

    def test_agent_timeline_replay_is_stable(self):
        events = [
            SessionEvent(1, 0, "turn_started", "main_agent", {"turn": 1}),
            SessionEvent(2, 0, "agent_spawned", "scout", {
                "agent_id": "agent_1", "name": "scout", "task": "inspect", "status": "queued",
            }),
            SessionEvent(3, 0, "thought_started", "scout", {"agent_id": "agent_1"}),
            SessionEvent(4, 0, "thought_finished", "scout", {"agent_id": "agent_1", "duration_seconds": 1.2}),
            SessionEvent(5, 0, "tool_call", "scout", {
                "agent_id": "agent_1", "tool": "read_file", "args": {"path": "a.py"},
                "tool_call_id": "c1",
            }),
            SessionEvent(6, 0, "tool_result", "scout", {
                "agent_id": "agent_1", "tool": "read_file", "status": "success",
                "tool_call_id": "c1",
            }),
            SessionEvent(7, 0, "agent_status", "scout", {
                "agent_id": "agent_1", "name": "scout", "status": "completed", "summary": "done",
            }),
            SessionEvent(8, 0, "turn_finished", "main_agent", {"turn": 1, "status": "completed"}),
        ]

        def snapshot():
            state = TuiState(SessionStatusSnapshot("model", "provider", "workspace-write", "session", Path.cwd()))
            for event in events:
                block = state.apply_event(event)
                state.add_block(block)
            return [
                (block.kind, block.title, block.body, block.status, block.id, block.parent_id)
                for block in state.blocks
            ]

        self.assertEqual(snapshot(), snapshot())

    def test_live_and_replayed_subagent_rows_stay_in_the_original_turn(self):
        from harness_code_agent.sessions.store import SessionStore
        from harness_code_agent.tui_bridge import SessionBridge

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "state")
            session = store.create(cwd=root, model="model", permission_mode="workspace-write")
            messages = []
            bridge = SessionBridge(
                cwd=root, state_root=store.root, session_id=session.id, environment="local",
                emit=messages.append, session_action=None,
            )
            bus = store.event_bus(session, listener=bridge._event_listener)
            bus.emit("turn_started", agent="main_agent", payload={"turn": 1})
            bus.emit("agent_spawned", agent="scout", payload={
                "agent_id": "agent_1", "name": "scout", "role": "explorer",
                "task": "inspect", "status": "queued",
            })
            bus.emit("tool_call", agent="scout", payload={
                "agent_id": "agent_1", "tool": "bash",
                "args": {"command": "ls"}, "tool_call_id": "c1",
            })
            bus.emit("tool_result", agent="scout", payload={
                "agent_id": "agent_1", "tool": "bash", "status": "success",
                "return_code": 0, "output": "ok", "tool_call_id": "c1",
            })
            bus.emit("turn_finished", agent="main_agent", payload={"turn": 1, "status": "completed"})
            bus.emit("agent_status", agent="scout", payload={
                "agent_id": "agent_1", "name": "scout", "status": "running",
            })
            bus.emit("turn_started", agent="main_agent", payload={"turn": 2})
            bus.emit("agent_status", agent="scout", payload={
                "agent_id": "agent_1", "name": "scout",
                "status": "completed", "summary": "found it",
            })
            bus.emit("turn_finished", agent="main_agent", payload={"turn": 2, "status": "completed"})

            live_items = [
                message["event"]["item"] for message in messages
                if message["event"]["type"] == "transcript"
                and message["event"]["item"].get("id") == "agent-agent_1"
            ]
            self.assertGreaterEqual(len(live_items), 3)
            self.assertEqual({item["parentId"] for item in live_items}, {"assistant-group-1-1"})
            self.assertEqual(live_items[-1]["body"], "found it")
            items = bridge._history_items(session.id)
            messages.clear()
            bus.emit("turn_started", agent="main_agent", payload={"turn": 3})
            bus.emit("agent_status", agent="scout", payload={
                "agent_id": "agent_1", "name": "scout", "status": "running",
            })
            followup = next(
                event["item"] for event in (message["event"] for message in messages)
                if event["type"] == "transcript" and event["item"]["id"] == "agent-agent_1"
            )
            self.assertEqual(followup["parentId"], "history-group-1")

        agent_items = [item for item in items if item["kind"] == "agent" and item.get("role") == "group"]
        self.assertEqual(len(agent_items), 1)
        agent_item = agent_items[0]
        self.assertEqual(agent_item["id"], "agent-agent_1")
        self.assertEqual(agent_item["parentId"], "history-group-1")
        self.assertEqual(agent_item["title"], "scout")
        self.assertEqual(agent_item["state"], "success")
        self.assertEqual(agent_item["body"], "found it")

        children = [item for item in items if item.get("parentId") == agent_item["id"]]
        self.assertEqual([item["kind"] for item in children], ["tool"])
        self.assertEqual(children[0]["state"], "success")
        self.assertIn("命令已执行", children[0]["title"])

    def test_child_activity_does_not_reorder_main_output(self):
        state = TuiState(SessionStatusSnapshot("model", "provider", "workspace-write", "session", Path.cwd()))

        def apply(sequence, event_type, agent, payload):
            block = state.apply_event(SessionEvent(sequence, 0, event_type, agent, payload))
            state.add_block(block)
            return block

        apply(1, "turn_started", "main_agent", {"turn": 1})
        apply(2, "assistant_message", "main_agent", {"text": "第一步"})
        self.assertEqual(state.snapshot.status, "idle")
        spawned = apply(3, "agent_spawned", "worker", {
            "agent_id": "agent_1", "name": "worker", "task": "check", "status": "queued",
        })
        apply(4, "tool_call", "worker", {
            "agent_id": "agent_1", "tool": "bash", "args": {"command": "pytest -q"},
            "tool_call_id": "c1",
        })
        self.assertEqual(state.snapshot.status, "idle")
        apply(5, "tool_result", "worker", {
            "agent_id": "agent_1", "tool": "bash", "status": "success",
            "return_code": 0, "output": "ok", "tool_call_id": "c1",
        })
        apply(6, "assistant_message", "main_agent", {"text": "第二步"})

        self.assertEqual(
            [block.kind for block in state.blocks],
            ["assistant", "agent", "tool", "assistant"],
        )
        self.assertEqual(state.blocks[1].id, spawned.id)
        self.assertEqual(state.blocks[2].parent_id, spawned.id)
        self.assertEqual(state.snapshot.status, "idle")

    def test_late_spawn_event_does_not_reopen_a_finished_agent(self):
        state = TuiState(SessionStatusSnapshot("model", "provider", "workspace-write", "session", Path.cwd()))
        completed = state.apply_event(SessionEvent(1, 0, "agent_status", "fast", {
            "agent_id": "agent_f", "name": "fast", "status": "completed", "summary": "done",
        }))
        spawned = state.apply_event(SessionEvent(2, 0, "agent_spawned", "fast", {
            "agent_id": "agent_f", "name": "fast", "task": "inspect", "status": "queued",
        }))

        self.assertIs(spawned, completed)
        self.assertEqual(spawned.status, "success")
        self.assertEqual(spawned.body, "done")
        self.assertEqual(len(state.agent_blocks), 1)

    def test_agent_status_dedup_and_completed_wording(self):
        state = TuiState(SessionStatusSnapshot("model", "provider", "workspace-write", "session", Path.cwd()))
        spawned = state.apply_event(SessionEvent(1, 0, "agent_spawned", "worker", {
            "agent_id": "agent_w", "name": "worker", "task": "run", "status": "queued",
        }))
        running = state.apply_event(SessionEvent(2, 0, "agent_status", "worker", {
            "agent_id": "agent_w", "name": "worker", "status": "running",
        }))
        closed = state.apply_event(SessionEvent(3, 0, "agent_status", "worker", {
            "agent_id": "agent_w", "name": "worker", "status": "closed",
        }))
        completed = state.apply_event(SessionEvent(4, 0, "agent_status", "worker", {
            "agent_id": "agent_w", "name": "worker", "status": "completed", "summary": "all tests pass",
        }))
        failed = state.apply_event(SessionEvent(5, 0, "agent_spawned", "verifier", {
            "agent_id": "agent_v", "name": "verifier", "task": "check", "status": "queued",
        }))
        failed = state.apply_event(SessionEvent(6, 0, "agent_status", "verifier", {
            "agent_id": "agent_v", "name": "verifier", "status": "failed", "error": "pytest failed",
        }))
        interrupted = state.apply_event(SessionEvent(7, 0, "agent_spawned", "scout", {
            "agent_id": "agent_s", "name": "scout", "task": "look", "status": "queued",
        }))
        interrupted = state.apply_event(SessionEvent(8, 0, "agent_status", "scout", {
            "agent_id": "agent_s", "name": "scout", "status": "interrupted", "error": "cancelled",
        }))

        self.assertIs(running, spawned)
        self.assertIsNone(closed)
        self.assertIs(completed, spawned)
        self.assertEqual(completed.status, "success")
        self.assertEqual(completed.body, "all tests pass")
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.body, "pytest failed")
        self.assertEqual(interrupted.status, "interrupted")
        self.assertEqual(interrupted.body, "cancelled")
        self.assertEqual(len(state.agent_blocks), 3)

    def test_project_allowlist_matches_persisted_prefixes_across_arguments_and_redirections(self):
        with tempfile.TemporaryDirectory() as tmp:
            allowlist = ApprovalAllowlist(tmp)
            for command in ("pytest -q", "git log --oneline -5 2>$null; git status --short 2>&1"):
                request = ApprovalRequest("bash", {"command": command}, "shell_risky", "confirm")
                prefix = _persistent_prefix_for_request(request)
                self.assertIsNotNone(prefix)
                allowlist.add_prefix_rule(prefix, command=command)
            self.assertTrue(allowlist.matches("pytest --all"))
            self.assertTrue(allowlist.matches("git log --oneline -3 2>$null; git status --short 2>&1"))
            self.assertFalse(allowlist.matches("ruff check ."))
            self.assertFalse(allowlist.matches("git push origin main 2>$null"))

    @unittest.skipUnless(os.name == "nt", "PowerShell commands are parsed as PowerShell only on Windows")
    def test_project_allowlist_supports_pipeline_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            allowlist = ApprovalAllowlist(tmp)
            command = (
                "Get-Item calc.html | Select-Object Name,Length; "
                "(Get-Content calc.html | Measure-Object -Line).Lines"
            )
            request = ApprovalRequest("bash", {"command": command}, "shell_risky", "confirm")
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

    def test_fallback_block_ids_are_unique_and_replay_stable(self):
        def build_ids() -> list[str]:
            state = TuiState(SessionStatusSnapshot("model", "provider", "workspace-write", "session", Path.cwd()))
            state.add_block(TranscriptBlock("notice", "same", "first", turn=2))
            state.add_block(TranscriptBlock("notice", "same", "second", turn=2))
            return [block.id for block in state.blocks]

        live_ids = build_ids()
        replay_ids = build_ids()

        self.assertEqual(live_ids, replay_ids)
        self.assertEqual(len(set(live_ids)), 2)

    def test_llm_usage_updates_context_tokens(self):
        from harness_code_agent.config import CONTEXT_WINDOW_TOKENS

        state = TuiState(SessionStatusSnapshot("model", "provider", "workspace-write", "session", Path.cwd()))
        result = state.apply_event(SessionEvent(1, 0, "llm_usage", "main", {"prompt_tokens": 16000}))

        self.assertIsNone(result)
        self.assertEqual(state.snapshot.context_tokens, 16000)
        self.assertEqual(state.snapshot.context_window_tokens, int(CONTEXT_WINDOW_TOKENS))

    def test_run_bash_failure_keeps_raw_output(self):
        state = TuiState(SessionStatusSnapshot("model", "provider", "workspace-write", "session", Path.cwd()))
        call = state.apply_event(SessionEvent(1, 0, "tool_call", "main", {
            "tool": "bash",
            "args": {"command": "pytest tests/test_thing.py"},
        }))
        result = state.apply_event(SessionEvent(2, 0, "tool_result", "main", {
            "tool": "bash",
            "status": "success",
            "return_code": 1,
            "output": "3 failed\nModuleNotFoundError: No module named 'xxx'",
        }))

        self.assertIs(result, call)
        self.assertEqual(result.status, "failed")
        self.assertIn("命令执行失败", result.title)
        self.assertIn("ModuleNotFoundError", result.body)
        self.assertIn("退出码 1", result.body)

    def test_model_protocol_errors_stay_out_of_the_transcript(self):
        state = TuiState(SessionStatusSnapshot("model", "provider", "workspace-write", "session", Path.cwd()))
        state.snapshot.status = "thinking"
        events = [
            ("tool_call", {"tool": "write_file", "args": {}, "failure_visibility": "protocol"}),
            ("tool_result", {"tool": "write_file", "status": "failed", "error": "invalid_json: duplicate key",
                             "metadata": {"failure_visibility": "protocol"}}),
            ("failure", {"tool": "write_file", "message": "invalid_json: duplicate key",
                         "metadata": {"failure_visibility": "protocol"}}),
        ]
        for sequence, (kind, payload) in enumerate(events, 1):
            self.assertIsNone(state.apply_event(SessionEvent(sequence, 0, kind, "main", payload)))
        self.assertEqual(state.snapshot.status, "thinking")
        self.assertEqual(state.active_tool_blocks, {})
        self.assertIsNone(state.apply_event(SessionEvent(4, 0, "agent_fallback", "main", {
            "reason": "retry_budget_exhausted", "limit_type": "schema_failures", "last_tool": "write_file",
        })))

    def test_approval_and_policy_rejections_collapse_to_a_short_line(self):
        state = TuiState(SessionStatusSnapshot("model", "provider", "workspace-write", "session", Path.cwd()))
        state.apply_event(SessionEvent(1, 0, "tool_call", "main", {
            "tool": "bash",
            "args": {"command": "Remove-Item x"},
        }))
        denied = state.apply_event(SessionEvent(2, 0, "tool_result", "main", {
            "tool": "bash",
            "status": "failed",
            "error": "[approval_denied] 用户已拒绝该操作。",
            "metadata": {"status_source": "approval"},
        }))
        self.assertEqual(denied.status, "failed")
        self.assertEqual(denied.body, "操作已拒绝")

        state.apply_event(SessionEvent(3, 0, "tool_call", "main", {
            "tool": "bash",
            "args": {"command": "Get-ChildItem . -Recurse"},
        }))
        # A guard block is an intermediate self-correction step: the user
        # never sees it, and the running block is cleaned up.
        blocked = state.apply_event(SessionEvent(4, 0, "tool_result", "main", {
            "tool": "bash",
            "status": "failed",
            "error": "[blocked] Recursive repository listing...",
            "metadata": {"status_source": "tool_policy"},
        }))
        self.assertIsNone(blocked)

    def test_tool_failures_render_once_and_denial_stop_stays_quiet(self):
        state = TuiState(SessionStatusSnapshot("model", "provider", "workspace-write", "session", Path.cwd()))
        state.apply_event(SessionEvent(1, 0, "tool_call", "main", {
            "tool": "bash",
            "args": {"command": "pytest"},
        }))
        state.apply_event(SessionEvent(2, 0, "tool_result", "main", {
            "tool": "bash",
            "status": "failed",
            "error": "[approval_denied] 用户已拒绝该操作。",
            "metadata": {"status_source": "approval"},
        }))
        # The tool_result already showed "操作已拒绝": a failure for the same
        # tool would duplicate it, and the approval_denied stop must not add
        # a "代理已停止" block on top of a decision the user made.
        failure = state.apply_event(SessionEvent(3, 0, "failure", "main", {
            "tool": "bash",
            "category": "approval_denied",
            "message": "用户已拒绝该操作。",
        }))
        self.assertIsNone(failure)
        fallback = state.apply_event(SessionEvent(4, 0, "agent_fallback", "main", {
            "reason": "approval_denied",
            "last_tool": "bash",
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
