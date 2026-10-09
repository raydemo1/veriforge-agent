from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harness_code_agent.agent.change_proposal import ChangeProposalStore
from harness_code_agent.sessions.events import SessionEvent
from harness_code_agent.tui.projection import WorkStateProjection
from harness_code_agent.tui.protocol import UI_PROTOCOL_VERSION, validate_ui_event


def event(event_type: str, payload: dict | None = None, agent: str | None = None) -> SessionEvent:
    return SessionEvent(
        sequence=1,
        timestamp=0.0,
        type=event_type,
        agent=agent,
        payload=payload or {},
    )


class WorkStateProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.projection = WorkStateProjection(self.cwd)
        (self.cwd / "global_plan" / "current").mkdir(parents=True)
        (self.cwd / "global_plan" / "current" / "plan.md").write_text(
            "# Plan\n\n1. First step\n2. Second step\n3. Third step\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_protocol_version_is_v5_and_accepts_work_events(self) -> None:
        self.assertEqual(UI_PROTOCOL_VERSION, 6)
        validate_ui_event({"type": "plan_updated", "plan": None})
        validate_ui_event({"type": "tasks_updated", "tasks": {"items": [], "completed": 0, "total": 0}})
        validate_ui_event({"type": "agent_run_updated", "agents": []})
        validate_ui_event({"type": "changes_updated", "changes": {"workspace": [], "proposals": []}})
        validate_ui_event({"type": "verification_updated", "checks": []})
        validate_ui_event({"type": "artifact_updated", "artifacts": []})
        with self.assertRaises(ValueError):
            validate_ui_event({"type": "plan_updated"})

    def test_skipped_verification_is_preserved(self) -> None:
        self.projection.apply_event(event("checks_recorded", {
            "checks": [{"name": "TypeScript", "status": "skipped", "detail": "not installed"}],
        }))
        self.assertEqual(self.projection.section("checks")[0]["status"], "skipped")

    def test_plan_lifecycle_parses_steps_and_tracks_progress(self) -> None:
        changed = self.projection.apply_event(event("plan_ready", {
            "plan_path": "global_plan/current/plan.md",
            "plan_revision": 2,
        }))
        self.assertEqual(changed, {"plan"})
        plan = self.projection.section("plan")
        self.assertEqual(plan["status"], "ready")
        self.assertEqual(plan["revision"], 2)
        self.assertEqual(plan["totalCount"], 3)
        self.assertEqual([step["text"] for step in plan["steps"]], [
            "First step", "Second step", "Third step",
        ])

        self.projection.apply_event(event("profile_switched", {
            "profile": "coding-agent",
            "reason": "execute approved plan",
        }))
        self.assertEqual(self.projection.section("plan")["status"], "executing")

        # Todo progress with matching length drives plan step statuses.
        self.projection.apply_event(event("tool_result", {
            "tool": "update_todo",
            "status": "success",
            "metadata": {"todo_state": {"items": [
                {"id": "a", "text": "First step", "status": "completed"},
                {"id": "b", "text": "Second step", "status": "in_progress"},
                {"id": "c", "text": "Third step", "status": "pending"},
            ]}},
        }))
        plan = self.projection.section("plan")
        self.assertEqual(plan["completedCount"], 1)
        self.assertEqual([step["status"] for step in plan["steps"]], [
            "completed", "in_progress", "pending",
        ])

        # A turn ending mid-plan must not fake completion: the plan is
        # incomplete and step statuses stay aligned to todo evidence.
        self.projection.apply_event(event("turn_finished", {}))
        plan = self.projection.section("plan")
        self.assertEqual(plan["status"], "incomplete")
        self.assertEqual([step["status"] for step in plan["steps"]], [
            "completed", "in_progress", "pending",
        ])

        # A follow-up turn resumes the plan.
        self.projection.apply_event(event("turn_started", {"turn": 2}))
        self.assertEqual(self.projection.section("plan")["status"], "executing")

        # Only full todo evidence completes the plan.
        self.projection.apply_event(event("tool_result", {
            "tool": "update_todo",
            "status": "success",
            "metadata": {"todo_state": {"items": [
                {"id": "a", "text": "First step", "status": "completed"},
                {"id": "b", "text": "Second step", "status": "completed"},
                {"id": "c", "text": "Third step", "status": "completed"},
            ]}},
        }))
        self.projection.apply_event(event("turn_finished", {}))
        plan = self.projection.section("plan")
        self.assertEqual(plan["status"], "completed")
        self.assertEqual(plan["completedCount"], 3)

    def test_plan_without_todo_evidence_stays_incomplete_at_turn_end(self) -> None:
        self.projection.apply_event(event("plan_ready", {
            "plan_path": "global_plan/current/plan.md",
            "plan_revision": 1,
            "plan_markdown": "# Plan\n\n1. Do the work\n2. Verify\n",
        }))
        self.projection.apply_event(event("profile_switched", {"reason": "execute approved plan"}))
        self.projection.apply_event(event("turn_finished", {}))
        plan = self.projection.section("plan")
        self.assertEqual(plan["status"], "incomplete")
        self.assertTrue(all(step["status"] == "pending" for step in plan["steps"]))

    def test_plan_markdown_in_payload_is_the_replay_truth(self) -> None:
        # A later session overwrites plan.md on disk; the journaled snapshot
        # from the older session must still reproduce that session's plan.
        (self.cwd / "global_plan" / "current" / "plan.md").write_text(
            "# Plan B (current disk artifact)\n\n1. Current step only\n",
            encoding="utf-8",
        )
        self.projection.apply_event(event("plan_ready", {
            "plan_path": "global_plan/current/plan.md",
            "plan_revision": 1,
            "plan_markdown": "# Plan A\n\n1. Historical step one\n2. Historical step two\n",
        }))
        plan = self.projection.section("plan")
        self.assertEqual([step["text"] for step in plan["steps"]], [
            "Historical step one", "Historical step two",
        ])

    def test_tasks_section_uses_todo_metadata(self) -> None:
        self.projection.apply_event(event("tool_result", {
            "tool": "update_todo",
            "status": "success",
            "metadata": {"todo_state": {"items": [
                {"id": "t1", "text": "one", "status": "completed"},
                {"id": "t2", "text": "two", "status": "pending"},
            ]}},
        }))
        tasks = self.projection.section("tasks")
        self.assertEqual(tasks["completed"], 1)
        self.assertEqual(tasks["total"], 2)
        self.assertEqual(tasks["items"][0]["id"], "t1")

    def test_agent_spawn_status_and_close(self) -> None:
        self.projection.apply_event(event("agent_spawned", {
            "agent_id": "a1", "name": "worker-1", "role": "worker",
            "task": "build the thing", "status": "queued",
        }))
        self.projection.apply_event(event("agent_spawned", {
            "agent_id": "a2", "name": "explorer-1", "role": "explorer",
            "task": "read code", "status": "queued",
        }))
        agents = {item["id"]: item for item in self.projection.section("agents")}
        self.assertEqual(agents["a1"]["isolation"], "isolated workspace")
        self.assertEqual(agents["a2"]["isolation"], "read-only")
        self.assertEqual(agents["a1"]["task"], "build the thing")

        self.projection.apply_event(event("agent_status", {
            "agent_id": "a1", "status": "completed", "duration_seconds": 4.25,
        }))
        agent = next(item for item in self.projection.section("agents") if item["id"] == "a1")
        self.assertEqual(agent["status"], "completed")
        self.assertEqual(agent["durationSeconds"], 4.2)

        self.projection.apply_event(event("agent_status", {"agent_id": "a2", "status": "closed"}))
        self.assertEqual([item["id"] for item in self.projection.section("agents")], ["a1"])

    def test_workspace_changes_aggregate_counts_and_ignore_subagents(self) -> None:
        self.projection.apply_event(event("file_change", {
            "path": "a.py", "operation": "modify", "diff": "--- a\n+++ b\n-old\n+new1\n+new2\n",
        }))
        # A later event for the same path replaces the row.
        self.projection.apply_event(event("file_change", {
            "path": "a.py", "operation": "modify", "additions": 5, "deletions": 1,
        }))
        self.projection.apply_event(event("file_change", {
            "path": "b.py", "operation": "create", "additions": 2, "deletions": 0,
        }))
        # Subagent file changes live in sandboxes and must not hit the workspace view.
        self.projection.apply_event(event("file_change", {
            "path": "sandbox.py", "operation": "modify", "additions": 9, "deletions": 9,
        }, agent="worker-1"))

        changes = self.projection.section("changes")
        self.assertEqual([item["path"] for item in changes["workspace"]], ["a.py", "b.py"])
        a_py = changes["workspace"][0]
        self.assertEqual((a_py["additions"], a_py["deletions"]), (5, 1))
        self.assertEqual((changes["additions"], changes["deletions"]), (7, 1))

    def test_checks_upsert_and_turn_reset_with_artifacts_persisting(self) -> None:
        self.projection.apply_event(event("tool_call", {
            "tool": "browser_test",
            "args": {"url": "http://localhost:5173"},
        }))
        browser = next(item for item in self.projection.section("checks") if item["id"] == "browser-verification")
        self.assertEqual(browser["status"], "running")

        self.projection.apply_event(event("tool_result", {
            "tool": "browser_test",
            "status": "success",
            "metadata": {
                "url": "http://localhost:5173",
                "artifacts": [{
                    "kind": "image",
                    "path": ".harness/artifacts/browser.png",
                    "title": "browser.png",
                    "detail": "http://localhost:5173",
                }],
            },
        }))
        checks = {item["id"]: item for item in self.projection.section("checks")}
        self.assertEqual(checks["browser-verification"]["status"], "passed")
        artifact = self.projection.section("artifacts")[0]
        self.assertEqual(artifact["kind"], "image")

        # Static verifier evidence upserts by check name.
        self.projection.apply_event(event("checks_recorded", {"checks": [
            {"name": "Python syntax", "status": "passed", "detail": "2 files parsed"},
            {"name": "Ruff lint", "status": "failed", "detail": "[F811] x.py:1"},
        ]}))
        self.assertEqual({item["id"] for item in self.projection.section("checks")}, {
            "browser-verification", "python-syntax", "ruff-lint",
        })

        # A new turn clears checks but keeps artifacts.
        self.projection.apply_event(event("turn_started", {"turn": 2}))
        self.assertEqual(self.projection.section("checks"), [])
        self.assertEqual(len(self.projection.section("artifacts")), 1)

    def test_proposal_merge_decorates_agent_and_filters_applied(self) -> None:
        self.projection.apply_event(event("agent_spawned", {
            "agent_id": "w1", "name": "worker-1", "role": "worker",
            "task": "edit", "status": "running",
        }))
        self.assertTrue(self.projection.merge_proposals([
            {
                "id": "p1", "agentId": "w1", "status": "ready",
                "files": [{"path": "x.py", "operation": "modify", "additions": 3, "deletions": 1}],
                "additions": 3, "deletions": 1, "invalidReasons": [], "conflict": None,
            },
            {
                "id": "p2", "agentId": "gone", "status": "applied",
                "files": [], "additions": 0, "deletions": 0, "invalidReasons": [], "conflict": None,
            },
        ]))
        changes = self.projection.section("changes")
        self.assertEqual([item["id"] for item in changes["proposals"]], ["p1"])
        self.assertEqual(changes["proposals"][0]["agentName"], "worker-1")
        # Identical snapshots are no-ops.
        self.assertFalse(self.projection.merge_proposals(self.projection._proposals))

    def test_event_replay_reconstructs_identical_state(self) -> None:
        events = [
            event("plan_ready", {"plan_path": "global_plan/current/plan.md", "plan_revision": 1}),
            event("profile_switched", {"reason": "execute approved plan"}),
            event("tool_result", {"tool": "update_todo", "status": "success", "metadata": {"todo_state": {"items": [
                {"id": "a", "text": "First step", "status": "completed"},
                {"id": "b", "text": "Second step", "status": "in_progress"},
                {"id": "c", "text": "Third step", "status": "pending"},
            ]}}}),
            event("agent_spawned", {"agent_id": "a1", "name": "worker-1", "role": "worker", "task": "t", "status": "queued"}),
            event("file_change", {"path": "a.py", "operation": "modify", "additions": 2, "deletions": 1}),
            event("checks_recorded", {"checks": [{"name": "Python syntax", "status": "passed", "detail": "ok"}]}),
        ]
        for item in events:
            self.projection.apply_event(item)

        replay = WorkStateProjection(self.cwd)
        for item in events:
            replay.apply_event(item)
        self.assertEqual(replay.changed_sections(), self.projection.changed_sections())


class ChangeProposalSnapshotTests(unittest.TestCase):
    def test_snapshot_reports_files_stats_and_conflict_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "a.py").write_text("before\n", encoding="utf-8")
            store = ChangeProposalStore()
            sandbox = store.create_sandbox("agent-1", root)
            (sandbox.workspace / "src" / "a.py").write_text("before\nafter\n", encoding="utf-8")
            (sandbox.workspace / "src" / "new.py").write_text("new\n", encoding="utf-8")
            proposal = store.finalize("agent-1", ["src"])

            snapshots = store.snapshot_proposals()
            self.assertEqual(len(snapshots), 1)
            snapshot = snapshots[0]
            self.assertEqual(snapshot["id"], proposal.id)
            self.assertEqual(snapshot["status"], "ready")
            self.assertIsNone(snapshot["conflict"])
            paths = {item["path"]: item for item in snapshot["files"]}
            self.assertEqual(paths["src/a.py"]["additions"], 1)
            self.assertEqual(paths["src/new.py"]["additions"], 1)
            self.assertEqual(snapshot["additions"], 2)
            store.close()


if __name__ == "__main__":
    unittest.main()
