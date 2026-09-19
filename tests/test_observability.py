import json
import shutil
import tempfile
import unittest
from pathlib import Path

from harness_code_agent.sessions.store import SessionStore


def _event(sequence, event_type, payload, agent="main_agent"):
    return {
        "sequence": sequence,
        "timestamp": 0.0,
        "type": event_type,
        "agent": agent,
        "payload": payload,
    }


class ObservabilityAggregationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_session_observability_aggregates_tokens_tools_and_audit_events(self):
        from harness_code_agent.sessions.observability import (
            build_session_observability,
        )

        metadata = {
            "id": "session-a",
            "profile": "coding-agent",
            "model": "model-a",
            "created_at": "2026-06-01T00:00:00+00:00",
        }
        events = [
            _event(1, "llm_usage", {"prompt_tokens": 100, "cached_tokens": 80, "cache_miss_tokens": 20, "completion_tokens": 20, "total_tokens": 120}),
            _event(2, "llm_usage", {"prompt_tokens": None, "cached_tokens": 4, "completion_tokens": None, "total_tokens": None}),
            _event(3, "tool_call", {"tool": "read_file", "args": {"path": "README.md"}}),
            _event(4, "tool_result", {"tool": "read_file", "status": "success", "output": "ok"}),
            _event(5, "tool_call", {"tool": "run_bash", "args": {"command": "pytest"}}),
            _event(6, "tool_result", {"tool": "run_bash", "status": "failed", "error": "exit 1"}),
            _event(7, "tool_call", {"tool": "write_file", "args": {"path": "app.py"}}),
            _event(8, "tool_result", {"tool": "inspect_state", "status": "unknown"}),
            _event(9, "failure", {"category": "runtime_error", "message": "exit 1"}),
            _event(10, "failure", {"category": "validation_error", "message": "bad args"}),
            _event(11, "agent_fallback", {"reason": "token_budget_exceeded"}),
            _event(12, "context_compaction_committed", {"tokens_saved": 40000}),
            _event(13, "approval_requested", {"tool": "run_bash"}),
            _event(14, "approval_decided", {"approved": True}),
            _event(15, "approval_decided", {"approved": False}),
            _event(16, "file_change", {"path": "app.py"}),
            _event(17, "file_change", {"path": "app.py"}),
            _event(18, "context_anxiety_observed", {"score": 2, "reasons": ["due to context limit"]}),
            _event(19, "llm_response_finished", {"duration_ms": 100, "first_token_ms": 25}),
            _event(20, "llm_response_finished", {"duration_ms": 300, "first_token_ms": 75}),
            _event(21, "turn_finished", {"duration_seconds": 2.5}),
        ]

        snapshot = build_session_observability(metadata, events)

        self.assertEqual(snapshot.session_id, "session-a")
        self.assertEqual(snapshot.tokens.llm_calls, 2)
        self.assertEqual(snapshot.tokens.prompt_tokens, 100)
        self.assertEqual(snapshot.tokens.cached_tokens, 84)
        self.assertEqual(snapshot.tokens.cache_miss_tokens, 20)
        self.assertEqual(snapshot.tokens.completion_tokens, 20)
        self.assertEqual(snapshot.tokens.total_tokens, 120)
        self.assertEqual(snapshot.tokens.cache_hit_ratio, 0.84)
        self.assertEqual(snapshot.tools.tool_results, 3)
        self.assertEqual(snapshot.tools.successes, 1)
        self.assertEqual(snapshot.tools.failures, 1)
        self.assertEqual(snapshot.tools.unknown, 1)
        self.assertEqual(snapshot.tools.pending_calls, 1)
        self.assertAlmostEqual(snapshot.tools.success_rate, 1 / 3)
        self.assertEqual(snapshot.tools.by_tool["read_file"].successes, 1)
        self.assertEqual(snapshot.tools.by_tool["run_bash"].failures, 1)
        self.assertEqual(snapshot.audit.failure_categories["runtime_error"], 1)
        self.assertEqual(snapshot.audit.failure_categories["validation_error"], 1)
        self.assertEqual(snapshot.audit.fallbacks, 1)
        self.assertEqual(snapshot.audit.latest_fallback, "token_budget_exceeded")
        self.assertEqual(snapshot.audit.compactions_committed, 1)
        self.assertEqual(snapshot.audit.tokens_saved, 40000)
        self.assertEqual(snapshot.audit.context_anxiety_observed, 1)
        self.assertEqual(snapshot.audit.approvals_requested, 1)
        self.assertEqual(snapshot.audit.approvals_approved, 1)
        self.assertEqual(snapshot.audit.approvals_denied, 1)
        self.assertEqual(snapshot.audit.changed_files, ["app.py"])
        self.assertEqual(snapshot.performance.llm_response_latency_ms.count, 2)
        self.assertEqual(snapshot.performance.llm_response_latency_ms.p95, 300)
        self.assertEqual(snapshot.performance.llm_first_token_ms.p50, 25)
        self.assertEqual(snapshot.performance.turn_duration_ms.p99, 2500)

    def test_project_observability_aggregates_sessions_and_skips_bad_records(self):
        from harness_code_agent.sessions.observability import (
            build_project_observability,
        )

        store = SessionStore(self.root / ".harness")
        first = store.create(profile="coding-agent", cwd=self.root, model="model-a", permission_mode="workspace-write")
        second = store.create(profile="plan", cwd=self.root, model="model-a", permission_mode="workspace-write")
        store.event_bus(first).emit("llm_usage", payload={"prompt_tokens": 100, "cached_tokens": 50, "total_tokens": 150})
        store.event_bus(first).emit("tool_result", payload={"tool": "run_bash", "status": "failed"})
        store.event_bus(first).emit("failure", payload={"category": "runtime_error", "message": "failed"})
        store.event_bus(second).emit("llm_usage", payload={"prompt_tokens": 20, "cached_tokens": 20, "total_tokens": 30})
        bad = store.sessions_dir / "bad-session"
        bad.mkdir(parents=True)
        (bad / "session.json").write_text("{not json", encoding="utf-8")

        snapshot = build_project_observability(store)

        self.assertEqual(snapshot.session_count, 2)
        self.assertEqual(snapshot.tokens.prompt_tokens, 120)
        self.assertEqual(snapshot.tokens.cached_tokens, 70)
        self.assertEqual(snapshot.tokens.total_tokens, 180)
        self.assertEqual(snapshot.tools.tool_results, 1)
        self.assertEqual(snapshot.audit.failures, 1)
        self.assertEqual(snapshot.top_token_sessions[0].session_id, first.id)
        self.assertEqual(snapshot.top_failure_sessions[0].session_id, first.id)
        self.assertEqual(snapshot.low_cache_sessions[0].session_id, first.id)

    def test_observability_report_renders_and_exports_markdown_and_json(self):
        from harness_code_agent.sessions.observability import (
            export_observability_report,
            format_session_observability,
        )

        store = SessionStore(self.root / ".harness")
        session = store.create(profile="coding-agent", cwd=self.root, model="model-a", permission_mode="workspace-write")
        store.event_bus(session).emit("llm_usage", payload={"prompt_tokens": 100, "cached_tokens": 75, "total_tokens": 120})
        store.event_bus(session).emit("tool_result", payload={"tool": "read_file", "status": "success"})

        report = format_session_observability(store, session.id)
        exported = export_observability_report(store, mode="current", session_id=session.id)

        self.assertIn("Observability dashboard", report)
        self.assertIn("cache hit ratio: 75.0%", report)
        self.assertTrue(exported.markdown_path.exists())
        self.assertTrue(exported.json_path.exists())
        self.assertIn("Observability dashboard", exported.markdown_path.read_text(encoding="utf-8"))
        payload = json.loads(exported.json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["mode"], "current")
        self.assertEqual(payload["snapshot"]["session_id"], session.id)


class ObservationArtifactAdoptionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_streamed_artifact_is_adopted_without_second_spill(self):
        import hashlib

        from harness_code_agent.agent.observations import FactTracker, ObservationStore
        from harness_code_agent.runtime.tool_result import ToolResult

        art_dir = self.root / "artifacts"
        art_dir.mkdir(parents=True)
        artifact = art_dir / "shell_output_deadbeef.txt"
        full_output = "x" * 20000
        artifact.write_text(full_output, encoding="utf-8")
        sha = hashlib.sha256(full_output.encode("utf-8")).hexdigest()
        preview = "x" * 100 + "...omitted..." + "x" * 100

        result = ToolResult(
            tool="run_bash",
            status="success",
            output=preview,
            metadata={
                "status_source": "shell",
                "output_spilled": True,
                "artifact_path": str(artifact),
                "artifact_sha256": sha,
                "output_chars": 20000,
                "output_bytes": 20000,
            },
        )
        store = ObservationStore(art_dir)
        observation = store.create(
            tool="run_bash",
            args={"command": "big"},
            result=result,
            fact_tracker=FactTracker(),
        )

        # The observation points at the streamed artifact, not a new .txt.
        self.assertEqual(observation.raw_output_path, artifact)
        self.assertTrue(observation.artifact_adopted)
        self.assertEqual(observation.output_chars, 20000)
        self.assertEqual(observation.output_hash, sha[:16])
        self.assertEqual(list(art_dir.glob("obs_*.txt")), [])

        message = store.observed_message(observation, result)
        self.assertIn(f"raw_output: {artifact}", message)
        self.assertIn("omitted", message)

    def test_artifact_outside_store_is_not_adopted_or_unlinked(self):
        from harness_code_agent.agent.observations import FactTracker, ObservationStore
        from harness_code_agent.runtime.tool_result import ToolResult

        external = self.root / "outside.txt"
        external.write_text("do not delete me", encoding="utf-8")
        result = ToolResult(
            tool="run_bash",
            status="success",
            output="preview text",
            metadata={
                "status_source": "shell",
                "output_spilled": True,
                "artifact_path": str(external),
                "output_chars": 17,
            },
        )
        store = ObservationStore(self.root / "obs")
        observation = store.create(
            tool="run_bash",
            args={"command": "big"},
            result=result,
            fact_tracker=FactTracker(),
        )
        # Not adopted; the store spills its own copy and leaves the external
        # file untouched.
        self.assertFalse(observation.artifact_adopted)
        self.assertNotEqual(observation.raw_output_path, external)
        self.assertTrue(external.exists())
        self.assertEqual(external.read_text(encoding="utf-8"), "do not delete me")
        self.assertTrue(observation.raw_output_path.exists())

    def test_small_output_is_spilled_by_the_store_as_before(self):
        from harness_code_agent.agent.observations import FactTracker, ObservationStore
        from harness_code_agent.runtime.tool_result import ToolResult

        result = ToolResult(
            tool="run_bash",
            status="success",
            output="small",
            metadata={"status_source": "shell"},
        )
        store = ObservationStore(self.root / "obs")
        observation = store.create(
            tool="run_bash",
            args={"command": "echo small"},
            result=result,
            fact_tracker=FactTracker(),
        )
        self.assertTrue(observation.raw_output_path.exists())
        self.assertEqual(observation.raw_output_path.read_text(encoding="utf-8"), "small")


if __name__ == "__main__":
    unittest.main()
