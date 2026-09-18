import json
import tempfile
import unittest
from pathlib import Path

from eval.benchmarks.hca_terminal_runner import (
    export_session_artifacts,
    parse_args,
    write_session_manifest,
)


class TerminalRunnerArtifactTests(unittest.TestCase):
    def test_parse_args_accepts_task_name_for_profile_metadata(self):
        args = parse_args([
            "--workspace",
            "/app",
            "--task-name",
            "terminal-bench/overfull-hbox",
            "solve it",
        ])

        self.assertEqual(args.workspace, "/app")
        self.assertEqual(args.task_name, "terminal-bench/overfull-hbox")
        self.assertEqual(args.prompt, "solve it")

    def test_exports_raw_session_observations_and_todo_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            harness_root = root / ".harness"
            session_id = "session-123"
            session_root = harness_root / "sessions" / session_id
            todo_root = session_root / "todo"
            observations_root = harness_root / "observations" / session_id
            traces_root = harness_root / "traces"
            artifacts_root = root / "artifacts"
            todo_root.mkdir(parents=True)
            observations_root.mkdir(parents=True)
            traces_root.mkdir(parents=True)

            events = [
                {
                    "sequence": 1,
                    "type": "tool_call",
                    "agent": "main_agent",
                    "payload": {
                        "tool": "update_todo",
                        "args": {
                            "items": [
                                {"id": "inspect", "text": "inspect", "status": "in_progress"},
                                {"id": "verify", "text": "verify", "status": "pending"},
                            ]
                        },
                    },
                },
                {
                    "sequence": 2,
                    "type": "tool_result",
                    "agent": "main_agent",
                    "payload": {
                        "tool": "update_todo",
                        "status": "success",
                        "metadata": {
                            "todo_state": {
                                "revision": 1,
                                "items": [
                                    {"id": "inspect", "text": "inspect", "status": "in_progress"},
                                    {"id": "verify", "text": "verify", "status": "pending"},
                                ],
                            }
                        },
                    },
                },
                {
                    "sequence": 3,
                    "type": "tool_call",
                    "agent": "main_agent",
                    "payload": {
                        "tool": "run_bash",
                        "args": {"command": "pytest -q"},
                    },
                },
                {
                    "sequence": 4,
                    "type": "tool_result",
                    "agent": "main_agent",
                    "payload": {
                        "tool": "run_bash",
                        "status": "failed",
                        "return_code": 1,
                        "output": "preview",
                    },
                },
                {
                    "sequence": 5,
                    "type": "tool_call",
                    "agent": "main_agent",
                    "payload": {
                        "tool": "update_todo",
                        "args": {
                            "items": [
                                {"id": "inspect", "text": "inspect", "status": "completed"},
                                {"id": "verify", "text": "verify", "status": "completed"},
                            ]
                        },
                    },
                },
            ]
            (session_root / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            (session_root / "session.json").write_text(
                json.dumps({"id": session_id, "profile": "terminal"}),
                encoding="utf-8",
            )
            (todo_root / "state.json").write_text(
                json.dumps({"revision": 2}),
                encoding="utf-8",
            )
            (observations_root / "obs_0001.txt").write_text(
                "complete pytest output",
                encoding="utf-8",
            )
            (traces_root / "trace_main_agent.jsonl").write_text(
                '{"event":"finish","data":{"reason":"exception"}}\n',
                encoding="utf-8",
            )

            exported = export_session_artifacts(
                harness_root=harness_root,
                session_id=session_id,
                artifacts_root=artifacts_root,
                runner_error="Traceback: synthetic runner failure",
            )

            export_root = artifacts_root / "hca" / session_id
            self.assertEqual(exported, export_root)
            self.assertEqual(
                (export_root / "session" / "events.jsonl").read_text(encoding="utf-8"),
                (session_root / "events.jsonl").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (export_root / "observations" / "obs_0001.txt").read_text(encoding="utf-8"),
                "complete pytest output",
            )
            self.assertEqual(
                (export_root / "traces" / "trace_main_agent.jsonl").read_text(encoding="utf-8"),
                '{"event":"finish","data":{"reason":"exception"}}\n',
            )
            self.assertEqual(
                (export_root / "runner_error.txt").read_text(encoding="utf-8"),
                "Traceback: synthetic runner failure\n",
            )

            todo_history = [
                json.loads(line)
                for line in (export_root / "todo_history.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([item["sequence"] for item in todo_history], [1, 2, 5])
            self.assertEqual(todo_history[0]["payload"]["args"]["items"][0]["text"], "inspect")
            self.assertEqual(
                todo_history[1]["payload"]["metadata"]["todo_state"]["revision"],
                1,
            )
            self.assertEqual(
                todo_history[2]["payload"]["args"]["items"][1]["status"],
                "completed",
            )

            trajectory = [
                json.loads(line)
                for line in (export_root / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(trajectory), len(events))
            self.assertEqual(trajectory[2]["payload"]["args"]["command"], "pytest -q")
            self.assertEqual(trajectory[3]["payload"]["return_code"], 1)

            manifest = json.loads((export_root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["session_id"], session_id)
            self.assertEqual(manifest["event_count"], len(events))
            self.assertEqual(manifest["todo_event_count"], 3)
            self.assertTrue(manifest["observations_exported"])
            self.assertTrue(manifest["traces_exported"])
            self.assertTrue(manifest["runner_error_exported"])

    def test_writes_early_session_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            exported = write_session_manifest(
                session_id="session-early",
                workspace="/app",
                harness_root=root / ".harness",
                artifacts_root=root / "artifacts",
                status="started",
            )

            manifest = json.loads((exported / "early_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["session_id"], "session-early")
            self.assertEqual(manifest["workspace"], "/app")
            self.assertEqual(manifest["status"], "started")
            self.assertIn(".harness", manifest["harness_root"])

    def test_export_session_artifacts_is_repeatable_for_partial_sessions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            harness_root = root / ".harness"
            session_id = "session-repeat"
            session_root = harness_root / "sessions" / session_id
            session_root.mkdir(parents=True)
            (session_root / "events.jsonl").write_text("", encoding="utf-8")
            (session_root / "session.json").write_text("{}", encoding="utf-8")
            artifacts_root = root / "artifacts"

            first = export_session_artifacts(
                harness_root=harness_root,
                session_id=session_id,
                artifacts_root=artifacts_root,
                runner_error="first",
            )
            second = export_session_artifacts(
                harness_root=harness_root,
                session_id=session_id,
                artifacts_root=artifacts_root,
                runner_error="second",
            )

            self.assertEqual(first, second)
            self.assertEqual((second / "runner_error.txt").read_text(encoding="utf-8"), "second\n")
            manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["event_count"], 0)

    def test_export_session_artifacts_skips_concurrently_written_partial_jsonl_line(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            harness_root = root / ".harness"
            session_id = "session-partial-jsonl"
            session_root = harness_root / "sessions" / session_id
            session_root.mkdir(parents=True)
            complete_event = {"sequence": 1, "type": "tool_call", "payload": {"tool": "run_bash"}}
            (session_root / "events.jsonl").write_text(
                json.dumps(complete_event) + "\n" + '{"sequence": 2, "type":',
                encoding="utf-8",
            )
            (session_root / "session.json").write_text("{}", encoding="utf-8")

            exported = export_session_artifacts(
                harness_root=harness_root,
                session_id=session_id,
                artifacts_root=root / "artifacts",
            )

            trajectory = [
                json.loads(line)
                for line in (exported / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            manifest = json.loads((exported / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(trajectory, [complete_event])
            self.assertEqual(manifest["event_count"], 1)

    def test_export_session_artifacts_preserves_early_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            harness_root = root / ".harness"
            session_id = "session-early-preserve"
            session_root = harness_root / "sessions" / session_id
            session_root.mkdir(parents=True)
            (session_root / "events.jsonl").write_text("", encoding="utf-8")
            (session_root / "session.json").write_text("{}", encoding="utf-8")
            artifacts_root = root / "artifacts"
            early_dir = write_session_manifest(
                session_id=session_id,
                workspace="/app",
                harness_root=harness_root,
                artifacts_root=artifacts_root,
            )

            export_session_artifacts(
                harness_root=harness_root,
                session_id=session_id,
                artifacts_root=artifacts_root,
            )

            self.assertTrue((early_dir / "early_manifest.json").exists())
            self.assertTrue((early_dir / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
