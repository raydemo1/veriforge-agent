import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from harness_code_agent.memory import MemoryService, MemoryStore, MemoryWriteCommand
from harness_code_agent.memory.store import resolve_repo_key
from harness_code_agent.sessions.journal import SessionJournal


class MemorySystemTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.workspace = self.temp_dir / "workspace"
        self.workspace.mkdir()
        self.root = self.temp_dir / "memory"
        self.store = MemoryStore(self.root, workspace=self.workspace)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def write(self, **overrides):
        values = {
            "topic": "Parser recovery",
            "body": "Replay the failing fixture before changing the parser.",
            "source_paths": ["parser.py"],
        }
        values.update(overrides)
        return self.store.write(MemoryWriteCommand(**values))

    def test_markdown_is_authoritative_and_index_is_rebuildable(self):
        (self.workspace / "parser.py").write_text("old", encoding="utf-8")
        doc = self.write()

        entry = self.root / "entries" / f"{doc.id}.md"
        self.assertTrue(entry.exists())
        self.assertIn("Parser recovery", entry.read_text(encoding="utf-8"))
        self.assertIn(doc.id, (self.root / "MEMORY.md").read_text(encoding="utf-8"))

        self.store.db_path.unlink()
        self.store.rebuild_index()
        with self.store.connect() as db:
            row = db.execute("SELECT topic FROM documents WHERE id=?", (doc.id,)).fetchone()
        self.assertEqual(row["topic"], "Parser recovery")

    def test_repo_key_is_shared_by_worktrees_with_different_directory_names(self):
        common = self.temp_dir / "main-repo" / ".git"
        common.mkdir(parents=True)
        completed = SimpleNamespace(stdout=str(common) + "\n")
        with patch("harness_code_agent.memory.store.subprocess.run", return_value=completed):
            first = resolve_repo_key(self.temp_dir / "worktree-one")
            second = resolve_repo_key(self.temp_dir / "worktree-two")
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("main-repo-"))

    def test_manual_markdown_edit_advances_version_before_indexing(self):
        doc = self.write(source_paths=[])
        entry = self.root / "entries" / f"{doc.id}.md"
        entry.write_text(
            entry.read_text(encoding="utf-8").replace(
                "Replay the failing fixture", "Manually inspect the failing fixture",
            ),
            encoding="utf-8",
        )
        indexed = self.store.list_documents()[0]
        self.assertEqual(indexed.version, 2)
        self.assertIn("Manually inspect", indexed.body)
        with self.assertRaisesRegex(ValueError, "version changed"):
            self.write(memory_id=doc.id, expected_version=1, source_paths=[])

    def test_update_requires_current_version(self):
        doc = self.write(source_paths=[])
        updated = self.write(
            memory_id=doc.id,
            expected_version=doc.version,
            body="Use the minimized fixture first.",
            source_paths=[],
        )
        self.assertEqual(updated.version, 2)
        with self.assertRaisesRegex(ValueError, "version changed"):
            self.write(memory_id=doc.id, expected_version=1, source_paths=[])

    def test_same_path_can_hold_independent_memories(self):
        first = self.write(topic="Parser recovery")
        second = self.write(topic="Parser ownership", body="Parser changes require compiler review.")
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(len(self.store.list_documents()), 2)

    def test_changed_evidence_marks_only_recalled_memory_for_review(self):
        path = self.workspace / "parser.py"
        path.write_text("old", encoding="utf-8")
        doc = self.write()
        path.write_text("new", encoding="utf-8")

        refreshed = self.store.refresh_applicability(self.store.read(doc.id))
        self.assertEqual(refreshed.status, "review_required")
        self.assertEqual(refreshed.version, 2)

        validated = self.store.validate(doc.id, expected_version=2)
        self.assertEqual(validated.status, "active")
        self.assertEqual(validated.evidence_fingerprints["parser.py"], self.store.read(doc.id).evidence_fingerprints["parser.py"])

    def test_revision_supersedes_old_memory_without_deleting_history(self):
        old = self.write(source_paths=[])
        new = self.write(
            topic="Parser recovery v2", supersedes=old.id,
            expected_version=old.version, source_paths=[],
        )
        old_after = self.store.read(old.id)
        self.assertEqual(old_after.status, "superseded")
        self.assertEqual(old_after.superseded_by, new.id)
        self.assertNotIn(old.id, {item.id for item in self.store.list_documents()})
        self.assertIn(old.id, {item.id for item in self.store.list_documents(include_superseded=True)})

    def test_revision_rejects_stale_expected_version_before_writing(self):
        old = self.write(source_paths=[])
        with self.assertRaisesRegex(ValueError, "version changed"):
            self.write(
                topic="Stale correction", supersedes=old.id,
                expected_version=old.version + 1, source_paths=[],
            )
        self.assertEqual(len(self.store.list_documents(include_superseded=True)), 1)
        self.assertEqual(self.store.read(old.id).status, "active")

    def test_forget_removes_content_and_keeps_content_free_suppression(self):
        doc = self.write(source_sessions=["session-secret"], source_paths=[])
        self.store.forget(doc.id, expected_version=1)
        self.assertFalse((self.root / "entries" / f"{doc.id}.md").exists())
        with self.assertRaises(KeyError):
            self.store.read(doc.id)
        with self.store.connect() as db:
            keys = [row["key"] for row in db.execute("SELECT key FROM suppressions")]
        self.assertIn(f"id:{doc.id}", keys)
        self.assertNotIn("session-secret", json.dumps(keys))

    def test_forget_removes_entire_revision_lineage(self):
        old = self.write(source_sessions=["old-session"], source_paths=[])
        new = self.write(
            topic="Corrected parser recovery", supersedes=old.id,
            expected_version=old.version, source_sessions=["new-session"], source_paths=[],
        )
        self.store.forget(new.id, expected_version=new.version)
        self.assertEqual(self.store.list_documents(include_superseded=True), [])
        self.assertFalse((self.root / "entries" / f"{old.id}.md").exists())

    def test_chinese_and_path_recall(self):
        with patch.dict("os.environ", {"HARNESS_MEMORY_ROOT": str(self.root)}):
            service = MemoryService(self.workspace)
            service.write(
                MemoryWriteCommand(
                    topic="解析器调试流程",
                    body="出现词法错误时先回放最小失败样例。",
                    source_paths=["src/parser.py"],
                )
            )
            hits = service.search("词法报错怎么调试", paths=["src/parser.py"])
        self.assertTrue(hits)
        self.assertEqual(hits[0].document.topic, "解析器调试流程")

    def test_extraction_queue_is_deduplicated(self):
        journal = self.temp_dir / "journal.jsonl"
        journal.write_text("", encoding="utf-8")
        self.store.enqueue_extraction("s1", 12, journal)
        self.store.enqueue_extraction("s1", 12, journal)
        with self.store.connect() as db:
            count = db.execute("SELECT COUNT(*) FROM extraction_jobs").fetchone()[0]
        self.assertEqual(count, 1)

    def test_expired_extraction_lease_is_reclaimed(self):
        from harness_code_agent.memory.background import _claim_next_job

        journal = self.temp_dir / "journal.jsonl"
        journal.write_text("", encoding="utf-8")
        self.store.enqueue_extraction("s1", 12, journal)
        with self.store.connect() as db:
            db.execute(
                "UPDATE extraction_jobs SET status='running',attempts=1,lease_until=?,available_at=?",
                (time.time() - 1, time.time() - 2),
            )
        row = _claim_next_job(self.store)
        self.assertEqual(row["session_id"], "s1")
        with self.store.connect() as db:
            status = db.execute("SELECT status FROM extraction_jobs").fetchone()[0]
        self.assertEqual(status, "running")

    def test_user_scope_forget_suppresses_project_extraction_job(self):
        from harness_code_agent.memory.background import _claim_next_job, _process_job

        journal = self.temp_dir / "journal.jsonl"
        journal.write_text("{}\n", encoding="utf-8")
        with patch.dict("os.environ", {"HARNESS_MEMORY_ROOT": str(self.root)}):
            service = MemoryService(self.workspace)
            user_doc = service.write(MemoryWriteCommand(
                topic="Preference", body="Use compact output.", scope="user",
                source_sessions=["s-user"],
            ))
            service.forget(user_doc.id, user_doc.version, scope="user")
            project = service.stores["project"]
            project.enqueue_extraction("s-user", 1, journal)
            with project.connect() as db:
                db.execute("UPDATE extraction_jobs SET available_at=?", (time.time() - 1,))
            row = _claim_next_job(project)
            with patch("harness_code_agent.memory.background._extract_candidates") as extract:
                _process_job(service, project, row)
            with project.connect() as db:
                status = db.execute("SELECT status FROM extraction_jobs").fetchone()[0]
        extract.assert_not_called()
        self.assertEqual(status, "suppressed")


class MemoryToolTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.context = SimpleNamespace(
            workspace=SimpleNamespace(root=self.temp_dir),
            session_id="session-1",
        )
        self.memory_root = self.temp_dir / "memory"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_read_search_validate_and_forget(self):
        from harness_code_agent.runtime.builtins.memory_tools import (
            memory_forget,
            memory_read,
            memory_search,
            memory_validate,
            memory_write,
        )

        with patch.dict("os.environ", {"HARNESS_MEMORY_ROOT": str(self.memory_root)}):
            written = memory_write(
                "Build command", "Run python -m unittest.",
                source_paths=[], tool_context=self.context,
            )
            memory_id = written.metadata["memory_id"]
            read = memory_read(memory_id, tool_context=self.context)
            found = memory_search("unittest build command", tool_context=self.context)
            validated = memory_validate(memory_id, 1, tool_context=self.context)
            forgotten = memory_forget(memory_id, 2, tool_context=self.context)

        self.assertEqual(written.status, "success")
        self.assertIn("Run python", read.output)
        self.assertIn(memory_id, found.output)
        self.assertEqual(validated.status, "success")
        self.assertEqual(forgotten.status, "success")

    def test_session_memory_toggles_are_enforced_by_tools(self):
        from harness_code_agent.runtime.builtins.memory_tools import (
            memory_search,
            memory_write,
        )

        self.context.memory_use_enabled = False
        self.context.memory_generate_enabled = False
        with patch.dict("os.environ", {"HARNESS_MEMORY_ROOT": str(self.memory_root)}):
            searched = memory_search("anything", tool_context=self.context)
            written = memory_write("Topic", "Body", tool_context=self.context)
        self.assertEqual(searched.status, "failed")
        self.assertEqual(written.status, "failed")


class SessionJournalTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.journal = SessionJournal(self.temp_dir / "journal.jsonl")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_recovery_uses_latest_summary_and_kept_messages(self):
        self.journal.append_message({"role": "system", "content": "rules"})
        self.journal.append_message({"role": "user", "content": "old task"})
        kept = self.journal.append_message({"role": "user", "content": "current task"})
        self.journal.append_compaction(
            summary="## Task goal\nFinish current task",
            first_kept_sequence=kept.sequence,
            phase="manual",
        )

        messages = self.journal.recovery_messages("new rules")
        self.assertEqual(messages[0]["content"], "new rules")
        self.assertIn("Finish current task", messages[1]["content"])
        self.assertEqual(messages[2]["content"], "current task")
        self.assertNotIn("old task", json.dumps(messages))


class ContextManagerTests(unittest.TestCase):
    def test_failed_summary_does_not_replace_messages(self):
        from harness_code_agent.agent.context_manager import ContextManager

        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "old" * 100},
            {"role": "user", "content": "current"},
        ]

        def fail(_messages):
            raise RuntimeError("offline")

        result = ContextManager().compact(
            messages, fail, current_turn_start_index=2, state={}, force=True,
        )
        self.assertIsNone(result)
        self.assertEqual(messages[1]["content"], "old" * 100)

    def test_summary_has_structured_contract_and_keeps_current_turn(self):
        from harness_code_agent.agent.context_manager import (
            SUMMARY_MARKER,
            ContextManager,
        )

        calls = []
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "old task"},
            {"role": "assistant", "content": "old result"},
            {"role": "user", "content": "current task"},
        ]

        def summarize(prompt):
            calls.extend(prompt)
            return "## Task goal\ncurrent task\n## Next action\ncontinue"

        result = ContextManager().compact(
            messages, summarize, current_turn_start_index=3,
            state={"changed_files": ["a.py"]}, force=True,
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.messages[1]["content"].startswith(SUMMARY_MARKER))
        self.assertEqual(result.messages[-1]["content"], "current task")
        self.assertIn("## Evidence references", calls[1]["content"])


if __name__ == "__main__":
    unittest.main()
