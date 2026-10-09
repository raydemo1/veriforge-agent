from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness_code_agent.workspace.change_journal import WorkspaceChangeJournal
from harness_code_agent.workspace.service import WorkspaceService


class WorkspaceChangeJournalTests(unittest.TestCase):
    def test_cursor_returns_only_later_changes_and_deduplicates_paths(self):
        journal = WorkspaceChangeJournal()
        journal.record("before.py", operation="write_file")
        cursor = journal.cursor()
        journal.record("after.py", operation="write_file")
        journal.record("after.py", operation="apply_patch")

        changes = journal.changes_since(cursor)

        self.assertEqual([change.sequence for change in changes], [2, 3])
        self.assertEqual([change.operation for change in changes], ["write_file", "apply_patch"])
        self.assertEqual(journal.paths_since(cursor), (Path("after.py"),))

    def test_workspace_records_operation_and_returns_an_immutable_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceService(root=tmp)
            workspace.write_text("note.txt", "one")
            workspace.apply_text_patch("note.txt", search="one", replace="two")

            snapshot = workspace.changed_files
            snapshot.append(Path("fake.txt"))
            changes = workspace.change_journal.changes_since()

            self.assertEqual(workspace.changed_files, [Path("note.txt")])
            self.assertEqual([change.operation for change in changes], ["write_file", "apply_patch"])
            self.assertIsNotNone(changes[1].snapshot_path)

    def test_batch_rollback_preserves_non_utf8_file_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceService(root=tmp)
            binary_path = Path(tmp) / "payload.bin"
            failing_path = (Path(tmp) / "later.txt").resolve()
            original_bytes = b"\xff\xfe\x00original"
            binary_path.write_bytes(original_bytes)
            original_write_text = Path.write_text

            def fail_second_write(path: Path, data: str, *args, **kwargs):
                if path == failing_path:
                    raise OSError("injected write failure")
                return original_write_text(path, data, *args, **kwargs)

            with (
                patch.object(Path, "write_text", fail_second_write),
                self.assertRaisesRegex(OSError, "injected write failure"),
            ):
                workspace.write_text_batch({
                    "payload.bin": "replacement",
                    "later.txt": "never committed",
                })

            self.assertEqual(binary_path.read_bytes(), original_bytes)
            self.assertFalse(failing_path.exists())


if __name__ == "__main__":
    unittest.main()
