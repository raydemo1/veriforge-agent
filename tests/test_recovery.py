from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from harness_code_agent.runtime.questions import QuestionResult
from harness_code_agent.runtime.recovery import (
    RecoveryStore,
)


class RecoveryStoreTests(unittest.TestCase):
    def test_capture_preserves_file_bytes_and_excludes_runtime_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.bin").write_bytes(b"\xff\x00\r\n")
            (root / "empty").mkdir()
            (root / ".git").mkdir()
            (root / ".git/state").write_bytes(b"private")
            store = RecoveryStore(root)
            snapshot = store.capture()
            manifest = store.manifest(snapshot)
            self.assertEqual(store.blob(manifest["a.bin"]["hash"]), b"\xff\x00\r\n")
            self.assertEqual(manifest["empty"]["kind"], "directory")
            self.assertNotIn(".git/state", manifest)
            self.assertEqual(snapshot, store.capture())

    def test_corrupt_blob_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RecoveryStore(tmp)
            digest = store._put_blob(b"original")
            (store.root / "blobs" / digest).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "校验失败"):
                store.blob(digest)


from harness_code_agent.runtime.recovery import atomic_bytes
from harness_code_agent.workspace.access import (
    WorkspaceAccessCoordinator,
    WorkspaceConflict,
)


@pytest.mark.session
def test_repeated_rewind_and_resume_keep_only_effective_history(runtime_factory):
    create, _ = runtime_factory
    session = create()
    session.submit("第一条测试请求")
    first = max(session.recovery_points().values(), key=lambda p: p["turn"])
    session.submit("第二条测试请求")
    session.tool_context.workspace.write_text("new", "temporary")
    assert session.rewind_to_point(first["id"])
    session.submit("替换后的测试请求")
    assert "第二条测试请求" not in str(session.conversation.messages)
    assert session.rewind_to_point(first["id"])
    assert "替换后的测试请求" not in str(session.conversation.messages)
    identity = session.session.id
    session.close()
    restored = create(session_id=identity)
    assert "第一条测试请求" in str(restored.conversation.messages)
    assert "第二条测试请求" not in str(restored.conversation.messages)
    assert "替换后的测试请求" not in str(restored.conversation.messages)




@pytest.mark.session
def test_rewind_cancel_and_external_edit_preserve_state(runtime_factory, tmp_path):
    session = runtime_factory[0]()
    session.submit("第一条测试请求")
    point = max(session.recovery_points().values(), key=lambda p: p["turn"])
    session.tool_context.workspace.write_text("a", "owned")
    old = session.conversation
    session.question_provider = SimpleNamespace(ask=lambda _: QuestionResult(value="cancel"))
    assert not session.rewind_to_point(point["id"])
    assert session.conversation is old
    assert (tmp_path / "a").read_text() == "owned"
    (tmp_path / "a").write_text("external", encoding="utf-8")
    session.question_provider = SimpleNamespace(ask=lambda _: QuestionResult(value="rewind"))
    with pytest.raises(WorkspaceConflict):
        session.rewind_to_point(point["id"])
    assert (tmp_path / "a").read_text() == "external"
    assert session.conversation is old




@pytest.mark.session
def test_transaction_failure_rolls_back_all_files_and_journal(tmp_path):
    access = WorkspaceAccessCoordinator(tmp_path)
    (tmp_path / "a").write_bytes(b"old-a")
    (tmp_path / "b").write_bytes(b"old-b")
    journal = tmp_path / ".harness/sessions/A/journal.jsonl"
    atomic_bytes(journal, b"old-journal")
    real = atomic_bytes
    def write(path, data):
        if path == tmp_path / "b" and data == b"new-b":
            raise OSError("disk unavailable")
        real(path, data)
    with (
        patch("harness_code_agent.runtime.recovery.atomic_bytes", side_effect=write),
        pytest.raises(OSError),
        access.transaction({"a": b"new-a", "b": b"new-b"}, participants=[journal]),
    ):
        journal.write_bytes(b"new-journal")
    assert (tmp_path / "a").read_bytes() == b"old-a"
    assert (tmp_path / "b").read_bytes() == b"old-b"
    assert journal.read_bytes() == b"old-journal"


@pytest.mark.session
def test_process_crash_restores_files_and_session_participants(tmp_path):
    import subprocess
    import sys
    journal = tmp_path / ".harness/sessions/A/journal.jsonl"
    atomic_bytes(journal, b"old-journal")
    (tmp_path / "a").write_bytes(b"before")
    script = """
import os, sys
from pathlib import Path
from harness_code_agent.workspace.access import WorkspaceAccessCoordinator
root=Path(sys.argv[1]); a=WorkspaceAccessCoordinator(root)
with a.acquire([a.write_claim()]):
 with a.transaction({'a':b'after'}, participants=[root/'.harness/sessions/A/journal.jsonl']):
  (root/'.harness/sessions/A/journal.jsonl').write_bytes(b'new-journal')
  os._exit(17)
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], timeout=10, check=False)
    assert result.returncode == 17
    WorkspaceAccessCoordinator(tmp_path)
    assert (tmp_path / "a").read_bytes() == b"before"
    assert journal.read_bytes() == b"old-journal"


@pytest.mark.session
def test_untracked_operation_blocks_only_rewinds_across_its_boundary(tmp_path):
    access = WorkspaceAccessCoordinator(tmp_path)
    before = access.cursor()
    access.limit_recovery("A", "turn-A", "unknown shell")
    after = access.cursor()
    access = WorkspaceAccessCoordinator(tmp_path)
    with pytest.raises(WorkspaceConflict):
        access.rewind("A", before)
    access.check_rewind("A", after)
    access.check_rewind("B", before)


@pytest.mark.session
def test_rewind_keeps_session_background_shell_alive(runtime_factory, tmp_path):
    import os
    session = runtime_factory[0]()
    jobs = session.conversation.runtime_state.shell_job_manager
    job = jobs.start("Start-Sleep -Seconds 30" if os.name == "nt" else "sleep 30")
    process = job.process
    try:
        session.tool_context.workspace.access.limit_recovery(session.session.id, "background", "long shell")
        session.submit("稳定边界")
        point = max(session.recovery_points().values(), key=lambda value: value["turn"])
        session.tool_context.workspace.write_text("owned.txt", "later")
        assert session.rewind_to_point(point["id"])
        assert not (tmp_path / "owned.txt").exists()
        assert session.conversation.runtime_state.shell_job_manager is jobs
        assert process.poll() is None
        assert jobs.get(job.job_id).status == "running"
    finally:
        jobs.close()
