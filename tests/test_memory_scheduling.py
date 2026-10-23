import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from harness_code_agent import config
from harness_code_agent.core.interactive import InteractiveSession
from harness_code_agent.memory import MemoryService, MemoryWriteCommand
from harness_code_agent.memory import background as bg
from harness_code_agent.sessions.journal import SessionJournal


@pytest.fixture
def memory_runtime(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("HARNESS_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setattr(config, "MEMORY_EXTRACTION_IDLE_SECONDS", 60)
    service = MemoryService(workspace)
    store = service.stores["project"]
    journal = SessionJournal(workspace / "journal.jsonl")
    yield workspace, service, store, journal
    deadline = time.monotonic() + 5
    while workspace in bg._running_workspaces and time.monotonic() < deadline:
        time.sleep(0.01)
    with bg._worker_lock:
        timer = bg._scheduled_workspaces.pop(workspace, None)
        if timer is not None:
            timer.cancel()
        bg._active_sessions.pop(workspace, None)


def test_turn_checkpoints_coalesce_and_postpone_extraction(memory_runtime):
    workspace, _, store, journal = memory_runtime
    with patch.object(bg, "start_memory_worker"), patch.object(bg.time, "time", return_value=1000):
        journal.append_message({"role": "user", "content": "first completed task"})
        bg.finish_memory_turn(workspace, "s1", journal.path)
        journal.append_message({"role": "assistant", "content": "another completed turn"})
        bg.finish_memory_turn(workspace, "s1", journal.path)
    with store.connect() as db:
        rows = db.execute("SELECT * FROM extraction_jobs ORDER BY id").fetchall()
    assert [row["status"] for row in rows] == ["superseded", "pending"]
    assert rows[-1]["boundary"] == journal.path.stat().st_size
    assert rows[-1]["available_at"] == 1060
    with patch.object(bg.time, "time", return_value=1059):
        assert bg._claim_next_job(store) is None
    with patch.object(bg.time, "time", return_value=1060):
        assert bg._claim_next_job(store)["boundary"] == rows[-1]["boundary"]


def test_active_turn_cannot_be_claimed_even_if_idle_deadline_passes(memory_runtime):
    workspace, _, store, journal = memory_runtime
    journal.append_message({"role": "user", "content": "completed"})
    store.enqueue_extraction("s1", journal.path.stat().st_size, journal.path)
    with patch.object(bg, "start_memory_worker"):
        bg.begin_memory_turn(workspace, "s1")
        with store.connect() as db:
            db.execute("UPDATE extraction_jobs SET available_at=0")
        assert bg._claim_next_job(store) is None
        bg.finish_memory_turn(workspace, "s1")
    assert "s1" not in bg._active_for(workspace)


def test_idle_deadline_wakes_worker_without_another_turn_or_close(memory_runtime):
    workspace, _, store, journal = memory_runtime
    journal.append_message({"role": "user", "content": "remember UTF-8"})
    with (
        patch.object(config, "MEMORY_EXTRACTION_IDLE_SECONDS", 0.05),
        patch.object(bg, "_extract_candidates", return_value=[]) as extract,
    ):
        bg.finish_memory_turn(workspace, "s1", journal.path)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with store.connect() as db:
                row = db.execute("SELECT status FROM extraction_jobs").fetchone()
            if row and row["status"] == "done":
                break
            time.sleep(0.01)
        assert row["status"] == "done"
        extract.assert_called_once()


def test_restart_rearms_persisted_idle_deadline(memory_runtime):
    workspace, _, store, journal = memory_runtime
    journal.append_message({"role": "user", "content": "completed"})
    store.enqueue_extraction("s1", journal.path.stat().st_size, journal.path, idle_seconds=60)
    timer = MagicMock()
    with patch.object(bg.threading, "Timer", return_value=timer) as timer_factory:
        bg._run_once(workspace)
    assert 0 < timer_factory.call_args.args[0] <= 60
    timer.start.assert_called_once()


def test_retry_is_scheduled_and_uses_only_successful_incremental_boundary(memory_runtime):
    workspace, service, store, journal = memory_runtime
    journal.append_message({"role": "user", "content": "first"})
    first = journal.path.stat().st_size
    store.enqueue_extraction("s1", first, journal.path)
    with patch.object(bg, "_extract_candidates", return_value=[]):
        bg._process_job(service, store, bg._claim_next_job(store))
    journal.append_message({"role": "assistant", "content": "second"})
    second = journal.path.stat().st_size
    store.enqueue_extraction("s1", second, journal.path)
    with (
        patch.object(bg, "_extract_candidates", side_effect=RuntimeError("temporary")),
        patch.object(bg.threading, "Timer") as timer,
    ):
        bg._run_once(workspace)
        assert 299 <= timer.call_args.args[0] <= 300
    with store.connect() as db:
        db.execute("UPDATE extraction_jobs SET available_at=0 WHERE status='pending'")
    with patch.object(bg, "_extract_candidates", return_value=[]) as extract:
        bg._process_job(service, store, bg._claim_next_job(store))
    assert extract.call_args.kwargs == {"start_boundary": first, "end_boundary": second}


def test_extractor_obeys_snapshot_boundary_and_excludes_reasoning_and_epoch_duplicates(memory_runtime):
    _, service, _, journal = memory_runtime
    journal.append_message({"role": "user", "content": "ALREADY_EXTRACTED"})
    start = journal.path.stat().st_size
    journal.append_message({"role": "assistant", "content": "NEW_VERIFIED_FACT", "reasoning_content": "PRIVATE_REASONING"})
    journal.append("context_epoch", {"base_messages": [{"role": "user", "content": "DUPLICATED_EPOCH_HISTORY"}]})
    end = journal.path.stat().st_size
    journal.append_message({"role": "user", "content": "LATER_ACTIVE_TURN"})
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))],
    )
    with patch.object(bg, "client_scope") as scope:
        scope.return_value.__enter__.return_value = client
        assert bg._extract_candidates(service, journal.path, "s1", start_boundary=start, end_boundary=end) == []
    prompt = json.loads(client.chat.completions.create.call_args.kwargs["messages"][-1]["content"])
    assert "NEW_VERIFIED_FACT" in prompt["journal"]
    assert all(text not in prompt["journal"] for text in (
        "ALREADY_EXTRACTED", "PRIVATE_REASONING", "DUPLICATED_EPOCH_HISTORY", "LATER_ACTIVE_TURN",
    ))


def test_cancel_during_extraction_prevents_writes_and_retry(memory_runtime):
    _, service, store, journal = memory_runtime
    journal.append_message({"role": "user", "content": "fact"})
    store.enqueue_extraction("s1", journal.path.stat().st_size, journal.path)
    row = bg._claim_next_job(store)

    def cancel(*args, **kwargs):
        store.cancel_extraction("s1")
        return [MemoryWriteCommand(topic="fact", body="must not be written")]

    with patch.object(bg, "_extract_candidates", side_effect=cancel), patch.object(service, "write") as write:
        bg._process_job(service, store, row)
    write.assert_not_called()
    with store.connect() as db:
        assert db.execute("SELECT status FROM extraction_jobs").fetchone()[0] == "cancelled"


def test_activity_after_claim_defers_extraction_before_model_call(memory_runtime):
    workspace, service, store, journal = memory_runtime
    journal.append_message({"role": "user", "content": "fact"})
    store.enqueue_extraction("s1", journal.path.stat().st_size, journal.path)
    row = bg._claim_next_job(store)
    with patch.object(bg, "start_memory_worker"), patch.object(bg, "_extract_candidates") as extract:
        bg.begin_memory_turn(workspace, "s1")
        bg._process_job(service, store, row)
        bg.finish_memory_turn(workspace, "s1")
    extract.assert_not_called()


def test_forget_during_extraction_does_not_recreate_the_memory(memory_runtime):
    _, service, store, journal = memory_runtime
    journal.append_message({"role": "user", "content": "fact"})
    doc = service.write(MemoryWriteCommand(topic="fact", body="forgotten", source_sessions=["s1"]))
    store.enqueue_extraction("s1", journal.path.stat().st_size, journal.path)
    row = bg._claim_next_job(store)

    def forget(*args, **kwargs):
        service.forget(doc.id, doc.version)
        return [MemoryWriteCommand(topic="fact", body="forgotten", source_sessions=["s1"])]

    with patch.object(bg, "_extract_candidates", side_effect=forget):
        bg._process_job(service, store, row)
    assert store.list_documents() == []
    with store.connect() as db:
        assert db.execute("SELECT status FROM extraction_jobs").fetchone()[0] == "suppressed"


@pytest.mark.parametrize("succeeds", [True, False])
def test_interactive_turn_checkpoints_only_completed_turns(memory_runtime, succeeds):
    workspace, _, _, journal = memory_runtime
    session = InteractiveSession.__new__(InteractiveSession)
    session.cwd = workspace
    session.session = SimpleNamespace(id="s1", journal_path=journal.path)
    session.memory_auto_extract_enabled = True
    with (
        patch.object(session, "_run_turn_impl", side_effect=None if succeeds else RuntimeError("interrupted")),
        patch("harness_code_agent.core.interactive.begin_memory_turn") as begin,
        patch("harness_code_agent.core.interactive.finish_memory_turn") as finish,
    ):
        if succeeds:
            session._run_turn("task")
        else:
            with pytest.raises(RuntimeError, match="interrupted"):
                session._run_turn("task")
    begin.assert_called_once_with(workspace, "s1")
    finish.assert_called_once_with(workspace, "s1", journal.path if succeeds else None)
