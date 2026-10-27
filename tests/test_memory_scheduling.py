import time

from harness_code_agent.memory import MemoryService
from harness_code_agent.sessions.journal import SessionJournal


def test_resume_skips_copied_history_and_generation_cancel_fences_jobs(tmp_path):
    service = MemoryService(tmp_path)
    journal = SessionJournal(tmp_path / "journal.jsonl")
    journal.append_user_input("copied old source", turn_id="old", memory_generation=service.extraction.generation)
    boundary = journal.path.stat().st_size
    service.extraction.register("fork", journal.path)
    assert service.extraction.checkpoint("fork") == (boundary, 0)
    journal.append_user_input("new source", turn_id="new", memory_generation=service.extraction.generation)
    service.extraction.enqueue("fork", journal.path)
    task = service.extraction.claim()
    assert task is not None
    service.extraction.cancel("fork")
    assert service.extraction.process(task, invoke=lambda **kwargs: None)["state"] == "obsolete"
    service.extraction.resume("fork")
    with service.stores["project"].connect(write=True) as db:
        row = db.execute("SELECT status,available_at FROM extraction_tasks WHERE id=?", (task["id"],)).fetchone()
        assert row["status"] == "pending"
        assert row["available_at"] <= time.time()
    resumed = service.extraction.claim()
    assert resumed["id"] == task["id"] and resumed["lease_token"] != task["lease_token"]
