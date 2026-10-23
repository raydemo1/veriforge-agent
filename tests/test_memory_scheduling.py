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
    service.extraction.enqueue("fork", journal.path, idle_seconds=0)
    task = service.extraction.claim()
    assert task is not None
    service.extraction.cancel("fork")
    assert service.extraction.process(task, invoke=lambda **kwargs: None)["state"] == "obsolete"
    service.extraction.resume("fork")
    assert service.extraction.claim() is None
