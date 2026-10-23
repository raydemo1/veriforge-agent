from eval.scripts.run_memory_extraction_eval import assess


def test_extraction_grader_rejects_correct_fact_in_wrong_scope():
    case = {"facts": [{"id": "command", "scope": "project", "patterns": ["uv sync --locked"]}]}
    result = assess(case, [{"id": "new", "scope": "user", "body": "uv sync --locked"}], [], [])
    assert result["facts"][0]["found"]
    assert not result["passed"]


def test_extraction_grader_requires_replacement_of_old_active_fact():
    case = {"facts": [], "requires_supersession": True}
    assert not assess(case, [], [{"id": "old"}], ["old"])["passed"]
    assert assess(case, [], [{"id": "new"}], ["old"])["passed"]


def test_extraction_grader_rejects_memory_from_negative_control():
    case = {"facts": [], "expect_no_new_memories": True}
    assert assess(case, [], [], [])["passed"]
    assert not assess(case, [{"id": "new"}], [], [])["passed"]
