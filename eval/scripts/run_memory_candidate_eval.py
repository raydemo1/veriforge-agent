"""Two-request live check of the isolated candidate-extraction stage."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.scripts.memory_eval_common import RequestRecorder
from harness_code_agent import config
from harness_code_agent.memory import extraction
from harness_code_agent.memory.database import MemoryDatabase
from harness_code_agent.memory.extraction import ExtractionQueue
from harness_code_agent.sessions.journal import SessionJournal

CASES = [
    {
        "id": "seven_conventions",
        "source": "下面七项是已确认的长期项目发布规范，每项应独立记忆：构建入口 BUILD_LIVE31；"
                  "测试门禁 GATE_LIVE32；负责人 OWNER_LIVE33；包仓库别名 REGISTRY_LIVE34；"
                  "回滚入口 ROLLBACK_LIVE35；观察窗口 WINDOW_LIVE36；审计清单 AUDIT_LIVE37。",
        "expected": ["BUILD_LIVE31", "GATE_LIVE32", "OWNER_LIVE33", "REGISTRY_LIVE34",
                     "ROLLBACK_LIVE35", "WINDOW_LIVE36", "AUDIT_LIVE37"],
    },
    {
        "id": "transient_progress",
        "source": "这次临时跑完了第12步，工具调用18次，还有一些日志未看；没有新的长期规范，"
                  "不要记住这些进度。",
        "expected": [],
    },
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    protocol = {"request_cap": 2, "automatic_retries": False, "profile": asdict(config.resolve_model_profile("fast")),
                "fixture_sha256": hashlib.sha256(json.dumps(CASES, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                "scope": "Candidate extraction only, not semantic reconciliation, retrieval or full-agent quality."}
    if not args.run:
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return
    root = PROJECT_ROOT / "eval/results" / ("memory-candidates-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    root.mkdir(parents=True)
    (root / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    recorder = RequestRecorder(root / "requests.jsonl", 2)
    results = []
    with patch.object(extraction, "client_scope", recorder.scope):
        for case in CASES:
            case_root = root / case["id"]
            case_root.mkdir()
            database = MemoryDatabase(case_root / "memory", "project:synthetic")
            queue = ExtractionQueue(database)
            journal = SessionJournal(case_root / "journal.jsonl")
            journal.append_user_input(case["source"], turn_id="turn-1", memory_generation=queue.generation)
            journal.append_message({"role": "user", "content": "RECALLED_UNTRUSTED_MEMORY_NOT_A_SOURCE"})
            queue.enqueue("synthetic-" + case["id"], journal.path)
            recorder.label = case["id"]
            outcome = queue.process(queue.claim(), max_calls=1)
            candidates = queue.candidates()
            covered = [token for token in case["expected"] if any(token in c["payload"]["body"] for c in candidates)]
            passed = (outcome["state"] == "done" and len(covered) == len(case["expected"])
                      and len(candidates) == len(case["expected"]) and database.records() == [])
            results.append({"case": case["id"], "outcome": outcome, "candidates": candidates,
                            "covered": covered, "published_records": len(database.records()), "passed": passed})
            print(f"{case['id']}: passed={passed}, candidates={len(candidates)}", flush=True)
    summary = {"protocol": protocol, "requests": len(recorder.records), "results": results}
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results: {root}", flush=True)


if __name__ == "__main__":
    main()
