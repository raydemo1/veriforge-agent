"""Bounded real-model extraction, consolidation and SiliconFlow retrieval evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.scripts.memory_eval_common import RequestRecorder
from harness_code_agent import config
from harness_code_agent.memory import extraction, reconciliation
from harness_code_agent.memory.database import EvidenceReference, MemoryDatabase
from harness_code_agent.memory.extraction import ExtractionQueue
from harness_code_agent.memory.reconciliation import (
    ConsolidationQueue,
    CurrentMemoryView,
)
from harness_code_agent.memory.vectors import SiliconFlowEmbeddings, VectorIndex
from harness_code_agent.sessions.journal import SessionJournal

CASES = [
    {"id": "confirmed_update", "turns": [
        "已确认的长期项目规范：数据库引擎使用 SQLite。",
        "数据库迁移已完成，旧的 SQLite 规范作废，今后统一使用 PostgreSQL。"],
     "query": "项目数据库用什么？", "current": ["PostgreSQL"], "count": 1},
    {"id": "paraphrase_duplicate", "turns": [
        "请长期记住我的个人偏好：所有解释都使用简体中文。",
        "再次确认同一个个人偏好：跟我交流时用简体中文，不用英文。"],
     "query": "What wording is most comfortable for this person?", "current": ["中文"], "count": 1,
     "user": True, "distractors": 40},
    {"id": "scoped_coexistence", "turns": [
        "已确认的项目约定：默认数据库引擎使用 SQLite。",
        "新增已确认的局部约定：仅 tools/ 子目录使用 PostgreSQL，项目其他部分仍然使用 SQLite。"],
     "query": "数据库的目录约定是什么？", "current": ["SQLite", "PostgreSQL"], "count": 2},
    {"id": "partial_update", "turns": [
        "以下是两项独立的长期发布规范：构建命令 BUILD_M31；测试命令 TEST_M32。",
        "已经确认：仅将构建命令从 BUILD_M31 改成 BUILD_M33，测试命令不变。"],
     "query": "发布时如何构建和测试？", "current": ["BUILD_M33", "TEST_M32"], "count": 2,
     "forbidden": ["BUILD_M31"]},
    {"id": "successive_updates", "turns": [
        "已确认的长期配置：默认服务端口为 PORT_M41。",
        "正式更正：默认服务端口已从 PORT_M41 切换到 PORT_M42。",
        "再次正式更正：PORT_M42 已停用，目前默认服务端口统一为 PORT_M43。"],
     "query": "默认服务端口是什么？", "current": ["PORT_M43"], "count": 1,
     "forbidden": ["PORT_M41", "PORT_M42"]},
    {"id": "invalidation_without_replacement", "turns": [
        "已确认的长期部署约定：使用发布入口 DEPLOY_M51。",
        "重要更正：DEPLOY_M51 已经停用，禁止再用。目前新发布入口尚未确定。"],
     "query": "发布入口现在可用吗？", "current": ["停用"], "count": 1},
]


class EmbeddingRecorder:
    def __init__(self, path, cap=48):
        self.path, self.cap = path, cap
        self.client = SiliconFlowEmbeddings()
        self.contract = self.client.contract
        self.records = []

    def embed(self, texts):
        if len(self.records) >= self.cap:
            raise RuntimeError("Embedding evaluation cap reached")
        record = {"index": len(self.records) + 1, "count": len(texts),
                  "input_hash": hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()}
        self.records.append(record)
        started = time.monotonic()
        try:
            vectors, metadata = self.client.embed(texts)
            record.update(metadata)
            return vectors, metadata
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            raise
        finally:
            record["elapsed_ms"] = round((time.monotonic() - started) * 1000)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_case(case, root, model_recorder, embeddings):
    folder = root / case["id"]
    folder.mkdir()
    project = MemoryDatabase(folder / "memory", "project:synthetic")
    user = project.scoped("user")
    extraction_queue = ExtractionQueue(project)
    journal = SessionJournal(folder / "journal.jsonl")
    factory = lambda db: VectorIndex(db, embeddings=embeddings)
    worker = ConsolidationQueue(project, vector_factory=factory)
    stages = []
    for turn, text in enumerate(case["turns"]):
        journal.append_user_input(text, turn_id=f"turn-{turn}", memory_generation=extraction_queue.generation)
        extraction_queue.enqueue(case["id"], journal.path, idle_seconds=0)
        model_recorder.label = f"extract:{case['id']}:{turn}"
        result = extraction_queue.process(extraction_queue.claim(), max_calls=1)
        stages.append({"extract": result})
        if result["state"] != "done":
            return {"passed": False, "stages": stages, "failure": "extraction"}
        task = worker.claim()
        if task:
            model_recorder.label = f"consolidate:{case['id']}:{turn}"
            result = worker.process(task)
            stages[-1]["consolidate"] = result
            if result["state"] != "done":
                return {"passed": False, "stages": stages, "failure": "consolidation"}
        for store in (project, user):
            indexed = factory(store).sync(max_requests=4)
            stages[-1].setdefault("index", []).append(indexed)
            if indexed.get("error_type") or factory(store).pending():
                return {"passed": False, "stages": stages, "failure": "index"}
        if turn == 0 and case.get("distractors"):
            ref = EvidenceReference("manual-fixture", 1, "manual", "f" * 64)
            with user.connect(write=True) as sql:
                for number in range(case["distractors"]):
                    user._write(sql, topic=f"unrelated preference {number}", fact_key=f"fixture.{number}",
                                kind="preference", body=f"Favorite archive label is LABEL_{number}.", evidence=[ref])
            factory(user).sync(max_requests=4)
    return {"stages": stages, **assess_current(case, project, factory)}


def assess_current(case, project, factory):
    store = project.scoped("user") if case.get("user") else project
    view = CurrentMemoryView(store, vectors=factory(store))
    hits = view.search(case["query"], limit=10)
    active = store.records()
    relevant = [r for r in active if r.topic != "" and not r.fact_key.startswith("fixture.")]
    bodies = "\n".join(r.body for r in relevant)
    returned = "\n".join(i.get("body", "") for i in hits["items"])
    passed = (len(relevant) == case["count"]
              and all(token in bodies and token in returned for token in case["current"])
              and not any(token in bodies or token in returned for token in case.get("forbidden", []))
              and hits["semantic_status"] == "ready" and hits["pending_vectors"] == 0)
    return {"passed": passed, "records": [r.__dict__ for r in relevant], "hits": hits,
            "topic_count": len({r.topic_id for r in relevant})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=[c["id"] for c in CASES])
    args = parser.parse_args()
    selected = [c for c in CASES if not args.cases or c["id"] in args.cases]
    protocol = {"model_request_cap": 28, "embedding_request_cap": 48, "retries": 0,
                "cases": [c["id"] for c in selected], "profile": config.resolve_model_profile("fast").__dict__,
                "fixture_hash": hashlib.sha256(json.dumps(CASES, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                "scope": "Isolated memory pipeline; no production memory changes or full-agent claims."}
    if not args.run:
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return
    if not os.environ.get("SILICONFLOW_API_KEY", "").strip():
        raise SystemExit("Configure SILICONFLOW_API_KEY in local .env before running.")
    root = ROOT / "eval/results" / ("memory-consolidation-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    root.mkdir(parents=True)
    (root / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    recorder = RequestRecorder(root / "model-requests.jsonl", 28)
    embeddings = EmbeddingRecorder(root / "embedding-requests.jsonl")
    results = []
    with (patch.object(extraction, "client_scope", recorder.scope),
          patch.object(reconciliation, "client_scope", recorder.scope)):
        for case in selected:
            try:
                result = run_case(case, root, recorder, embeddings)
            except Exception as exc:  # noqa: BLE001 - record failures without automatically expanding the experiment
                result = {"passed": False, "error_type": type(exc).__name__}
            results.append({"case": case["id"], **result})
            (root / "summary.json").write_text(json.dumps({"protocol": protocol, "results": results,
                "model_requests": len(recorder.records), "embedding_requests": len(embeddings.records)},
                ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{case['id']}: passed={result['passed']}", flush=True)
    print(f"Results: {root}")


if __name__ == "__main__":
    main()
