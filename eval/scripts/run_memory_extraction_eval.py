"""Bounded live extraction and controlled recall evaluation in isolated stores."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from openai import APIError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness_code_agent import config
from harness_code_agent.agent.providers import ProviderAdapter, client_scope
from harness_code_agent.agent.utils import _usage_to_dict
from harness_code_agent.memory import MemoryService, MemoryWriteCommand, background
from harness_code_agent.sessions.journal import SessionJournal


def fact_present(text: str, fact: dict) -> bool:
    return all(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in fact["patterns"])


def document_text(document: dict) -> str:
    return "\n".join(str(document.get(key) or "") for key in ("topic", "body", "applicability"))


def assess(case: dict, candidates: list[dict], active: list[dict], seed_ids: list[str]) -> dict:
    facts = []
    for fact in case["facts"]:
        matches = [doc for doc in candidates if fact_present(document_text(doc), fact)]
        facts.append({
            "id": fact["id"], "found": bool(matches),
            "correct_scope": any(doc["scope"] == fact["scope"] for doc in matches),
        })
    no_extra = not case.get("expect_no_new_memories") or not candidates
    supersession = not case.get("requires_supersession") or not set(seed_ids) & {doc["id"] for doc in active}
    return {
        "facts": facts, "new_memory_count": len(candidates),
        "negative_or_dedup_pass": no_extra, "supersession_pass": supersession,
        "passed": all(fact["correct_scope"] for fact in facts) and no_extra and supersession,
    }


class RequestRecorder:
    def __init__(self, path: Path, limit: int):
        self.path = path
        self.limit = limit
        self.records: list[dict] = []
        self.label = ""

    @contextmanager
    def scope(self):
        with client_scope() as client:
            original = client.chat.completions.create

            def create(**kwargs):
                if len(self.records) >= self.limit:
                    raise RuntimeError("Evaluation request cap reached")
                profile = config.resolve_model_profile("fast")
                payload = json.dumps(kwargs["messages"], ensure_ascii=False, sort_keys=True)
                record = {
                    "index": len(self.records) + 1, "label": self.label,
                    "provider": profile.provider, "model": kwargs["model"],
                    "input_chars": len(payload), "input_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                }
                self.records.append(record)
                started = time.monotonic()
                try:
                    result = original(**kwargs)
                    record["usage"] = _usage_to_dict(result.usage)
                    record["response"] = result.choices[0].message.content or ""
                    record["finish_reason"] = result.choices[0].finish_reason
                    return result
                except Exception as exc:
                    record["error_type"] = type(exc).__name__
                    raise
                finally:
                    record["elapsed_ms"] = round((time.monotonic() - started) * 1000)
                    with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            with patch.object(client.chat.completions, "create", side_effect=create):
                yield client


def all_documents(service: MemoryService, *, include_superseded=False) -> list[dict]:
    return [asdict(doc) for store in service.stores.values()
            for doc in store.list_documents(include_superseded=include_superseded)]


def run_extraction(case: dict, repeat: int, root: Path, recorder: RequestRecorder) -> tuple[dict, dict | None]:
    case_root = root / f"{case['id']}-{repeat}"
    workspace = case_root / "workspace"
    workspace.mkdir(parents=True)
    with patch.dict(os.environ, {
        "HARNESS_MEMORY_ROOT": str(case_root / "stores" / "project"),
        "HARNESS_MEMORY_APPLICABILITY_CHECK": "1", "HARNESS_MEMORY_DISABLED": "0",
    }):
        service = MemoryService(workspace)
        for name, content in case.get("files", {}).items():
            target = workspace / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        seeds = [service.write(MemoryWriteCommand(**value)) for value in case.get("seed", [])]
        for name, content in case.get("changed_files", {}).items():
            (workspace / name).write_text(content, encoding="utf-8")
        journal = SessionJournal(case_root / "journal.jsonl")
        for message in case["messages"]:
            journal.append_message(message)
        if case.get("filler_chars"):
            journal.append_message({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "logs", "function": {"name": "read_log", "arguments": "{}"}}],
            })
            journal.append_message({
                "role": "tool", "tool_call_id": "logs",
                "content": ("temporary progress line, no durable facts.\n" * 2000)[:case["filler_chars"]],
            })
            journal.append_message({"role": "assistant", "content": "临时日志读取结束，没有新的长期结论。"})
        result = {"case": case["id"], "repeat": repeat, "stress_case": bool(case.get("stress_case"))}
        recorder.label = f"extract:{case['id']}:{repeat}"
        try:
            with patch.object(background, "client_scope", recorder.scope):
                commands = background._extract_candidates(
                    service, journal.path, f"eval-{case['id']}-{repeat}",
                    start_boundary=0, end_boundary=journal.path.stat().st_size,
                )
            result["candidates"] = [asdict(command) for command in commands]
            for command in commands:
                service.write(command)
            active = all_documents(service)
            new = [doc for doc in active if doc["id"] not in {seed.id for seed in seeds}]
            result["stored"] = all_documents(service, include_superseded=True)
            result["assessment"] = assess(case, new, active, [seed.id for seed in seeds])
            if case.get("query") and repeat == 1:
                hits = service.search(case["query"], scope="both")
                block = MemoryService.format_hits(hits)
                recalled = [asdict(hit.document) for hit in hits]
                result["recall"] = {
                    "hit_ids": [doc["id"] for doc in recalled],
                    "fact_hits": [fact["id"] for fact in case["facts"]
                                  if any(fact_present(document_text(doc), fact) for doc in recalled)],
                    "stale_hit_ids": [doc["id"] for doc in recalled if doc["status"] != "active"],
                }
                return result, {"case": case, "memory": block}
        except (APIError, OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            result["error_type"] = type(exc).__name__
            result["assessment"] = {
                "passed": False,
                "facts": [{"id": fact["id"], "found": False, "correct_scope": False} for fact in case["facts"]],
            }
        return result, None


def run_reuse(job: dict, recorder: RequestRecorder) -> list[dict]:
    results = []
    case = job["case"]
    profile = config.resolve_model_profile("fast")
    for variant in ("without_memory", "with_memory"):
        recorder.label = f"reuse:{case['id']}:{variant}"
        content = case["query"]
        if variant == "with_memory":
            content = job["memory"] + "\n\nUser turn:\n" + content
        kwargs = ProviderAdapter(profile.provider).chat_kwargs(
            profile=profile,
            messages=[
                {"role": "system", "content": "只依据本次提供的资料回答；资料不足时回答 UNKNOWN，不猜测。保持简洁，保留命令和标识符的原样拼写。"},
                {"role": "user", "content": content},
            ], max_tokens=800,
        )
        result = {"case": case["id"], "variant": variant}
        try:
            with recorder.scope() as client:
                answer = client.chat.completions.create(**kwargs).choices[0].message.content or ""
            result.update({"answer": answer, "fact_hits": [fact["id"] for fact in case["facts"] if fact_present(answer, fact)]})
        except (APIError, OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            result["error_type"] = type(exc).__name__
        results.append(result)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Make live model requests; default only prints protocol")
    args = parser.parse_args(argv)
    task_path = PROJECT_ROOT / "eval" / "tasks" / "memory_extraction.json"
    cases = json.loads(task_path.read_text(encoding="utf-8"))["cases"]
    protocol = {
        "suite": "memory_extraction", "cases": len(cases), "repeats": 2,
        "request_cap": 24, "extraction_implementation": "memory.background._extract_candidates",
        "reuse_protocol": "four paired single-response probes; no tools or repository access",
        "grading": "frozen regex fact coverage and scope checks; inspect raw outputs for semantic grounding",
        "profile": asdict(config.resolve_model_profile("fast")),
        "fixture_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
    }
    if not args.run:
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return
    if not config.API_KEY:
        raise RuntimeError("No configured model credential")
    root = PROJECT_ROOT / "eval" / "results" / ("memory-extraction-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    root.mkdir(parents=True)
    (root / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    recorder = RequestRecorder(root / "requests.jsonl", 24)
    extractions, reuse, jobs = [], [], []
    with patch.object(config, "LLM_REQUEST_TIMEOUT_SECONDS", 60):
        for case in cases:
            for repeat in (1, 2):
                result, job = run_extraction(case, repeat, root, recorder)
                extractions.append(result)
                if job:
                    jobs.append(job)
                with (root / "extractions.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(f"extract {case['id']} repeat={repeat} passed={result['assessment']['passed']}", flush=True)
        for job in jobs:
            reuse.extend(run_reuse(job, recorder))
            (root / "reuse.json").write_text(json.dumps(reuse, ensure_ascii=False, indent=2), encoding="utf-8")
    facts = [fact for result in extractions for fact in result["assessment"]["facts"]]
    summary = {
        "protocol": protocol, "requests": len(recorder.records),
        "extraction_cases_passed": sum(result["assessment"]["passed"] for result in extractions),
        "extraction_cases_total": len(extractions),
        "fact_units": len(facts), "facts_found": sum(fact["found"] for fact in facts),
        "facts_with_correct_scope": sum(fact["correct_scope"] for fact in facts),
        "errors": sum("error_type" in record for record in recorder.records),
        "usage": {key: sum((record.get("usage") or {}).get(key) or 0 for record in recorder.records)
                  for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cache_hit_tokens", "cache_miss_tokens")},
        "extraction_results": extractions, "reuse_results": reuse,
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results: {root}", flush=True)


if __name__ == "__main__":
    main()
