"""Two real Harness sessions, bounded at six model and five embedding requests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness_code_agent import config
from harness_code_agent.core.interactive import InteractiveSession
from harness_code_agent.memory import background
from harness_code_agent.runtime.approvals import NoApprovalProvider


class HttpLedger:
    def __init__(self, root):
        self.root = root
        self.records = []
        self.lock = threading.Lock()
        self.original = httpx.Client.send

    def send(self, client, request, **kwargs):
        path = request.url.path
        purpose = "embedding" if path.endswith("/embeddings") else "model" if path.endswith("/chat/completions") else None
        if purpose is None:
            raise RuntimeError("Unexpected external HTTP request during memory evaluation")
        with self.lock:
            used = sum(row["purpose"] == purpose for row in self.records)
            if used >= (5 if purpose == "embedding" else 6):
                raise RuntimeError("Memory runtime evaluation request cap reached")
            record = {"index": len(self.records) + 1, "purpose": purpose,
                      "input_hash": hashlib.sha256(request.content).hexdigest()}
            self.records.append(record)
        start = time.monotonic()
        try:
            response = self.original(client, request, **kwargs)
            record["status_code"] = response.status_code
            if not kwargs.get("stream"):
                body = response.json()
                record["usage"] = body.get("usage")
            return response
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            raise
        finally:
            record["elapsed_ms"] = round((time.monotonic() - start) * 1000)
            with self.lock, (self.root / "requests.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    protocol = {"model_cap": 6, "embedding_cap": 5, "retries_included": True,
                "scope": "Native memory_write, cross-session startup context, raw-source separation."}
    if not args.run:
        print(json.dumps(protocol, indent=2))
        return
    root = ROOT / "eval/results" / ("memory-runtime-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    root.mkdir(parents=True)
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    workspace = root / "workspace"
    (workspace / ".harness").mkdir(parents=True)
    (workspace / ".harness/mcp.json").write_text('{"servers":{}}', encoding="utf-8")
    ledger = HttpLedger(root)
    def send(client, request, **kwargs):
        return ledger.send(client, request, **kwargs)
    fast = config.resolve_model_profile("fast")
    result = {"protocol": protocol, "sessions": []}
    with (patch.dict(os.environ, {"HARNESS_MEMORY_ROOT": str(root / "memory"), "HARNESS_MEMORY_DISABLED": "0",
                                 "HARNESS_MEMORY_GENERATION_DISABLED": "1"}),
          patch.object(config, "MODEL_INTENSITY", "fast"), patch.object(config, "MODEL", fast.model),
          patch.object(config, "MODEL_PROFILE", fast),
          patch.object(httpx.Client, "send", send)):
        try:
            for turn, task in enumerate([
                ("请记住这个长期偏好：和我交流一律用简体中文。请调用 memory_write 保存到 user 空间，"
                "topic 为交流语言、fact_key 为 reply.language、kind 为 preference。source_quote 逐字引用本条消息中的偏好。"
                "不需要其他工具，保存后简短回复。"),
                "我之前要求你用哪种语言和我交流？只回答语言，不调用工具，不保存新记忆。",
            ]):
                session = InteractiveSession(cwd=workspace, profile_name="general", profile_explicit=True,
                                             approval_provider=NoApprovalProvider(), enable_turn_summary=False)
                try:
                    response = session.submit(task)
                    records = session.memory_service.stores["user"].records()
                    sources = [e for e in session.conversation.journal.read() if e.kind == "user_input"]
                    references = [m for m in session.conversation.messages if "[HARNESS_MEMORY_" in str(m.get("content"))]
                    result["sessions"].append({"id": session.session.id, "reply": response.text,
                        "user_memories": [r.__dict__ for r in records], "raw_source_count": len(sources),
                        "source_is_original": len(sources) == 1 and sources[0].payload["text"] == task,
                        "reference_count": len(references), "turn": turn})
                finally:
                    session.close()
                deadline = time.monotonic() + 35
                while background._running and time.monotonic() < deadline:
                    time.sleep(0.1)
            first, second = result["sessions"]
            result["passed"] = (len(first["user_memories"]) == len(second["user_memories"]) == 1
                                and "中文" in second["reply"] and second["reference_count"] > 0
                                and first["source_is_original"] and second["source_is_original"]
                                and not background._clients and not background._timers)
        except Exception as exc:  # noqa: BLE001 - no automatic extension of a failed experiment
            result.update(passed=False, error_type=type(exc).__name__)
    result["model_requests"] = sum(r["purpose"] == "model" for r in ledger.records)
    result["embedding_requests"] = sum(r["purpose"] == "embedding" for r in ledger.records)
    (root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"passed={result['passed']}, model={result['model_requests']}, embeddings={result['embedding_requests']}")
    print(f"Results: {root}")


if __name__ == "__main__":
    main()
