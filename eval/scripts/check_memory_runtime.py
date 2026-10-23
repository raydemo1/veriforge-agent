"""Explicit end-to-end runtime checks with deterministic model and embedding fixtures."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openai.types.chat import ChatCompletion

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.scripts.check_memory_consolidation import LocalVectors, response
from harness_code_agent import config
from harness_code_agent.core.interactive import InteractiveSession
from harness_code_agent.memory import (
    MemoryService,
    background,
    extraction,
    reconciliation,
)
from harness_code_agent.memory.background import drain_memory_jobs
from harness_code_agent.memory.vectors import SiliconFlowEmbeddings
from harness_code_agent.runtime.approvals import NoApprovalProvider
from harness_code_agent.sessions.journal import SessionJournal


def completion(text="", tool=None):
    message = {"role": "assistant", "content": text}
    if tool:
        message["tool_calls"] = [{"id": "save-pref", "type": "function", "function": {
            "name": "memory_write", "arguments": json.dumps(tool, ensure_ascii=False)}}]
    return ChatCompletion.model_validate({"id": "fixture", "object": "chat.completion", "created": 0,
        "model": "fixture", "choices": [{"index": 0, "finish_reason": "tool_calls" if tool else "stop", "message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}})


def wait_workers():
    deadline = time.monotonic() + 10
    while background._running and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not background._running, "memory worker did not stop"


def checks(root):
    workspace = root / "workspace"
    workspace.mkdir()
    (workspace / ".harness").mkdir()
    (workspace / ".harness/mcp.json").write_text('{"servers":{}}', encoding="utf-8")
    raw = "请记住：和我交流使用简体中文。"
    replies = [completion(tool={"topic": "交流", "fact_key": "reply.language", "kind": "preference",
                               "body": "用户偏好简体中文交流。", "source_quote": raw, "scope": "user"}),
               completion("已记录。"), completion("简体中文。")]
    requests = []
    def create(**kwargs):
        requests.append(deepcopy(kwargs))
        return replies.pop(0)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None)
    with (patch.dict(os.environ, {"HARNESS_MEMORY_ROOT": str(root / "memory"), "HARNESS_MEMORY_DISABLED": "0"}),
          patch.object(config, "MEMORY_EXTRACTION_IDLE_SECONDS", 3600),
          patch("harness_code_agent.agent.conversation.get_client", return_value=client),
          patch.object(SiliconFlowEmbeddings, "embed", lambda self, texts: LocalVectors().embed(texts))):
        first = InteractiveSession(cwd=workspace, profile_name="general", profile_explicit=True,
                                   approval_provider=NoApprovalProvider(), enable_turn_summary=False)
        try:
            first.memory_command(["generate", "off"])
            first.submit(raw)
            stored = first.memory_service.stores["user"].records()
            assert len(stored) == 1, requests
            assert first.conversation.runtime_state.current_user_task == raw
            source = [e for e in first.conversation.journal.read() if e.kind == "user_input"]
            assert len(source) == 1 and source[0].payload["text"] == raw
            assert source[0].payload["auto_extract"] is False
            assert first.tool_context.memory_user_sequence == source[0].sequence
            first.memory_command(["generate", "on"])
            old = deepcopy(first.conversation.messages)
            first.submit("用哪种语言回复我？")
            assert first.conversation.messages[:len(old)] == old
            assert any("[HARNESS_MEMORY_REFERENCE]" in str(m.get("content")) for m in requests[-1]["messages"])
        finally:
            first.close()
        wait_workers()
        second = InteractiveSession(cwd=workspace, profile_name="general", profile_explicit=True,
                                    approval_provider=NoApprovalProvider(), enable_turn_summary=False)
        try:
            assert any("[HARNESS_MEMORY_CURRENT]" in str(m.get("content")) for m in second.conversation.messages)
            assert second.memory_service.read(stored[0].id, "user")["body"] == "用户偏好简体中文交流。"
        finally:
            second.close()
        wait_workers()
        assert not background._clients and not background._timers and not background._heartbeats
        service = MemoryService(workspace)
        journal = SessionJournal(root / "source.jsonl")
        service.extraction.register("background", journal.path)
        text = "已确认的长期构建约定：使用 BUILD_RUNTIME_1。"
        journal.append_user_input(text, turn_id="new", memory_generation=service.extraction.generation)
        service.extraction.enqueue("background", journal.path, idle_seconds=0)
        item = {"topic": "构建", "fact_key": "build.command", "kind": "convention", "body": "使用 BUILD_RUNTIME_1。",
                "scope": "project", "quotes": [text]}
        def consolidate(**kwargs):
            value = json.loads(kwargs["messages"][-1]["content"])
            return response({"decisions": [{"candidate_id": c["id"], "target_id": None, "action": "add",
                "topic": "构建", "fact_key": "build.command", "applicability": "", "body": "使用 BUILD_RUNTIME_1。",
                "reason": "source-confirmed"} for c in value["candidates"]]})
        with (patch.object(extraction, "_call_model", return_value=response({"complete": True, "episode": "", "candidates": [item]})),
              patch.object(reconciliation, "_call_model", side_effect=consolidate)):
            result = drain_memory_jobs(service, max_model_calls=4, max_embedding_requests=8)
        assert result["model_calls"] == 2, result
        assert service.stores["project"].records()[0].body == "使用 BUILD_RUNTIME_1。"
    return ["source isolation and native tool execution", "append-only references and bounded automatic recall",
            "explicit native save with automatic generation disabled",
            "cross-session pinned preference", "session close stops future background scheduling",
            "independent extraction/consolidation/index queues through the runtime worker"]


def main():
    with tempfile.TemporaryDirectory(prefix="memory-runtime-") as directory:
        results = checks(Path(directory))
    for label in results:
        print("PASS:", label)
    print(f"{len(results)} runtime checks passed without network calls.")


if __name__ == "__main__":
    main()
