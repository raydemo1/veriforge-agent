import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import patch

import pytest
from harness_code_agent.support.cancellation import (
    CancellationToken,
    CancelledError,
    cancellation_scope,
)

from harness_code_agent.agent.trace import TraceWriter
from harness_code_agent.tui_bridge import BridgeServer

pytestmark = pytest.mark.session

def _wait(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "execution did not reach the expected state"
        time.sleep(.01)


def test_sessions_limit_actual_streams_and_cancel_waiters(runtime_factory):
    create, clients = runtime_factory
    sessions = [create(stream_sink=lambda _: None) for _ in range(5)]
    release = threading.Event()
    tokens = [CancellationToken() for _ in sessions]
    outcomes = []

    def request(**kwargs):
        def stream():
            assert release.wait(5)
            yield {"choices": [{"delta": {"content": "完成"}, "finish_reason": "stop"}]}
        return stream()

    def run(session, token):
        try:
            with cancellation_scope(token):
                session.submit("执行")
            outcomes.append("completed")
        except CancelledError:
            outcomes.append("cancelled")

    for client in clients:
        client.create = request
    threads = [threading.Thread(target=run, args=pair) for pair in zip(sessions, tokens)]
    try:
        for thread in threads:
            thread.start()
        budget = sessions[0].project_resources
        _wait(lambda: len(budget.snapshot()["waiting"]) == 2)
        snapshot = budget.snapshot()
        assert len(snapshot["active"]) == 3
        waiting = snapshot["waiting"][0]["sessionId"]
        tokens[next(i for i, session in enumerate(sessions) if session.session.id == waiting)].cancel()
        _wait(lambda: "cancelled" in outcomes)
        assert len(budget.snapshot()["active"]) == 3
    finally:
        release.set()
        for thread in threads:
            thread.join(5)
    assert sorted(outcomes) == ["cancelled", "completed", "completed", "completed", "completed"]


def test_automatic_dormancy_preserves_history_and_waits_for_tool_exit(runtime_factory, tmp_path):
    with patch.object(BridgeServer, "_write"):
        bridge = BridgeServer(cwd=tmp_path)
        try:
            record = bridge.registry.get(bridge.registry.active_id)
            host = record.host
            assert host._session is None
            session = host._require_session()
            session.submit("保留这段历史")
            session_id = session.session.id
            history = bridge.registry.snapshot(session_id)["items"]
            pending = Future()
            pending.set_running_or_notify_cancel()
            session.tool_context.tool_tasks.track(pending)
            bridge.registry.maintain(now=host._last_activity + 121)
            assert host._session is session
            pending.set_result(None)
            bridge.registry.maintain(now=host._last_activity + 121)
            assert host._session is None
            assert session.conversation._closed
            assert not any(event.type == "session_finished" for event in session.event_bus.events)
            restored = host._require_session()
            assert restored.session.id == session_id
            assert "保留这段历史" in str(restored.conversation.messages)
            assert restored is not session
            assert bridge.registry.snapshot(session_id)["items"] == history
        finally:
            bridge.close()


def test_trace_files_keep_concurrent_session_records_separate(tmp_path):
    writers = [TraceWriter("main", workspace=tmp_path, session_id=name,
                           turn_id=lambda: "turn", request_id=lambda: "request") for name in ("A", "B")]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda index: writers[index % 2].tool_call("read", {}, "ok", tool_call_id=str(index)), range(100)))
    for writer in writers:
        rows = [json.loads(line) for line in writer._path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 50
        assert all(row["session_id"] == writer.session_id and row["turn_id"] == "turn"
                   and row["ui_request_id"] == "request" and row["tool_call_id"] for row in rows)


def test_queued_agent_can_be_cancelled_before_initialization(runtime_factory):
    session = runtime_factory[0]()
    budget = session.project_resources
    gate = threading.Event()
    occupied = [budget.submit("agents", gate.wait) for _ in range(budget.limits["agents"])]
    try:
        _wait(lambda: len(budget.snapshot()["active"]) == budget.limits["agents"])
        coordinator = session.tool_context.agent_coordinator
        queued = coordinator.spawn(name="waiting", role="explorer", task="不会开始")
        assert coordinator.interrupt(queued["agent_id"])["status"] == "interrupted"
        record = coordinator._record(queued["agent_id"])
        assert record.future.cancelled()
        assert record.conversation is None
    finally:
        gate.set()
        for future in occupied:
            future.result(5)


def test_mcp_cached_catalog_connects_separate_sessions_only_on_call(tmp_path):
    import sys

    from harness_code_agent.runtime.mcp import McpClientManager

    server = tmp_path / "server.py"
    server.write_text('''import os
from mcp.server.fastmcp import FastMCP
app = FastMCP("isolated")
count = 0
@app.tool(structured_output=False)
def counter() -> str:
    global count
    count += 1
    return f"{os.getpid()}:{count}"
app.run(transport="stdio")
''', encoding="utf-8")
    directory = tmp_path / ".harness"
    directory.mkdir()
    (directory / "mcp.json").write_text(json.dumps({"servers": {"isolated": {
        "transport": "stdio", "command": sys.executable, "args": [str(server)],
    }}}), encoding="utf-8")
    first = McpClientManager.from_workspace(tmp_path, session_id="A")
    first.timeout_seconds = 5
    second = McpClientManager.from_workspace(tmp_path, session_id="B")
    try:
        first.discover("isolated")
        binding = first.tool_bindings[0]
        second.prime_catalog()
        assert second._loop_thread is None
        assert second.tool_bindings[0] is not binding
        a = first.call_tool(binding.exposed_name, {}).to_text()
        b = second.call_tool(binding.exposed_name, {}).to_text()
        assert a != b
        assert ":1" in a and ":1" in b
        from contextlib import ExitStack
        first.tool_timeout_seconds = .5
        with ThreadPoolExecutor(max_workers=1) as pool:
            with ExitStack() as leases:
                for _ in range(2):
                    leases.enter_context(first.project_resources.acquire("mcp:isolated"))
                result = pool.submit(first.call_tool, binding.exposed_name, {})
                _wait(lambda: any(entry["resource"] == "mcp:isolated" for entry in first.project_resources.snapshot()["waiting"]))
                time.sleep(.6)
                assert not result.done()
            assert ":2" in result.result(5).to_text()
    finally:
        first.close()
        second.close()
