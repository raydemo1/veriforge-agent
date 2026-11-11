import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from harness_code_agent.agent.coordinator import AgentCoordinator
from harness_code_agent.runtime.task_supervisor import ToolTaskSupervisor
from harness_code_agent.support.cancellation import CancellationToken

pytestmark = pytest.mark.session


def test_lingering_foreground_future_blocks_next_turn(runtime_factory):
    session = runtime_factory[0]()
    session.conversation.runtime_state.turn_id = "previous"
    future = Future()
    future.set_running_or_notify_cancel()
    session.tool_context.tool_tasks.track(future, turn_id="previous")
    try:
        with pytest.raises(RuntimeError, match="前台工具尚未退出"), session._turn_scope():
            pytest.fail("new turn shared state with lingering tool")
        assert session.turn_count == 0
        assert session.recovery.origin is None
    finally:
        future.set_result(None)
    with session._turn_scope():
        pass


def test_coordinator_close_is_bounded_and_preserves_dependencies():
    release, started = threading.Event(), threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    def blocked():
        started.set()
        release.wait()
    future = executor.submit(blocked)
    assert started.wait(1)
    coordinator = AgentCoordinator.__new__(AgentCoordinator)
    coordinator._condition = threading.Condition()
    coordinator._version = 0
    coordinator._resource_owner = Mock()
    record = SimpleNamespace(token=CancellationToken(), future=future, conversation=Mock())
    coordinator._records = {"blocked": record}
    coordinator.context = SimpleNamespace(tool_tasks=ToolTaskSupervisor())
    coordinator.changes = Mock()
    coordinator._publish_outputs = Mock()
    try:
        before = time.monotonic()
        with pytest.raises(RuntimeError, match="resources retained"):
            coordinator.close(timeout=0.05)
        assert time.monotonic() - before < 1
        record.conversation.close.assert_not_called()
        coordinator.changes.close.assert_not_called()
        assert record.token.is_cancelled
    finally:
        release.set()
        executor.shutdown(wait=True)
    coordinator.close(timeout=1)
    record.conversation.close.assert_called_once()
    coordinator.changes.close.assert_called_once()


def test_busy_mcp_toggle_does_not_write_config(tmp_path, runtime_factory):
    session = runtime_factory[0]()
    config_path = tmp_path / "mcp.json"
    config_path.write_text('{"enabled":true}', encoding="utf-8")
    original = config_path.read_bytes()
    session._turn_thread = -1
    session._ensure_mcp_tools_loaded = Mock()
    session.mcp_manager = Mock()
    session.mcp_manager.configured_server_names.return_value = ["server"]
    session.mcp_manager.config.servers = {"server": object()}
    session.mcp_manager.set_server_enabled.side_effect = lambda *_: config_path.write_text(
        '{"enabled":false}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="当前回合仍在执行"):
        session.toggle_mcp_server("server")
    assert config_path.read_bytes() == original
    session.mcp_manager.set_server_enabled.assert_not_called()
    session._turn_thread = None


def test_managed_background_agent_tools_do_not_block_main_turn(runtime_factory):
    session = runtime_factory[0]()
    future = Future()
    future.set_running_or_notify_cancel()
    session.tool_context.tool_tasks.track(future, turn_id="child-agent-turn", foreground=False)
    try:
        with session._turn_scope():
            assert not future.done()
    finally:
        future.set_result(None)
