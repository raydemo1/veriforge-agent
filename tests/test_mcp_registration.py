import asyncio
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import mcp
import pytest
from mcp.client import stdio as mcp_stdio

from harness_code_agent.agent.coordinator import AgentRecord
from harness_code_agent.agent.tool_executor import ToolExecutor
from harness_code_agent.runtime.builtins.discovery import search_tools
from harness_code_agent.runtime.builtins.registry import BUILTIN_TOOL_REGISTRY
from harness_code_agent.runtime.mcp import (
    McpClientManager,
    McpConfigError,
    McpServerStatus,
    McpToolBinding,
    build_server_config,
)
from harness_code_agent.support.cancellation import CancellationToken, CancelledError

MISSING_COMMAND = "__veriforge_missing_command__"


def _write_config(root, servers: dict) -> None:
    path = root / ".harness" / "mcp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"servers": servers}, indent=2), encoding="utf-8")


def _read_config(root) -> dict:
    path = root / ".harness" / "mcp.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_build_server_config_requires_exactly_one_transport_source():
    with pytest.raises(McpConfigError):
        build_server_config(name="both", url="http://127.0.0.1:1/mcp", command="npx")
    with pytest.raises(McpConfigError):
        build_server_config(name="neither")
    with pytest.raises(McpConfigError):
        build_server_config(name="bad name", url="http://127.0.0.1:1/mcp")


def test_register_server_persists_entry_when_probe_fails(tmp_path):
    manager = McpClientManager.from_workspace(tmp_path)
    manager.timeout_seconds = 5
    server = build_server_config(name="ghost", command=MISSING_COMMAND)
    registration = manager.register_server(server)
    assert registration.state == "failed"
    assert registration.tool_count == 0
    entry = _read_config(tmp_path)["servers"]["ghost"]
    assert entry["transport"] == "stdio"
    assert entry["command"] == MISSING_COMMAND
    assert "description" not in entry


def test_discover_isolates_unreachable_servers_and_reports_them(tmp_path):
    _write_config(tmp_path, {"ghost": {"transport": "stdio", "command": MISSING_COMMAND}})
    manager = McpClientManager.from_workspace(tmp_path)
    manager.timeout_seconds = 5
    try:
        failures = manager.discover()
        assert failures["ghost"].startswith("FileNotFoundError")
        assert manager.statuses["ghost"].state == "failed"
        requested = manager.discover("ghost")
        assert "ghost" in requested
    finally:
        manager.close()


def test_reregister_server_keeps_existing_metadata(tmp_path):
    _write_config(
        tmp_path,
        {
            "docs": {
                "transport": "stdio",
                "command": MISSING_COMMAND,
                "title": "Docs",
                "description": "Searches project documentation.",
                "tool_permissions": {"search": "read"},
            }
        },
    )
    manager = McpClientManager.from_workspace(tmp_path)
    manager.timeout_seconds = 5
    server = build_server_config(name="docs", url="http://127.0.0.1:1/mcp", permission="network_read")
    registration = manager.register_server(server)
    assert registration.replaced is True
    assert registration.server.description == "Searches project documentation."
    assert registration.server.title == "Docs"
    assert registration.server.tool_permissions == {"search": "read"}
    entry = _read_config(tmp_path)["servers"]["docs"]
    assert entry["url"] == "http://127.0.0.1:1/mcp"
    assert entry["permission"] == "network_read"
    assert "command" not in entry
    assert entry["description"] == "Searches project documentation."
    assert entry["title"] == "Docs"
    assert entry["tool_permissions"] == {"search": "read"}


def test_remove_server_keeps_other_entries_and_env_placeholders(tmp_path):
    _write_config(
        tmp_path,
        {
            "alpha": {"transport": "stdio", "command": MISSING_COMMAND},
            "beta": {
                "transport": "stdio",
                "command": "server-b",
                "env": {"TOKEN": "${MY_TOKEN}"},
            },
        },
    )
    manager = McpClientManager.from_workspace(tmp_path)
    message = manager.remove_server("alpha")
    assert "alpha" in message
    data = _read_config(tmp_path)
    assert "alpha" not in data["servers"]
    assert data["servers"]["beta"]["env"]["TOKEN"] == "${MY_TOKEN}"


def test_generate_server_description_persists_and_respects_refresh(tmp_path):
    _write_config(tmp_path, {"docs": {"transport": "stdio", "command": "server"}})
    manager = McpClientManager.from_workspace(tmp_path)
    manager.tool_bindings = [
        McpToolBinding("mcp__docs__search", "docs", "search", "Search docs", {}, "read")
    ]

    def first_model(messages):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"docs": "Searches documentation."}')
                )
            ]
        )

    message = manager.generate_server_description("docs", llm_create=first_model)
    assert "Searches documentation." in message
    assert _read_config(tmp_path)["servers"]["docs"]["description"] == "Searches documentation."

    def unexpected_model(messages):
        raise AssertionError("model must not be called when a description already exists")

    manager.generate_server_description("docs", llm_create=unexpected_model)

    def refreshed_model(messages):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"docs": "Searches and reads project docs."}')
                )
            ]
        )

    manager.generate_server_description("docs", refresh=True, llm_create=refreshed_model)
    assert _read_config(tmp_path)["servers"]["docs"]["description"] == "Searches and reads project docs."


def test_slash_mcp_commands_route_to_registration_api():
    from harness_code_agent.tui.commands import default_command_registry

    class Session:
        def __init__(self):
            self.registered = None
            self.removed = None
            self.described = None

        def register_mcp_server(self, **kwargs):
            self.registered = kwargs
            return "registered"

        def remove_mcp_server(self, name):
            self.removed = name
            return "removed"

        def refresh_mcp_server_description(self, name, *, refresh):
            self.described = (name, refresh)
            return "described"

        def mcp_status(self):
            return "status"

    registry = default_command_registry(skill_registry=SimpleNamespace(skills=[]))
    session = Session()

    assert registry.execute(
        "/mcp add docs https://example.test/mcp --permission network_read", session
    ).text == "registered"
    assert session.registered["url"] == "https://example.test/mcp"
    assert session.registered["permission"] == "network_read"

    assert registry.execute("/mcp add fs --command npx -y example", session).text == "registered"
    assert session.registered["command"] == "npx"
    assert session.registered["args"] == ["-y", "example"]

    assert registry.execute("/mcp remove docs", session).text == "removed"
    assert session.removed == "docs"

    assert registry.execute("/mcp describe docs --refresh", session).text == "described"
    assert session.described == ("docs", True)

    assert registry.execute("/mcp list", session).text == "status"


def test_reload_server_leaves_other_servers_untouched(tmp_path):
    _write_config(
        tmp_path,
        {
            "target": {"transport": "stdio", "command": MISSING_COMMAND},
            "other": {"transport": "stdio", "command": "server-b"},
        },
    )
    manager = McpClientManager.from_workspace(tmp_path)
    binding = McpToolBinding("mcp__other__read", "other", "read", "Read", {}, "read")
    manager.tool_bindings = [binding]
    manager._bindings_by_exposed_name = {binding.exposed_name: binding}
    manager.statuses["other"] = McpServerStatus(
        name="other", transport="stdio", state="connected", tool_count=1
    )
    try:
        message = manager.reload_server("target")
        assert "连接失败" in message
        assert manager.statuses["other"].state == "connected"
        assert [item.exposed_name for item in manager.tool_bindings] == ["mcp__other__read"]
        assert manager.statuses["target"].state == "failed"
    finally:
        manager.close()


def _binding(server, name):
    return McpToolBinding(f"mcp__{server}__{name}", server, name, name, {"type": "object"}, "read")


async def _wait_for_close(queue):
    kind, _, response = await queue.get()
    assert kind == "close"
    response.set_result(None)


def test_concurrent_connects_share_a_handshake_and_isolate_timeout(tmp_path, monkeypatch):
    _write_config(tmp_path, {
        name: {"transport": "stdio", "command": name} for name in ("slow", "fast")
    })
    manager = McpClientManager.from_workspace(tmp_path)
    manager.timeout_seconds = 3
    started = threading.Event()
    calls = []

    async def worker(server, queue, ready):
        calls.append(server.name)
        if server.name == "slow":
            started.set()
            await asyncio.Event().wait()
        ready.set_result([_binding(server.name, "read")])
        await _wait_for_close(queue)

    monkeypatch.setattr(manager, "_connection_worker", worker)
    token = CancellationToken()

    async def both_waiting():
        while manager._connecting["slow"].waiters < 2:
            await asyncio.sleep(0.01)

    with ThreadPoolExecutor(max_workers=3) as pool:
        try:
            cancelled = pool.submit(manager.ensure_connected, "slow", cancellation_token=token)
            assert started.wait(2)
            waiting = pool.submit(manager.ensure_connected, "slow")
            fast = pool.submit(manager.ensure_connected, "fast")
            fast.result(2)
            manager._run(both_waiting(), timeout=1)
            token.cancel()
            with pytest.raises(CancelledError):
                cancelled.result(2)
            with pytest.raises(TimeoutError):
                waiting.result(5)
            assert calls.count("slow") == 1
            assert manager.discovery_statuses()["slow"].state == "failed"
            assert manager.discovery_statuses()["fast"].state == "connected"
        finally:
            manager.close()


def test_discovery_scopes_and_mcp_catalog_replacement(tmp_path, monkeypatch, runtime_factory):
    _write_config(tmp_path, {
        "agents": {"transport": "stdio", "command": "agents", "permission": "read", "description": "External tasks."},
        "docs": {"transport": "stdio", "command": "docs", "permission": "read", "description": "Read documentation."},
        "broken": {"transport": "stdio", "command": "broken", "permission": "read"},
        "invalid": {"transport": "invalid"},
        "disabled": {"enabled": False},
    })
    session = runtime_factory[0]()
    manager = session.mcp_manager
    session._load_mcp_tools()
    context = session.tool_context
    catalog = session.agent.stable_system_prompt
    assert "builtin:agents" in catalog and "mcp:agents" in catalog
    assert "mcp:invalid" not in catalog and "mcp:disabled" not in catalog
    assert "server invalid: invalid" in manager.status_report()
    assert "server disabled: disabled" in manager.status_report()

    catalogs = {
        "agents": [_binding("agents", "delegate_external")],
        "docs": [_binding("docs", "read"), _binding("docs", "list")],
        "broken": [
            _binding("broken", "good"),
            replace(_binding("broken", "bad"), input_schema={"type": "object", "required": "not-an-array"}),
        ],
    }
    connected = []

    async def worker(server, queue, ready):
        connected.append(server.name)
        ready.set_result(catalogs[server.name])
        await _wait_for_close(queue)

    monkeypatch.setattr(manager, "_connection_worker", worker)
    builtins = search_tools("spawn agent", "builtin:agents", tool_context=context)
    assert "spawn_agent" in builtins.metadata["revealed_tool_names"]
    assert "send_parent_message" not in builtins.metadata["revealed_tool_names"]
    assert all(context.tool_registry.spec_for(name).origin_server is None for name in builtins.metadata["revealed_tool_names"])
    assert search_tools("read", "mcp:invalid", tool_context=context).metadata["status"] == "invalid"
    assert search_tools("read", "mcp:disabled", tool_context=context).metadata["status"] == "disabled"
    assert manager._loop_thread is None

    external = search_tools("delegate external", "agents", tool_context=context)
    assert external.metadata["scope"] == "mcp:agents"
    assert external.metadata["revealed_tool_names"] == ["mcp__agents__delegate_external"]
    assert search_tools("delegate external", "mcp:agents", tool_context=context).metadata["level"] == "repeated"
    failed = search_tools("good", "mcp:broken", tool_context=context)
    assert failed.metadata["status"] == "failed"
    assert context.tool_registry.spec_for("mcp__broken__good") is None
    assert connected == ["agents", "broken"]

    assert not manager.discover("docs")
    session._sync_mcp_connections()
    revealed = {"mcp__docs__read", "mcp__docs__list", "mcp__agents__delegate_external"}
    context.discovered_tool_names.update(revealed)
    session._refresh_agent_tool_schemas()
    executor = ToolExecutor(session.conversation)
    prepared, = executor._prepare_calls([{
        "id": "old-read",
        "type": "function",
        "function": {"name": "mcp__docs__read", "arguments": "{}"},
    }])
    catalogs["docs"] = [replace(
        _binding("docs", "read"), input_schema={"type": "object", "properties": {"id": {"type": "string"}}},
    )]
    assert "已重新连接" in session.reload_mcp_server("docs")
    session._sync_mcp_connections()
    assert context.tool_registry.get("mcp__docs__list") is None
    assert context.revealed_tool_names == {"mcp__agents__delegate_external"}
    assert session.agent.allowed_tool_names.isdisjoint(revealed - context.revealed_tool_names)
    assert executor._execute_one(prepared, None).result.metadata["reason"] == "stale_binding"
    assert search_tools("read", "mcp:docs", tool_context=context).metadata["revealed_tool_names"] == ["mcp__docs__read"]


def test_malformed_mcp_document_leaves_builtins_available(tmp_path, runtime_factory):
    _write_config(tmp_path, {})
    path = tmp_path / ".harness" / "mcp.json"
    path.write_text('{"servers":', encoding="utf-8")
    session = runtime_factory[0]()
    context = session.tool_context
    assert "Invalid MCP config JSON" in session.mcp_manager.status_report()
    assert search_tools("spawn agent", "builtin:agents", tool_context=context).status == "success"
    result = search_tools("read", "mcp:docs", tool_context=context)
    assert result.error == "mcp_unavailable"
    assert "Invalid MCP config JSON" in result.output
    assert session.mcp_manager._loop_thread is None


@pytest.mark.parametrize("failure", ["exception", "repeated_cursor", "timeout", "invalid_schema"])
def test_mcp_discovers_all_pages_and_preserves_history_on_failed_reconnect(tmp_path, monkeypatch, runtime_factory, failure):
    _write_config(tmp_path, {"docs": {
        "transport": "stdio", "command": "docs", "permission": "read", "description": "Read documentation.",
    }})
    session = runtime_factory[0]()
    manager = session.mcp_manager
    session._load_mcp_tools()
    cursors = []
    reconnecting = False

    @asynccontextmanager
    async def transport(_):
        yield object(), object()

    class PagedSession:
        def __init__(self, read_stream, write_stream):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            pass

        async def list_tools(self, cursor=None):
            cursors.append(cursor)
            if cursor is None:
                name = "new_first_page" if reconnecting else "first_page"
                next_cursor = "next"
            else:
                if reconnecting and failure == "exception":
                    raise RuntimeError("second page failed")
                if reconnecting and failure == "timeout":
                    await asyncio.Event().wait()
                name = "second_page"
                next_cursor = "next" if reconnecting and failure == "repeated_cursor" else None
            schema = {"type": "object"}
            if reconnecting and cursor is not None and failure == "invalid_schema":
                schema["required"] = "invalid"
            return SimpleNamespace(nextCursor=next_cursor, tools=[
                SimpleNamespace(name=name, description=name.replace("_", " "), inputSchema=schema),
            ])

    monkeypatch.setattr(mcp, "ClientSession", PagedSession)
    monkeypatch.setattr(mcp_stdio, "stdio_client", transport)
    context = session.tool_context
    result = search_tools("second page", "mcp:docs", limit=1, tool_context=context)
    assert result.metadata["revealed_tool_names"] == ["mcp__docs__second_page"]
    assert cursors == [None, "next"]
    known = {"mcp__docs__first_page", "mcp__docs__second_page"}
    assert {binding.exposed_name for binding in manager.catalog_snapshot().bindings} == known

    reconnecting = True
    if failure == "timeout":
        manager.timeout_seconds = .15
    assert "连接失败" in session.reload_mcp_server("docs")
    assert cursors == [None, "next", None, "next"]
    status = manager.discovery_statuses()["docs"]
    assert (status.state, status.catalog_source) == ("failed", "cache")
    assert {binding.exposed_name for binding in manager.catalog_snapshot().bindings} == known
    assert context.tool_registry.get("mcp__docs__new_first_page") is None


def test_main_and_child_requests_apply_a_disconnect_once(tmp_path, monkeypatch, runtime_factory):
    _write_config(tmp_path, {
        name: {"transport": "stdio", "command": name, "permission": "read"}
        for name in ("docs", "other")
    })
    session = runtime_factory[0]()
    manager = session.mcp_manager

    async def worker(server, queue, ready):
        ready.set_result([_binding(server.name, "read")])
        kind, _, response = await queue.get()
        if kind == "close":
            response.set_result(None)

    monkeypatch.setattr(manager, "_connection_worker", worker)
    session._load_mcp_tools()
    assert not manager.discover()
    session._sync_mcp_connections()
    record = AgentRecord("child", "explorer", "explorer", "Inspect", "", [], "none", None)
    session.tool_context.agent_coordinator._ensure_conversation(record)
    child = record.conversation
    main = session.conversation
    main.add_user_turn("Inspect")
    try:
        for conversation in (main, child):
            for server in ("docs", "other"):
                result = search_tools("read", f"mcp:{server}", tool_context=conversation.agent.tool_context)
                ToolExecutor(conversation)._reveal_tool_schemas_from_result(result)

        async def disconnect():
            connection = manager._connections["docs"]
            connection.request_queue.put_nowait(("disconnect", None, None))
            await connection.task

        barrier = threading.Barrier(2, action=lambda: manager._run(disconnect(), timeout=2))
        register_tools = manager.register_tools
        publications = []

        def counted_register(registry, *, snapshot):
            publications.append(snapshot.revision)
            return register_tools(registry, snapshot=snapshot)

        def unexpected_validation(_):
            pytest.fail("a disconnect must reuse previously validated specs")

        monkeypatch.setattr(manager, "register_tools", counted_register)
        monkeypatch.setattr(manager, "_server_specs", unexpected_validation)
        for conversation in (main, child):
            def request(kwargs, *, cancellation_token=None):
                names = {schema["function"]["name"] for schema in kwargs["tools"]}
                assert "mcp__other__read" in names
                if "mcp__docs__read" in names:
                    barrier.wait(timeout=3)
                    return {"role": "assistant", "content": "", "tool_calls": [{
                        "id": "list_files", "type": "function",
                        "function": {"name": "list_files", "arguments": "{}"},
                    }]}, "tool_calls"
                return {"role": "assistant", "content": "Done"}, "stop"

            monkeypatch.setattr(conversation.llm, "request_assistant_message", request)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(conversation.run_until_idle) for conversation in (main, child)]
            assert [future.result(8) for future in futures] == ["Done", "Done"]
        assert publications == [manager.connection_revision]
        assert "mcp__docs__read" not in main.agent.allowed_tool_names
        assert "mcp__docs__read" not in child.agent.allowed_tool_names
    finally:
        child.close()


def test_stdio_disconnect_keeps_history_without_mounting_tools(tmp_path):
    server = tmp_path / "server.py"
    server.write_text("""import os
import threading
import time
from pathlib import Path
from mcp.server.fastmcp import FastMCP

app = FastMCP("docs")

@app.tool(structured_output=False)
def find_docs() -> str:
    return "docs"

def watch():
    marker = Path(__file__).with_name("stop")
    while not marker.exists():
        time.sleep(0.02)
    os._exit(0)

threading.Thread(target=watch, daemon=True).start()
app.run(transport="stdio")
""", encoding="utf-8")
    _write_config(tmp_path, {"docs": {
        "transport": "stdio", "command": sys.executable, "args": [str(server)], "permission": "read",
    }})
    manager = McpClientManager.from_workspace(tmp_path)
    try:
        assert not manager.discover("docs")
        registry = BUILTIN_TOOL_REGISTRY.copy()
        manager.register_tools(registry)
        handler = registry.get("mcp__docs__find_docs")
        assert handler().status == "success"
        (tmp_path / "stop").touch()
        deadline = time.monotonic() + 5
        while manager.discovery_statuses()["docs"].state == "connected" and time.monotonic() < deadline:
            time.sleep(0.02)
        status = manager.discovery_statuses()["docs"]
        assert (status.state, status.catalog_source) == ("failed", "cache")
        assert handler().status == "failed"
        context = SimpleNamespace(
            tool_registry=registry,
            tool_capability="main",
            discover_mcp_tools=manager.discover,
            mcp_discovery_statuses=manager.discovery_statuses,
            allowed_tool_permissions={"read"},
            blocked_tool_names=set(),
            revealed_tool_names=set(),
            search_tools_queries=set(),
            search_tools_signature=(),
            search_tools_no_progress=0,
        )
        result = search_tools("find docs", "mcp:docs", tool_context=context)
        assert result.metadata["revealed_tool_names"] == []
        assert result.metadata["catalog_source"] == "cache"
        assert "Historical tools (unavailable)" in result.output
        assert "find_docs" in result.output
        assert manager.discovery_statuses()["docs"].state == "failed"
    finally:
        manager.close()
