import json
from types import SimpleNamespace

import pytest

from harness_code_agent.runtime.mcp import (
    McpClientManager,
    McpConfigError,
    McpServerStatus,
    McpToolBinding,
    build_server_config,
)

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


def test_register_server_replaces_existing_entry(tmp_path):
    _write_config(
        tmp_path,
        {"docs": {"transport": "stdio", "command": MISSING_COMMAND}},
    )
    manager = McpClientManager.from_workspace(tmp_path)
    manager.timeout_seconds = 5
    server = build_server_config(
        name="docs",
        url="http://127.0.0.1:1/mcp",
        permission="network_read",
    )
    registration = manager.register_server(server)
    assert registration.replaced is True
    entry = _read_config(tmp_path)["servers"]["docs"]
    assert entry["url"] == "http://127.0.0.1:1/mcp"
    assert entry["permission"] == "network_read"
    assert "command" not in entry


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
    server = build_server_config(name="docs", url="http://127.0.0.1:1/mcp")
    registration = manager.register_server(server)
    assert registration.replaced is True
    assert registration.server.description == "Searches project documentation."
    assert registration.server.title == "Docs"
    assert registration.server.tool_permissions == {"search": "read"}
    entry = _read_config(tmp_path)["servers"]["docs"]
    assert entry["url"] == "http://127.0.0.1:1/mcp"
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
