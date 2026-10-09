from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import psutil
import pytest

from harness_code_agent.runtime import language_service as lsp
from harness_code_agent.runtime import source_watch
from harness_code_agent.runtime.builtins.code_intelligence import code_intelligence
from harness_code_agent.runtime.builtins.registry import BUILTIN_TOOL_REGISTRY
from harness_code_agent.runtime.execution_planner import (
    CallEffect,
    effects_conflict,
    workspace_claim,
)
from harness_code_agent.runtime.permissions import PermissionPolicy
from harness_code_agent.runtime.tool_context import ToolContext
from harness_code_agent.sessions.events import EventBus
from harness_code_agent.workspace.service import WorkspaceService


@pytest.fixture
def service(tmp_path, monkeypatch):
    servers = []

    def create(mode="normal"):
        script = Path(__file__).parent / "fixtures" / "language_server.py"
        monkeypatch.setattr(
            lsp, "discover_server", lambda *_: [sys.executable, str(script), mode]
        )
        monkeypatch.setattr(lsp, "_typescript_server", lambda *_: Path(sys.executable))
        manager = lsp.LanguageService(tmp_path)
        servers.append(manager)
        return manager

    yield create
    for manager in servers:
        manager.close()


@pytest.mark.parametrize(
    "operation", ["definition", "references", "symbols", "diagnostics"]
)
def test_stdio_operations_and_close(service, tmp_path, operation):
    (tmp_path / "main.py").write_text("hello\n", encoding="utf-8")
    manager = service()
    assert manager._loop is None
    result = manager.query(operation, "main.py", 1, 2)
    assert result["available"], result
    assert result["items"][0]["location"].startswith("main.py:1:")
    if operation == "symbols":
        assert result["items"][1]["name"] == "hello.child"
    server = next(iter(manager._servers.values()))
    pid = server.client._server.pid
    manager.close()
    assert not psutil.pid_exists(pid)


@pytest.mark.parametrize("mode", ["normal", "push", "push-canonical"])
def test_diagnostics_refresh_after_edit_and_rewind(service, tmp_path, mode):
    source = tmp_path / "main.py"
    source.write_text("before", encoding="utf-8")
    manager = service(mode)
    for text in ["before", "after", "before"]:
        source.write_text(text, encoding="utf-8")
        result = manager.query("diagnostics", "main.py")
        assert result["available"], result
        assert result["items"][0]["message"] == text
    assert len(manager._servers) == 1


def test_typescript_diagnostics_ignore_unversioned_stale_notifications(
    service, tmp_path
):
    source = tmp_path / "main.ts"
    manager = service("ts-sync")
    for text in ["before", "after", "before"]:
        source.write_text(text, encoding="utf-8")
        result = manager.query("diagnostics", "main.ts")
        assert result["available"], result
        assert result["items"][0]["message"] == text


@pytest.mark.parametrize(
    "mode", ["crash", "query-crash", "timeout", "malformed", "bad-json"]
)
def test_server_fault_is_unavailable_and_process_is_closed(
    service, tmp_path, monkeypatch, mode
):
    (tmp_path / "main.py").write_text("hello", encoding="utf-8")
    monkeypatch.setattr(lsp, "_TIMEOUT", 0.3)
    manager = service(mode)
    result = manager.query("definition", "main.py", 1, 1)
    assert result["available"] is False, result
    assert result["reason"]
    assert manager._servers == {}


def test_missing_server_does_not_start_thread(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("hello", encoding="utf-8")
    monkeypatch.setattr(lsp, "discover_server", lambda *_: None)
    manager = lsp.LanguageService(tmp_path)
    result = manager.query("definition", "main.py", 1, 1)
    assert result["available"] is False
    assert manager._loop is None


@pytest.mark.integration
def test_read_only_tool_does_not_execute_workspace_language_server(
    tmp_path, monkeypatch, require_integration_tool
):
    node = shutil.which("node")
    require_integration_tool(
        node is not None, "Node.js is required for the execution-boundary regression"
    )
    monkeypatch.setenv("PATH", str(Path(node).parent))
    package = tmp_path / "node_modules/typescript-language-server"
    package.mkdir(parents=True)
    (package / "package.json").write_text(
        '{"bin":{"typescript-language-server":"evil.js"}}', encoding="utf-8"
    )
    (package / "evil.js").write_text(
        "require('fs').writeFileSync('executed', 'yes'); process.exit(7);",
        encoding="utf-8",
    )
    (tmp_path / "main.ts").write_text("const value = 1;", encoding="utf-8")
    context = ToolContext(
        workspace=WorkspaceService(root=tmp_path),
        permission_policy=PermissionPolicy(mode="read-only"),
        event_bus=EventBus(),
    )
    try:
        result = code_intelligence("symbols", "main.ts", tool_context=context)
        assert not (tmp_path / "executed").exists()
        assert json.loads(result.output)["available"] is False
        assert context.get_language_service()._loop is None
    finally:
        context.close_language_service()


def test_position_utf16_and_location_link(service, tmp_path):
    (tmp_path / "main.py").write_text("a😀b", encoding="utf-8")
    manager = service("link")
    result = manager.query("definition", "main.py", 1, 3)
    assert result["available"], result
    assert result["items"][0]["line"] == 1
    assert result["items"][0]["column"] == 3


def test_rejects_server_workspace_edit(service, tmp_path):
    source = tmp_path / "main.py"
    source.write_text("unchanged", encoding="utf-8")
    result = service("apply-edit").query("definition", "main.py", 1, 1)
    assert result["available"], result
    assert source.read_text(encoding="utf-8") == "unchanged"


def test_answers_server_configuration_requests(service, tmp_path):
    (tmp_path / "main.py").write_text("hello", encoding="utf-8")
    result = service("configuration").query("definition", "main.py", 1, 1)
    assert result["available"], result


def test_changed_unopened_dependencies_notify_server(service, tmp_path):
    (tmp_path / "main.py").write_text("hello", encoding="utf-8")
    (tmp_path / "other.py").write_text("before", encoding="utf-8")
    manager = service("watch")
    initial = manager.query("symbols", "main.py")
    assert initial["available"], initial
    (tmp_path / "other.py").write_text("after", encoding="utf-8")
    result = manager.query("definition", "main.py", 1, 1)
    assert result["available"], result


def test_warm_query_reads_only_requested_document_without_source_rescan(
    service, tmp_path, monkeypatch
):
    (tmp_path / "main.py").write_text("main", encoding="utf-8")
    other = tmp_path / "other.py"
    other.write_text("other", encoding="utf-8")
    manager = service()
    assert manager.query("symbols", "main.py")["available"]
    assert manager.query("symbols", "other.py")["available"]
    read_text = Path.read_text

    def checked_read(path, *args, **kwargs):
        assert path != other, "Unchanged opened dependencies must not be reread"
        return read_text(path, *args, **kwargs)

    def unexpected_scan(*_args):
        pytest.fail("Warm queries must not rescan source files")

    monkeypatch.setattr(Path, "read_text", checked_read)
    monkeypatch.setattr(source_watch, "iter_source_files", unexpected_scan)
    monkeypatch.setattr(lsp, "discover_server", unexpected_scan)
    result = manager.query("symbols", "main.py")
    assert result["available"], result


def test_failed_server_can_restart_on_next_request(service, tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("hello", encoding="utf-8")
    manager = service("malformed")
    assert not manager.query("definition", "main.py", 1, 1)["available"]
    script = Path(__file__).parent / "fixtures" / "language_server.py"
    monkeypatch.setattr(
        lsp, "discover_server", lambda *_: [sys.executable, str(script), "normal"]
    )
    result = manager.query("symbols", "main.py")
    assert result["available"], result


def system_executable(directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    executable = directory / (name + ".exe" if os.name == "nt" else name)
    executable.write_text("fixture", encoding="utf-8")
    executable.chmod(0o755)
    return executable


def test_system_npm_server_is_used_instead_of_workspace_server(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    local = workspace / "node_modules/typescript-language-server"
    local.mkdir(parents=True)
    (local / "package.json").write_text(
        '{"bin":{"typescript-language-server":"evil.js"}}', encoding="utf-8"
    )
    (local / "evil.js").write_text("evil", encoding="utf-8")
    system = tmp_path / "system"
    system.mkdir()
    package = system / "node_modules/typescript-language-server"
    package.mkdir(parents=True)
    (package / "package.json").write_text(
        '{"bin":{"typescript-language-server":"cli.mjs"}}', encoding="utf-8"
    )
    (package / "cli.mjs").write_text("", encoding="utf-8")
    if os.name == "nt":
        (system / "typescript-language-server.cmd").write_text(
            "fixture", encoding="utf-8"
        )
    else:
        (system / "typescript-language-server").symlink_to(package / "cli.mjs")
        (package / "cli.mjs").chmod(0o755)
    node = system_executable(system, "node")
    system_executable(workspace, "node")
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("PATH", os.pathsep.join([str(workspace), str(system)]))
    expected = [str(node), str(package / "cli.mjs"), "--stdio"]
    assert lsp.discover_server("typescript", workspace) == expected


def test_workspace_virtualenv_server_is_not_discovered(tmp_path, monkeypatch):
    system_executable(
        tmp_path / ".venv" / ("Scripts" if os.name == "nt" else "bin"),
        "pyright-langserver",
    )
    monkeypatch.setenv("PATH", "")
    assert lsp.discover_server("python", tmp_path) is None


@pytest.mark.parametrize(
    "language,name",
    [("python", "pyright-langserver"), ("go", "gopls"), ("rust", "rust-analyzer")],
)
def test_workspace_path_entries_are_skipped_before_system_server(
    tmp_path, monkeypatch, language, name
):
    workspace = tmp_path / "workspace"
    local = system_executable(workspace / "bin", name)
    external = system_executable(tmp_path / "system", name)
    monkeypatch.setenv(
        "PATH", os.pathsep.join([str(local.parent), str(external.parent)])
    )
    assert lsp.discover_server(language, workspace)[0] == str(external)
    monkeypatch.setenv("PATH", str(local.parent))
    assert lsp.discover_server(language, workspace) is None


def test_external_npm_entry_cannot_point_into_workspace(tmp_path, monkeypatch):
    if os.name != "nt":
        pytest.skip("Windows npm shim boundary")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    evil = workspace / "evil.js"
    evil.write_text("evil", encoding="utf-8")
    system = tmp_path / "system"
    system_executable(system, "node")
    (system / "typescript-language-server.cmd").write_text("fixture", encoding="utf-8")
    package = system / "node_modules/typescript-language-server"
    package.mkdir(parents=True)
    (package / "package.json").write_text(
        json.dumps({"bin": {"typescript-language-server": str(evil)}}), encoding="utf-8"
    )
    monkeypatch.setenv("PATH", str(system))
    assert lsp.discover_server("typescript", workspace) is None


def test_system_symlink_to_workspace_server_is_rejected(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    local = system_executable(workspace, "gopls")
    system = tmp_path / "system"
    system.mkdir()
    try:
        (system / local.name).symlink_to(local)
    except OSError as exc:
        pytest.skip(f"Cannot create executable symlink: {exc}")
    monkeypatch.setenv("PATH", str(system))
    assert lsp.discover_server("go", workspace) is None


def test_workspace_typescript_compiler_is_not_selected(tmp_path, monkeypatch):
    package = tmp_path / "node_modules/typescript"
    (package / "lib").mkdir(parents=True)
    (package / "package.json").write_text('{"version":"6.0.3"}', encoding="utf-8")
    (package / "lib/tsserver.js").write_text("evil", encoding="utf-8")
    system_executable(tmp_path, "tsserver")
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ValueError, match="system-installed TypeScript compiler"):
        lsp._typescript_server(tmp_path)


def test_context_service_is_bound_to_worker_workspace(tmp_path):
    main = ToolContext(
        workspace=WorkspaceService(root=tmp_path),
        permission_policy=PermissionPolicy(),
        event_bus=EventBus(),
    )
    worker = ToolContext(
        workspace=WorkspaceService(root=tmp_path / "worker"),
        permission_policy=PermissionPolicy(),
        event_bus=EventBus(),
    )
    assert main.get_language_service().workspace == tmp_path
    assert worker.get_language_service().workspace == tmp_path / "worker"
    assert main.get_language_service() is not worker.get_language_service()
    main.close_language_service()
    worker.close_language_service()


def test_read_only_settings_disable_implicit_builds_and_downloads():
    assert lsp._read_only_settings("typescript")["disableAutomaticTypingAcquisition"]
    rust = lsp._read_only_settings("rust")
    assert rust["cargo"]["buildScripts"]["enable"] is False
    assert rust["procMacro"]["enable"] is False
    assert rust["checkOnSave"] is False
    assert "--frozen" in rust["cargo"]["extraArgs"]


def test_paths_stay_in_workspace_and_invalid_position_is_unavailable(service, tmp_path):
    manager = service()
    assert not manager.query("symbols", "../outside.py")["available"]
    (tmp_path / "main.py").write_text("x", encoding="utf-8")
    assert not manager.query("definition", "main.py", 2, 1)["available"]


def test_tool_is_read_only_for_all_roles_and_unavailable_is_not_failure(
    tmp_path, monkeypatch
):
    context = ToolContext(
        workspace=WorkspaceService(root=tmp_path),
        permission_policy=PermissionPolicy(mode="read-only"),
        event_bus=EventBus(tmp_path / "events.jsonl"),
    )
    (tmp_path / "main.py").write_text("hello", encoding="utf-8")
    monkeypatch.setattr(lsp, "discover_server", lambda *_: None)
    spec = BUILTIN_TOOL_REGISTRY.spec_for("code_intelligence")
    assert spec.permission == "read"
    assert spec.capabilities == {"main", "readonly_agent", "worker_agent"}
    effect = spec.effect_resolver({}, context)
    assert all(claim.access == "read" for claim in effect.resources)
    assert not effect.barrier
    assert not effects_conflict(
        effect,
        CallEffect((workspace_claim(tmp_path, ".", scope="global", access="read"),)),
    )
    assert effects_conflict(
        effect,
        CallEffect(
            (workspace_claim(tmp_path, "main.py", scope="exact", access="write"),)
        ),
    )
    result = code_intelligence("symbols", "main.py", tool_context=context)
    assert result.ok is True
    assert json.loads(result.output)["available"] is False
    assert context.workspace.change_journal.cursor() == 0
    context.close_language_service()
