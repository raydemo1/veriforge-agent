from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from harness_code_agent.runtime.middleware.verification import StaticVerifierMiddleware
from harness_code_agent.runtime.verification import (
    CheckResult,
    VerificationEngine,
    languages,
)
from harness_code_agent.runtime.verification.languages import (
    GoProvider,
    RustProvider,
    TypeScriptProvider,
)
from harness_code_agent.sessions.events import EventBus
from harness_code_agent.workspace.service import WorkspaceService


def write(root, path, text=""):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def test_non_git_shell_change_is_turn_local(tmp_path, monkeypatch):
    write(tmp_path, "old.py", "def broken(\n")
    write(tmp_path, "edit.py", "x = 1\n")
    middleware = StaticVerifierMiddleware(workspace_root=str(tmp_path))
    monkeypatch.setattr(
        "harness_code_agent.runtime.verification.python._check_ruff",
        lambda *_: ([], False),
    )
    state = SimpleNamespace(verification=None)
    middleware.after_turn(runtime_state=state)
    assert state.verification is None
    write(tmp_path, "edit.py", "def broken(\n")
    middleware.after_turn(runtime_state=state)
    assert state.verification["files"] == ["edit.py"]
    syntax = state.verification["checks"][0]
    assert syntax["status"] == "failed"
    assert "edit.py" in syntax["detail"]
    assert "old.py" not in syntax["detail"]


@pytest.mark.parametrize("language", ["eslint", "go", "rust"])
def test_automatic_verification_never_executes_project_code(
    tmp_path, monkeypatch, language
):
    script = (
        "from pathlib import Path; Path('executed').write_text('yes', encoding='utf-8')"
    )
    if language == "eslint":
        write(tmp_path, "package.json", '{"devDependencies":{"eslint":"*"}}')
        write(tmp_path, "runner", script)
        write(tmp_path, "main.js", "const value = 1;")
        monkeypatch.setattr(
            languages,
            "node_command",
            lambda *_: [sys.executable, str(tmp_path / "runner")],
        )
        provider, files = TypeScriptProvider(), ["main.js"]
    elif language == "go":
        write(tmp_path, "go.mod", "module test")
        write(tmp_path, "main.go", "package main")
        write(tmp_path, "test", script)
        monkeypatch.setattr(
            languages.shutil,
            "which",
            lambda name: sys.executable if name == "go" else None,
        )
        provider, files = GoProvider(), ["main.go"]
    else:
        write(tmp_path, "Cargo.toml", '[package]\nname="example"')
        write(tmp_path, "lib.rs", "fn main() {}")
        write(tmp_path, "check", script)
        monkeypatch.setattr(
            languages.shutil,
            "which",
            lambda name: sys.executable if name == "cargo" else None,
        )
        provider, files = RustProvider(), ["lib.rs"]
    results = provider.verify(files, tmp_path)
    assert not (tmp_path / "executed").exists()
    assert any(
        result.status == "skipped" and "run_bash" in " ".join(result.details)
        for result in results
    )


def test_deleted_files_are_dispatched_but_removed_python_is_not_parsed(tmp_path):
    write(tmp_path, "old.py", "x = 1")
    middleware = StaticVerifierMiddleware(workspace_root=str(tmp_path))
    (tmp_path / "old.py").unlink()
    state = SimpleNamespace()
    middleware.after_turn(runtime_state=state)
    assert state.verification["files"] == ["old.py"]
    assert state.verification["checks"] == []


def test_skipped_tool_published_without_blocking(tmp_path, monkeypatch):
    write(tmp_path, "tsconfig.json", "{}")
    middleware = StaticVerifierMiddleware(workspace_root=str(tmp_path))
    write(tmp_path, "main.ts", "const x = 1;")
    monkeypatch.setattr(languages, "node_command", lambda *_: None)
    events = []
    state = SimpleNamespace(
        event_bus=SimpleNamespace(emit=lambda *a, **kw: events.append(kw["payload"]))
    )
    middleware.after_turn(runtime_state=state)
    assert events[0]["checks"][0]["status"] == "skipped"


def test_project_execution_skip_records_command_without_blocking_exit(tmp_path):
    write(tmp_path, "Cargo.toml", "[package]")
    middleware = StaticVerifierMiddleware(workspace_root=str(tmp_path))
    write(tmp_path, "lib.rs", "fn main() {}")
    state = SimpleNamespace()
    middleware.after_turn(runtime_state=state)
    check = state.verification["checks"][0]
    assert check["status"] == "skipped"
    assert "run_bash" in check["detail"]
    assert '"cargo", "check"' in check["detail"]


@pytest.mark.parametrize("restore_original", [False, True])
def test_workspace_edit_during_diagnostics_marks_evidence_stale(tmp_path, monkeypatch, restore_original):
    workspace = WorkspaceService(root=tmp_path, session_id="verifying-session")
    other = WorkspaceService(root=tmp_path, session_id="writing-session")
    middleware = StaticVerifierMiddleware(workspace=workspace)
    write(tmp_path, "app.py", "x = 1\n")
    state = SimpleNamespace(event_bus=EventBus())

    def verify(*_):
        original = other.read_text("app.py")
        other.write_text("app.py", "x = 2\n")
        if restore_original:
            other.write_text("app.py", original)
        return [CheckResult("Python syntax", "passed")]

    monkeypatch.setattr(middleware.engine, "verify", verify)
    middleware.after_turn(runtime_state=state)

    assert state.verification_stale is True
    [event] = state.event_bus.events
    assert event.type == "checks_recorded"
    assert event.payload["stale"] is True


def test_stop_after_tool_persists_diagnostics_with_memory_disabled(runtime_factory, monkeypatch):
    create, _ = runtime_factory
    monkeypatch.setenv("HARNESS_PERMISSION_MODE", "workspace-write")
    monkeypatch.setattr(
        "harness_code_agent.runtime.verification.python._check_ruff", lambda *_: ([], False)
    )
    session = create(profile_name="coding-agent")
    session.ensure_profile_bound_for_first_task("edit a source file")
    requests = []

    def reply(*_, **__):
        requests.append(1)
        return {
            "role": "assistant", "content": "done",
            "tool_calls": [{"id": "edit", "type": "function", "function": {
                "name": "write_file", "arguments": json.dumps({"path": "broken.py", "content": "def broken(\n"}),
            }}],
        }, "stop"

    monkeypatch.setattr(session.conversation.llm, "request_assistant_message", reply)
    assert session.submit("edit a source file").text == "done"
    assert len(requests) == 1
    events = session.session_store.read_events(session.session_id)
    [record] = [event for event in events if event["type"] == "checks_recorded"]
    assert record["payload"]["checks"][0]["status"] == "failed"
    assert record["payload"]["stale"] is False
    [entry] = [entry for entry in session.conversation.journal.read() if entry.kind == "verification"]
    assert entry.payload["outcome"] == "failed"
    assert entry.payload["auto_extract"] is False


def test_local_typescript_and_changed_file_eslint(tmp_path, monkeypatch):
    write(tmp_path, "app/package.json", '{"devDependencies":{"eslint":"*"}}')
    write(tmp_path, "app/tsconfig.json", "{}")
    write(tmp_path, "app/main.ts", "const x=1")
    write(tmp_path, "app/old.ts", "const old=1")
    write(tmp_path, "node_modules/typescript/bin/tsc")
    write(tmp_path, "node_modules/eslint/bin/eslint.js")
    monkeypatch.setattr(
        languages.shutil, "which", lambda name: "node" if name == "node" else None
    )
    calls = []

    def check(name, command, cwd, **kw):
        calls.append((name, command, cwd))
        return CheckResult(name, "passed")

    monkeypatch.setattr(languages, "command_check", check)
    results = TypeScriptProvider().verify(["app/main.ts"], tmp_path)
    assert len(calls) == 1
    assert calls[0][1][1] == str(tmp_path / "node_modules/typescript/bin/tsc")
    assert "--noEmit" in calls[0][1]
    assert results[1].status == "skipped"
    argv = json.loads(results[1].details[0].split("; argv: ", 1)[1])
    assert argv[:2] == ["node", str(tmp_path / "node_modules/eslint/bin/eslint.js")]
    assert "main.ts" in results[1].details[0]
    assert "old.ts" not in results[1].details[0]
    assert all(call[2] == tmp_path / "app" for call in calls)


def test_javascript_without_config_does_not_invoke_unrelated_tools(
    tmp_path, monkeypatch
):
    write(tmp_path, "package.json", "{}")
    write(tmp_path, "main.js")
    monkeypatch.setattr(
        languages, "command_check", lambda *_a, **_kw: pytest.fail("unexpected command")
    )
    assert TypeScriptProvider().verify(["main.js"], tmp_path)[0].status == "skipped"


def test_missing_eslint_does_not_offer_an_unresolved_command(tmp_path, monkeypatch):
    write(tmp_path, "package.json", '{"devDependencies":{"eslint":"*"}}')
    write(tmp_path, "main.js", "const value = 1;")
    monkeypatch.setattr(languages, "node_command", lambda *_: None)
    results = TypeScriptProvider().verify(["main.js"], tmp_path)
    assert results[0].status == "skipped"
    assert "not installed" in results[0].details[0]
    assert "argv:" not in results[0].details[0]


def test_go_checks_only_changed_packages_in_each_module(tmp_path, monkeypatch):
    write(tmp_path, "go.mod", "module test")
    write(tmp_path, "a/main.go")
    write(tmp_path, "b/main.go")
    write(tmp_path, "nested/go.mod", "module nested")
    write(tmp_path, "nested/c/main.go")
    monkeypatch.setattr(languages.shutil, "which", lambda name: name)
    calls = []

    def check(name, command, cwd, **kw):
        calls.append((name, command, cwd, kw))
        return CheckResult(name, "passed")

    monkeypatch.setattr(languages, "command_check", check)
    results = GoProvider().verify(["a/main.go", "nested/c/main.go"], tmp_path)
    assert calls[0][1] == ["gofmt", "-d", "a/main.go", "nested/c/main.go"]
    assert calls[0][3]["diff"]
    assert len(calls) == 1
    assert "./a" in results[1].details[0]
    assert "./c" in results[2].details[0]
    assert "nested" in results[2].details[0]
    assert all(r.status == "skipped" for r in results[1:])


def test_rust_check_uses_nearest_manifest(tmp_path, monkeypatch):
    write(tmp_path, "crates/a/Cargo.toml", "[package]")
    write(tmp_path, "crates/a/src/lib.rs")
    monkeypatch.setattr(languages.shutil, "which", lambda name: name)
    calls = []

    def check(name, command, cwd, **kw):
        calls.append((command, cwd))
        return CheckResult(name, "passed")

    monkeypatch.setattr(languages, "command_check", check)
    results = RustProvider().verify(["crates/a/src/lib.rs"], tmp_path)
    assert calls == []
    assert results[0].status == "skipped"
    assert "cargo" in results[0].details[0]
    assert "crates/a" in results[0].details[0]


@pytest.mark.parametrize(
    "code,diff,status",
    [(0, False, "passed"), (1, False, "failed"), (0, True, "failed")],
)
def test_real_command_status_and_diff(tmp_path, code, diff, status):
    command = [sys.executable, "-c", f"print('result');raise SystemExit({code})"]
    result = languages.command_check("test", command, tmp_path, diff=diff)
    assert result.status == status


def test_missing_and_timeout_are_skipped(tmp_path):
    assert languages.command_check("test", None, tmp_path).status == "skipped"
    assert (
        languages.command_check(
            "test",
            [sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path,
            timeout=0.1,
        ).status
        == "skipped"
    )


def test_engine_dispatches_each_matching_provider_once(tmp_path):
    engine = VerificationEngine(
        [
            SimpleNamespace(
                supports=lambda *_: True,
                verify=lambda *_: [CheckResult("fixture", "passed")],
            )
        ]
    )
    assert [check.name for check in engine.verify(["main.ts"], tmp_path)] == ["fixture"]


def test_eslint_inherited_by_nested_package(tmp_path, monkeypatch):
    write(tmp_path, "package.json", '{"devDependencies":{"eslint":"*"}}')
    write(tmp_path, "nested/package.json", "{}")
    write(tmp_path, "nested/main.js")
    calls = []
    monkeypatch.setattr(languages, "node_command", lambda *_: ["eslint"])

    def check(name, command, cwd, **kw):
        calls.append(cwd)
        return CheckResult(name, "passed")

    monkeypatch.setattr(languages, "command_check", check)
    results = TypeScriptProvider().verify(["nested/main.js"], tmp_path)
    assert calls == []
    assert results[0].status == "skipped"
    assert "run_bash" in results[0].details[0]


def test_typescript_configuration_only_change_is_verified(tmp_path, monkeypatch):
    write(tmp_path, "tsconfig.json", "{}")
    provider = TypeScriptProvider()
    monkeypatch.setattr(languages, "node_command", lambda *_: None)
    assert provider.supports(["tsconfig.json"], tmp_path)
    assert provider.verify(["tsconfig.json"], tmp_path)[0].status == "skipped"


def test_removing_last_go_file_does_not_test_nonexistent_package(tmp_path, monkeypatch):
    write(tmp_path, "go.mod", "module test")
    write(tmp_path, "gone/main.go")
    (tmp_path / "gone/main.go").unlink()
    monkeypatch.setattr(
        languages,
        "command_check",
        lambda *_a, **_kw: pytest.fail("empty package must not run"),
    )
    assert GoProvider().verify(["gone/main.go"], tmp_path) == []


def test_mixed_configured_and_unconfigured_projects_have_separate_evidence(
    tmp_path, monkeypatch
):
    write(tmp_path, "app/tsconfig.json", "{}")
    write(tmp_path, "app/main.ts")
    write(tmp_path, "standalone/main.ts")
    monkeypatch.setattr(languages, "node_command", lambda *_: ["tsc"])
    monkeypatch.setattr(
        languages, "command_check", lambda name, *_a, **_kw: CheckResult(name, "passed")
    )
    results = TypeScriptProvider().verify(
        ["app/main.ts", "standalone/main.ts"], tmp_path
    )
    assert [r.status for r in results] == ["passed", "skipped"]
    assert len({r.name for r in results}) == 2
