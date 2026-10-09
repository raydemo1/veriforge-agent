from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from eval.benchmarks import hca_claw_runner, hca_terminal_runner
from harness_code_agent import cli, config
from harness_code_agent.core.interactive import InteractiveSession
from harness_code_agent.headless import run_task
from harness_code_agent.runtime.approvals import ApprovalRequest
from harness_code_agent.runtime.questions import QuestionOption, QuestionRequest


@pytest.fixture(autouse=True)
def isolate_runner_environment():
    with patch.dict(os.environ):
        yield


@pytest.fixture
def model(monkeypatch, tmp_path):
    harness_root = tmp_path / ".harness"
    harness_root.mkdir()
    (harness_root / "mcp.json").write_text('{"servers": {}}', encoding="utf-8")
    monkeypatch.delenv("HARNESS_MCPORTER_CONFIG", raising=False)
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="Task response", tool_calls=None),
            finish_reason="stop",
        )],
        usage=None,
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=response))),
        close=Mock(),
    )
    monkeypatch.setattr("harness_code_agent.agent.conversation.get_client", lambda: client)
    monkeypatch.setattr("harness_code_agent.core.logging_config.setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(config, "STREAM", "0")
    monkeypatch.setenv("HARNESS_MEMORY_DISABLED", "1")
    monkeypatch.setenv("HARNESS_MEMORY_GENERATION_DISABLED", "1")
    monkeypatch.setenv("HARNESS_MENTION_MODE", "off")
    monkeypatch.setenv("HARNESS_PERMISSION_MODE", "workspace-write")
    monkeypatch.setenv("HCA_ARTIFACT_EXPORT_INTERVAL_SECONDS", "0")
    monkeypatch.setattr(hca_terminal_runner.atexit, "register", Mock())
    monkeypatch.setattr(hca_terminal_runner.signal, "signal", Mock())
    return client


def test_run_task_records_session_before_execution_and_keeps_recovery(tmp_path, model):
    started = []

    def on_started(info):
        assert model.chat.completions.create.call_count == 0
        assert (info.harness_root / "sessions" / info.session_id / "session.json").is_file()
        started.append(info)

    result = run_task(cwd=tmp_path, task="Inspect the code", profile="coding-agent", on_session_started=on_started)

    assert result.status == "completed"
    assert result.exit_code == 0
    assert result.turn_result.text == "Task response"
    assert result.error == ""
    assert result.session_id == started[0].session_id
    assert result.harness_root == started[0].harness_root
    metadata = json.loads((result.harness_root / "sessions" / result.session_id / "session.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "closed"
    assert list((result.harness_root / "recovery" / "points").glob("*.json"))
    model.close.assert_called_once()


def test_empty_task_fails_without_starting_session(tmp_path, model):
    result = run_task(cwd=tmp_path, task="  ")
    assert result.status == "failed"
    assert result.session_id is None
    assert "No task provided" in result.error
    assert not (tmp_path / ".harness" / "sessions").exists()
    model.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("entry", ["cli", "terminal", "claw"])
@pytest.mark.parametrize("failure", [False, True])
def test_entrypoints_execute_real_session_and_report_failures(tmp_path, model, monkeypatch, capsys, entry, failure):
    monkeypatch.chdir(tmp_path)
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("HCA_ARTIFACTS_ROOT", str(artifacts))
    if failure:
        model.chat.completions.create.side_effect = RuntimeError("model request failed")

    if entry == "cli":
        exit_code = cli.main(["-p", "--profile", "coding-agent", "Inspect the code"])
    elif entry == "terminal":
        exit_code = hca_terminal_runner.main(["--workspace", str(tmp_path), "Inspect the code"])
    else:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Inspect the code", encoding="utf-8")
        exit_code = hca_claw_runner.main([str(prompt_file), "--workspace", str(tmp_path)])

    output = capsys.readouterr()
    assert exit_code == (1 if failure else 0)
    assert model.chat.completions.create.call_count >= 1
    model.close.assert_called_once()
    session_id = re.search(r"veriforge session: (\S+)", output.out).group(1)
    assert (tmp_path / ".harness" / "sessions" / session_id / "session.json").is_file()
    if failure:
        assert "model request failed" in output.err
    else:
        assert "Task response" in output.out
    if entry == "terminal":
        export_root = artifacts / "hca" / session_id
        assert (export_root / "early_manifest.json").is_file()
        assert (export_root / "manifest.json").is_file()
        assert "HCA_EVAL_METRICS:" in output.out
        if failure:
            assert "model request failed" in (export_root / "runner_error.txt").read_text(encoding="utf-8")


def test_terminal_manifest_is_written_before_model_request(tmp_path, model, monkeypatch):
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("HCA_ARTIFACTS_ROOT", str(artifacts))

    def create(**kwargs):
        assert len(list(artifacts.glob("hca/*/early_manifest.json"))) == 1
        raise RuntimeError("stop after checking early artifacts")

    model.chat.completions.create.side_effect = create
    assert hca_terminal_runner.main(["--workspace", str(tmp_path), "Inspect the code"]) == 1


def test_terminal_signal_exports_partial_session_and_closes_runtime(tmp_path, model, monkeypatch):
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("HCA_ARTIFACTS_ROOT", str(artifacts))
    handlers = {}
    monkeypatch.setattr(hca_terminal_runner.signal, "signal", lambda sig, handler: handlers.update({sig: handler}))

    def create(**kwargs):
        signum = hca_terminal_runner.signal.SIGTERM
        handlers[signum](signum, None)

    model.chat.completions.create.side_effect = create
    with pytest.raises(SystemExit) as exc:
        hca_terminal_runner.main(["--workspace", str(tmp_path), "Inspect the code"])

    assert exc.value.code == 128 + int(hca_terminal_runner.signal.SIGTERM)
    model.close.assert_called_once()
    export_root = next(artifacts.glob("hca/*"))
    assert (export_root / "early_manifest.json").is_file()
    assert (export_root / "trajectory.jsonl").is_file()
    assert "received signal" in (export_root / "runner_error.txt").read_text(encoding="utf-8")


def test_periodic_export_reads_public_session_paths(tmp_path, model, monkeypatch):
    import threading

    started = []
    exported = threading.Event()

    result = run_task(cwd=tmp_path, task="Inspect the code", profile="coding-agent", on_session_started=started.append)
    original_export = hca_terminal_runner.export_session_artifacts
    monkeypatch.setenv("HCA_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))

    def export(**kwargs):
        try:
            return original_export(**kwargs)
        finally:
            exported.set()

    monkeypatch.setattr(hca_terminal_runner, "export_session_artifacts", export)
    state = {"session_id": started[0].session_id, "harness_root": started[0].harness_root}
    stop = hca_terminal_runner.start_periodic_artifact_export(state, interval_seconds=0.01)
    try:
        assert exported.wait(3)
    finally:
        stop.set()
    manifest = tmp_path / "artifacts" / "hca" / result.session_id / "manifest.json"
    assert manifest.is_file()


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit(143)])
def test_interrupts_close_real_session(tmp_path, model, interrupt):
    model.chat.completions.create.side_effect = interrupt
    if isinstance(interrupt, SystemExit):
        with pytest.raises(SystemExit) as exc:
            run_task(cwd=tmp_path, task="Inspect the code", profile="coding-agent")
        assert exc.value.code == 143
    else:
        result = run_task(cwd=tmp_path, task="Inspect the code", profile="coding-agent")
        assert result.status == "interrupted"
        assert result.exit_code == 130
        assert result.session_id
    model.close.assert_called_once()


@pytest.mark.parametrize("submit_fails", [False, True])
def test_close_failure_preserves_result_and_primary_error(tmp_path, model, monkeypatch, submit_fails):
    original_close = InteractiveSession.close

    def close(session):
        original_close(session)
        raise RuntimeError("session close failed")

    monkeypatch.setattr(InteractiveSession, "close", close)
    if submit_fails:
        model.chat.completions.create.side_effect = RuntimeError("primary submit error")

    result = run_task(cwd=tmp_path, task="Inspect the code", profile="coding-agent")

    assert result.status == "failed"
    assert result.exit_code == 1
    assert "session close failed" in result.error
    if submit_fails:
        assert result.error.index("primary submit error") < result.error.index("session close failed")
    else:
        assert result.turn_result.text == "Task response"
    model.close.assert_called_once()


def test_start_callback_failure_closes_session_without_submitting(tmp_path, model):
    def started(info):
        raise RuntimeError("startup callback failed")

    result = run_task(cwd=tmp_path, task="Inspect the code", profile="coding-agent", on_session_started=started)

    assert result.status == "failed"
    assert result.session_id
    assert "startup callback failed" in result.error
    model.chat.completions.create.assert_not_called()
    model.close.assert_called_once()


def test_initialization_failure_releases_already_registered_resources(tmp_path, model, monkeypatch):
    original_report = InteractiveSession._report_startup

    def report(session, stage):
        original_report(session, stage)
        if stage == "ready":
            raise RuntimeError("initialization failed")

    monkeypatch.setattr(InteractiveSession, "_report_startup", report)
    result = run_task(cwd=tmp_path, task="Inspect the code", profile="coding-agent")

    assert result.status == "failed"
    assert result.session_id is None
    assert "initialization failed" in result.error
    model.chat.completions.create.assert_not_called()
    model.close.assert_called_once()
    metadata_path = next((tmp_path / ".harness" / "sessions").glob("*/session.json"))
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["status"] == "failed"


def test_headless_does_not_read_input_for_approval_or_questions(tmp_path, model, monkeypatch):
    def forbidden_input(*args):
        pytest.fail("headless execution requested terminal input")

    monkeypatch.setattr("builtins.input", forbidden_input)
    original_submit = InteractiveSession.submit

    def submit(session, task):
        approval = session.approval_provider.request(ApprovalRequest("run_bash", {}, "risk", "reason"))
        question = session.question_provider.ask(QuestionRequest("Continue?", [QuestionOption("Yes")]))
        assert not approval.approved
        assert question.cancelled
        return original_submit(session, task)

    monkeypatch.setattr(InteractiveSession, "submit", submit)
    result = run_task(cwd=tmp_path, task="Inspect the code", profile="coding-agent")
    assert result.status == "completed"


def test_cli_print_still_handles_local_slash_commands(tmp_path, model, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["-p", "--profile", "coding-agent", "/context"]) == 0
    assert "上下文估算" in capsys.readouterr().out
    model.chat.completions.create.assert_not_called()
    model.close.assert_called_once()
