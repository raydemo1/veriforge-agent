from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from harness_code_agent.core.interactive import InteractiveSession
from harness_code_agent.runtime.questions import QuestionResult

from harness_code_agent import config
from harness_code_agent.memory import extraction, reconciliation


@pytest.fixture(autouse=True)
def isolate_model_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "API_KEY", "test-key")
    monkeypatch.setattr(config, "BASE_URL", "https://model.example.invalid/v1")
    monkeypatch.setattr(config, "ROUTER_API_KEY", "")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "")
    monkeypatch.setenv("SILICONFLOW_BASE_URL", "https://embedding.example.invalid/v1")
    monkeypatch.setenv("HARNESS_MEMORY_ROOT", str(tmp_path / "isolated-memory"))
    def reject_memory_request(**kwargs):
        raise AssertionError("Background memory model calls require an explicit offline fixture")
    monkeypatch.setattr(extraction, "_call_model", reject_memory_request)
    monkeypatch.setattr(reconciliation, "_call_model", reject_memory_request)


def pytest_addoption(parser):
    parser.addoption(
        "--require-integration-tools",
        action="store_true",
        help="Fail instead of skipping when an integration test needs a missing tool",
    )


@pytest.fixture
def require_integration_tool(request):
    def require(available, reason):
        if available:
            return
        if request.config.getoption("--require-integration-tools"):
            pytest.fail(reason, pytrace=False)
        pytest.skip(reason)

    return require




class ModelClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=self)
        self.requests = []

    def with_options(self, **kwargs):
        return self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if hasattr(self, "gate"):
            self.started.set()
            self.gate.wait()
        if kwargs.get("stream"):
            return iter([{"choices": [{"delta": {"content": "任务完成"}, "finish_reason": "stop"}]}])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="任务完成", tool_calls=None), finish_reason="stop")], usage=None)

    def close(self):
        if hasattr(self, "gate"):
            self.gate.set()


@pytest.fixture
def runtime_factory(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_MEMORY_DISABLED", "1")
    monkeypatch.setenv("HARNESS_MEMORY_GENERATION_DISABLED", "1")
    clients = []
    def client():
        item = ModelClient()
        clients.append(item)
        return item
    runtimes = []
    with patch("harness_code_agent.agent.conversation.get_client", side_effect=client), \
         patch("harness_code_agent.core.interactive.validate_shell_configuration"), \
         patch.object(InteractiveSession, "_ensure_mcp_tools_loaded"):
        def create(**kwargs):
            item = InteractiveSession(cwd=tmp_path, profile_explicit=True, enable_turn_summary=False,
                                      question_provider=SimpleNamespace(ask=lambda _: QuestionResult(value="rewind")), **kwargs)
            runtimes.append(item)
            return item
        yield create, clients
        for item in runtimes:
            item.close()
