from __future__ import annotations

import pytest

from harness_code_agent import config


@pytest.fixture(autouse=True)
def isolate_model_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "API_KEY", "test-key")
    monkeypatch.setattr(config, "BASE_URL", "https://model.example.invalid/v1")
    monkeypatch.setattr(config, "ROUTER_API_KEY", "")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "")
    monkeypatch.setenv("SILICONFLOW_BASE_URL", "https://embedding.example.invalid/v1")
    monkeypatch.setenv("HARNESS_MEMORY_ROOT", str(tmp_path / "isolated-memory"))


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
