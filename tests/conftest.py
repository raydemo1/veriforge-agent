from __future__ import annotations

import pytest


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
