"""The dotenv loader must never override an explicitly exported variable.

Priority, highest first: process environment > ``.env.ab`` > ``.env`` > the
defaults in :mod:`harness_code_agent.config`.  These tests run the loader
against a temporary project root so the working copy's real .env is irrelevant.
"""
from __future__ import annotations

import os

import pytest

from harness_code_agent import config

_KEY = "HARNESS_DOTENV_PRIORITY_TEST"

#: Every variable these fixtures may load or unset.  The loader writes straight
#: into ``os.environ``, so each test needs an explicit snapshot: pytest's
#: ``monkeypatch.delenv(raising=False)`` does not register a rollback for a
#: variable that was absent, which would leak the values below into every later
#: test in the same process.
_MANAGED_KEYS = (
    _KEY,
    "HARNESS_EVAL_PROFILE",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "HARNESS_PROVIDER",
    "HARNESS_MODEL",
    "HARNESS_MODEL_FAST",
    "HARNESS_MODEL_NORMAL",
    "HARNESS_MODEL_HARD",
    "HARNESS_MODEL_MAX",
)


@pytest.fixture
def clean_env():
    saved = {name: os.environ.get(name) for name in _MANAGED_KEYS}
    for name in _MANAGED_KEYS:
        os.environ.pop(name, None)
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _write(path, text):
    path.write_text(text, encoding="utf-8")


def test_process_environment_wins_over_dotenv(tmp_path, monkeypatch, clean_env):
    monkeypatch.setenv(_KEY, "from-process")
    _write(tmp_path / ".env", f"{_KEY}=from-file\n")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)

    config._load_dotenv()

    assert os.environ[_KEY] == "from-process"


def test_dotenv_fills_variables_that_are_not_exported(tmp_path, monkeypatch, clean_env):
    _write(tmp_path / ".env", f"{_KEY}=from-file\n")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)

    config._load_dotenv()

    assert os.environ[_KEY] == "from-file"


def test_ab_overlay_wins_over_dotenv_but_not_over_the_process_environment(tmp_path, monkeypatch, clean_env):
    _write(tmp_path / ".env", f"{_KEY}=from-file\n")
    _write(
        tmp_path / ".env.ab",
        "\n".join(
            [
                f"{_KEY}=from-overlay",
                "OPENAI_API_KEY=test-key",
                "OPENAI_BASE_URL=https://example.invalid/v1",
                "HARNESS_PROVIDER=openai-compatible",
                "HARNESS_MODEL=test-model",
                "HARNESS_MODEL_FAST=fast-model",
                "HARNESS_MODEL_NORMAL=normal-model",
                "HARNESS_MODEL_HARD=hard-model",
                "HARNESS_MODEL_MAX=max-model",
            ]
        )
        + "\n",
    )
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("HARNESS_EVAL_PROFILE", "ab")

    config._load_dotenv()

    assert os.environ[_KEY] == "from-overlay"


def test_process_environment_wins_over_the_ab_overlay(tmp_path, monkeypatch, clean_env):
    _write(tmp_path / ".env", f"{_KEY}=from-file\n")
    _write(
        tmp_path / ".env.ab",
        "\n".join(
            [
                f"{_KEY}=from-overlay",
                "OPENAI_API_KEY=test-key",
                "OPENAI_BASE_URL=https://example.invalid/v1",
                "HARNESS_PROVIDER=openai-compatible",
                "HARNESS_MODEL=test-model",
                "HARNESS_MODEL_FAST=fast-model",
                "HARNESS_MODEL_NORMAL=normal-model",
                "HARNESS_MODEL_HARD=hard-model",
                "HARNESS_MODEL_MAX=max-model",
            ]
        )
        + "\n",
    )
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("HARNESS_EVAL_PROFILE", "ab")
    monkeypatch.setenv(_KEY, "from-process")

    config._load_dotenv()

    assert os.environ[_KEY] == "from-process"


def test_shell_job_durations_must_be_finite(monkeypatch):
    monkeypatch.setenv("HARNESS_SHELL_JOB_EXIT_DRAIN_SECONDS", "nan")
    with pytest.raises(ValueError):
        config._finite_seconds("HARNESS_SHELL_JOB_EXIT_DRAIN_SECONDS", 2.0)

    monkeypatch.setenv("HARNESS_SHELL_JOB_EXIT_DRAIN_SECONDS", "-1")
    with pytest.raises(ValueError):
        config._finite_seconds("HARNESS_SHELL_JOB_EXIT_DRAIN_SECONDS", 2.0)
