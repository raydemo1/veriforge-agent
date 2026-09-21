"""Environment forwarding helpers for Harbor-installed benchmark agents."""
from __future__ import annotations

import os
from urllib.parse import urlparse

RUNNER_ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "HARNESS_PROVIDER",
    "HARNESS_MODEL",
    "HARNESS_MODEL_INTENSITY",
    "HARNESS_MODEL_FAST",
    "HARNESS_MODEL_NORMAL",
    "HARNESS_MODEL_HARD",
    "HARNESS_MODEL_MAX",
    "MAX_AGENT_TOOL_CALLS",
    "MAX_AGENT_TOTAL_TOKENS",
    "AGENT_BUDGET_WARN_FRACTION",
    "HCA_NO_PROXY_HOSTS",
)
MODEL_API_NO_PROXY_ENV = "HCA_MODEL_API_NO_PROXY_HOSTS"


def model_api_no_proxy_hosts(base_url: str | None) -> list[str]:
    if not base_url:
        return []
    parsed = urlparse(base_url if "://" in base_url else f"https://{base_url}")
    host = parsed.hostname
    if not host:
        return []
    return [host]


def runner_env_vars() -> dict[str, str]:
    env_vars: dict[str, str] = {}
    for key in RUNNER_ENV_KEYS:
        val = os.environ.get(key)
        if val:
            env_vars[key] = val
    no_proxy_hosts = model_api_no_proxy_hosts(env_vars.get("OPENAI_BASE_URL"))
    if no_proxy_hosts:
        env_vars[MODEL_API_NO_PROXY_ENV] = ",".join(no_proxy_hosts)
    return env_vars
