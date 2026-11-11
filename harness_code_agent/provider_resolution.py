"""Provider name resolution from explicit config or the endpoint host."""
from __future__ import annotations

from urllib.parse import urlsplit


def _hostname(base_url: str | None) -> str:
    """Host part of a base URL, with or without an explicit scheme."""
    value = (base_url or "").strip()
    if not value:
        return ""
    if "//" not in value:
        value = "//" + value
    try:
        return (urlsplit(value).hostname or "").lower()
    except ValueError:
        return ""


def resolve_provider_name(*, provider: str | None, base_url: str | None) -> str:
    """Resolve the provider name from explicit config or the endpoint host.

    When *provider* is set to a known value (``"openai"``, ``"deepseek"``,
    ``"openai-compatible"``), it is returned as-is.  When *provider* is ``"auto"``
    (the default), only the endpoint hostname is inspected, and only against the
    official domains.  The model name is deliberately not part of the decision:
    an OpenAI-compatible gateway can serve a model whose name contains
    ``deepseek`` while speaking the plain Chat Completions protocol, and guessing
    from the name would send it DeepSeek-only request fields.  For precision, set
    ``HARNESS_PROVIDER`` explicitly.
    """
    requested = (provider or "auto").strip().lower()
    if requested in {"openai-compatible", "compatible"}:
        return "openai-compatible"
    if requested in {"openai", "deepseek"}:
        return requested
    if requested != "auto":
        raise ValueError(f"Unsupported HARNESS_PROVIDER: {provider}")

    host = _hostname(base_url)
    if host == "openai.com" or host.endswith(".openai.com"):
        return "openai"
    if host == "deepseek.com" or host.endswith(".deepseek.com"):
        return "deepseek"
    return "openai-compatible"
