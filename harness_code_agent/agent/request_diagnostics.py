"""Content-free diagnostics captured at the actual transport boundary."""

from __future__ import annotations

import time

from .history import fingerprint


def _shape(value):
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_shape(item) for item in value]
    return {
        "hash": fingerprint(value),
        "length": len(value) if isinstance(value, str) else None,
    }


def _first_difference(old, new, path=""):
    if old == new:
        return None
    if (
        isinstance(old, dict)
        and isinstance(new, dict)
        and "hash" not in old
        and "hash" not in new
    ):
        for key in sorted(old.keys() | new.keys()):
            found = _first_difference(
                old.get(key), new.get(key), f"{path}.{key}".lstrip(".")
            )
            if found:
                return found
    if isinstance(old, list) and isinstance(new, list):
        for index in range(max(len(old), len(new))):
            found = _first_difference(
                old[index] if index < len(old) else None,
                new[index] if index < len(new) else None,
                f"{path}[{index}]",
            )
            if found:
                return found
    return path or "request"


class RequestDiagnostics:
    def __init__(self):
        self.previous = None
        self.attempt = 0
        self._request_attempts = {}

    def begin(self, kwargs: dict, *, provider: str, epoch=None, request_id="") -> dict:
        self.attempt += 1
        self._request_attempts[request_id] = (
            self._request_attempts.get(request_id, 0) + 1
        )
        context = {
            key: kwargs.get(key)
            for key in (
                "model",
                "tools",
                "tool_choice",
                "messages",
                "extra_body",
                "reasoning_effort",
            )
        }
        current = _shape(context)
        messages = current.get("messages", [])
        previous_messages = (self.previous or {}).get("messages", [])
        common = 0
        for old, new in zip(previous_messages, messages):
            if old != new:
                break
            common += 1
        record = {
            "request_id": request_id,
            "attempt": self._request_attempts[request_id],
            "request_sequence": self.attempt,
            "provider": provider,
            "model": kwargs.get("model"),
            "epoch_id": getattr(epoch, "id", None),
            "epoch_reason": getattr(epoch, "reason", None),
            "contract_hash": getattr(epoch, "contract", None),
            "request_hash": fingerprint(context),
            "tools_hash": fingerprint(kwargs.get("tools")),
            "messages": messages,
            "common_message_prefix": common,
            "first_difference": _first_difference(self.previous, current)
            if self.previous
            else None,
            "started": time.perf_counter(),
        }
        self.previous = current
        return record

    @staticmethod
    def finish(record: dict, *, usage=None, error=None):
        result = dict(record)
        result["latency_ms"] = round(
            (time.perf_counter() - result.pop("started")) * 1000
        )
        result["prompt_tokens"] = (usage or {}).get("prompt_tokens")
        result["completion_tokens"] = (usage or {}).get("completion_tokens")
        result["total_tokens"] = (usage or {}).get("total_tokens")
        result["cache_hit_tokens"] = (usage or {}).get("cache_hit_tokens")
        result["cache_miss_tokens"] = (usage or {}).get("cache_miss_tokens")
        result["error_status"] = getattr(error, "status_code", None)
        result["error_type"] = type(error).__name__ if error else None
        return result
