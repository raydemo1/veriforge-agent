"""Shared utilities for the agent package."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conversation import Agent


def _get(value, key: str, default=None):
    """Get an attribute/key from either an object or a dict."""
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _usage_to_dict(usage) -> dict | None:
    """Convert an OpenAI Usage object (or dict) to a plain dict."""
    if usage is None:
        return None
    prompt_tokens = _get(usage, "prompt_tokens")
    completion_tokens = _get(usage, "completion_tokens")
    total_tokens = _get(usage, "total_tokens")
    details = _get(usage, "prompt_tokens_details") or {}
    cache_hit_tokens = _usage_int(usage, "prompt_cache_hit_tokens")
    if cache_hit_tokens == 0:
        cache_hit_tokens = _int_or_zero(_get(details, "cached_tokens", 0))
    cache_miss_tokens = _usage_int(usage, "prompt_cache_miss_tokens")
    if cache_miss_tokens == 0 and cache_hit_tokens and prompt_tokens is not None:
        try:
            cache_miss_tokens = max(0, int(prompt_tokens) - cache_hit_tokens)
        except (TypeError, ValueError):
            cache_miss_tokens = 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cache_hit_tokens,
        "cache_hit_tokens": cache_hit_tokens,
        "cache_miss_tokens": cache_miss_tokens,
    }


@dataclass(frozen=True)
class PromptCacheShape:
    system_hash: str
    memory_index_hash: str
    tools_hash: str
    prefix_hash: str
    log_rewrite_version: int
    tool_schema_tokens: int

    def to_dict(self) -> dict:
        return {
            "system_hash": self.system_hash,
            "memory_index_hash": self.memory_index_hash,
            "tools_hash": self.tools_hash,
            "prefix_hash": self.prefix_hash,
            "log_rewrite_version": self.log_rewrite_version,
            "tool_schema_tokens": self.tool_schema_tokens,
        }


def capture_prompt_cache_shape(
    agent: Agent,
    tool_schemas: list[dict] | None,
    *,
    log_rewrite_version: int = 0,
) -> PromptCacheShape:
    """Snapshot the stable request prefix inputs used for cache diagnostics."""
    tool_text = json.dumps(
        _canonical_tool_schemas(tool_schemas or []),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    system_prompt = getattr(agent, "system_prompt", "")
    memory_index = getattr(agent, "memory_index", None) or ""
    return PromptCacheShape(
        system_hash=_short_hash(system_prompt),
        memory_index_hash=_short_hash(memory_index),
        tools_hash=_short_hash(tool_text),
        prefix_hash=_short_hash(
            json.dumps(
                {"system": system_prompt, "tools": tool_text},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        log_rewrite_version=int(log_rewrite_version),
        tool_schema_tokens=max(0, len(tool_text) // 4),
    )


def compare_prompt_cache_shapes(
    previous: PromptCacheShape | None,
    current: PromptCacheShape,
    usage: dict | None,
) -> dict:
    reasons: list[str] = []
    if previous is not None:
        if previous.system_hash != current.system_hash:
            reasons.append("system")
        if previous.memory_index_hash != current.memory_index_hash:
            reasons.append("memory_index")
        if previous.tools_hash != current.tools_hash:
            reasons.append("tools")
        if previous.log_rewrite_version != current.log_rewrite_version:
            reasons.append("log_rewrite")
    return {
        **current.to_dict(),
        "prefix_changed": bool(reasons),
        "prefix_change_reasons": reasons,
        "cache_hit_tokens": int((usage or {}).get("cache_hit_tokens") or (usage or {}).get("cached_tokens") or 0),
        "cache_miss_tokens": int((usage or {}).get("cache_miss_tokens") or 0),
    }


def _prompt_cache_key(
    agent: Agent,
    tool_schemas: list[dict] | None,
    *,
    model: str,
) -> str:
    tool_text = json.dumps(
        _canonical_tool_schemas(tool_schemas or []),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    stable_prefix = getattr(agent, "prompt_cache_identity", None) or {
        "system_prompt_hash": _short_hash(agent.system_prompt),
    }
    payload = {
        "agent": agent.name,
        "model": model,
        "stable_prefix": stable_prefix,
        "tools_hash": _short_hash(tool_text),
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "hca:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:48]


def _canonical_tool_schemas(tool_schemas: list[dict]) -> list[dict]:
    return sorted(
        (_canonical_schema_value(schema) for schema in tool_schemas),
        key=lambda schema: str(schema.get("function", {}).get("name") or ""),
    )


def _canonical_schema_value(value, *, parent_key: str = ""):
    if isinstance(value, dict):
        return {
            key: _canonical_schema_value(item, parent_key=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        items = [_canonical_schema_value(item, parent_key=parent_key) for item in value]
        if parent_key == "required" and all(isinstance(item, str) for item in items):
            return sorted(items)
        return items
    return value


def _usage_int(usage, key: str) -> int:
    value = _get(usage, key, None)
    if value is None:
        extra = _get(usage, "model_extra") or {}
        value = _get(extra, key, 0)
    return _int_or_zero(value)


def _int_or_zero(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
