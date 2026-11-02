"""Server-scoped tool discovery with confidence tiers."""
from __future__ import annotations

import hashlib
import json

from ..mcp import MCP_TOOL_PREFIX
from ..permissions import VALID_TOOL_PERMISSIONS
from ..tool_context import ToolContext
from ..tool_registry import ToolSpec
from ..tool_result import ToolResult
from ..tool_search import SearchDocument, search_bm25

SEARCH_TOOLS_NAME = "search_tools"
HIGH_CONFIDENCE = 0.60
MEDIUM_CONFIDENCE = 0.30
NO_PROGRESS_LIMIT = 3
DEFAULT_LIMIT = 5
MAX_LIMIT = 20

_DESCRIPTION_LIMIT = 180


def search_tools(
    query: str,
    server: str | None = None,
    limit: int = DEFAULT_LIMIT,
    tool_context: ToolContext | None = None,
) -> ToolResult:
    """Search the deferred tool catalog and mount matching tool schemas.

    ``server`` narrows the search to one MCP server.  Zero-score results never
    mount a schema: they return that server's tool names and descriptions so
    the model can retry with better keywords or switch servers.
    """
    query = str(query or "").strip()
    server_name = str(server or "").strip() or None
    try:
        limit = max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))
    except (TypeError, ValueError):
        limit = DEFAULT_LIMIT

    if tool_context is None or tool_context.tool_registry is None:
        return _failure("no_tool_registry", "search_tools requires a tool registry in the current tool context.")
    if not query:
        return _failure("missing_query", "search_tools requires a non-empty query.")

    registry = tool_context.tool_registry
    allowed_permissions = (
        set(tool_context.allowed_tool_permissions)
        if tool_context.allowed_tool_permissions is not None
        else set(VALID_TOOL_PERMISSIONS)
    )
    blocked_names = set(tool_context.blocked_tool_names or set())
    revealed = set(tool_context.revealed_tool_names or set())

    deferred = [spec for spec in registry.specs() if spec.disclosure == "deferred"]
    servers = sorted({_server_of(spec) for spec in deferred if _server_of(spec)})
    if server_name is not None and server_name not in servers:
        valid = ", ".join(servers) if servers else "none"
        return _failure(
            "unknown_server",
            f"Unknown MCP server: {server_name}. Valid server ids: {valid}.",
            metadata={"query": query, "server": server_name},
        )

    catalog = [
        spec for spec in deferred if spec.name not in blocked_names and spec.permission in allowed_permissions
    ]
    spec_by_name = {spec.name: spec for spec in catalog}
    documents = [
        _document(spec)
        for spec in catalog
        if spec.name not in revealed and (server_name is None or _server_of(spec) == server_name)
    ]

    signature = _catalog_signature(catalog)
    if signature != tool_context.search_tools_signature:
        # A changed visible catalog (including a permission or profile change)
        # invalidates the previous query history; the old repeat guard must not
        # outlive the view it was recorded against.
        tool_context.search_tools_queries.clear()
        tool_context.search_tools_signature = signature
        tool_context.search_tools_no_progress = 0

    query_key = (server_name or "", " ".join(query.lower().split()), limit)
    if query_key in tool_context.search_tools_queries:
        tool_context.search_tools_no_progress += 1
        return _result(
            f"You already searched '{query}' in {_scope_label(server_name)}. Do not repeat it: "
            "use different English keywords, or search another server.\n"
            + _directory(catalog, server_name)
            + _no_progress_block(tool_context),
            metadata={
                "query": query,
                "server": server_name,
                "level": "repeated",
                "revealed_tool_names": [],
                "no_progress": tool_context.search_tools_no_progress,
            },
        )
    tool_context.search_tools_queries.add(query_key)

    if not documents:
        tool_context.search_tools_no_progress += 1
        return _result(
            f"No unmounted tools remain in {_scope_label(server_name)}. Proceed with the mounted tools, "
            "or search another server.\n"
            + _directory(catalog, server_name)
            + _no_progress_block(tool_context),
            metadata={
                "query": query,
                "server": server_name,
                "level": "none",
                "revealed_tool_names": [],
                "no_progress": tool_context.search_tools_no_progress,
            },
        )

    hits = search_bm25(documents, query, limit=limit)
    if not hits:
        tool_context.search_tools_no_progress += 1
        return _result(
            f"No match for '{query}' in {_scope_label(server_name)} (coverage 0.00). "
            "No tool schema was mounted, so calling an unmounted tool will fail.\n"
            + _directory(catalog, server_name)
            + "\nRetry with different English keywords, or search another server if the capability "
            "lives elsewhere. If no server has it, tell the user you cannot do it."
            + _no_progress_block(tool_context),
            metadata={
                "query": query,
                "server": server_name,
                "coverage": 0.0,
                "level": "none",
                "revealed_tool_names": [],
                "no_progress": tool_context.search_tools_no_progress,
            },
        )

    coverage = hits[0].coverage
    level = _confidence_level(coverage)
    note = {
        "high": "",
        "medium": " (medium confidence: verify before using)",
        "low": " (LOW confidence: mounted anyway for inspection; verify it really fits this step before calling)",
    }[level]
    mounted = [hit.key for hit in hits]
    new_names = {name for name in mounted if name not in revealed}
    if new_names:
        tool_context.search_tools_no_progress = 0

    listing = "\n".join(
        f"- {_tool_of(spec_by_name[hit.key])}: {_one_line(hit.metadata.get('description'))}"
        for hit in hits
        if hit.key in spec_by_name
    )
    if new_names:
        heading = f"Mounted {len(new_names)} new tool(s) from {_scope_label(server_name)}{note}:"
    else:
        heading = (
            f"No new tools: all {len(mounted)} match(es) from {_scope_label(server_name)} were "
            "already mounted. Change the query, or proceed with the mounted tools."
        )

    return _result(
        f"{heading}\n{listing}" + _no_progress_block(tool_context),
        metadata={
            "query": query,
            "server": server_name,
            "coverage": coverage,
            "level": level,
            "revealed_tool_names": mounted,
            "no_progress": tool_context.search_tools_no_progress,
        },
    )


def _catalog_signature(deferred: list[ToolSpec]) -> tuple[str, ...]:
    """Fingerprint the deferred catalog with fields that affect discovery.

    Names alone are insufficient: a permission change (for example
    ``dangerous`` -> ``read``) can make a tool usable under a profile without
    changing its schema, and a reconnect can change the schema while keeping
    the name. Both invalidate the repeat-query guard.
    """
    return tuple(
        hashlib.sha256(
            json.dumps(
                {
                    "name": spec.name,
                    "permission": spec.permission,
                    "origin_server": spec.origin_server,
                    "origin_tool": spec.origin_tool,
                    "schema": spec.schema,
                },
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:16]
        for spec in sorted(deferred, key=lambda item: item.name)
    )


def _confidence_level(coverage: float) -> str:
    if coverage >= HIGH_CONFIDENCE:
        return "high"
    if coverage >= MEDIUM_CONFIDENCE:
        return "medium"
    return "low"


def _no_progress_block(context: ToolContext) -> str:
    if context.search_tools_no_progress < NO_PROGRESS_LIMIT:
        return ""
    return (
        f"\n\nNo usable tool has been found in the last {context.search_tools_no_progress} searches. "
        "Consider a different approach, or tell the user this step is not supported. "
        "You may still search again later for other steps of the task."
    )


def _scope_label(server_name: str | None) -> str:
    return f"server '{server_name}'" if server_name else "the catalog"


def _directory(catalog: list[ToolSpec], server_name: str | None) -> str:
    selected = [
        spec for spec in catalog if server_name is None or _server_of(spec) == server_name
    ]
    if not selected:
        return "No tools are available in this catalog."
    lines = ["Available tools (names and descriptions only):"]
    grouped: dict[str, list[ToolSpec]] = {}
    for spec in selected:
        grouped.setdefault(_server_of(spec) or "", []).append(spec)
    for name in sorted(grouped):
        if name and server_name is None:
            lines.append(f"[{name}]")
        for spec in grouped[name]:
            description = _one_line(_description_of(spec))
            lines.append(f"- {_tool_of(spec)}: {description}")
    return "\n".join(lines)


def _document(spec: ToolSpec) -> SearchDocument:
    description = _description_of(spec)
    parameters = spec.schema.get("function", {}).get("parameters") if isinstance(spec.schema, dict) else {}
    text = " ".join(
        part
        for part in [
            spec.name.replace("_", " "),
            description,
            _schema_parameter_text(parameters if isinstance(parameters, dict) else {}),
        ]
        if part
    )
    return SearchDocument(
        key=spec.name,
        text=text,
        metadata={
            "description": description,
            "server": _server_of(spec),
            "tool": _tool_of(spec),
            "permission": spec.permission,
        },
    )


def _description_of(spec: ToolSpec) -> str:
    function = spec.schema.get("function", {}) if isinstance(spec.schema, dict) else {}
    return str(function.get("description") or "")


def _schema_parameter_text(schema: dict) -> str:
    parts: list[str] = []
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, raw in properties.items():
            parts.append(str(name).replace("_", " "))
            if isinstance(raw, dict):
                description = raw.get("description")
                if description:
                    parts.append(str(description))
                nested = raw.get("items")
                if isinstance(nested, dict):
                    parts.append(_schema_parameter_text(nested))
    return " ".join(part for part in parts if part)


def _server_of(spec: ToolSpec) -> str | None:
    if spec.origin_server:
        return spec.origin_server
    name = spec.name
    if not name.startswith(MCP_TOOL_PREFIX):
        return None
    remainder = name[len(MCP_TOOL_PREFIX):]
    if "__" not in remainder:
        return None
    return remainder.split("__", 1)[0] or None


def _tool_of(spec: ToolSpec) -> str:
    if spec.origin_tool:
        return spec.origin_tool
    name = spec.name
    if not name.startswith(MCP_TOOL_PREFIX):
        return name
    remainder = name[len(MCP_TOOL_PREFIX):]
    return remainder.split("__", 1)[1] if "__" in remainder else remainder


def _one_line(value: object) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > _DESCRIPTION_LIMIT:
        text = text[: _DESCRIPTION_LIMIT - 3] + "..."
    return text


def _failure(reason: str, message: str, *, metadata: dict | None = None) -> ToolResult:
    return ToolResult(
        tool=SEARCH_TOOLS_NAME,
        status="failed",
        output=f"[error] {message}",
        error=reason,
        metadata={"status_source": "validation", **(metadata or {})},
    )


def _result(output: str, *, metadata: dict) -> ToolResult:
    return ToolResult(
        tool=SEARCH_TOOLS_NAME,
        status="success",
        output=output,
        metadata={"status_source": "native", **metadata},
    )
