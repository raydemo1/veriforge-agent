from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import logging
import os
import re
import threading
import time
import weakref
from collections.abc import Callable
from concurrent.futures import Future, InvalidStateError
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from anyio import BrokenResourceError, ClosedResourceError, EndOfStream
from anyio.abc import ObjectReceiveStream
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from ..support.cancellation import CancelledError
from .permissions import (
    TOOL_PERMISSION_DANGEROUS,
    TOOL_PERMISSION_NETWORK_READ,
    TOOL_PERMISSION_READ,
    VALID_TOOL_PERMISSIONS,
)
from .project_resources import ProjectResourceBudget
from .recovery import atomic_bytes
from .tool_registry import ToolSpec
from .tool_result import ToolResult

_static_lock = threading.RLock()
_config_cache = {}
_schema_cache = {}

MCP_CONFIG_RELATIVE_PATH = Path(".harness") / "mcp.json"
MCP_TOOL_PREFIX = "mcp__"
MCP_TOOL_NAME_LIMIT = 64
MCP_TRANSPORTS = ("stdio", "streamable_http")

log = logging.getLogger(__name__)


class McpConfigError(ValueError):
    """Raised when .harness/mcp.json cannot be parsed or validated."""


@dataclass(frozen=True)
class McpServerConfig:
    name: str
    transport: str
    permission: str = TOOL_PERMISSION_DANGEROUS
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    tool_permissions: dict[str, str] = field(default_factory=dict)
    title: str = ""
    description: str = ""


@dataclass(frozen=True)
class McpConfig:
    path: Path
    servers: dict[str, McpServerConfig] = field(default_factory=dict)
    server_errors: dict[str, str] = field(default_factory=dict)
    disabled_servers: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class McpToolBinding:
    exposed_name: str
    server_name: str
    tool_name: str
    description: str
    input_schema: dict[str, Any]
    permission: str
    annotations: dict[str, Any] = field(default_factory=dict)

    def schema(self) -> dict[str, Any]:
        description = self.description or f"MCP tool {self.server_name}/{self.tool_name}"
        return {
            "type": "function",
            "function": {
                "name": self.exposed_name,
                "description": description,
                "parameters": self.input_schema,
            },
        }


@dataclass(frozen=True)
class McpCatalogSnapshot:
    """The catalog and its revision captured together for Registry publication."""

    revision: int
    bindings: tuple[McpToolBinding, ...]
    statuses: dict[str, McpServerStatus]


@dataclass(frozen=True)
class McpServerStatus:
    name: str
    transport: str
    state: str
    tool_count: int = 0
    error: str | None = None
    catalog_source: str = "none"


@dataclass(frozen=True)
class PublishedMcpCatalog:
    manager: McpClientManager
    revision: int
    specs: tuple[ToolSpec, ...]
    statuses: dict[str, McpServerStatus]


@dataclass(frozen=True)
class McpRegistration:
    server: McpServerConfig
    state: str
    tool_count: int = 0
    error: str | None = None
    replaced: bool = False


@dataclass
class _McpConnection:
    config: McpServerConfig
    request_queue: asyncio.Queue
    task: asyncio.Task | None


@dataclass
class _ConnectFlight:
    task: asyncio.Task
    waiters: int = 0


def load_mcp_config(workspace: str | Path) -> McpConfig:
    root = Path(workspace).resolve()
    path = root / MCP_CONFIG_RELATIVE_PATH
    if not path.exists():
        return McpConfig(path=path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise McpConfigError(f"Invalid MCP config JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise McpConfigError("MCP config must be a JSON object")
    raw_servers = data.get("servers")
    if not isinstance(raw_servers, dict):
        raise McpConfigError("MCP config field 'servers' must be an object")

    config = McpConfig(path=path)
    for name, raw in raw_servers.items():
        if isinstance(raw, dict) and raw.get("enabled") is False:
            config.disabled_servers.add(name)
            continue
        try:
            config.servers[name] = _parse_server_entry(name, raw)
        except McpConfigError as exc:
            config.server_errors[name] = str(exc)
    return config


def _parse_server_entry(name: str, raw: Any) -> McpServerConfig:
    _validate_server_name(name)
    if not isinstance(raw, dict):
        raise McpConfigError(f"MCP server {name!r} must be an object")
    expanded = _expand_env(raw)
    transport = str(expanded.get("transport") or "").strip()
    if transport == "streamable-http":
        transport = "streamable_http"
    if transport not in MCP_TRANSPORTS:
        raise McpConfigError(f"MCP server {name!r} has unsupported transport: {transport!r}")

    permission = str(expanded.get("permission") or TOOL_PERMISSION_DANGEROUS)
    _validate_permission(permission, f"server {name!r}")
    tool_permissions = _string_dict(expanded.get("tool_permissions", {}), f"server {name!r} tool_permissions")
    for tool_name, tool_permission in tool_permissions.items():
        _validate_permission(tool_permission, f"server {name!r} tool {tool_name!r}")

    args = expanded.get("args", [])
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise McpConfigError(f"MCP server {name!r} args must be a list of strings")

    common = {
        "name": name,
        "transport": transport,
        "permission": permission,
        "tool_permissions": tool_permissions,
        "title": str(expanded.get("title") or "").strip(),
        "description": str(expanded.get("description") or "").strip(),
    }
    if transport == "stdio":
        command = expanded.get("command")
        if not isinstance(command, str) or not command.strip():
            raise McpConfigError(f"MCP stdio server {name!r} requires command")
        return McpServerConfig(
            **common,
            command=command.strip(),
            args=args,
            env=_string_dict(expanded.get("env", {}), f"server {name!r} env"),
            cwd=str(expanded.get("cwd")) if expanded.get("cwd") else None,
        )

    url = expanded.get("url")
    if not isinstance(url, str) or not url.strip():
        raise McpConfigError(f"MCP streamable_http server {name!r} requires url")
    return McpServerConfig(
        **common,
        url=url.strip(),
        headers=_string_dict(expanded.get("headers", {}), f"server {name!r} headers"),
    )


def build_server_config(
    *,
    name: str,
    url: str | None = None,
    command: str | None = None,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    headers: dict[str, str] | None = None,
    permission: str | None = None,
    title: str = "",
    description: str = "",
) -> McpServerConfig:
    """Validate one user-supplied MCP server definition for registration."""
    _validate_server_name(name)
    url = (url or "").strip()
    command = (command or "").strip()
    if bool(url) == bool(command):
        raise McpConfigError("MCP server requires exactly one of url or command")
    permission = (permission or TOOL_PERMISSION_DANGEROUS).strip()
    _validate_permission(permission, f"server {name!r}")
    if command:
        return McpServerConfig(
            name=name,
            transport="stdio",
            permission=permission,
            command=command,
            args=list(args or []),
            env=dict(env or {}),
            cwd=(cwd or "").strip() or None,
            title=title.strip(),
            description=description.strip(),
        )
    return McpServerConfig(
        name=name,
        transport="streamable_http",
        permission=permission,
        url=url,
        headers=dict(headers or {}),
        title=title.strip(),
        description=description.strip(),
    )


class McpClientManager:
    def __init__(
        self,
        *,
        workspace: str | Path,
        config: McpConfig | None = None,
        config_error: str | None = None,
        timeout_seconds: float = 30.0,
        tool_timeout_seconds: float = 120.0,
        project_resources=None,
        session_id=None,
    ):
        self.workspace = Path(workspace).resolve()
        self.project_resources = project_resources or ProjectResourceBudget.for_project(self.workspace)
        self._resource_owner = weakref.finalize(self, self.project_resources.hold())
        self.session_id = session_id
        self._state_lock = threading.RLock()
        self._catalog_only = set()
        self._closed = False
        self.config = config or McpConfig(path=self.workspace / MCP_CONFIG_RELATIVE_PATH)
        self.config_error = config_error
        self.timeout_seconds = timeout_seconds
        self.tool_timeout_seconds = tool_timeout_seconds
        self.statuses: dict[str, McpServerStatus] = {}
        self.tool_bindings: list[McpToolBinding] = []
        self._bindings_by_exposed_name: dict[str, McpToolBinding] = {}
        self._validated_specs: dict[str, tuple[tuple[McpToolBinding, ...], tuple[ToolSpec, ...]]] = {}
        self._connections: dict[str, _McpConnection] = {}
        self._connecting: dict[str, _ConnectFlight] = {}
        self._disconnecting: set[str] = set()
        self._workers: set[asyncio.Task] = set()
        self.connection_revision = 0
        self._loop_thread: _AsyncLoopThread | None = None

    @classmethod
    def from_workspace(cls, workspace: str | Path, *, project_resources=None, session_id=None) -> McpClientManager:
        try:
            path = Path(workspace).resolve() / MCP_CONFIG_RELATIVE_PATH
            info = path.stat() if path.exists() else None
            signature = (info.st_mtime_ns, info.st_size) if info else None
            with _static_lock:
                cached = _config_cache.get(path)
                if cached is None or cached[0] != signature:
                    cached = (signature, load_mcp_config(workspace))
                    _config_cache[path] = cached
                config = copy.deepcopy(cached[1])
            return cls(workspace=workspace, config=config, project_resources=project_resources, session_id=session_id)
        except (McpConfigError, OSError) as exc:
            root = Path(workspace).resolve()
            return cls(
                workspace=root,
                config=McpConfig(path=root / MCP_CONFIG_RELATIVE_PATH),
                config_error=str(exc),
                project_resources=project_resources, session_id=session_id,
            )

    def _schema_key(self, server):
        entry = _server_entry(server)
        entry.pop("title", None)
        entry.pop("description", None)
        data = {"workspace": str(self.workspace), "server": entry}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def prime_catalog(self, server_name: str | None = None) -> None:
        with self._state_lock:
            known = {binding.server_name for binding in self.tool_bindings}
            for server in self.config.servers.values():
                if server_name is not None and server.name != server_name:
                    continue
                if server.name in known or server.name in self._connections:
                    continue
                with _static_lock:
                    cached = copy.deepcopy(_schema_cache.get(self._schema_key(server), ()))
                if cached and not any(item.exposed_name in self._bindings_by_exposed_name for item in cached):
                    self._replace_bindings(server.name, cached, cached=True)

    def discover(self, server_name=None, *, cancellation_token=None) -> dict[str, str]:
        """Connect the requested server, preserving cached catalogs on failure."""
        if self.config_error:
            return {server_name or "config": self.config_error}
        self.prime_catalog(server_name)
        with self._state_lock:
            failures = dict(self.config.server_errors)
            names = list(self.config.servers)
        if server_name is not None:
            failures = {server_name: failures[server_name]} if server_name in failures else {}
            names = [server_name] if server_name in names else []
        if names:
            failures.update(self._ensure_loop().run(
                self._discover_connected(names), timeout=None, cancellation_token=cancellation_token,
            ))
        return failures

    def ensure_connected(self, server_name, *, cancellation_token=None):
        if cancellation_token is not None:
            cancellation_token.check()
        loop = self._ensure_loop()
        loop.run(self._ensure_connected(server_name), timeout=None, cancellation_token=cancellation_token)

    async def _ensure_connected(self, server_name: str) -> None:
        if self._closed:
            raise RuntimeError("MCP manager is closed")
        if server_name in self._disconnecting:
            raise RuntimeError(f"MCP server is being reconfigured: {server_name}")
        connection = self._connections.get(server_name)
        if connection is not None and connection.task is not None and not connection.task.done():
            return
        if connection is not None:
            self._on_worker_done(connection)
        if error := self.config.server_errors.get(server_name):
            raise McpConfigError(error)
        server = self.config.servers[server_name]
        flight = self._connecting.get(server_name)
        if flight is None:
            flight = _ConnectFlight(asyncio.create_task(self._connect_one_and_store(server)))
            self._connecting[server_name] = flight

            def finished(done: asyncio.Task) -> None:
                if self._connecting.get(server_name) is flight:
                    self._connecting.pop(server_name)
                if not done.cancelled():
                    done.exception()

            flight.task.add_done_callback(finished)
        flight.waiters += 1
        try:
            await asyncio.shield(flight.task)
        finally:
            flight.waiters -= 1
            if not flight.waiters and not flight.task.done():
                if self._connecting.get(server_name) is flight:
                    self._connecting.pop(server_name)
                flight.task.cancel()

    def connect_all(self) -> None:
        self.discover()

    async def _discover_connected(self, names: list[str]) -> dict[str, str]:
        results = await asyncio.gather(*(self._ensure_connected(name) for name in names), return_exceptions=True)
        return {
            name: f"{type(result).__name__}: {result}"
            for name, result in zip(names, results)
            if isinstance(result, BaseException)
        }

    async def _connect_one(
        self,
        server: McpServerConfig,
    ) -> tuple[_McpConnection, list[McpToolBinding], float]:
        """Connect to a single server with a per-server timeout."""
        async with self.project_resources.acquire_async("mcp_connect", session_id=self.session_id, label=server.name):
            deadline = time.monotonic() + self.timeout_seconds
            request_queue: asyncio.Queue = asyncio.Queue()
            ready: asyncio.Future = asyncio.get_running_loop().create_future()
            connection = _McpConnection(config=server, request_queue=request_queue, task=None)
            task = asyncio.create_task(self._connection_worker(server, request_queue, ready))
            connection.task = task
            self._workers.add(task)
            task.add_done_callback(lambda _task: self._on_worker_done(connection))
            try:
                bindings = await asyncio.wait_for(asyncio.shield(ready), timeout=self.timeout_seconds)
            except (asyncio.CancelledError, Exception) as exc:
                ready.cancel()
                task.cancel()
                if isinstance(exc, asyncio.TimeoutError):
                    raise TimeoutError(f"MCP handshake exceeded {self.timeout_seconds:g} seconds") from exc
                raise
            return connection, bindings, deadline

    def register_tools(self, registry, snapshot: McpCatalogSnapshot | None = None) -> dict[str, str]:
        view = snapshot if snapshot is not None else self.catalog_snapshot()
        groups: dict[str, list[McpToolBinding]] = {}
        for binding in view.bindings:
            groups.setdefault(binding.server_name, []).append(binding)

        failures: dict[str, str] = {}
        published = []
        names = {spec.name for spec in registry.specs() if spec.origin_server is None}
        for server_name, bindings in groups.items():
            try:
                specs = self._validated_server_specs(bindings)
                if collision := names.intersection(spec.name for spec in specs):
                    raise ValueError(f"Duplicate tool name: {min(collision)!r}")
            except (ValueError, TypeError) as exc:
                failures[server_name] = str(exc)
                continue
            published.extend(specs)
            names.update(spec.name for spec in specs)
        registry.replace_mcp_specs(published)
        return failures

    def _validated_server_specs(self, bindings: list[McpToolBinding]) -> tuple[ToolSpec, ...]:
        if not bindings:
            return ()
        server_name = bindings[0].server_name
        with self._state_lock:
            previous, specs = self._validated_specs.get(server_name, ((), ()))
        if len(previous) == len(bindings) and all(old is new for old, new in zip(previous, bindings)):
            return specs

        specs = tuple(self._server_specs(bindings))
        with self._state_lock:
            if all(self._bindings_by_exposed_name.get(item.exposed_name) is item for item in bindings):
                self._validated_specs[server_name] = (tuple(bindings), specs)
        return specs

    def _server_specs(self, bindings):
        from .execution_planner import CallEffect, ResourceClaim
        from .tool_registry import (
            TOOL_CAPABILITY_MAIN,
            TOOL_CAPABILITY_READONLY_AGENT,
            TOOL_CAPABILITY_WORKER_AGENT,
            ToolRegistry,
        )

        registry = ToolRegistry()
        for binding in bindings:
            if registry.spec_for(binding.exposed_name) is not None:
                raise ValueError(f"Duplicate tool name: {binding.tool_name!r}")
            parameters = binding.input_schema
            if not isinstance(parameters, dict) or parameters.get("type") != "object":
                raise ValueError(f"MCP tool {binding.tool_name!r} requires an object input schema")
            try:
                Draft202012Validator.check_schema(parameters)
            except SchemaError as exc:
                raise ValueError(f"MCP tool {binding.tool_name!r} has invalid input schema: {exc.message}") from exc
            if binding.annotations.get("readOnlyHint") is True:
                effect = CallEffect(
                    (ResourceClaim("mcp", binding.server_name, "exact", "read"),),
                    concurrency_key="network",
                    kind="mcp_read",
                )
            else:
                effect = CallEffect(
                    (ResourceClaim("mcp", binding.server_name, "exact", "write"),),
                    kind="mcp_server_write",
                )
            registry.register(
                binding.schema(),
                self._handler_for(binding),
                permission=binding.permission,
                effect=effect,
                disclosure="deferred",
                capabilities=(
                    {
                        TOOL_CAPABILITY_MAIN,
                        TOOL_CAPABILITY_READONLY_AGENT,
                        TOOL_CAPABILITY_WORKER_AGENT,
                    }
                    if binding.permission in {TOOL_PERMISSION_READ, TOOL_PERMISSION_NETWORK_READ}
                    else {TOOL_CAPABILITY_MAIN}
                ),
                origin_server=binding.server_name,
                origin_tool=binding.tool_name,
            )

        return registry.specs()

    def catalog_snapshot(self) -> McpCatalogSnapshot:
        """Return a versioned view of the current tool catalog."""
        with self._state_lock:
            return McpCatalogSnapshot(
                revision=self.connection_revision,
                bindings=tuple(self.tool_bindings),
                statuses=self.discovery_statuses(),
            )

    def discovery_statuses(self) -> dict[str, McpServerStatus]:
        with self._state_lock:
            counts: dict[str, int] = {}
            for binding in self.tool_bindings:
                counts[binding.server_name] = counts.get(binding.server_name, 0) + 1
            statuses = {}
            for name, server in self.config.servers.items():
                count = counts.get(name, 0)
                current = self.statuses.get(name)
                if current is not None and current.state == "connected":
                    statuses[name] = current
                    continue
                source = "cache" if count else "none"
                statuses[name] = McpServerStatus(
                    name, server.transport,
                    current.state if current else ("cached" if count else "unconnected"),
                    count, current.error if current else None, source,
                )
            statuses.update(
                (name, McpServerStatus(name, "config", "invalid", error=error))
                for name, error in self.config.server_errors.items()
            )
            statuses.update(
                (name, McpServerStatus(name, "config", "disabled"))
                for name in self.config.disabled_servers
            )
            return statuses

    def call_tool(
        self,
        exposed_name: str,
        arguments: dict[str, Any],
        *,
        cancellation_token=None,
        expected_binding: McpToolBinding | None = None,
    ) -> ToolResult:
        """Validate the binding and dispatch in one event-loop operation.

        ``expected_binding`` is the binding captured when the tool handler was
        registered. Identity is compared inside the loop immediately before
        selecting the connection, so a Registry snapshot cannot pass a check
        and then be routed to a newer binding from another connection.
        """
        try:
            return self._ensure_loop().run(
                self._call_with_connection(exposed_name, dict(arguments or {}), expected_binding),
                cancellation_token=cancellation_token,
                timeout=None,
            )
        except CancelledError:
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            return ToolResult(
                tool=exposed_name,
                status="failed",
                output=f"[error] MCP tool call failed: {error}",
                error=error,
                metadata={"status_source": "mcp"},
            )

    async def _call_with_connection(
        self,
        exposed_name: str,
        arguments: dict[str, Any],
        expected_binding: McpToolBinding | None,
    ) -> ToolResult:
        with self._state_lock:
            binding = self._bindings_by_exposed_name.get(exposed_name)
            status = self.statuses.get(binding.server_name) if binding is not None else None
            cached = exposed_name in self._catalog_only
        if binding is not None and (expected_binding is None or binding is expected_binding):
            if status is not None and status.state == "failed":
                raise RuntimeError(f"MCP server {binding.server_name!r} is unavailable: {status.error}")
            if cached:
                await self._ensure_connected(binding.server_name)
                expected_binding = binding
        return await self._dispatch_tool(exposed_name, arguments, expected_binding)

    async def _dispatch_tool(
        self,
        exposed_name: str,
        arguments: dict[str, Any],
        expected_binding: McpToolBinding | None,
    ) -> ToolResult:
        binding = self._bindings_by_exposed_name.get(exposed_name)
        if expected_binding is not None and binding is not expected_binding:
            error = (
                f"MCP tool '{exposed_name}' has been updated or removed. "
                "Search again before calling it."
            )
            return ToolResult(
                tool=exposed_name,
                status="failed",
                output=f"[error] {error}",
                error=error,
                metadata={
                    "status_source": "mcp",
                    "reason": "stale_binding",
                    "server": expected_binding.server_name,
                    "tool": expected_binding.tool_name,
                },
            )
        if binding is None:
            error = f"Unknown MCP tool: {exposed_name}"
            return ToolResult(
                tool=exposed_name,
                status="failed",
                output=f"[error] {error}",
                error=error,
                metadata={"status_source": "mcp"},
            )
        connection = self._connections.get(binding.server_name)
        if connection is None or connection.task is None or connection.task.done():
            error = f"MCP server is not connected: {binding.server_name}"
            return ToolResult(
                tool=exposed_name,
                status="failed",
                output=f"[error] {error}",
                error=error,
                metadata={
                    "status_source": "mcp",
                    "server": binding.server_name,
                    "tool": binding.tool_name,
                },
            )
        return await self._call_tool(connection, binding, arguments)

    def status_report(self) -> str:
        lines = ["MCP status", f"config: {self.config.path}"]
        if self.config_error:
            lines.append(f"config: failed - {self.config_error}")
            return "\n".join(lines)
        statuses = self.discovery_statuses()
        if not statuses:
            lines.append("no MCP servers configured")
        for status in statuses.values():
            details = f" ({status.tool_count} tool(s), {status.transport})" if status.catalog_source != "none" else ""
            error = f" - {status.error}" if status.error else ""
            lines.append(f"server {status.name}: {status.state}{details}{error}")
        return "\n".join(lines)

    def tools_report(self) -> str:
        lines = ["MCP tools"]
        if not self.tool_bindings:
            lines.append("no MCP tools registered")
            return "\n".join(lines)
        for binding in self.tool_bindings:
            lines.append(
                f"{binding.exposed_name} -> {binding.server_name}/{binding.tool_name} "
                f"permission={binding.permission} disclosure=deferred"
            )
        return "\n".join(lines)

    def discovery_scopes(self) -> dict[str, str]:
        with self._state_lock:
            return {
                f"mcp:{name}": f"{server.title} — {server.description}" if server.title and server.description
                else server.description or server.title or _MISSING_DESCRIPTION_TEXT
                for name, server in sorted(self.config.servers.items())
            }

    def configured_server_names(self) -> list[str]:
        """Return server names from the raw config, including disabled ones."""
        if not self.config.path.exists():
            return sorted(self.config.servers)
        try:
            data = json.loads(self.config.path.read_text(encoding="utf-8"))
            servers = data.get("servers") if isinstance(data, dict) else None
            return sorted(str(name) for name in servers if isinstance(name, str))
        except (OSError, ValueError, AttributeError):
            return sorted(self.config.servers)

    def registered_report(self) -> str:
        """List registered servers from the config file without connecting."""
        lines = [f"MCP config: {self.config.path}"]
        if self.config_error:
            lines.append(f"config: failed - {self.config_error}")
            return "\n".join(lines)
        if not self.config.path.exists():
            lines.append("no MCP servers registered")
            return "\n".join(lines)
        try:
            data = json.loads(self.config.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            lines.append(f"config: failed - {exc}")
            return "\n".join(lines)
        servers = data.get("servers") if isinstance(data, dict) else None
        if not isinstance(servers, dict) or not servers:
            lines.append("no MCP servers registered")
            return "\n".join(lines)
        for name in sorted(str(item) for item in servers):
            if error := self.config.server_errors.get(name):
                lines.append(f"- {name}: invalid - {error}")
                continue
            entry = servers[name]
            if not isinstance(entry, dict):
                lines.append(f"- {name}: invalid entry")
                continue
            disabled = " (disabled)" if entry.get("enabled", True) is False else ""
            transport = str(entry.get("transport") or ("stdio" if entry.get("command") else "streamable_http"))
            permission = str(entry.get("permission") or TOOL_PERMISSION_DANGEROUS)
            description = str(entry.get("description") or "").strip() or "(no description)"
            lines.append(f"- {name}{disabled} [{transport}, permission={permission}]: {description}")
        return "\n".join(lines)

    def _raw_server_entry(self, server_name: str) -> dict[str, Any]:
        """Return the raw config entry for one server, including disabled ones."""
        if not self.config.path.exists():
            return {}
        try:
            data = json.loads(self.config.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        servers = data.get("servers") if isinstance(data, dict) else None
        entry = servers.get(server_name) if isinstance(servers, dict) else None
        return entry if isinstance(entry, dict) else {}

    def _merge_previous_metadata(self, server: McpServerConfig) -> McpServerConfig:
        """Keep description/title/tool permissions when re-registering a server."""
        previous = self._raw_server_entry(server.name)
        updates: dict[str, Any] = {}
        if not server.description and isinstance(previous.get("description"), str):
            description = previous["description"].strip()
            if description:
                updates["description"] = description
        if not server.title and isinstance(previous.get("title"), str):
            title = previous["title"].strip()
            if title:
                updates["title"] = title
        if not server.tool_permissions and isinstance(previous.get("tool_permissions"), dict):
            permissions = {
                str(name): value
                for name, value in previous["tool_permissions"].items()
                if isinstance(name, str) and isinstance(value, str) and value in VALID_TOOL_PERMISSIONS
            }
            if permissions:
                updates["tool_permissions"] = permissions
        return replace(server, **updates) if updates else server

    def register_server(self, server: McpServerConfig) -> McpRegistration:
        """Persist one server and probe it, keeping the entry if the probe fails."""
        if self.config_error:
            raise McpConfigError(f"现有 MCP 配置无法解析，无法注册服务：{self.config_error}")
        existed = server.name in self.configured_server_names()
        if existed:
            server = self._merge_previous_metadata(server)
        self._update_config_document(lambda servers: servers.__setitem__(server.name, _server_entry(server)))
        self._reconfigure_server(server.name, server)
        self.prime_catalog(server.name)
        try:
            self.ensure_connected(server.name)
        except Exception as exc:
            return McpRegistration(server, "failed", error=f"{type(exc).__name__}: {exc}", replaced=existed)
        status = self.discovery_statuses()[server.name]
        return McpRegistration(server, status.state, status.tool_count, status.error, replaced=existed)

    def remove_server(self, server_name: str) -> str:
        """Remove one registered server from the config and drop its runtime state."""
        if self.config_error:
            raise McpConfigError(f"现有 MCP 配置无法解析，无法移除服务：{self.config_error}")
        if server_name not in self.configured_server_names():
            raise McpConfigError(f"MCP 服务不存在：{server_name}")
        self._update_config_document(lambda servers: servers.pop(server_name, None))
        self._reconfigure_server(server_name, None)
        return f"MCP 服务已移除：{server_name}"

    def _reconfigure_server(self, name: str, server: McpServerConfig | None) -> None:
        def update_config() -> None:
            if server is None:
                self.config.servers.pop(name, None)
            else:
                self.config.servers[name] = server
            self.config.server_errors.pop(name, None)
            self.config.disabled_servers.discard(name)

        async def apply() -> None:
            with self._state_lock:
                update_config()
            await self._drop_server_async(name)

        with self._state_lock:
            if self._loop_thread is None:
                update_config()
                self._drop_server_state(name)
                return
        self._run(apply(), timeout=self.timeout_seconds + 5.0)

    def generate_server_description(
        self,
        server_name: str,
        *,
        refresh: bool = False,
        llm_create: Callable[..., Any] | None = None,
    ) -> str:
        """Generate one server description from its discovered tool catalog."""
        server = self.config.servers.get(server_name)
        if server is None:
            raise McpConfigError(f"MCP 服务不存在或未启用：{server_name}")
        if server.description and not refresh:
            return f"MCP 服务已有描述：{server_name} — {server.description}"
        tools = [
            (binding.tool_name, binding.description)
            for binding in self.tool_bindings
            if binding.server_name == server_name
        ]
        if not tools:
            status = self.statuses.get(server_name)
            detail = status.error if status is not None and status.error else "服务未连接"
            raise McpConfigError(f"无法生成描述：{server_name} 尚未发现工具（{detail}）")
        descriptions = describe_mcp_servers({server_name: tools}, llm_create=llm_create)
        text = descriptions.get(server_name, "").strip()
        if not text:
            raise McpConfigError(f"描述生成失败：{server_name}（服务本身仍可正常使用）")
        self.persist_server_descriptions({server_name: text}, overwrite=refresh)
        return f"MCP 服务描述已{'刷新' if refresh else '生成'}：{server_name} — {text}"

    def reload_server(self, server_name: str) -> str:
        """Reconnect exactly one server, leaving other servers untouched."""
        if error := self.config_error or self.config.server_errors.get(server_name):
            return f"MCP 服务配置无效：{server_name}（{error}）"
        server = self.config.servers.get(server_name)
        if server is None:
            return f"MCP 服务不存在或未启用：{server_name}"
        self._drop_server(server_name)
        self.prime_catalog(server_name)
        try:
            self.ensure_connected(server_name)
        except Exception as exc:
            return f"MCP 服务连接失败：{server_name}（{type(exc).__name__}: {exc}）"
        status = self.statuses.get(server_name)
        tool_count = status.tool_count if status is not None else 0
        if not server.description:
            self._backfill_missing_descriptions([server_name])
        return f"MCP 服务已重新连接：{server_name}（{tool_count} 个工具）"

    def _backfill_missing_descriptions(self, server_names: list[str] | None = None) -> None:
        """Fill descriptions for explicitly reconnected servers that have none.

        An empty configured description *is* the pending fact, so no in-memory
        bookkeeping is needed. This runs only on user-initiated reconnect (or
        the explicit ``add``/``describe`` paths); startup and full reloads never
        spend model calls.
        """
        names = server_names if server_names is not None else list(self.config.servers)
        for name in names:
            server = self.config.servers.get(name)
            status = self.statuses.get(name)
            if server is None or server.description:
                continue
            if status is None or status.state != "connected":
                continue
            try:
                self.generate_server_description(name)
            except McpConfigError as exc:
                log.warning("Failed to backfill MCP description for %s: %s", name, exc)

    def _store_connection(
        self,
        server: McpServerConfig,
        connection: _McpConnection,
        bindings: list[McpToolBinding],
        deadline: float,
    ) -> None:
        """Validate a full catalog before replacing the previous server snapshot."""
        if connection.task is not None and connection.task.done():
            raise RuntimeError("MCP connection worker exited before registration")
        with self._state_lock:
            bindings = [
                self._bindings_by_exposed_name[item.exposed_name]
                if item.exposed_name in self._catalog_only
                and self._bindings_by_exposed_name[item.exposed_name] == item
                and server.name not in self.statuses
                else item
                for item in bindings
            ]
        specs = self._validated_server_specs(bindings)
        cache_key = self._schema_key(server)
        cached = copy.deepcopy(tuple(bindings))
        with self._state_lock, _static_lock:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"MCP handshake exceeded {self.timeout_seconds:g} seconds")
            for item in bindings:
                existing = self._bindings_by_exposed_name.get(item.exposed_name)
                if existing is not None and existing.server_name != server.name:
                    raise ValueError(f"Duplicate exposed MCP tool name {item.exposed_name!r}")
            self._validated_specs[server.name] = (tuple(bindings), specs)
            self._connections[server.name] = connection
            self.statuses[server.name] = McpServerStatus(
                server.name, server.transport, "connected", len(bindings), catalog_source="live",
            )
            self._replace_bindings(server.name, bindings)
            _schema_cache[cache_key] = cached

    def _on_worker_done(self, connection: _McpConnection) -> None:
        server_name = connection.config.name
        task = connection.task
        error = "connection worker exited"
        if task is not None:
            self._workers.discard(task)
            if not task.cancelled() and (exc := task.exception()) is not None:
                error = f"{type(exc).__name__}: {exc}"
        with self._state_lock:
            if self._connections.get(server_name) is not connection:
                return
            self._connections.pop(server_name)
            history = [item for item in self.tool_bindings if item.server_name == server_name]
            self._replace_bindings(server_name, history, cached=True)
            self._mark_failed(connection.config, error)
        log.warning("MCP server %s disconnected unexpectedly: %s", server_name, error)

    def _replace_bindings(self, server_name: str, bindings, *, cached: bool = False) -> None:
        for item in self.tool_bindings:
            if item.server_name == server_name:
                self._bindings_by_exposed_name.pop(item.exposed_name, None)
                self._catalog_only.discard(item.exposed_name)
        self.tool_bindings = [item for item in self.tool_bindings if item.server_name != server_name] + list(bindings)
        self._bindings_by_exposed_name.update((item.exposed_name, item) for item in bindings)
        if cached:
            self._catalog_only.update(item.exposed_name for item in bindings)
        self.connection_revision += 1

    def _drop_server(self, server_name: str) -> None:
        with self._state_lock:
            if self._loop_thread is None:
                self._drop_server_state(server_name)
                return
        self._run(self._drop_server_async(server_name), timeout=self.timeout_seconds + 5.0)

    async def _drop_server_async(self, server_name: str) -> None:
        self._disconnecting.add(server_name)
        try:
            pending = self._connecting.get(server_name)
            if pending is not None:
                pending.task.cancel()
                await asyncio.gather(pending.task, return_exceptions=True)
            with self._state_lock:
                connection = self._connections.pop(server_name, None)
                self._drop_server_state(server_name)
            if connection is not None:
                try:
                    await asyncio.wait_for(self._close_connections([connection]), timeout=self.timeout_seconds)
                except Exception as exc:
                    log.warning("Failed to close MCP server %s: %s", server_name, exc)
                    if connection.task is not None and not connection.task.done():
                        connection.task.cancel()
        finally:
            self._disconnecting.discard(server_name)

    def _drop_server_state(self, server_name: str) -> None:
        self.statuses.pop(server_name, None)
        self._validated_specs.pop(server_name, None)
        self._replace_bindings(server_name, ())

    def _mark_failed(self, server: McpServerConfig, error: str) -> None:
        with self._state_lock:
            self.statuses[server.name] = McpServerStatus(
                server.name, server.transport, "failed", error=error,
            )
            self.connection_revision += 1

    async def _connect_one_and_store(self, server: McpServerConfig) -> None:
        connection = None
        try:
            connection, bindings, deadline = await self._connect_one(server)
            self._store_connection(server, connection, bindings, deadline)
        except Exception as exc:
            if connection is not None and connection.task is not None:
                connection.task.cancel()
            self._mark_failed(server, f"{type(exc).__name__}: {exc}")
            raise

    def set_server_enabled(self, server_name: str, enabled: bool) -> str:
        """Toggle one raw config entry while preserving environment placeholders."""
        if server_name not in set(self.configured_server_names()):
            return f"MCP 服务不存在：{server_name}"

        def mutate(servers: dict[str, Any]) -> None:
            entry = servers.get(server_name)
            if not isinstance(entry, dict):
                raise McpConfigError(f"MCP 服务配置无效：{server_name}")
            if enabled:
                entry.pop("enabled", None)
            else:
                entry["enabled"] = False

        self._update_config_document(mutate)
        return f"MCP 服务已{'启用' if enabled else '停用'}：{server_name}"

    def persist_server_descriptions(self, descriptions: dict[str, str], *, overwrite: bool = False) -> None:
        """Write generated descriptions into the config file."""
        if not descriptions:
            return
        written: dict[str, str] = {}

        def mutate(servers: dict[str, Any]) -> None:
            for name, description in descriptions.items():
                entry = servers.get(name)
                if not isinstance(entry, dict):
                    continue
                if entry.get("description") and not overwrite:
                    continue
                entry["description"] = description
                written[name] = description

        self._update_config_document(mutate)
        for name, description in written.items():
            with self._state_lock:
                server = self.config.servers.get(name)
                if server is not None and server.description != description:
                    self.config.servers[name] = replace(server, description=description)
                    self.connection_revision += 1

    def _update_config_document(self, mutate: Callable[[dict[str, Any]], None]) -> None:
        """Apply a mutation to the raw config document with an atomic write."""
        path = self.config.path
        data: dict[str, Any] = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise McpConfigError(f"无法读取 MCP 配置：{exc}") from exc
            if not isinstance(data, dict):
                raise McpConfigError("MCP 配置必须是 JSON 对象")
        servers = data.get("servers")
        if servers is None:
            servers = {}
            data["servers"] = servers
        if not isinstance(servers, dict):
            raise McpConfigError("MCP 配置字段 'servers' 必须是对象")
        mutate(servers)
        try:
            atomic_bytes(path, (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        except OSError as exc:
            raise McpConfigError(f"无法写入 MCP 配置（{path}）：{exc}") from exc

    def doctor_status(self) -> tuple[bool, str]:
        if self.config_error:
            return False, self.config_error
        if not self.config.path.exists():
            return True, "not configured"
        failed = [status for status in self.discovery_statuses().values() if status.state in {"failed", "invalid"}]
        if failed:
            details = "; ".join(f"{item.name}: {item.error}" for item in failed)
            return False, details
        return True, f"{len(self.config.servers)} server(s), {len(self.tool_bindings)} tool(s)"

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            loop = self._loop_thread
        try:
            if loop is not None:
                try:
                    loop.run(self._close_async(), timeout=self.timeout_seconds + 5.0)
                except Exception as exc:
                    log.warning("MCP shutdown reported an error: %s", exc)
                finally:
                    loop.close()
        finally:
            with self._state_lock:
                self._loop_thread = None
            self._resource_owner()

    async def _connection_worker(
        self,
        server: McpServerConfig,
        request_queue: asyncio.Queue,
        ready: asyncio.Future,
    ) -> None:
        """Own a server connection in one task from connect through close."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamable_http_client

        stack = contextlib.AsyncExitStack()
        close_response: asyncio.Future | None = None
        #: In-flight tool calls keyed by their response future, so cancelling
        #: the response can be propagated to the task running
        #: ``session.call_tool()``.
        active_calls: dict[asyncio.Future, asyncio.Task] = {}

        def _track_call(response: asyncio.Future, task: asyncio.Task) -> None:
            active_calls[response] = task

            def _discard(_):
                active_calls.pop(response, None)

            def _cancel_task(future):
                if future.cancelled() and not task.done():
                    task.cancel()

            task.add_done_callback(_discard)
            response.add_done_callback(_cancel_task)

        try:
            if server.transport == "stdio":
                params = StdioServerParameters(
                    command=server.command or "",
                    args=list(server.args),
                    env=dict(server.env) if server.env else None,
                    cwd=_resolve_server_cwd(self.workspace, server.cwd),
                )
                read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
            else:
                import httpx

                http_client = httpx.AsyncClient(
                    headers=dict(server.headers),
                    timeout=httpx.Timeout(self.tool_timeout_seconds, connect=self.timeout_seconds),
                )
                await stack.enter_async_context(http_client)
                read_stream, write_stream, _ = await stack.enter_async_context(
                    streamable_http_client(server.url or "", http_client=http_client)
                )

            def disconnected() -> None:
                request_queue.put_nowait(("disconnect", None, None))

            session = await stack.enter_async_context(
                ClientSession(_ObservedReadStream(read_stream, disconnected), write_stream)
            )
            await session.initialize()
            list_result = await session.list_tools()
            tools = list(list_result.tools)
            cursors = set()
            while (cursor := getattr(list_result, "nextCursor", None)) is not None:
                if cursor in cursors:
                    raise ValueError("MCP tools/list repeated a pagination cursor")
                cursors.add(cursor)
                list_result = await session.list_tools(cursor=cursor)
                tools.extend(list_result.tools)
            bindings = _bindings_for_server(server, tools)
            if not ready.done():
                ready.set_result(bindings)

            async def execute_call(binding, arguments, response) -> None:
                if response.cancelled():
                    return
                try:
                    async with self.project_resources.acquire_async(f"mcp:{server.name}", session_id=self.session_id, label=binding.exposed_name):
                        result = await asyncio.wait_for(session.call_tool(binding.tool_name, arguments),
                                                        timeout=self.tool_timeout_seconds)
                    tool_result = mcp_result_to_tool_result(binding.exposed_name, result, binding=binding)
                except Exception as exc:
                    if not response.done():
                        response.set_exception(exc)
                else:
                    if not response.done():
                        response.set_result(tool_result)

            while True:
                kind, payload, response = await request_queue.get()
                if kind == "disconnect":
                    raise RuntimeError("MCP transport closed")
                if kind == "close":
                    # Do not wait for remote calls to finish naturally; the
                    # finally block cancels and reaps them.
                    close_response = response
                    break
                if kind != "call":
                    if not response.done():
                        response.set_exception(ValueError(f"Unknown MCP worker request: {kind}"))
                    continue
                binding, arguments = payload
                task = asyncio.create_task(execute_call(binding, arguments, response))
                _track_call(response, task)
        except Exception as exc:
            if not ready.done():
                ready.set_exception(exc)
            raise
        finally:
            close_error: Exception | None = None
            for task in active_calls.values():
                task.cancel()
            if active_calls:
                await asyncio.gather(*active_calls.values(), return_exceptions=True)
            try:
                await stack.aclose()
            except Exception as exc:
                close_error = exc
            if close_response is not None and not close_response.done():
                if close_error is None:
                    close_response.set_result(None)
                else:
                    close_response.set_exception(close_error)
            if close_error is not None:
                raise close_error

    async def _call_tool(
        self,
        connection: _McpConnection,
        binding: McpToolBinding,
        arguments: dict[str, Any],
    ) -> ToolResult:
        task = connection.task
        if task is None or task.done():
            raise RuntimeError("MCP connection worker is not running")
        response = asyncio.get_running_loop().create_future()
        await connection.request_queue.put(("call", (binding, arguments), response))
        try:
            done, _ = await asyncio.wait(
                {response, task}, return_when=asyncio.FIRST_COMPLETED
            )
            if response in done:
                return response.result()
            response.cancel()
            raise RuntimeError("MCP connection worker exited before responding")
        except asyncio.CancelledError:
            response.cancel()
            raise

    async def _close_async(self) -> None:
        pending = [flight.task for flight in self._connecting.values()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        with self._state_lock:
            connections = list(self._connections.values())
            self._connections.clear()
        try:
            results = await asyncio.gather(
                *(self._close_connections([connection]) for connection in connections),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    raise result
        finally:
            workers = list(self._workers)
            if workers:
                await asyncio.gather(*workers, return_exceptions=True)

    async def _close_connections(self, connections: list[_McpConnection]) -> None:
        errors: list[tuple[str, Exception]] = []
        for connection in reversed(connections):
            task = connection.task
            if task is None:
                continue
            if task.done():
                if not task.cancelled():
                    error = task.exception()
                    if error is not None:
                        errors.append((connection.config.name, error))
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
                continue
            response = asyncio.get_running_loop().create_future()
            await connection.request_queue.put(("close", None, response))
            done, _ = await asyncio.wait(
                {response, task}, return_when=asyncio.FIRST_COMPLETED
            )
            if response in done and not response.cancelled():
                # Read the close result: a failed cleanup must stay visible
                # instead of being reported as a successful close.
                error = response.exception()
                if error is not None:
                    errors.append((connection.config.name, error))
                else:
                    response.result()
            else:
                response.cancel()
                if task.done() and not task.cancelled():
                    error = task.exception()
                    if error is not None:
                        errors.append((connection.config.name, error))
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if errors:
            for name, error in errors:
                log.warning("MCP server %s close failed: %s", name, error)
            raise errors[0][1]

    def _handler_for(self, binding: McpToolBinding) -> Callable[..., ToolResult]:
        def handler(*, cancellation_token=None, **kwargs) -> ToolResult:
            arguments = {
                key: value
                for key, value in kwargs.items()
                if key not in {"runtime_state", "agent_name", "tool_context"}
            }
            # Identity is verified against the live binding inside the event
            # loop, immediately before the connection is selected and the
            # request enqueued.
            return self.call_tool(
                binding.exposed_name,
                arguments,
                cancellation_token=cancellation_token,
                expected_binding=binding,
            )

        return handler

    def _ensure_loop(self) -> _AsyncLoopThread:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("MCP manager is closed")
            if self._loop_thread is None:
                self._loop_thread = _AsyncLoopThread()
            return self._loop_thread

    def _run(self, coro, *, cancellation_token=None, timeout: float | None = None):
        try:
            loop = self._ensure_loop()
        except Exception:
            coro.close()
            raise
        return loop.run(
            coro,
            timeout=self.timeout_seconds if timeout is None else timeout,
            cancellation_token=cancellation_token,
        )


class _ObservedReadStream(ObjectReceiveStream):
    def __init__(self, stream, disconnected):
        self._stream = stream
        self._disconnected = disconnected

    async def receive(self):
        try:
            return await self._stream.receive()
        except (EndOfStream, ClosedResourceError, BrokenResourceError):
            self._disconnected()
            raise

    async def aclose(self) -> None:
        await self._stream.aclose()


def mcp_result_to_tool_result(
    exposed_name: str,
    result: Any,
    *,
    binding: McpToolBinding | None = None,
) -> ToolResult:
    parts: list[str] = []
    for item in list(getattr(result, "content", []) or []):
        parts.append(_content_item_to_text(item))
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        parts.append("structuredContent:\n" + json.dumps(structured, ensure_ascii=False, indent=2, sort_keys=True))
    output = "\n\n".join(part for part in parts if part)
    is_error = bool(getattr(result, "isError", False))
    metadata: dict[str, Any] = {"status_source": "mcp"}
    if binding is not None:
        metadata.update(
            {
                "server": binding.server_name,
                "tool": binding.tool_name,
                "permission": binding.permission,
                "annotations": binding.annotations,
            }
        )
    return ToolResult(
        tool=exposed_name,
        status="failed" if is_error else "success",
        output=output,
        error="MCP tool returned isError=true" if is_error else None,
        metadata=metadata,
    )


def _bindings_for_server(
    server: McpServerConfig,
    mcp_tools: list[Any],
) -> list[McpToolBinding]:
    bindings: list[McpToolBinding] = []
    for tool in mcp_tools:
        tool_name = getattr(tool, "name", "")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("MCP tool name is missing")
        exposed_name = _exposed_tool_name(server.name, tool_name)
        permission = server.tool_permissions.get(tool_name, server.permission)
        description = str(getattr(tool, "description", "") or getattr(tool, "title", "") or "")
        annotations = _model_to_dict(getattr(tool, "annotations", None))
        bindings.append(
            McpToolBinding(
                exposed_name=exposed_name,
                server_name=server.name,
                tool_name=tool_name,
                description=description,
                input_schema=getattr(tool, "inputSchema", None),
                permission=permission,
                annotations=annotations,
            )
        )
    return bindings


_MISSING_DESCRIPTION_TEXT = (
    "Use search_tools to inspect its tools."
)


def describe_mcp_servers(
    servers: dict[str, list[tuple[str, str]]],
    *,
    llm_create: Callable[..., Any] | None = None,
) -> dict[str, str]:
    """Ask the fast model for one-sentence descriptions of newly registered servers.

    ``servers`` maps a server id to its ``(tool name, description)`` pairs.  The
    result is persisted by the caller so later runs reuse it instead of calling
    the model again.
    """
    if not servers:
        return {}
    payload = {
        name: [{"name": tool, "description": description} for tool, description in tools]
        for name, tools in servers.items()
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You write directory entries for MCP servers used by a coding agent. "
                "For each server id in the user's JSON, reply with one short English sentence "
                "(max 140 characters) describing what that server is useful for, based on its tools. "
                "Do not list tool names, do not mention JSON, and do not use markdown. "
                'Reply with a JSON object mapping every server id to its sentence, for example '
                '{"market": "..."}'
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        if llm_create is not None:
            response = llm_create(messages)
        else:
            from .. import config
            from ..support.llm import ProviderAdapter, client_scope

            profile = config.resolve_model_profile("fast")
            adapter = ProviderAdapter(profile.provider)
            kwargs = adapter.chat_kwargs(profile=profile, messages=messages, max_tokens=400)
            with client_scope() as client:
                response = client.chat.completions.create(**kwargs)
        content = str(response.choices[0].message.content or "")
    except Exception as exc:
        log.warning("Failed to generate MCP server descriptions: %s", exc)
        return {}

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match is None:
        log.warning("MCP server description response was not JSON: %s", content[:200])
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        log.warning("Failed to parse MCP server descriptions: %s", exc)
        return {}
    descriptions: dict[str, str] = {}
    for name, value in data.items() if isinstance(data, dict) else []:
        if name not in servers or not isinstance(value, str):
            continue
        text = " ".join(value.split())
        if text:
            descriptions[name] = text[:200]
    return descriptions


def _server_entry(server: McpServerConfig) -> dict[str, Any]:
    """Serialize one server back to its config-file representation."""
    entry: dict[str, Any] = {"transport": server.transport}
    if server.transport == "stdio":
        entry["command"] = server.command or ""
        if server.args:
            entry["args"] = list(server.args)
        if server.env:
            entry["env"] = dict(server.env)
        if server.cwd:
            entry["cwd"] = server.cwd
    else:
        entry["url"] = server.url or ""
        if server.headers:
            entry["headers"] = dict(server.headers)
    entry["permission"] = server.permission
    if server.tool_permissions:
        entry["tool_permissions"] = dict(server.tool_permissions)
    if server.title:
        entry["title"] = server.title
    if server.description:
        entry["description"] = server.description
    return entry


def _validate_server_name(name: str) -> None:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise McpConfigError(f"Invalid MCP server name: {name!r}")


def _content_item_to_text(item: Any) -> str:
    item_type = str(getattr(item, "type", "") or "")
    if item_type == "text":
        return str(getattr(item, "text", "") or "")
    if item_type in {"image", "audio"}:
        data = str(getattr(item, "data", "") or "")
        mime = str(getattr(item, "mimeType", "") or "application/octet-stream")
        return f"[{item_type} {mime}, {len(data)} base64 chars omitted]"
    if item_type == "resource":
        resource = getattr(item, "resource", None)
        if resource is None:
            return "[resource omitted]"
        uri = str(getattr(resource, "uri", "") or "")
        mime = str(getattr(resource, "mimeType", "") or "application/octet-stream")
        if hasattr(resource, "text"):
            return f"[resource {uri} {mime}]\n{getattr(resource, 'text', '')}"
        blob = str(getattr(resource, "blob", "") or "")
        return f"[resource {uri} {mime}, {len(blob)} base64 chars omitted]"
    if item_type == "resource_link":
        uri = str(getattr(item, "uri", "") or "")
        name = str(getattr(item, "name", "") or "")
        return f"[resource_link {name} {uri}]"
    return f"[{item_type or type(item).__name__} content omitted]"


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _expand_env(item) for key, item in value.items()}
    return value


def _string_dict(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise McpConfigError(f"MCP {label} must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise McpConfigError(f"MCP {label} must contain only string keys and values")
        result[key] = item
    return result


def _validate_permission(permission: str, label: str) -> None:
    if permission not in VALID_TOOL_PERMISSIONS:
        raise McpConfigError(f"Invalid MCP permission for {label}: {permission}")


def _resolve_server_cwd(workspace: Path, cwd: str | None) -> str | None:
    if not cwd:
        return None
    path = Path(cwd)
    if not path.is_absolute():
        path = workspace / path
    return str(path.resolve())


def _exposed_tool_name(server_name: str, tool_name: str) -> str:
    """Derive a stable exposed name from the raw ``(server, tool)`` identity.

    Names must not depend on connection completion order: a tool that was
    addressable in one run has to resolve to the same server in the next run.
    The mapping is therefore computed only from the raw identifiers; when
    sanitization is lossy (``foo__bar`` -> ``foo_bar``) the raw value's digest
    is embedded so distinct servers never swap names.
    """
    server_segment = _stable_name_segment(server_name)
    tool_segment = _stable_name_segment(tool_name)
    base = f"{MCP_TOOL_PREFIX}{server_segment}__{tool_segment}"
    if len(base) > MCP_TOOL_NAME_LIMIT:
        digest = hashlib.sha256(f"{server_name}/{tool_name}".encode()).hexdigest()[:8]
        server_budget = min(len(server_segment), 20)
        tool_budget = max(1, MCP_TOOL_NAME_LIMIT - len(MCP_TOOL_PREFIX) - server_budget - 4 - 8)
        base = (
            f"{MCP_TOOL_PREFIX}{server_segment[:server_budget]}__"
            f"{tool_segment[:tool_budget]}__{digest}"
        )
    return base[:MCP_TOOL_NAME_LIMIT]


def _stable_name_segment(value: str) -> str:
    raw = value.strip()
    safe = _safe_name_segment(raw)
    if safe == raw:
        return safe
    # Sanitization collapsed or stripped characters; suffix the raw digest
    # behind a ``__x`` marker. Sanitization never emits consecutive
    # underscores, so a marked segment cannot be produced by a different raw
    # name that happens to look like ``<safe>_<digest>``.
    return f"{safe}__x{hashlib.sha256(raw.encode()).hexdigest()[:8]}"


def _safe_name_segment(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", value.strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "tool"


def _model_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return dict(value)
    return {}


class _AsyncLoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._submit_lock = threading.Lock()
        self._stopping = False
        self._queue = None
        self._actor_task: asyncio.Task | None = None
        self._thread = threading.Thread(target=self._run_loop, name="hca-mcp-loop", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    def run(self, coro, *, timeout: float | None, cancellation_token=None):
        future: Future = Future()
        actor_holder: list[asyncio.Task | None] = [None]

        def submit() -> None:
            assert self._queue is not None
            self._queue.put_nowait((coro, future, actor_holder))

        with self._submit_lock:
            if self._stopping:
                coro.close()
                raise RuntimeError("MCP event loop is closed")
            self.loop.call_soon_threadsafe(submit)
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            if cancellation_token is not None and cancellation_token.is_cancelled:
                self._cancel(future, actor_holder)
                raise CancelledError("Turn cancelled by user")
            remaining = deadline - time.monotonic() if deadline is not None else .05
            if remaining <= 0:
                self._cancel(future, actor_holder)
                raise TimeoutError("MCP operation timed out and was cancelled")
            try:
                return future.result(timeout=min(0.05, remaining))
            except FutureTimeoutError:
                if future.done():
                    return future.result()

    def _cancel(self, future: Future, actor_holder: list[asyncio.Task | None]) -> None:
        future.cancel()
        actor_task = actor_holder[0]
        if actor_task is not None and not actor_task.done():
            with contextlib.suppress(RuntimeError):
                self.loop.call_soon_threadsafe(actor_task.cancel)

    def close(self) -> None:
        if self._queue is not None:
            future: Future = Future()

            def submit_stop() -> None:
                assert self._queue is not None
                self._queue.put_nowait((None, future))

            with self._submit_lock:
                self._stopping = True
                self.loop.call_soon_threadsafe(submit_stop)
            with contextlib.suppress(Exception):
                future.result(timeout=5)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)
        self.loop.close()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self._queue = asyncio.Queue()
        self._actor_task = self.loop.create_task(self._actor())
        self._ready.set()
        self.loop.run_forever()

    async def _actor(self) -> None:
        assert self._queue is not None
        tasks: set[asyncio.Task] = set()
        while True:
            item = await self._queue.get()
            coro, future = item[0], item[1]
            actor_holder: list[asyncio.Task | None] | None = item[2] if len(item) > 2 else None
            if coro is None:
                for task in list(tasks):
                    task.cancel()
                if tasks:
                    with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                        await asyncio.wait_for(
                            asyncio.gather(*tasks, return_exceptions=True),
                            timeout=2.0,
                        )
                future.set_result(None)
                return
            if future.cancelled():
                coro.close()
                continue
            # Run each request as its own task so one slow call cannot block
            # other servers, reconnects, or the caller that submitted it.
            task = asyncio.create_task(coro)
            tasks.add(task)
            if actor_holder is not None:
                # Publish the task before re-reading the cancellation flag.
                # The caller cancels the Future and then consults the holder,
                # so either it sees this task (and cancels it), or this re-check
                # observes the cancellation; the gap between the two is closed.
                actor_holder[0] = task
                if future.cancelled():
                    task.cancel()

            def _finish(done: asyncio.Task, *, target: Future = future) -> None:
                tasks.discard(done)
                with contextlib.suppress(InvalidStateError):
                    if done.cancelled():
                        target.set_exception(TimeoutError("MCP operation was cancelled"))
                    elif (error := done.exception()) is not None:
                        target.set_exception(error)
                    else:
                        target.set_result(done.result())

            task.add_done_callback(_finish)
