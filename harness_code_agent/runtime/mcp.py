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
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..support.cancellation import CancelledError
from .permissions import (
    TOOL_PERMISSION_DANGEROUS,
    TOOL_PERMISSION_NETWORK_READ,
    TOOL_PERMISSION_READ,
    VALID_TOOL_PERMISSIONS,
)
from .project_resources import ProjectResourceBudget
from .recovery import atomic_bytes
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
        parameters = self.input_schema or {"type": "object", "properties": {}}
        if parameters.get("type") != "object":
            parameters = {"type": "object", "properties": {}, "additionalProperties": True}
        description = self.description or f"MCP tool {self.server_name}/{self.tool_name}"
        return {
            "type": "function",
            "function": {
                "name": self.exposed_name,
                "description": description,
                "parameters": parameters,
            },
        }


@dataclass(frozen=True)
class McpCatalogSnapshot:
    """Immutable catalog view published by the connection event loop.

    ``revision`` and ``bindings`` come from the same event loop tick, so a
    consumer can build a Registry from the bindings and record the revision
    without racing a concurrent connect or disconnect.
    """

    revision: int
    bindings: tuple[McpToolBinding, ...]


@dataclass
class McpServerStatus:
    name: str
    transport: str
    state: str
    tool_count: int = 0
    error: str | None = None


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


def load_mcp_config(workspace: str | Path) -> McpConfig:
    root = Path(workspace).resolve()
    path = root / MCP_CONFIG_RELATIVE_PATH
    if not path.exists():
        return McpConfig(path=path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise McpConfigError(f"Invalid MCP config JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise McpConfigError("MCP config must be a JSON object")
    raw_servers = data.get("servers")
    if not isinstance(raw_servers, dict):
        raise McpConfigError("MCP config field 'servers' must be an object")

    servers: dict[str, McpServerConfig] = {}
    for name, raw in raw_servers.items():
        _validate_server_name(name)
        if not isinstance(raw, dict):
            raise McpConfigError(f"MCP server {name!r} must be an object")
        if raw.get("enabled", True) is False:
            continue

        expanded = _expand_env(raw)
        transport = str(expanded.get("transport") or "").strip()
        if transport == "streamable-http":
            transport = "streamable_http"
        if transport not in MCP_TRANSPORTS:
            raise McpConfigError(f"MCP server {name!r} has unsupported transport: {transport!r}")

        permission = str(expanded.get("permission") or TOOL_PERMISSION_DANGEROUS)
        _validate_permission(permission, f"server {name!r}")
        tool_permissions = _string_dict(expanded.get("tool_permissions") or {}, f"server {name!r} tool_permissions")
        for tool_name, tool_permission in tool_permissions.items():
            _validate_permission(tool_permission, f"server {name!r} tool {tool_name!r}")

        args = expanded.get("args") or []
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            raise McpConfigError(f"MCP server {name!r} args must be a list of strings")

        if transport == "stdio":
            command = str(expanded.get("command") or "").strip()
            if not command:
                raise McpConfigError(f"MCP stdio server {name!r} requires command")
            servers[name] = McpServerConfig(
                name=name,
                transport=transport,
                permission=permission,
                command=command,
                args=args,
                env=_string_dict(expanded.get("env") or {}, f"server {name!r} env"),
                cwd=str(expanded.get("cwd")) if expanded.get("cwd") else None,
                tool_permissions=tool_permissions,
                title=str(expanded.get("title") or "").strip(),
                description=str(expanded.get("description") or "").strip(),
            )
        else:
            url = str(expanded.get("url") or "").strip()
            if not url:
                raise McpConfigError(f"MCP streamable_http server {name!r} requires url")
            servers[name] = McpServerConfig(
                name=name,
                transport=transport,
                permission=permission,
                url=url,
                headers=_string_dict(expanded.get("headers") or {}, f"server {name!r} headers"),
                tool_permissions=tool_permissions,
                title=str(expanded.get("title") or "").strip(),
                description=str(expanded.get("description") or "").strip(),
            )

    return McpConfig(path=path, servers=servers)


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
        self._connect_lock = threading.RLock()
        self._catalog_only = set()
        self._closed = False
        self.config = config or McpConfig(path=self.workspace / MCP_CONFIG_RELATIVE_PATH)
        self.config_error = config_error
        self.timeout_seconds = timeout_seconds
        self.tool_timeout_seconds = tool_timeout_seconds
        self.statuses: dict[str, McpServerStatus] = {}
        self.tool_bindings: list[McpToolBinding] = []
        self._bindings_by_exposed_name: dict[str, McpToolBinding] = {}
        self._connections: dict[str, _McpConnection] = {}
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
        data = {"workspace": str(self.workspace), "server": _server_entry(server)}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def prime_catalog(self):
        with self._connect_lock, _static_lock:
            for server in self.config.servers.values():
                if any(binding.server_name == server.name for binding in self.tool_bindings):
                    continue
                cached = _schema_cache.get(self._schema_key(server), ())
                for binding in copy.deepcopy(cached):
                    self.tool_bindings.append(binding)
                    self._bindings_by_exposed_name[binding.exposed_name] = binding
                    self._catalog_only.add(binding.exposed_name)

    def discover(self, server_name=None, *, cancellation_token=None):
        self.prime_catalog()
        for server in self.config.servers.values():
            if server_name is not None and server.name != server_name:
                continue
            if not any(binding.server_name == server.name for binding in self.tool_bindings):
                self.ensure_connected(server.name, cancellation_token=cancellation_token)

    def ensure_connected(self, server_name, *, cancellation_token=None):
        while not self._connect_lock.acquire(timeout=.05):
            if cancellation_token is not None:
                cancellation_token.check()
        try:
            connection = self._connections.get(server_name)
            if connection is not None and connection.task is not None and not connection.task.done():
                return
            if self._closed:
                raise RuntimeError("MCP manager is closed")
            server = self.config.servers[server_name]
            self._ensure_loop()
            self._loop_thread.run(self._connect_one_and_store(server), timeout=None,
                                  cancellation_token=cancellation_token)
        finally:
            self._connect_lock.release()

    def connect_all(self) -> None:
        if self.config_error:
            self.statuses["config"] = McpServerStatus(
                name="config",
                transport="config",
                state="failed",
                error=self.config_error,
            )
            return
        if not self.config.servers:
            return
        for server in self.config.servers.values():
            try:
                self.ensure_connected(server.name)
            except Exception as exc:
                self.statuses[server.name] = McpServerStatus(
                    name=server.name, transport=server.transport, state="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )

    async def _connect_one(
        self,
        server: McpServerConfig,
    ) -> tuple[_McpConnection, list[McpToolBinding]]:
        """Connect to a single server with a per-server timeout."""
        async with self.project_resources.acquire_async("mcp_connect", session_id=self.session_id, label=server.name):
            request_queue: asyncio.Queue = asyncio.Queue()
            ready: asyncio.Future = asyncio.get_running_loop().create_future()
            connection = _McpConnection(config=server, request_queue=request_queue, task=None)
            task = asyncio.create_task(self._connection_worker(server, request_queue, ready))
            connection.task = task
            task.add_done_callback(lambda _task, item=connection: self._on_worker_done(item))
            try:
                bindings = await asyncio.wait_for(
                    asyncio.shield(ready), timeout=self.timeout_seconds
                )
            except asyncio.CancelledError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                raise
            except Exception:
                task.cancel()
                with contextlib.suppress(Exception):
                    await task
                raise
            return connection, bindings

    def register_tools(self, registry, snapshot: McpCatalogSnapshot | None = None) -> None:
        from .execution_planner import CallEffect, ResourceClaim
        from .tool_registry import (
            TOOL_CAPABILITY_MAIN,
            TOOL_CAPABILITY_READONLY_AGENT,
            TOOL_CAPABILITY_WORKER_AGENT,
        )

        view = snapshot if snapshot is not None else self.catalog_snapshot()
        for binding in view.bindings:
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

    def catalog_snapshot(self) -> McpCatalogSnapshot:
        """Return a versioned, immutable view of the current tool catalog."""
        if self._loop_thread is None:
            return McpCatalogSnapshot(
                revision=self.connection_revision,
                bindings=tuple(self.tool_bindings),
            )
        return self._run(self._catalog_snapshot())

    async def _catalog_snapshot(self) -> McpCatalogSnapshot:
        return McpCatalogSnapshot(
            revision=self.connection_revision,
            bindings=tuple(self.tool_bindings),
        )

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
            binding = self._bindings_by_exposed_name.get(exposed_name)
            if binding is not None:
                self.ensure_connected(binding.server_name, cancellation_token=cancellation_token)
            self._ensure_loop()
            return self._loop_thread.run(
                self._dispatch_tool(exposed_name, dict(arguments or {}), expected_binding),
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

    async def _dispatch_tool(
        self,
        exposed_name: str,
        arguments: dict[str, Any],
        expected_binding: McpToolBinding | None,
    ) -> ToolResult:
        binding = self._bindings_by_exposed_name.get(exposed_name)
        if binding is None:
            error = f"Unknown MCP tool: {exposed_name}"
            return ToolResult(
                tool=exposed_name,
                status="failed",
                output=f"[error] {error}",
                error=error,
                metadata={"status_source": "mcp"},
            )
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
        lines = ["MCP status"]
        if self.config_error:
            lines.append(f"config: failed - {self.config_error}")
            return "\n".join(lines)
        if not self.config.servers:
            lines.append(f"config: {self.config.path}")
            lines.append("no MCP servers configured")
            return "\n".join(lines)
        lines.append(f"config: {self.config.path}")
        for status in self.statuses.values():
            if status.state == "connected":
                lines.append(f"server {status.name}: connected ({status.tool_count} tool(s), {status.transport})")
            else:
                lines.append(f"server {status.name}: failed - {status.error}")
        missing = set(self.config.servers) - set(self.statuses)
        for name in sorted(missing):
            server = self.config.servers[name]
            lines.append(f"server {name}: not connected ({server.transport})")
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

    def catalog_prompt(self) -> str:
        """Describe connected MCP servers without listing their tools.

        The model picks a server id from this catalog and calls ``search_tools``
        to mount the tools it needs; tool schemas stay out of the prompt.
        """
        if self.config_error or not self.config.servers:
            return ""
        lines = [
            "## MCP Servers",
            "",
            (
                "Configured MCP servers, connected on demand. Use the server id when "
                "calling search_tools:"
            ),
        ]
        for name in sorted(self.config.servers):
            server = self.config.servers[name]
            description = server.description or _MISSING_DESCRIPTION_TEXT
            label = f"{name} ({server.title})" if server.title else name
            lines.append(f"- {label}: {description}")
        lines.extend(
            [
                "",
                (
                    "When a needed capability is not in your mounted tools, call search_tools with the "
                    "MCP server id and short English keywords. Matching tools are mounted for later turns "
                    "and stay available. Never repeat the same server and query; if nothing is mounted, "
                    "try different English keywords or another server, and if no server can do it, tell "
                    "the user instead of substituting a similar tool."
                ),
            ]
        )
        return "\n".join(lines)

    def configured_server_names(self) -> list[str]:
        """Return server names from the raw config, including disabled ones."""
        if not self.config.path.exists():
            return sorted(self.config.servers)
        try:
            data = json.loads(self.config.path.read_text(encoding="utf-8"))
            servers = data.get("servers") if isinstance(data, dict) else None
            return sorted(str(name) for name in servers if isinstance(name, str))
        except (OSError, json.JSONDecodeError, AttributeError):
            return sorted(self.config.servers)

    def registered_report(self) -> str:
        """List registered servers from the config file without connecting."""
        lines = [f"MCP config: {self.config.path}"]
        if not self.config.path.exists():
            lines.append("no MCP servers registered")
            return "\n".join(lines)
        try:
            data = json.loads(self.config.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            lines.append(f"config: failed - {exc}")
            return "\n".join(lines)
        servers = data.get("servers") if isinstance(data, dict) else None
        if not isinstance(servers, dict) or not servers:
            lines.append("no MCP servers registered")
            return "\n".join(lines)
        for name in sorted(str(item) for item in servers):
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
                if isinstance(name, str) and value in VALID_TOOL_PERMISSIONS
            }
            if permissions:
                updates["tool_permissions"] = permissions
        return replace(server, **updates) if updates else server

    def register_server(self, server: McpServerConfig) -> McpRegistration:
        """Add or replace one server: probe connection, persist config, bind tools.

        The config entry is written even when the connection probe fails, so the
        user keeps a registration to retry later.  A write failure raises
        ``McpConfigError`` and leaves no in-memory state behind.
        """
        if self.config_error:
            raise McpConfigError(f"现有 MCP 配置无法解析，无法注册服务：{self.config_error}")
        existed = server.name in set(self.configured_server_names()) or server.name in self.config.servers
        if existed:
            server = self._merge_previous_metadata(server)
        connection: _McpConnection | None = None
        bindings: list[McpToolBinding] = []
        error: str | None = None
        try:
            self._ensure_loop()
            connection, bindings = self._run(self._connect_one(server))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        try:
            self._update_config_document(lambda servers: servers.__setitem__(server.name, _server_entry(server)))
        except McpConfigError:
            if connection is not None:
                with contextlib.suppress(Exception):
                    self._run(self._close_connections([connection]))
            raise
        self.config.servers[server.name] = server
        if connection is not None:
            try:
                self._run(
                    self._register_connection(server, connection, bindings),
                    timeout=self.timeout_seconds + 5.0,
                )
            except Exception as exc:
                if self._connections.get(server.name) is not connection:
                    error = f"{type(exc).__name__}: {exc}"
                    with contextlib.suppress(Exception):
                        self._run(self._close_connections([connection]))
                    connection = None
                else:
                    log.warning(
                        "MCP server %s was registered but replacement reported: %s",
                        server.name,
                        exc,
                    )
        else:
            self._drop_server(server.name)
        if connection is not None:
            state = "connected"
        else:
            self.statuses[server.name] = McpServerStatus(
                name=server.name,
                transport=server.transport,
                state="failed",
                error=error,
            )
            state = "failed"
        return McpRegistration(
            server=server,
            state=state,
            tool_count=len(bindings) if connection is not None else 0,
            error=error,
            replaced=existed,
        )

    def remove_server(self, server_name: str) -> str:
        """Remove one registered server from the config and drop its runtime state."""
        if self.config_error:
            raise McpConfigError(f"现有 MCP 配置无法解析，无法移除服务：{self.config_error}")
        if server_name not in set(self.configured_server_names()):
            raise McpConfigError(f"MCP 服务不存在：{server_name}")
        self._update_config_document(lambda servers: servers.pop(server_name, None))
        self._drop_server(server_name)
        self.config.servers.pop(server_name, None)
        return f"MCP 服务已移除：{server_name}"

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
        server = self.config.servers.get(server_name)
        if server is None:
            return f"MCP 服务不存在或未启用：{server_name}"
        self._drop_server(server_name)
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
    ) -> None:
        """Publish a connection; must run on the MCP event loop thread."""
        task = connection.task
        error = "connection worker exited before registration"
        if task is not None and task.done():
            if not task.cancelled():
                with contextlib.suppress(Exception):
                    exc = task.exception()
                    if exc is not None:
                        error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(error)
        for item in bindings:
            existing = self._bindings_by_exposed_name.get(item.exposed_name)
            if existing is not None and existing.server_name != server.name:
                raise RuntimeError(
                    f"duplicate exposed MCP tool name {item.exposed_name!r} "
                    f"for servers {existing.server_name!r} and {server.name!r}"
                )
        self._connections[server.name] = connection
        self.statuses[server.name] = McpServerStatus(
            name=server.name,
            transport=server.transport,
            state="connected",
            tool_count=len(bindings),
        )
        bindings = [self._bindings_by_exposed_name[item.exposed_name]
                    if item.exposed_name in self._catalog_only and self._bindings_by_exposed_name.get(item.exposed_name) == item
                    else item for item in bindings]
        self.tool_bindings = [item for item in self.tool_bindings if item.server_name != server.name] + bindings
        self._bindings_by_exposed_name.update({item.exposed_name: item for item in bindings})
        self._catalog_only.difference_update(item.exposed_name for item in bindings)
        with _static_lock:
            _schema_cache[self._schema_key(server)] = copy.deepcopy(tuple(bindings))
        self.connection_revision += 1

    async def _register_connection(
        self,
        server: McpServerConfig,
        connection: _McpConnection,
        bindings: list[McpToolBinding],
    ) -> None:
        """Replace one server, keeping the new worker owned if the old close hangs."""
        old = self._connections.pop(server.name, None)
        if old is not None and old is not connection:
            try:
                await asyncio.wait_for(
                    self._close_connections([old]), timeout=self.timeout_seconds
                )
            except Exception as exc:
                log.warning("Failed to close previous MCP server %s: %s", server.name, exc)
                if old.task is not None and not old.task.done():
                    old.task.cancel()
        self._drop_server_state(server.name)
        self._store_connection(server, connection, bindings)

    def _on_worker_done(self, connection: _McpConnection) -> None:
        """Reclaim a server whose connection worker exited outside a close."""
        server_name = connection.config.name
        if self._connections.get(server_name) is not connection:
            return
        self._connections.pop(server_name, None)
        task = connection.task
        error = "connection worker exited"
        if task is not None and not task.cancelled():
            with contextlib.suppress(Exception):
                exc = task.exception()
                if exc is not None:
                    error = f"{type(exc).__name__}: {exc}"
        self.statuses[server_name] = McpServerStatus(
            name=server_name,
            transport=connection.config.transport,
            state="failed",
            error=error,
        )
        self.tool_bindings = [
            binding for binding in self.tool_bindings if binding.server_name != server_name
        ]
        self._bindings_by_exposed_name = {
            binding.exposed_name: binding for binding in self.tool_bindings
        }
        self.connection_revision += 1
        log.warning("MCP server %s disconnected unexpectedly: %s", server_name, error)

    def _drop_server(self, server_name: str) -> None:
        """Drop one server's runtime state on the connection event loop."""
        if self._loop_thread is None:
            self._drop_server_state(server_name)
            return
        self._run(self._drop_server_async(server_name))

    async def _drop_server_async(self, server_name: str) -> None:
        connection = self._connections.pop(server_name, None)
        try:
            if connection is not None:
                await self._close_connections([connection])
        except Exception as exc:
            log.warning("Failed to close MCP server %s: %s", server_name, exc)
        finally:
            self._drop_server_state(server_name)

    def _drop_server_state(self, server_name: str) -> None:
        self.statuses.pop(server_name, None)
        self.tool_bindings = [
            binding for binding in self.tool_bindings if binding.server_name != server_name
        ]
        self._bindings_by_exposed_name = {
            binding.exposed_name: binding for binding in self.tool_bindings
        }
        self.connection_revision += 1

    async def _connect_one_and_store(self, server: McpServerConfig) -> None:
        try:
            connection, bindings = await self._connect_one(server)
            self._store_connection(server, connection, bindings)
        except Exception as exc:
            self.statuses[server.name] = McpServerStatus(
                name=server.name,
                transport=server.transport,
                state="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
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
            server = self.config.servers.get(name)
            if server is not None:
                self.config.servers[name] = replace(server, description=description)

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
        failed = [status for status in self.statuses.values() if status.state != "connected"]
        if failed:
            details = "; ".join(f"{item.name}: {item.error}" for item in failed)
            return False, details
        return True, f"{len(self.config.servers)} server(s), {len(self.tool_bindings)} tool(s)"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._loop_thread is None:
            self._resource_owner()
            return
        try:
            try:
                self._run(self._close_async())
            except Exception as exc:
                log.warning("MCP shutdown reported an error: %s", exc)
        finally:
            self._loop_thread.close()
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

            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            list_result = await session.list_tools()
            tools = list(getattr(list_result, "tools", []) or [])
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
        connections = list(self._connections.values())
        self._connections.clear()
        await self._close_connections(connections)

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

    def _ensure_loop(self) -> None:
        if self._loop_thread is None:
            self._loop_thread = _AsyncLoopThread()

    def _run(self, coro, *, cancellation_token=None, timeout: float | None = None):
        self._ensure_loop()
        assert self._loop_thread is not None
        return self._loop_thread.run(
            coro,
            timeout=self.timeout_seconds if timeout is None else timeout,
            cancellation_token=cancellation_token,
        )


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
        tool_name = str(getattr(tool, "name", "") or "").strip()
        if not tool_name:
            continue
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
                input_schema=dict(getattr(tool, "inputSchema", None) or {"type": "object", "properties": {}}),
                permission=permission,
                annotations=annotations,
            )
        )
    return bindings


_MISSING_DESCRIPTION_TEXT = (
    "description unavailable; call search_tools with this server id to inspect its tools"
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
                continue

    def _cancel(self, future: Future, actor_holder: list[asyncio.Task | None]) -> None:
        future.cancel()
        actor_task = actor_holder[0]
        if actor_task is not None and not actor_task.done():
            self.loop.call_soon_threadsafe(actor_task.cancel)

    def close(self) -> None:
        if self._queue is not None:
            future: Future = Future()

            def submit_stop() -> None:
                assert self._queue is not None
                self._queue.put_nowait((None, future))

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
                if target.done():
                    return
                if done.cancelled():
                    target.set_exception(TimeoutError("MCP operation timed out and was cancelled"))
                    return
                error = done.exception()
                if error is not None:
                    target.set_exception(error)
                else:
                    target.set_result(done.result())

            task.add_done_callback(_finish)
