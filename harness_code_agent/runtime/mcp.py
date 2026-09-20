from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agent.cancellation import CancelledError
from .permissions import (
    TOOL_PERMISSION_DANGEROUS,
    TOOL_PERMISSION_NETWORK_READ,
    TOOL_PERMISSION_READ,
    VALID_TOOL_PERMISSIONS,
)
from .tool_result import ToolResult

MCP_CONFIG_RELATIVE_PATH = Path(".harness") / "mcp.json"
MCP_TOOL_PREFIX = "mcp__"
MCP_TOOL_NAME_LIMIT = 64
MCP_OUTPUT_LIMIT = 60_000
MCPORTER_CONFIG_ENV = "HARNESS_MCPORTER_CONFIG"


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
        if _is_exa_search_tool(self):
            description += "\n\nPreferred search tool for web research."
        return {
            "type": "function",
            "function": {
                "name": self.exposed_name,
                "description": description,
                "parameters": parameters,
            },
        }


@dataclass
class McpServerStatus:
    name: str
    transport: str
    state: str
    tool_count: int = 0
    error: str | None = None


@dataclass
class _McpConnection:
    config: McpServerConfig
    request_queue: asyncio.Queue
    task: asyncio.Task


def load_mcp_config(workspace: str | Path) -> McpConfig:
    root = Path(workspace).resolve()
    path = root / MCP_CONFIG_RELATIVE_PATH
    if not path.exists():
        mcporter_path = _find_mcporter_config(root)
        if mcporter_path is None:
            return McpConfig(path=path)
        path = mcporter_path
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise McpConfigError(f"Invalid MCP config JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise McpConfigError("MCP config must be a JSON object")
    raw_servers = data.get("servers")
    if raw_servers is None and path.name == "mcporter.json":
        raw_servers = data.get("mcpServers", {})
    if not isinstance(raw_servers, dict):
        raise McpConfigError("MCP config field 'servers' must be an object")

    servers: dict[str, McpServerConfig] = {}
    for name, raw in raw_servers.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise McpConfigError(f"Invalid MCP server name: {name!r}")
        if not isinstance(raw, dict):
            raise McpConfigError(f"MCP server {name!r} must be an object")
        if raw.get("enabled", True) is False:
            continue
        if path.name == "mcporter.json" and name.lower() == "exa" and not _mcporter_exa_has_credentials(raw):
            continue

        expanded = _expand_env(raw)
        transport = str(expanded.get("transport") or "").strip()
        if not transport and path.name == "mcporter.json":
            transport = _mcporter_transport(expanded)
        if transport == "streamable-http":
            transport = "streamable_http"
        if transport not in {"stdio", "streamable_http"}:
            raise McpConfigError(f"MCP server {name!r} has unsupported transport: {transport!r}")

        default_permission = TOOL_PERMISSION_DANGEROUS
        if path.name == "mcporter.json" and name.lower() == "exa":
            default_permission = TOOL_PERMISSION_NETWORK_READ
        permission = str(expanded.get("permission") or default_permission)
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
            )
        else:
            url = str(expanded.get("url") or expanded.get("baseUrl") or "").strip()
            if not url:
                raise McpConfigError(f"MCP streamable_http server {name!r} requires url")
            servers[name] = McpServerConfig(
                name=name,
                transport=transport,
                permission=permission,
                url=url,
                headers=_string_dict(expanded.get("headers") or {}, f"server {name!r} headers"),
                tool_permissions=tool_permissions,
            )

    return McpConfig(path=path, servers=servers)


def _find_mcporter_config(workspace: Path) -> Path | None:
    """Find an existing MC Porter config without creating a second API config.

    A workspace-local VeriForge config always wins.  The global MC Porter config
    is intentionally considered only for the active process workspace, so unit
    tests and programmatic sessions pointed at another directory do not
    unexpectedly connect to the user's personal MCP servers.
    """
    configured = os.environ.get(MCPORTER_CONFIG_ENV, "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(workspace / ".mcporter" / "mcporter.json")
    if workspace == Path.cwd().resolve():
        candidates.append(Path.home() / ".mcporter" / "mcporter.json")
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return None


def _mcporter_transport(server: dict[str, Any]) -> str:
    if server.get("command"):
        return "stdio"
    if server.get("baseUrl") or server.get("url"):
        return "streamable_http"
    return ""


def _mcporter_exa_has_credentials(server: dict[str, Any]) -> bool:
    """Allow Exa when its MC Porter entry contains a literal credential.

    MC Porter may still use an environment placeholder in other setups, so
    that form remains supported.  This check only prevents registering an Exa
    server whose configured Authorization header is empty or unresolved.
    """
    headers = server.get("headers")
    if not isinstance(headers, dict):
        return False
    authorization = headers.get("Authorization") or headers.get("authorization")
    if not isinstance(authorization, str):
        return False
    value = authorization.strip()
    if not value or "${EXA_API_KEY}" in value:
        return bool(os.environ.get("EXA_API_KEY", "").strip())
    return True


class McpClientManager:
    def __init__(
        self,
        *,
        workspace: str | Path,
        config: McpConfig | None = None,
        config_error: str | None = None,
        timeout_seconds: float = 30.0,
    ):
        self.workspace = Path(workspace).resolve()
        self.config = config or McpConfig(path=self.workspace / MCP_CONFIG_RELATIVE_PATH)
        self.config_error = config_error
        self.timeout_seconds = timeout_seconds
        self.statuses: dict[str, McpServerStatus] = {}
        self.tool_bindings: list[McpToolBinding] = []
        self._bindings_by_exposed_name: dict[str, McpToolBinding] = {}
        self._connections: dict[str, _McpConnection] = {}
        self._loop_thread: _AsyncLoopThread | None = None

    @classmethod
    def from_workspace(cls, workspace: str | Path) -> McpClientManager:
        try:
            config = load_mcp_config(workspace)
            return cls(workspace=workspace, config=config)
        except McpConfigError as exc:
            root = Path(workspace).resolve()
            return cls(
                workspace=root,
                config=McpConfig(path=root / MCP_CONFIG_RELATIVE_PATH),
                config_error=str(exc),
            )

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
        self._ensure_loop()
        self._run(self._connect_all_parallel())

    async def _connect_all_parallel(self) -> None:
        """Connect to all configured MCP servers concurrently."""
        servers = list(self.config.servers.values())
        results = await asyncio.gather(
            *(self._connect_one(server) for server in servers),
            return_exceptions=True,
        )
        for server, result in zip(servers, results):
            if isinstance(result, BaseException):
                self.statuses[server.name] = McpServerStatus(
                    name=server.name,
                    transport=server.transport,
                    state="failed",
                    error=f"{type(result).__name__}: {result}",
                )
                continue
            connection, bindings = result
            self._connections[server.name] = connection
            self.statuses[server.name] = McpServerStatus(
                name=server.name,
                transport=server.transport,
                state="connected",
                tool_count=len(bindings),
            )
            self.tool_bindings.extend(bindings)
            self._bindings_by_exposed_name.update({binding.exposed_name: binding for binding in bindings})

    async def _connect_one(self, server: McpServerConfig) -> tuple[_McpConnection, list[McpToolBinding]]:
        """Connect to a single server with a per-server timeout."""
        request_queue: asyncio.Queue = asyncio.Queue()
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(self._connection_worker(server, request_queue, ready))
        try:
            bindings = await asyncio.wait_for(asyncio.shield(ready), timeout=self.timeout_seconds)
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
        return _McpConnection(config=server, request_queue=request_queue, task=task), bindings

    def register_tools(self, registry) -> None:
        from .execution_planner import CallEffect, ResourceClaim
        from .tool_registry import (
            TOOL_CAPABILITY_MAIN,
            TOOL_CAPABILITY_READONLY_AGENT,
            TOOL_CAPABILITY_WORKER_AGENT,
        )

        for binding in self.tool_bindings:
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
                disclosure="core" if _is_exa_search_tool(binding) else "deferred",
                capabilities=(
                    {
                        TOOL_CAPABILITY_MAIN,
                        TOOL_CAPABILITY_READONLY_AGENT,
                        TOOL_CAPABILITY_WORKER_AGENT,
                    }
                    if binding.permission in {TOOL_PERMISSION_READ, TOOL_PERMISSION_NETWORK_READ}
                    else {TOOL_CAPABILITY_MAIN}
                ),
            )

    def call_tool(
        self,
        exposed_name: str,
        arguments: dict[str, Any],
        *,
        cancellation_token=None,
    ) -> ToolResult:
        binding = self._bindings_by_exposed_name.get(exposed_name)
        if binding is None:
            return ToolResult(
                tool=exposed_name,
                status="failed",
                output=f"[error] Unknown MCP tool: {exposed_name}",
                error=f"Unknown MCP tool: {exposed_name}",
                metadata={"status_source": "mcp"},
            )
        if self._uses_mcporter_cli(binding):
            return self._call_via_mcporter(
                binding,
                arguments,
                cancellation_token=cancellation_token,
            )
        connection = self._connections.get(binding.server_name)
        if connection is None:
            return ToolResult(
                tool=exposed_name,
                status="failed",
                output=f"[error] MCP server is not connected: {binding.server_name}",
                error=f"MCP server is not connected: {binding.server_name}",
                metadata={"status_source": "mcp", "server": binding.server_name, "tool": binding.tool_name},
            )
        try:
            return self._run(
                self._call_tool(connection, binding, dict(arguments or {})),
                cancellation_token=cancellation_token,
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
                metadata={"status_source": "mcp", "server": binding.server_name, "tool": binding.tool_name},
            )

    def _uses_mcporter_cli(self, binding: McpToolBinding) -> bool:
        return self.config.path.name == "mcporter.json" and binding.server_name in self.config.servers

    def _call_via_mcporter(
        self,
        binding: McpToolBinding,
        arguments: dict[str, Any],
        *,
        cancellation_token=None,
    ) -> ToolResult:
        command = _mcporter_cli_command()
        if command is None:
            return ToolResult(
                tool=binding.exposed_name,
                status="failed",
                output="[error] mcporter command not found",
                error="mcporter command not found",
                metadata={"status_source": "mcporter", "server": binding.server_name, "tool": binding.tool_name},
            )
        argv = command + [
            "--config",
            str(self.config.path),
            "call",
            f"{binding.server_name}.{binding.tool_name}",
            "--args",
            json.dumps(dict(arguments or {}), ensure_ascii=False, separators=(",", ":")),
            "--output",
            "json",
        ]
        environment = os.environ.copy()
        server = self.config.servers.get(binding.server_name)
        if server is not None and _has_literal_authorization(server.headers):
            environment.pop("EXA_API_KEY", None)
        try:
            process = subprocess.Popen(
                argv,
                cwd=str(self.workspace),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=(
                    getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    if os.name == "nt"
                    else 0
                ),
                start_new_session=os.name != "nt",
            )
        except OSError as exc:
            error = f"{type(exc).__name__}: {exc}"
            return ToolResult(
                tool=binding.exposed_name,
                status="failed",
                output=f"[error] mcporter call failed: {error}",
                error=error,
                metadata={"status_source": "mcporter", "server": binding.server_name, "tool": binding.tool_name},
            )

        deadline = time.monotonic() + self.timeout_seconds
        while True:
            if cancellation_token is not None and cancellation_token.is_cancelled:
                _terminate_process_tree(process)
                raise CancelledError("Turn cancelled by user")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_process_tree(process)
                return ToolResult(
                    tool=binding.exposed_name,
                    status="failed",
                    output="[error] mcporter call timed out",
                    error="mcporter call timed out",
                    metadata={"status_source": "mcporter", "server": binding.server_name, "tool": binding.tool_name},
                )
            try:
                stdout, stderr = process.communicate(timeout=min(0.05, remaining))
                break
            except subprocess.TimeoutExpired:
                continue

        raw_output = stdout.strip()
        if process.returncode != 0:
            error = stderr.strip() or raw_output or f"mcporter exited with code {process.returncode}"
            return ToolResult(
                tool=binding.exposed_name,
                status="failed",
                output=f"[error] {error}",
                error=error,
                metadata={
                    "status_source": "mcporter",
                    "server": binding.server_name,
                    "tool": binding.tool_name,
                    "return_code": process.returncode,
                },
            )
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            error = f"mcporter returned invalid JSON: {exc}"
            return ToolResult(
                tool=binding.exposed_name,
                status="failed",
                output=f"[error] {error}",
                error=error,
                metadata={"status_source": "mcporter", "server": binding.server_name, "tool": binding.tool_name},
            )
        return _mcporter_json_to_tool_result(binding, payload)

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

    def configured_server_names(self) -> list[str]:
        """Return server names from the raw config, including disabled ones."""
        if not self.config.path.exists():
            return sorted(self.config.servers)
        try:
            data = json.loads(self.config.path.read_text(encoding="utf-8"))
            servers = data.get("servers") if isinstance(data, dict) else None
            if servers is None and self.config.path.name == "mcporter.json":
                servers = data.get("mcpServers", {}) if isinstance(data, dict) else {}
            return sorted(str(name) for name in servers if isinstance(name, str))
        except (OSError, json.JSONDecodeError, AttributeError):
            return sorted(self.config.servers)

    def reload_server(self, server_name: str) -> str:
        """Reconnect the requested MCP server without requiring a new TUI session.

        The connection loop is deliberately rebuilt as a small, deterministic
        operation. This keeps tool bindings in sync and avoids leaving stale
        handlers behind after a failed reconnect.
        """
        if server_name not in self.config.servers:
            return f"MCP 服务不存在或未启用：{server_name}"
        self.close()
        self.statuses.clear()
        self.tool_bindings.clear()
        self._bindings_by_exposed_name.clear()
        self.connect_all()
        return f"MCP 服务已重新连接：{server_name}"

    async def _connect_one_and_store(self, server: McpServerConfig) -> None:
        try:
            connection, bindings = await self._connect_one(server)
        except Exception as exc:
            self.statuses[server.name] = McpServerStatus(
                name=server.name,
                transport=server.transport,
                state="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._connections[server.name] = connection
        self.statuses[server.name] = McpServerStatus(
            name=server.name,
            transport=server.transport,
            state="connected",
            tool_count=len(bindings),
        )
        self.tool_bindings.extend(bindings)
        self._bindings_by_exposed_name.update({item.exposed_name: item for item in bindings})

    def set_server_enabled(self, server_name: str, enabled: bool) -> str:
        """Toggle one raw config entry while preserving environment placeholders."""
        path = self.config.path
        if not path.exists():
            return f"MCP 配置不存在：{path}"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise McpConfigError(f"无法读取 MCP 配置：{exc}") from exc
        servers = data.get("servers") if isinstance(data, dict) else None
        if servers is None and self.config.path.name == "mcporter.json":
            servers = data.get("mcpServers") if isinstance(data, dict) else None
        if not isinstance(servers, dict) or server_name not in servers:
            return f"MCP 服务不存在：{server_name}"
        entry = servers[server_name]
        if not isinstance(entry, dict):
            return f"MCP 服务配置无效：{server_name}"
        if enabled:
            entry.pop("enabled", None)
        else:
            entry["enabled"] = False
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return f"MCP 服务已{'启用' if enabled else '停用'}：{server_name}"

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
        if self._loop_thread is None:
            return
        try:
            self._run(self._close_async())
        finally:
            self._loop_thread.close()
            self._loop_thread = None

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

                http_client = httpx.AsyncClient(headers=dict(server.headers), timeout=self.timeout_seconds)
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
                    result = await session.call_tool(binding.tool_name, arguments)
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
        response = asyncio.get_running_loop().create_future()
        await connection.request_queue.put(("call", (binding, arguments), response))
        try:
            return await response
        except asyncio.CancelledError:
            response.cancel()
            raise

    async def _close_async(self) -> None:
        connections = list(self._connections.values())
        self._connections.clear()
        for connection in reversed(connections):
            response = asyncio.get_running_loop().create_future()
            await connection.request_queue.put(("close", None, response))
            await response
            with contextlib.suppress(Exception):
                await connection.task

    def _handler_for(self, binding: McpToolBinding) -> Callable[..., ToolResult]:
        def handler(*, cancellation_token=None, **kwargs) -> ToolResult:
            arguments = {
                key: value
                for key, value in kwargs.items()
                if key not in {"runtime_state", "agent_name", "tool_context"}
            }
            return self.call_tool(
                binding.exposed_name,
                arguments,
                cancellation_token=cancellation_token,
            )

        return handler

    def _ensure_loop(self) -> None:
        if self._loop_thread is None:
            self._loop_thread = _AsyncLoopThread()

    def _run(self, coro, *, cancellation_token=None):
        self._ensure_loop()
        assert self._loop_thread is not None
        return self._loop_thread.run(
            coro,
            timeout=self.timeout_seconds,
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
    if len(output) > MCP_OUTPUT_LIMIT:
        output = output[:MCP_OUTPUT_LIMIT] + f"\n\n[TRUNCATED: {len(output) - MCP_OUTPUT_LIMIT} chars omitted]"
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


def _bindings_for_server(server: McpServerConfig, mcp_tools: list[Any]) -> list[McpToolBinding]:
    used_names: set[str] = set()
    bindings: list[McpToolBinding] = []
    for tool in mcp_tools:
        tool_name = str(getattr(tool, "name", "") or "").strip()
        if not tool_name:
            continue
        exposed_name = _exposed_tool_name(server.name, tool_name, used_names)
        used_names.add(exposed_name)
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


def _is_exa_search_tool(binding: McpToolBinding) -> bool:
    identity = f"{binding.server_name} {binding.tool_name}".lower()
    return "exa" in identity and "search" in identity


def _mcporter_cli_command() -> list[str] | None:
    executable = shutil.which("mcporter")
    if not executable:
        return None
    path = Path(executable)
    suffix = path.suffix.lower()
    if suffix == ".ps1":
        shell = shutil.which("pwsh") or shutil.which("powershell")
        return [shell, "-NoProfile", "-NonInteractive", "-File", str(path)] if shell else None
    if suffix in {".cmd", ".bat"}:
        shell = shutil.which("cmd.exe") or shutil.which("cmd")
        return [shell, "/d", "/c", str(path)] if shell else None
    return [str(path)]


def _has_literal_authorization(headers: dict[str, str]) -> bool:
    authorization = headers.get("Authorization") or headers.get("authorization") or ""
    return bool(authorization.strip()) and "${" not in authorization


def _mcporter_json_to_tool_result(binding: McpToolBinding, payload: Any) -> ToolResult:
    if not isinstance(payload, dict):
        output = str(payload)
        return ToolResult(
            tool=binding.exposed_name,
            status="success",
            output=output,
            metadata={"status_source": "mcporter", "server": binding.server_name, "tool": binding.tool_name},
        )
    parts = [
        _mcporter_content_item_to_text(item)
        for item in list(payload.get("content") or [])
        if isinstance(item, dict)
    ]
    structured = payload.get("structuredContent")
    if structured is not None:
        parts.append("structuredContent:\n" + json.dumps(structured, ensure_ascii=False, indent=2, sort_keys=True))
    output = "\n\n".join(part for part in parts if part)
    if len(output) > MCP_OUTPUT_LIMIT:
        output = output[:MCP_OUTPUT_LIMIT] + f"\n\n[TRUNCATED: {len(output) - MCP_OUTPUT_LIMIT} chars omitted]"
    is_error = bool(payload.get("isError"))
    return ToolResult(
        tool=binding.exposed_name,
        status="failed" if is_error else "success",
        output=output,
        error="mcporter tool returned isError=true" if is_error else None,
        metadata={
            "status_source": "mcporter",
            "server": binding.server_name,
            "tool": binding.tool_name,
        },
    )


def _mcporter_content_item_to_text(item: dict[str, Any]) -> str:
    item_type = str(item.get("type") or "")
    if item_type == "text":
        return str(item.get("text") or "")
    if item_type in {"image", "audio"}:
        data = str(item.get("data") or "")
        mime = str(item.get("mimeType") or "application/octet-stream")
        return f"[{item_type} {mime}, {len(data)} base64 chars omitted]"
    if item_type == "resource_link":
        return f"[resource_link {item.get('name') or ''} {item.get('uri') or ''}]"
    return f"[{item_type or 'unknown'} content omitted]"


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


def _exposed_tool_name(server_name: str, tool_name: str, used_names: set[str]) -> str:
    safe_server = _safe_name_segment(server_name)
    safe_tool = _safe_name_segment(tool_name)
    base = f"{MCP_TOOL_PREFIX}{safe_server}__{safe_tool}"
    if len(base) > MCP_TOOL_NAME_LIMIT:
        digest = hashlib.sha256(f"{server_name}/{tool_name}".encode()).hexdigest()[:8]
        server_budget = min(len(safe_server), 20)
        tool_budget = max(1, MCP_TOOL_NAME_LIMIT - len(MCP_TOOL_PREFIX) - server_budget - len("__") - len("__") - 8)
        base = f"{MCP_TOOL_PREFIX}{safe_server[:server_budget]}__{safe_tool[:tool_budget]}__{digest}"
    candidate = base
    index = 2
    while candidate in used_names:
        suffix = f"_{index}"
        candidate = base[: MCP_TOOL_NAME_LIMIT - len(suffix)] + suffix
        index += 1
    return candidate


def _safe_name_segment(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", value.strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "tool"


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        with contextlib.suppress(Exception):
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
    else:
        with contextlib.suppress(Exception):
            os.killpg(process.pid, 15)
    if process.poll() is None:
        with contextlib.suppress(Exception):
            process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(Exception):
            process.kill()


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

    def run(self, coro, *, timeout: float, cancellation_token=None):
        future: Future = Future()
        actor_holder: list[asyncio.Task | None] = [None]

        def submit() -> None:
            assert self._queue is not None
            self._queue.put_nowait((coro, future, actor_holder))

        self.loop.call_soon_threadsafe(submit)
        deadline = time.monotonic() + timeout
        while True:
            if cancellation_token is not None and cancellation_token.is_cancelled:
                self._cancel(future, actor_holder)
                raise CancelledError("Turn cancelled by user")
            remaining = deadline - time.monotonic()
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
        while True:
            item = await self._queue.get()
            coro, future = item[0], item[1]
            actor_holder: list[asyncio.Task | None] | None = item[2] if len(item) > 2 else None
            if coro is None:
                future.set_result(None)
                return
            if future.cancelled():
                coro.close()
                continue
            if actor_holder is not None:
                actor_holder[0] = asyncio.current_task()
            try:
                result = await coro
            except asyncio.CancelledError:
                if not future.done():
                    future.set_exception(TimeoutError("MCP operation timed out and was cancelled"))
                continue
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
            else:
                if not future.done():
                    future.set_result(result)
