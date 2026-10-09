"""Lazy, read-only language-server access for one workspace."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import os
import re
import threading
from pathlib import Path
from urllib.parse import unquote, urlparse

import psutil
from pygls.client import JsonRPCClient
from pygls.protocol import (
    JsonRPCNotification,
    JsonRPCRequestMessage,
    JsonRPCResponseMessage,
    default_converter,
)

from .source_watch import SourceWatch
from .verification.languages import project_root

_LANGUAGES = {
    ".py": ("python", "pyproject.toml"),
    ".ts": ("typescript", "package.json"),
    ".tsx": ("typescriptreact", "package.json"),
    ".mts": ("typescript", "package.json"),
    ".cts": ("typescript", "package.json"),
    ".js": ("javascript", "package.json"),
    ".jsx": ("javascriptreact", "package.json"),
    ".mjs": ("javascript", "package.json"),
    ".cjs": ("javascript", "package.json"),
    ".go": ("go", "go.mod"),
    ".rs": ("rust", "Cargo.toml"),
}
_TIMEOUT = 8
_MAX_ITEMS = 200


def _external_file(path: Path, workspace: Path) -> Path | None:
    try:
        resolved = path.resolve(strict=True)
        if resolved.is_file() and not resolved.is_relative_to(workspace):
            return resolved
    except (OSError, RuntimeError):
        pass
    return None


def _system_executables(name: str, workspace: Path):
    extensions = (
        [
            ext
            for ext in os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";")
            if ext.lower() in {".exe", ".com", ".bat", ".cmd"}
        ]
        if os.name == "nt"
        else [""]
    )
    for entry in os.get_exec_path():
        directory = Path(os.path.expandvars(entry.strip('"')))
        if not directory.is_absolute():
            continue
        for extension in extensions:
            executable = _external_file(directory / (name + extension), workspace)
            if executable and os.access(executable, os.X_OK):
                yield executable


def _node_server(executable: Path, workspace: Path, package: str) -> list[str] | None:
    node = next(
        (
            path
            for path in _system_executables("node", workspace)
            if path.suffix.lower() not in {".cmd", ".bat"}
        ),
        None,
    )
    if node and executable.suffix.lower() in {".js", ".mjs", ".cjs"}:
        return [str(node), str(executable), "--stdio"]
    manifest = _external_file(
        executable.parent / "node_modules" / package / "package.json", workspace
    )
    if node is None or manifest is None:
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        entry = data.get("bin")
        if isinstance(entry, dict):
            entry = entry.get("pyright-langserver" if package == "pyright" else package)
        script = (
            _external_file(manifest.parent / entry, workspace)
            if isinstance(entry, str)
            else None
        )
        if script:
            return [str(node), str(script), "--stdio"]
    except (OSError, ValueError, AttributeError):
        pass
    return None


def _typescript_server(workspace: Path) -> Path:
    for executable in _system_executables("tsserver", workspace):
        for candidate in (
            executable.parent / "node_modules/typescript/lib/tsserver.js",
            executable.parent.parent / "lib/tsserver.js",
        ):
            server = _external_file(candidate, workspace)
            if server is None:
                continue
            manifest = _external_file(server.parent.parent / "package.json", workspace)
            if manifest is None:
                continue
            try:
                version = json.loads(manifest.read_text(encoding="utf-8")).get(
                    "version"
                )
                if isinstance(version, str) and re.fullmatch(
                    r"\d+\.\d+\.\d+(?:[-+][\w.-]+)?", version
                ):
                    return server
            except (OSError, ValueError, AttributeError):
                continue
    raise ValueError(
        "No system-installed TypeScript compiler; install typescript outside the workspace"
    )


def discover_server(language: str, workspace: Path) -> list[str] | None:
    workspace = workspace.resolve()
    names = (
        ("basedpyright-langserver", "pyright-langserver")
        if language == "python"
        else ("gopls",)
        if language == "go"
        else ("rust-analyzer",)
        if language == "rust"
        else ("typescript-language-server",)
    )
    package = "pyright" if language == "python" else "typescript-language-server"
    for name in names:
        for executable in _system_executables(name, workspace):
            if executable.suffix.lower() in {".cmd", ".bat", ".js", ".mjs", ".cjs"}:
                command = _node_server(executable, workspace, package)
                if command:
                    return command
                continue
            return (
                [str(executable), "--stdio"]
                if "langserver" in name or name == "typescript-language-server"
                else [str(executable)]
            )
    return None


def _converter():
    converter = default_converter()
    for message_type in (
        JsonRPCNotification,
        JsonRPCRequestMessage,
        JsonRPCResponseMessage,
    ):
        converter.register_structure_hook(message_type, lambda obj, cls: cls(**obj))
    return converter


class _Client(JsonRPCClient):
    def __init__(self):
        super().__init__(converter_factory=_converter)
        self.error: str | None = None

    def report_server_error(self, error, source):
        self.error = f"Invalid language-server response: {error}"


def _read_only_settings(family: str) -> dict:
    if family == "typescript":
        return {"disableAutomaticTypingAcquisition": True, "plugins": []}
    if family == "rust":
        return {
            "cargo": {"buildScripts": {"enable": False}, "extraArgs": ["--frozen"]},
            "procMacro": {"enable": False},
            "checkOnSave": False,
        }
    if family == "go":
        return {
            "env": {
                "GOFLAGS": "-mod=readonly",
                "GOPROXY": "off",
                "GOSUMDB": "off",
                "GOTOOLCHAIN": "local",
            }
        }
    return {}


class _Server:
    def __init__(self, root: Path, command: list[str], family: str, workspace: Path):
        self.root = root
        self.command = command
        self.client = _Client()
        self.capabilities = {}
        self.documents: dict[str, tuple[int, str]] = {}
        self.diagnostics: dict[Path, tuple[int | None, list]] = {}
        self.tasks: list[asyncio.Task] = []
        self.watch = None
        self.settings = _read_only_settings(family)
        if family == "typescript":
            self.settings["tsserver"] = {
                "path": str(_typescript_server(workspace)),
                "logVerbosity": "off",
            }
        self.family = family
        self.versions: dict[str, int] = {}

        @self.client.feature("textDocument/publishDiagnostics")
        def diagnostics(params):
            if not isinstance(params, dict) or not isinstance(
                params.get("diagnostics"), list
            ):
                self.client.error = "Malformed diagnostics notification"
                return
            self.diagnostics[_uri_path(params["uri"])] = (
                params.get("version"),
                params["diagnostics"],
            )

        @self.client.feature("workspace/configuration")
        def configuration(params):
            values = []
            for item in params.get("items", []):
                value = self.settings
                section = str(item.get("section") or "")
                namespace = {"rust": "rust-analyzer", "go": "gopls"}.get(self.family)
                if namespace and section.startswith(namespace):
                    section = section[len(namespace) :].lstrip(".")
                    for part in section.split(".") if section else []:
                        value = value.get(part, {}) if isinstance(value, dict) else {}
                elif section:
                    value = {}
                values.append(value)
            return values

        @self.client.feature("window/logMessage")
        def log_message(_params):
            pass

        @self.client.feature("$/typescriptVersion")
        def typescript_version(_params):
            pass

        @self.client.feature("workspace/workspaceFolders")
        def folders(_params):
            return [{"uri": root.as_uri(), "name": root.name}]

        @self.client.feature("workspace/applyEdit")
        def refuse_edit(_params):
            return {"applied": False, "failureReason": "Read-only language client"}

    async def start(self):
        self.watch = SourceWatch(self.root)
        await self.client.start_io(*self.command, cwd=self.root)
        self.tasks.append(asyncio.create_task(self._drain_stderr()))
        result = await self.request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": self.root.as_uri(),
                "workspaceFolders": [
                    {"uri": self.root.as_uri(), "name": self.root.name}
                ],
                "initializationOptions": self.settings,
                "capabilities": {
                    "general": {"positionEncodings": ["utf-16"]},
                    "workspace": {
                        "configuration": True,
                        "workspaceFolders": True,
                        "applyEdit": False,
                    },
                    "textDocument": {
                        "synchronization": {"didSave": True},
                        "definition": {"linkSupport": True},
                        "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                        "publishDiagnostics": {"versionSupport": True},
                        "diagnostic": {"dynamicRegistration": False},
                    },
                },
            },
        )
        if not isinstance(result, dict) or not isinstance(
            result.get("capabilities"), dict
        ):
            raise TypeError("Malformed initialize response")
        self.capabilities = result["capabilities"]
        if self.capabilities.get("positionEncoding", "utf-16") != "utf-16":
            raise ValueError("Unsupported server position encoding")
        self.client.protocol.notify("initialized", {})

    async def _drain_stderr(self):
        stream = self.client._server.stderr
        while await stream.read(4096):
            pass

    async def request(self, method, params):
        if self.client.error:
            raise ValueError(self.client.error)
        result = await asyncio.wait_for(
            self.client.protocol.send_request_async(method, params), _TIMEOUT
        )
        if self.client.error:
            raise ValueError(self.client.error)
        return result

    def sync(self, path: Path, language: str, text: str, *, refresh: bool = False):
        uri = path.as_uri()
        previous = self.documents.get(uri)
        if previous is None:
            version = self.versions.get(uri, 0) + 1
            self.diagnostics.pop(path, None)
            self.client.protocol.notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": language,
                        "version": version,
                        "text": text,
                    }
                },
            )
        elif previous[1] != text or refresh:
            version = previous[0] + 1
            self.diagnostics.pop(path, None)
            self.client.protocol.notify(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": version},
                    "contentChanges": [{"text": text}],
                },
            )
            self.client.protocol.notify(
                "textDocument/didSave", {"textDocument": {"uri": uri}, "text": text}
            )
        else:
            return previous[0]
        self.documents[uri] = (version, text)
        self.versions[uri] = version
        return version

    async def query(self, operation, path, language, text, line, column):
        changes = self.watch.changes()
        if changes:
            self.client.protocol.notify(
                "workspace/didChangeWatchedFiles", {"changes": changes}
            )
            self.diagnostics.clear()
        for uri in {item["uri"] for item in changes} & self.documents.keys():
            opened = _uri_path(uri)
            if not opened.is_file():
                self.client.protocol.notify(
                    "textDocument/didClose", {"textDocument": {"uri": uri}}
                )
                self.documents.pop(uri)
                self.diagnostics.pop(opened, None)
            else:
                self.sync(
                    opened,
                    _LANGUAGES[opened.suffix][0],
                    opened.read_text(encoding="utf-8"),
                )
        has_pull_diagnostics = (
            self.capabilities.get("diagnosticProvider") is not None
            and self.capabilities.get("diagnosticProvider") is not False
        )
        has_typescript_diagnostics = (
            self.family == "typescript"
            and "typescript.tsserverRequest"
            in self.capabilities.get("executeCommandProvider", {}).get("commands", [])
        )
        version = self.sync(
            path,
            language,
            text,
            refresh=operation == "diagnostics"
            and not has_pull_diagnostics
            and not has_typescript_diagnostics,
        )
        params = {"textDocument": {"uri": path.as_uri()}}
        if operation == "diagnostics":
            if has_pull_diagnostics:
                report = await self.request("textDocument/diagnostic", params)
                if (
                    not isinstance(report, dict)
                    or report.get("kind") != "full"
                    or not isinstance(report.get("items"), list)
                ):
                    raise ValueError("Malformed diagnostic report")
                return report["items"]
            if has_typescript_diagnostics:
                return await self._typescript_diagnostics(path)
            deadline = asyncio.get_running_loop().time() + _TIMEOUT
            while asyncio.get_running_loop().time() < deadline:
                if self.client.error:
                    raise ValueError(self.client.error)
                received = self.diagnostics.get(path)
                if received and (received[0] is None or received[0] == version):
                    return received[1]
                if self.client.stopped:
                    raise RuntimeError("Language server exited")
                await asyncio.sleep(0.05)
            raise TimeoutError("No fresh diagnostics received")
        capability = {
            "definition": "definitionProvider",
            "references": "referencesProvider",
            "symbols": "documentSymbolProvider",
        }[operation]
        if (
            self.capabilities.get(capability) is None
            or self.capabilities.get(capability) is False
        ):
            raise ValueError(f"Server does not support {operation}")
        if operation in {"definition", "references"}:
            rows = text.splitlines()
            if (
                line is None
                or column is None
                or line < 1
                or line > len(rows)
                or column < 1
                or column > len(rows[line - 1]) + 1
            ):
                raise ValueError(
                    "line/column must identify a position in the file (1-based)"
                )
            params["position"] = {
                "line": line - 1,
                "character": len(rows[line - 1][: column - 1].encode("utf-16-le")) // 2,
            }
        if operation == "references":
            params["context"] = {"includeDeclaration": True}
        method = {
            "definition": "definition",
            "references": "references",
            "symbols": "documentSymbol",
        }[operation]
        return await self.request("textDocument/" + method, params)

    async def _typescript_diagnostics(self, path: Path) -> list[dict]:
        reports = await asyncio.gather(
            *(
                self.request(
                    "workspace/executeCommand",
                    {
                        "command": "typescript.tsserverRequest",
                        "arguments": [
                            method,
                            {"file": path.as_uri()},
                            {"expectsResult": True, "lowPriority": False},
                        ],
                    },
                )
                for method in (
                    "syntacticDiagnosticsSync",
                    "semanticDiagnosticsSync",
                    "suggestionDiagnosticsSync",
                )
            )
        )
        items = []
        for report in reports:
            if (
                not isinstance(report, dict)
                or report.get("success") is not True
                or not isinstance(report.get("body"), list)
            ):
                raise ValueError("Malformed TypeScript diagnostic response")
            for detail in report["body"]:
                positions = {}
                for key in ("start", "end"):
                    position = detail[key]
                    if any(
                        type(position.get(field)) is not int or position[field] < 1
                        for field in ("line", "offset")
                    ):
                        raise ValueError("Malformed TypeScript diagnostic position")
                    positions[key] = {
                        "line": position["line"] - 1,
                        "character": position["offset"] - 1,
                    }
                items.append(
                    {
                        "range": positions,
                        "message": detail["text"],
                        "severity": {
                            "error": 1,
                            "warning": 2,
                            "message": 3,
                            "suggestion": 4,
                        }[detail["category"]],
                        "source": "typescript",
                    }
                )
        return items

    async def close(self):
        if self.watch is not None:
            self.watch.close()
            self.watch = None
        proc = self.client._server
        children = []
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(psutil.Error):
                children = psutil.Process(proc.pid).children(recursive=True)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    self.client.protocol.send_request_async("shutdown", None), 0.5
                )
                self.client.protocol.notify("exit", None)
                await asyncio.wait_for(proc.wait(), 0.5)
            if proc.returncode is None:
                with contextlib.suppress(psutil.Error):
                    parent = psutil.Process(proc.pid)
                    parent.kill()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(proc.wait(), 1)
        for child in reversed(children):
            with contextlib.suppress(psutil.Error):
                child.kill()
        if children:
            await asyncio.to_thread(psutil.wait_procs, children, timeout=1)
        for task in self.tasks + self.client._async_tasks:
            task.cancel()
        await asyncio.gather(
            *self.tasks, *self.client._async_tasks, return_exceptions=True
        )


def _uri_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValueError("Non-local file location")
    value = unquote(parsed.path)
    if os.name == "nt" and value.startswith("/") and len(value) > 2 and value[2] == ":":
        value = value[1:]
    return Path(value).resolve()


def _position(range_):
    if not isinstance(range_, dict) or not isinstance(range_.get("start"), dict):
        raise TypeError("Malformed location range")
    pos = range_["start"]
    if any(
        type(pos.get(key)) is not int or pos[key] < 0 for key in ("line", "character")
    ):
        raise ValueError("Malformed position")
    return pos["line"] + 1, pos["character"] + 1


def _normalize(operation: str, result, path: Path, workspace: Path) -> list[dict]:
    if result is None:
        return []
    items = (
        result
        if isinstance(result, list)
        else [result]
        if operation == "definition"
        else None
    )
    if items is None:
        raise ValueError("Malformed language-server result")
    normalized = []

    def add(item, parent=""):
        if not isinstance(item, dict):
            raise TypeError("Malformed result item")
        target = path
        name = ""
        if operation in {"definition", "references"}:
            target = _uri_path(item.get("uri") or item.get("targetUri") or "")
            range_ = item.get("range") or item.get("targetSelectionRange")
        elif operation == "symbols":
            name = str(item.get("name") or "")
            if not name:
                raise ValueError("Malformed symbol")
            name = f"{parent}.{name}" if parent else name
            location = item.get("location")
            if location:
                target = _uri_path(location.get("uri", ""))
                range_ = location.get("range")
            else:
                range_ = item.get("selectionRange") or item.get("range")
        else:
            if not isinstance(item.get("message"), str):
                raise ValueError("Malformed diagnostic")
            range_ = item.get("range")
        row, column = _position(range_)
        if target.is_relative_to(workspace) and target.is_file():
            rows = target.read_text(encoding="utf-8").splitlines()
            if row <= len(rows):
                prefix = rows[row - 1].encode("utf-16-le")[: (column - 1) * 2]
                column = len(prefix.decode("utf-16-le")) + 1
        display = (
            target.relative_to(workspace).as_posix()
            if target.is_relative_to(workspace)
            else str(target)
        )
        entry = {
            "path": display,
            "line": row,
            "column": column,
            "location": f"{display}:{row}:{column}",
        }
        if name:
            entry["name"] = name
        if operation == "diagnostics":
            entry.update(
                message=item["message"][:2_000],
                severity=item.get("severity"),
                source=item.get("source"),
            )
        normalized.append(entry)
        if operation == "symbols":
            children = item.get("children", [])
            if not isinstance(children, list):
                raise ValueError("Malformed symbol children")
            for child in children:
                if len(normalized) < _MAX_ITEMS:
                    add(child, name)

    for item in items:
        if len(normalized) >= _MAX_ITEMS:
            break
        add(item)
    return normalized


class LanguageService:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self._lock = threading.RLock()
        self._loop = None
        self._thread = None
        self._servers: dict[tuple[str, Path], _Server] = {}

    def _ensure_loop(self):
        if self._loop is not None:
            return
        ready = threading.Event()

        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            ready.set()
            self._loop.run_forever()
            self._loop.close()

        self._thread = threading.Thread(
            target=run, name="language-service", daemon=True
        )
        self._thread.start()
        ready.wait()

    def query(self, operation: str, path: str, line=None, column=None) -> dict:
        with self._lock:
            try:
                target = (self.workspace / path).resolve()
                if not target.is_relative_to(self.workspace):
                    raise ValueError("Path escapes workspace")
                if operation not in {
                    "definition",
                    "references",
                    "symbols",
                    "diagnostics",
                }:
                    raise ValueError("Unsupported operation")
                if target.suffix not in _LANGUAGES:
                    raise ValueError("Unsupported language")
                if target.stat().st_size > 2_000_000:
                    raise ValueError("File exceeds language-service size limit")
                text = target.read_text(encoding="utf-8")
                language, marker = _LANGUAGES[target.suffix]
                root = project_root(target, self.workspace, marker) or self.workspace
                family = (
                    "typescript"
                    if language.startswith(("typescript", "javascript"))
                    else language
                )
                key = (family, root)
                existing = self._servers.get(key)
                command = (
                    existing.command
                    if existing
                    else discover_server(language, self.workspace)
                )
                if command is None:
                    return {
                        "available": False,
                        "reason": f"No system-installed language server for {family}; workspace-local servers are not automatically executed",
                        "items": [],
                    }
                self._ensure_loop()
                future = asyncio.run_coroutine_threadsafe(
                    self._query(
                        key, command, operation, target, language, text, line, column
                    ),
                    self._loop,
                )
                try:
                    return future.result(timeout=_TIMEOUT * 2 + 4)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    raise TimeoutError("Language server did not respond") from None
            except Exception as exc:
                return {
                    "available": False,
                    "reason": str(exc) or type(exc).__name__,
                    "items": [],
                }

    async def _query(
        self, key, command, operation, target, language, text, line, column
    ):
        server = self._servers.get(key)
        try:
            if server is None:
                server = _Server(key[1], command, key[0], self.workspace)
                self._servers[key] = server
                await server.start()
            result = await server.query(operation, target, language, text, line, column)
            items = _normalize(operation, result, target, self.workspace)
            return {
                "available": True,
                "operation": operation,
                "items": items,
                "limit": _MAX_ITEMS,
                "server": Path(command[0]).name,
            }
        except (Exception, asyncio.CancelledError):
            if server is not None:
                await server.close()
            self._servers.pop(key, None)
            raise

    def close(self):
        with self._lock:
            if self._loop is None:
                return

            async def stop():
                await asyncio.gather(
                    *(server.close() for server in self._servers.values())
                )
                self._servers.clear()

            with contextlib.suppress(concurrent.futures.TimeoutError):
                asyncio.run_coroutine_threadsafe(stop(), self._loop).result(timeout=5)
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2)
            self._loop = None
            self._thread = None
