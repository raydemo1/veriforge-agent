"""Deterministic stdio server used to exercise the LSP transport boundary."""

import json
import os
import sys
import time

mode = sys.argv[1]
documents = {}
versions = {}
watched = []


def send(message):
    data = json.dumps({"jsonrpc": "2.0", **message}).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    sys.stdout.buffer.flush()


def receive():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line == b"\r\n":
            break
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    return json.loads(sys.stdin.buffer.read(int(headers["content-length"])))


def diagnostic(uri):
    text = documents[uri]
    return [
        {
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 0, "character": 1},
            },
            "severity": 1,
            "message": text.strip(),
            "source": "fixture",
        }
    ]


while message := receive():
    method = message.get("method")
    params = message.get("params") or {}
    if method == "exit":
        break
    if method == "workspace/didChangeWatchedFiles":
        watched = params["changes"]
    if method == "textDocument/didOpen":
        doc = params["textDocument"]
        documents[doc["uri"]] = doc["text"]
        versions[doc["uri"]] = doc["version"]
    elif method == "textDocument/didChange":
        doc = params["textDocument"]
        documents[doc["uri"]] = params["contentChanges"][0]["text"]
        versions[doc["uri"]] = doc["version"]
    if method in {"textDocument/didOpen", "textDocument/didChange"} and mode.startswith(
        "push"
    ):
        uri = doc["uri"]
        if mode == "push-canonical":
            uri = uri.replace("file:///", "file://localhost/")
            if os.name == "nt":
                uri = uri.replace(uri[17:19], uri[17:18].lower() + "%3A", 1)
        send(
            {
                "method": "textDocument/publishDiagnostics",
                "params": {
                    "uri": uri,
                    "version": versions[doc["uri"]],
                    "diagnostics": diagnostic(doc["uri"]),
                },
            }
        )
    if "id" not in message:
        continue
    if mode == "crash" and method != "shutdown":
        sys.exit(7)
    if mode == "timeout" and method != "shutdown":
        time.sleep(30)
    if method == "initialize":
        caps = {
            "definitionProvider": True,
            "referencesProvider": True,
            "documentSymbolProvider": True,
            "textDocumentSync": 1,
            "positionEncoding": "utf-16",
        }
        if mode == "ts-sync":
            caps["executeCommandProvider"] = {
                "commands": ["typescript.tsserverRequest"]
            }
        elif not mode.startswith("push"):
            caps["diagnosticProvider"] = {
                "interFileDependencies": False,
                "workspaceDiagnostics": False,
            }
        result = {"capabilities": caps}
    elif method == "shutdown":
        result = None
    elif mode == "query-crash":
        sys.exit(8)
    elif mode == "malformed":
        result = {"wrong": "shape"}
    elif mode == "bad-json":
        sys.stdout.buffer.write(b"Content-Length: 1\r\n\r\n{")
        sys.stdout.buffer.flush()
        continue
    elif method == "textDocument/diagnostic":
        result = {"kind": "full", "items": diagnostic(params["textDocument"]["uri"])}
    elif mode == "ts-sync" and method == "workspace/executeCommand":
        assert params["command"] == "typescript.tsserverRequest"
        command, args, config = params["arguments"]
        assert command in {
            "syntacticDiagnosticsSync",
            "semanticDiagnosticsSync",
            "suggestionDiagnosticsSync",
        }
        uri = args["file"]
        send(
            {
                "method": "textDocument/publishDiagnostics",
                "params": {
                    "uri": uri,
                    "diagnostics": [{**diagnostic(uri)[0], "message": "stale"}],
                },
            }
        )
        result = {
            "success": True,
            "body": [
                {
                    "start": {"line": 1, "offset": 1},
                    "end": {"line": 1, "offset": 2},
                    "text": documents[uri],
                    "category": "error",
                }
            ]
            if command == "semanticDiagnosticsSync"
            else [],
        }
    elif method == "textDocument/documentSymbol":
        result = [
            {
                "name": documents[params["textDocument"]["uri"]].strip(),
                "kind": 12,
                "range": {"start": {"line": 0, "character": 0}},
                "selectionRange": {"start": {"line": 0, "character": 0}},
                "children": [
                    {
                        "name": "child",
                        "kind": 13,
                        "range": {"start": {"line": 1, "character": 2}},
                    }
                ],
            }
        ]
    elif method in {"textDocument/definition", "textDocument/references"}:
        uri = params["textDocument"]["uri"]
        if mode == "watch":
            assert any(change["uri"].endswith("other.py") for change in watched)
        if mode == "configuration":
            send(
                {
                    "id": "config",
                    "method": "workspace/configuration",
                    "params": {"items": [{"section": "python"}]},
                }
            )
            assert receive()["result"] == [{}]
            send(
                {"id": "folders", "method": "workspace/workspaceFolders", "params": {}}
            )
            assert receive()["result"][0]["uri"]
        if mode == "apply-edit":
            send(
                {
                    "id": "edit",
                    "method": "workspace/applyEdit",
                    "params": {"edit": {"changes": {uri: []}}},
                }
            )
            reply = receive()
            assert reply["result"]["applied"] is False
        result = [{"uri": uri, "range": {"start": params["position"]}}]
        if mode == "link":
            result = [
                {
                    "targetUri": uri,
                    "targetSelectionRange": {"start": params["position"]},
                }
            ]
    else:
        result = {"pid": os.getpid()}
    send({"id": message["id"], "result": result})
