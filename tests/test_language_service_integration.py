from __future__ import annotations

import json
import time
from pathlib import Path

import psutil
import pytest

from harness_code_agent.runtime.language_service import (
    LanguageService,
    _typescript_server,
    discover_server,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("language", ["python", "typescript"])
def test_real_server_navigation_diagnostics_refresh_and_close(
    tmp_path, language, require_integration_tool
):
    require_integration_tool(
        discover_server(language, tmp_path) is not None,
        f"No system-installed {language} language server on PATH",
    )
    if language == "typescript":
        try:
            _typescript_server(tmp_path)
        except ValueError as exc:
            require_integration_tool(False, str(exc))
    python = language == "python"
    suffix = "py" if python else "ts"
    if python:
        (tmp_path / "pyrightconfig.json").write_text(
            '{"typeCheckingMode":"basic"}', encoding="utf-8"
        )
        library_text = "def greet(name: str) -> str:\n    return name\n"
        source_text = (
            'from library import greet\nresult = greet("ray")\nbad: int = "wrong"\n'
        )
    else:
        (tmp_path / "package.json").write_text(
            '{"name":"lsp-fixture","private":true}', encoding="utf-8"
        )
        (tmp_path / "tsconfig.json").write_text(
            json.dumps(
                {
                    "compilerOptions": {
                        "strict": True,
                        "noEmit": True,
                        "plugins": [{"name": "workspace-plugin"}],
                    },
                    "include": ["*.ts"],
                }
            ),
            encoding="utf-8",
        )
        library_text = "export function greet(name: string): string { return name; }\n"
        source_text = 'import { greet } from "./library";\nconst result = greet("ray");\nconst bad: number = "wrong";\n'
        local_compiler = tmp_path / "node_modules/typescript"
        (local_compiler / "lib").mkdir(parents=True)
        (local_compiler / "package.json").write_text(
            '{"version":"6.0.3"}', encoding="utf-8"
        )
        (local_compiler / "lib/tsserver.js").write_text(
            f"require('fs').writeFileSync({json.dumps(str(tmp_path / 'compiler-executed'))}, 'yes'); process.exit(7);",
            encoding="utf-8",
        )
        local_plugin = tmp_path / "node_modules/workspace-plugin"
        local_plugin.mkdir(parents=True)
        (local_plugin / "package.json").write_text(
            '{"main":"index.js"}', encoding="utf-8"
        )
        (local_plugin / "index.js").write_text(
            f"require('fs').writeFileSync({json.dumps(str(tmp_path / 'plugin-executed'))}, 'yes'); module.exports = () => ({{create: info => info.languageService}});",
            encoding="utf-8",
        )
    library = tmp_path / f"library.{suffix}"
    library.write_text(library_text, encoding="utf-8")
    source = tmp_path / f"main.{suffix}"
    source.write_text(source_text, encoding="utf-8")
    manager = LanguageService(tmp_path)
    processes = []

    def query(operation, path=source.name, line=None, column=None):
        result = manager.query(operation, path, line, column)
        assert result["available"], result
        return result["items"]

    def errors():
        return [item for item in query("diagnostics") if item["severity"] == 1]

    try:
        assert query("definition", line=2, column=10 if python else 16)
        references = query("references", library.name, 1, 5 if python else 17)
        assert {item["path"] for item in references} >= {source.name, library.name}
        assert {item["name"] for item in query("symbols")} >= {"result", "bad"}
        assert any(item["line"] == 3 for item in errors())
        source.write_text(source_text.replace('"wrong"', "1"), encoding="utf-8")
        assert errors() == []
        source.write_text(source_text, encoding="utf-8")
        assert any(item["line"] == 3 for item in errors())
        source.write_text(source_text.replace('"wrong"', "1"), encoding="utf-8")
        assert errors() == []
        # The imported file has not been opened by the client.
        library.write_text(library_text.replace("greet", "renamed"), encoding="utf-8")
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            findings = errors()
            if any("greet" in item["message"] for item in findings):
                break
        else:
            pytest.fail(f"Changed import was not reanalysed: {findings}")
        server = next(iter(manager._servers.values()))
        parent = psutil.Process(server.client._server.pid)
        processes = [parent, *parent.children(recursive=True)]
        if not python:
            assert not (tmp_path / "compiler-executed").exists()
            assert not (tmp_path / "plugin-executed").exists()
            assert not Path(server.settings["tsserver"]["path"]).is_relative_to(
                tmp_path
            )
    finally:
        manager.close()
    assert manager._thread is None
    assert not any(process.is_running() for process in processes)
