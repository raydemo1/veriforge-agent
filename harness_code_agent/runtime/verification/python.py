"""Python syntax and Ruff verification."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

from .engine import CheckResult

_RUFF_BLOCK_PREFIXES = ("E", "F")
_SUBPROCESS_TIMEOUT_SECONDS = 30


class PythonProvider:
    def supports(self, files, workspace):
        return any(path.endswith(".py") for path in files)

    def verify(self, files, workspace):
        files = [p for p in files if p.endswith(".py") and (workspace / p).is_file()]
        if not files:
            return []
        errors = _check_python_syntax(str(workspace), files)
        results = [
            CheckResult(
                "Python syntax",
                "failed" if errors else "passed",
                tuple(f"[syntax] {path}: {message}" for path, message in errors)
                or (f"{len(files)} file(s) parsed",),
            )
        ]
        findings, ran = _check_ruff(str(workspace), files)
        blocks = []
        warnings = []
        for path, code, message, row in findings:
            location = f"{path}:{row}" if row else path
            target = blocks if code[:1] in _RUFF_BLOCK_PREFIXES else warnings
            target.append(f"[{code}] {location}: {message}")
        results.append(
            CheckResult(
                "Ruff lint",
                "failed"
                if blocks
                else "warning"
                if warnings
                else "passed"
                if ran
                else "skipped",
                tuple(blocks or warnings)
                or (
                    ("no findings",)
                    if ran
                    else ("Ruff unavailable or unusable output",)
                ),
            )
        )
        return results


def _check_python_syntax(
    workspace_root: str | None,
    py_files: list[str],
) -> list[tuple[str, str]]:
    """``ast.parse`` each file: syntax errors without writing bytecode."""
    errors: list[tuple[str, str]] = []
    for rel_path in py_files:
        full_path = (
            Path(workspace_root) / rel_path if workspace_root else Path(rel_path)
        )
        try:
            source = full_path.read_text(encoding="utf-8", errors="replace")
            ast.parse(source, filename=str(full_path))
        except (OSError, SyntaxError) as exc:
            errors.append((rel_path, str(exc)))
    return errors


def _check_ruff(
    workspace_root: str | None,
    py_files: list[str],
) -> tuple[list[tuple[str, str, str, int | None]], bool]:
    """Run ruff on the given files via JSON output.

    Returns ``(findings, ran)`` where ``ran`` tells whether ruff actually
    executed (so an absent binary is never reported as a passing check).
    Ruff-invocation problems come back as a single non-blocking warning item
    (rule code not starting with E/F), so tooling trouble never blocks exit.
    """
    if not workspace_root or not py_files:
        return [], False
    try:
        result = subprocess.run(
            [
                "ruff",
                "check",
                "--output-format=json",
                "--no-cache",
                *py_files,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
            cwd=workspace_root,
        )
    except OSError:
        return [], False
    except subprocess.TimeoutExpired:
        return [
            (
                "",
                "RUFF-TIMEOUT",
                f"ruff check timed out after {_SUBPROCESS_TIMEOUT_SECONDS}s",
                None,
            )
        ], True

    stdout = (result.stdout or "").strip()
    if not stdout:
        # exit 0 = clean; exit >=2 with stderr is a ruff/config error.
        if result.returncode >= 2 and (result.stderr or "").strip():
            detail = result.stderr.strip().splitlines()[-1][:200]
            return [("", "RUFF", f"ruff could not run: {detail}", None)], True
        return [], result.returncode == 0
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return [], False
    if not isinstance(payload, list):
        return [], False

    root = Path(workspace_root).resolve()
    findings: list[tuple[str, str, str, int | None]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "").strip()
        if not code:
            continue
        rel_path = _relative_display_path(str(item.get("filename") or ""), root)
        message = str(item.get("message") or "").strip()
        location = item.get("location")
        row = location.get("row") if isinstance(location, dict) else None
        findings.append((rel_path, code, message, row))
    return findings, True


def _relative_display_path(filename: str, root: Path) -> str:
    if not filename:
        return ""
    path = Path(filename)
    try:
        return path.resolve().relative_to(root).as_posix()
    except (ValueError, OSError):
        return path.as_posix()
