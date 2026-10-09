"""Project toolchain checks for JavaScript, TypeScript, Go and Rust."""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import psutil

from .engine import CheckResult

SCRIPT_SUFFIXES = {".ts", ".tsx", ".js", ".jsx", ".mts", ".cts", ".mjs", ".cjs"}


def ancestors(directory: Path, workspace: Path):
    while directory.is_relative_to(workspace):
        yield directory
        if directory == workspace:
            break
        directory = directory.parent


def project_root(path: Path, workspace: Path, marker: str) -> Path | None:
    return next(
        (p for p in ancestors(path.parent, workspace) if (p / marker).is_file()), None
    )


def node_command(
    directory: Path, workspace: Path, package: str, script: str
) -> list[str] | None:
    node = shutil.which("node")
    for parent in ancestors(directory, workspace):
        entry = parent / "node_modules" / package / script
        if node and entry.is_file():
            return [node, str(entry)]
    executable = shutil.which("tsc" if package == "typescript" else package)
    if executable and Path(executable).suffix.lower() in {".cmd", ".bat"}:
        entry = Path(executable).parent / "node_modules" / package / script
        return [node, str(entry)] if node and entry.is_file() else None
    return [executable] if executable else None


def command_check(
    name: str,
    command: list[str] | None,
    cwd: Path,
    *,
    diff: bool = False,
    timeout: int = 60,
) -> CheckResult:
    if command is None:
        return CheckResult(name, "skipped", ("Tool is not installed",))
    try:
        with tempfile.TemporaryFile() as output_file:
            proc = subprocess.Popen(
                command,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=output_file,
                stderr=output_file,
            )
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(psutil.Error):
                    parent = psutil.Process(proc.pid)
                    for child in parent.children(recursive=True):
                        with contextlib.suppress(psutil.Error):
                            child.kill()
                    parent.kill()
                proc.wait(timeout=3)
                raise
            output_file.seek(0)
            output = output_file.read(4_000).decode("utf-8", errors="replace").strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult(name, "skipped", (f"Tool unavailable: {exc}",))
    failed = proc.returncode != 0 or (diff and bool(output))
    return CheckResult(
        name,
        "failed" if failed else "passed",
        ((output[:4_000] or f"Exited with {proc.returncode}"),)
        if failed
        else ("no findings",),
    )


def _name(check: str, directory: Path, workspace: Path) -> str:
    rel = directory.relative_to(workspace).as_posix()
    return check if rel == "." else f"{check} ({rel})"


def _execution_required(
    name: str, command: list[str], directory: Path, workspace: Path
) -> CheckResult:
    return CheckResult(
        name,
        "skipped",
        (
            (
                "Requires project-code execution through run_bash with the session's permissions. "
                f"cwd: {directory.relative_to(workspace).as_posix()}; argv: {json.dumps(command, ensure_ascii=False)}"
            ),
        ),
    )


def _groups(files, workspace, suffixes, marker):
    groups = {}
    for file in files:
        if Path(file).suffix not in suffixes:
            continue
        root = project_root(workspace / file, workspace, marker)
        if root is not None:
            groups.setdefault(root, []).append(file)
    return groups


class TypeScriptProvider:
    def supports(self, files, workspace):
        return any(
            Path(file).suffix in SCRIPT_SUFFIXES or Path(file).name == "tsconfig.json"
            for file in files
        )

    def verify(self, files, workspace):
        results = []
        scripts = [f for f in files if Path(f).suffix in SCRIPT_SUFFIXES]
        ts_groups = _groups(scripts, workspace, SCRIPT_SUFFIXES, "tsconfig.json")
        for file in files:
            if Path(file).name == "tsconfig.json" and (workspace / file).is_file():
                ts_groups.setdefault((workspace / file).parent, [])
        for directory in ts_groups:
            command = node_command(directory, workspace, "typescript", "bin/tsc")
            results.append(
                command_check(
                    _name("TypeScript", directory, workspace),
                    command + ["--noEmit", "--pretty", "false"] if command else None,
                    directory,
                )
            )
        uncovered = [
            f
            for f in scripts
            if Path(f).suffix in {".ts", ".tsx", ".mts", ".cts"}
            and project_root(workspace / f, workspace, "tsconfig.json") is None
        ]
        if uncovered:
            results.append(
                CheckResult(
                    "TypeScript (unconfigured files)",
                    "skipped",
                    ("No tsconfig.json for: " + ", ".join(uncovered),),
                )
            )
        js_groups = {}
        for file in scripts:
            directory = next(
                (
                    parent
                    for parent in ancestors((workspace / file).parent, workspace)
                    if _eslint_configured(parent)
                ),
                None,
            )
            if directory:
                js_groups.setdefault(directory, []).append(file)
        for directory, paths in js_groups.items():
            changed = [str(workspace / p) for p in paths if (workspace / p).is_file()]
            if changed:
                command = node_command(directory, workspace, "eslint", "bin/eslint.js")
                results.append(
                    _execution_required(
                        _name("ESLint", directory, workspace),
                        command + ["--no-cache", *changed],
                        directory,
                        workspace,
                    )
                    if command
                    else CheckResult(
                        _name("ESLint", directory, workspace),
                        "skipped",
                        ("ESLint is not installed or Node.js is unavailable",),
                    )
                )
        if not results:
            results.append(
                CheckResult(
                    "JavaScript verification",
                    "skipped",
                    ("No TypeScript or ESLint project configuration",),
                )
            )
        return results


def _eslint_configured(directory: Path) -> bool:
    try:
        manifest = json.loads((directory / "package.json").read_text(encoding="utf-8"))
        if (
            "eslint"
            in {
                **manifest.get("dependencies", {}),
                **manifest.get("devDependencies", {}),
            }
            or "eslintConfig" in manifest
        ):
            return True
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return any(
        (directory / name).is_file()
        for name in (
            "eslint.config.js",
            "eslint.config.mjs",
            "eslint.config.cjs",
            "eslint.config.ts",
            ".eslintrc",
            ".eslintrc.json",
            ".eslintrc.js",
            ".eslintrc.cjs",
            ".eslintrc.yml",
            ".eslintrc.yaml",
        )
    )


class GoProvider:
    def supports(self, files, workspace):
        return any(file.endswith(".go") for file in files)

    def verify(self, files, workspace):
        files = [f for f in files if f.endswith(".go")]
        results = []
        existing = [f for f in files if (workspace / f).is_file()]
        if existing:
            gofmt = shutil.which("gofmt")
            results.append(
                command_check(
                    "Go formatting",
                    [gofmt, "-d", *existing] if gofmt else None,
                    workspace,
                    diff=True,
                )
            )
        groups = _groups(files, workspace, {".go"}, "go.mod")
        uncovered = [
            file
            for file in files
            if project_root(workspace / file, workspace, "go.mod") is None
        ]
        if uncovered:
            results.append(
                CheckResult(
                    "Go tests (unconfigured packages)",
                    "skipped",
                    ("No go.mod for: " + ", ".join(uncovered),),
                )
            )
        for directory, paths in groups.items():
            packages = sorted(
                {
                    "./" + (workspace / p).parent.relative_to(directory).as_posix()
                    for p in paths
                    if any((workspace / p).parent.glob("*.go"))
                }
            )
            if packages:
                results.append(
                    _execution_required(
                        _name("Go tests", directory, workspace),
                        ["go", "test", *packages],
                        directory,
                        workspace,
                    )
                )
        return results


class RustProvider:
    def supports(self, files, workspace):
        return any(
            file.endswith(".rs") or Path(file).name == "Cargo.toml" for file in files
        )

    def verify(self, files, workspace):
        roots = {
            project_root(workspace / f, workspace, "Cargo.toml")
            for f in files
            if f.endswith(".rs") or Path(f).name == "Cargo.toml"
        }
        return [
            _execution_required(
                _name("Rust check", root, workspace),
                ["cargo", "check"],
                root,
                workspace,
            )
            if root
            else CheckResult(
                "Rust check", "skipped", ("No Cargo.toml for changed files",)
            )
            for root in sorted(roots, key=lambda p: str(p))
        ]
