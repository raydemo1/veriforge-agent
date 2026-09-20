"""Static pre-exit verification middleware.

Deterministic, objective gate (defense in depth — not a security boundary):
if Python files changed *during the current turn*, check them before the
agent is allowed to finish the turn.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .base import AgentMiddleware

#: Rule-code prefixes that block the exit; everything else is a one-time warn.
_RUFF_BLOCK_PREFIXES = ("E", "F")
_MAX_REPORT_ITEMS = 20
_SUBPROCESS_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class _TurnBaseline:
    """Turn-local snapshot taken in ``begin_turn`` (and at construction)."""

    #: Workspace change-journal cursor — covers edits made through the
    #: workspace service (write_file / apply_patch).
    journal_cursor: int
    #: Files already dirty in git when the turn started (tracked changes plus
    #: untracked files), POSIX-relative. Pre-existing dirty work must never be
    #: counted as a current-turn change.
    git_files: frozenset[str]
    #: SHA-256 of the ``.py`` files already dirty at baseline
    #: (POSIX-relative path -> digest). Content changes to such a file during
    #: the turn are detected even though it stays inside the baseline git set.
    dirty_py_fingerprints: dict[str, str]


class StaticVerifierMiddleware(AgentMiddleware):
    """Pre-exit lint gate for Python files changed in the current turn.

    Scope is strictly turn-local:

    * ``begin_turn`` records a baseline (change-journal cursor + the set of
      files already dirty in git). A file is verified only if it became dirty
      *after* the baseline, regardless of which tool changed it — this also
      covers files written through ``run_bash`` (scripts, formatters,
      heredocs), which never touch the change journal.
    * ``ast.parse`` (stdlib): a syntax error on any changed ``.py`` blocks the
      exit. Parsing emits no bytecode and needs no external tool.
    * ``ruff check --output-format=json`` (optional): scoped to the changed
      files only; E/F findings block, other rule families (W/C/N/...) produce
      a one-time non-blocking warning. Ruff missing, timing out or returning
      unusable output is skipped — tooling problems never block the exit.
    """

    def __init__(self, workspace_root: str | None = None, workspace=None):
        self._workspace_root = workspace_root
        self._workspace = workspace
        self._baseline: _TurnBaseline = self._capture_baseline()
        self._reported_warning_signatures: set[tuple] = set()

    def begin_turn(self, task: str, messages: list[dict], runtime_state=None,
                   agent_name: str | None = None) -> None:
        self._baseline = self._capture_baseline()
        self._reported_warning_signatures.clear()

    def pre_exit(self, messages: list[dict], runtime_state=None,
                 agent_name: str | None = None) -> str | None:
        py_files = _turn_changed_py_files(
            self._workspace_root,
            self._workspace,
            self._baseline,
        )
        if not py_files:
            return None

        blocks: list[str] = []
        # --- ast.parse: syntax errors on changed files ---
        for path, msg in _check_python_syntax(self._workspace_root, py_files):
            blocks.append(f"  [syntax] {path}: {msg}")

        # --- ruff (JSON): only the changed files; E/F -> block, rest -> warn ---
        warns: list[str] = []
        warning_ids: list[tuple] = []
        for rel_path, code, msg, row in _check_ruff(self._workspace_root, py_files):
            if code[:1] in _RUFF_BLOCK_PREFIXES:
                location = f"{rel_path}:{row}" if row else rel_path
                blocks.append(f"  [{code}] {location}: {msg}")
            else:
                location = f"{rel_path}:{row}" if row else rel_path
                warns.append(f"  [{code}] {location}: {msg}")
                warning_ids.append((rel_path, code, row))

        if blocks:
            details = "\n".join(blocks[:_MAX_REPORT_ITEMS])
            return (
                "[SYSTEM] LINT CHECK FAILED -- fix these errors before stopping:\n"
                f"{details}"
            )
        if warns:
            details = "\n".join(warns[:_MAX_REPORT_ITEMS])
            signature = tuple(warning_ids[:_MAX_REPORT_ITEMS])
            if signature in self._reported_warning_signatures:
                return None
            self._reported_warning_signatures.add(signature)
            return (
                f"[SYSTEM] Lint warnings (non-blocking):\n{details}\n"
                "Consider fixing before stopping."
            )
        return None

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    def _capture_baseline(self) -> _TurnBaseline:
        git_files = frozenset(_git_dirty_files(self._workspace_root))
        root = Path(
            self._workspace_root
            or getattr(self._workspace, "root", ".")
        ).resolve()
        dirty_py = {path for path in git_files if path.endswith(".py")}
        return _TurnBaseline(
            journal_cursor=_workspace_change_cursor(self._workspace),
            git_files=git_files,
            dirty_py_fingerprints=_file_fingerprints(root, dirty_py),
        )


def _turn_changed_py_files(
    workspace_root: str | None,
    workspace,
    baseline: _TurnBaseline,
) -> list[str]:
    """Return ``.py`` files that became dirty after the turn baseline.

    Two turn-local sources are unioned:

    1. the workspace change journal (edits through write_file/apply_patch);
    2. the git dirty-set delta (tracked + untracked), which also catches
       files created or modified through the shell, including in runs that
       have no :class:`WorkspaceService` at all.

    A file that was already dirty at baseline and is merely kept dirty is not
    reported. If its *content* changed during the turn, the baseline
    fingerprints catch it — including shell-only re-edits of an already-dirty
    file, the previous blind spot of the git-set-delta approach.
    """
    candidates: set[str] = set()
    root = Path(workspace_root or getattr(workspace, "root", ".")).resolve()

    if workspace is not None:
        journal = getattr(workspace, "change_journal", None)
        if journal is not None:
            candidates.update(
                Path(path).as_posix()
                for path in journal.paths_since(baseline.journal_cursor)
            )
        else:
            candidates.update(
                Path(path).as_posix()
                for path in getattr(workspace, "changed_files", [])[baseline.journal_cursor:]
            )

    if workspace_root:
        candidates.update(_git_dirty_files(workspace_root) - set(baseline.git_files))

    for rel, old_hash in baseline.dirty_py_fingerprints.items():
        new_hash = _file_fingerprint(root / rel)
        if new_hash is not None and new_hash != old_hash:
            candidates.add(rel)

    files = {
        rel
        for rel in candidates
        if rel.endswith(".py") and (root / rel).exists()
    }
    return sorted(files)


def _file_fingerprints(root: Path, rel_paths: set[str]) -> dict[str, str]:
    """Hash the given files under ``root``; unreadable files are omitted."""
    fingerprints: dict[str, str] = {}
    for rel in rel_paths:
        digest = _file_fingerprint(root / rel)
        if digest is not None:
            fingerprints[rel] = digest
    return fingerprints


def _file_fingerprint(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _workspace_change_cursor(workspace) -> int:
    if workspace is None:
        return 0
    journal = getattr(workspace, "change_journal", None)
    if journal is not None:
        return journal.cursor()
    return len(getattr(workspace, "changed_files", []))


def _git_dirty_files(workspace_root: str | None) -> set[str]:
    """Return files dirty vs HEAD (tracked changes + untracked), POSIX-relative."""
    if not workspace_root:
        return set()
    files: set[str] = set()
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=workspace_root,
            check=False,
        )
        if result.returncode == 0:
            files.update(result.stdout.splitlines())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()
    try:
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, timeout=10,
            check=False,
            cwd=workspace_root,
        )
        if result.returncode == 0:
            files.update(result.stdout.splitlines())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return {line.strip().replace("\\", "/") for line in files if line.strip()}


def _check_python_syntax(
    workspace_root: str | None, py_files: list[str],
) -> list[tuple[str, str]]:
    """``ast.parse`` each file: syntax errors without writing bytecode."""
    errors: list[tuple[str, str]] = []
    for rel_path in py_files:
        full_path = Path(workspace_root) / rel_path if workspace_root else Path(rel_path)
        try:
            source = full_path.read_text(encoding="utf-8", errors="replace")
            ast.parse(source, filename=str(full_path))
        except (OSError, SyntaxError) as exc:
            errors.append((rel_path, str(exc)))
    return errors


def _check_ruff(
    workspace_root: str | None, py_files: list[str],
) -> list[tuple[str, str, str, int | None]]:
    """Run ruff on the given files via JSON output.

    Returns ``[(relative_path, rule_code, message, row)]``. Returns an empty
    list when ruff is absent or its output is unusable; ruff-invocation
    problems come back as a single non-blocking warning item (rule code not
    starting with E/F), so tooling trouble never blocks an exit.
    """
    if not workspace_root or not py_files:
        return []
    try:
        result = subprocess.run(
            [
                "ruff", "check",
                "--output-format=json",
                "--no-cache",
                *py_files,
            ],
            check=False,
            capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT_SECONDS,
            cwd=workspace_root,
        )
    except FileNotFoundError:
        return []
    except subprocess.TimeoutExpired:
        return [("", "RUFF-TIMEOUT", f"ruff check timed out after {_SUBPROCESS_TIMEOUT_SECONDS}s", None)]

    stdout = (result.stdout or "").strip()
    if not stdout:
        # exit 0 = clean; exit >=2 with stderr is a ruff/config error.
        if result.returncode >= 2 and (result.stderr or "").strip():
            detail = result.stderr.strip().splitlines()[-1][:200]
            return [("", "RUFF", f"ruff could not run: {detail}", None)]
        return []
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []

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
    return findings


def _relative_display_path(filename: str, root: Path) -> str:
    if not filename:
        return ""
    path = Path(filename)
    try:
        return path.resolve().relative_to(root).as_posix()
    except (ValueError, OSError):
        return path.as_posix()
