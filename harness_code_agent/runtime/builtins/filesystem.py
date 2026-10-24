"""Workspace filesystem tools."""
from __future__ import annotations

import contextlib
import difflib
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

from ... import config
from ...agent.context import count_text_tokens
from ...workspace.service import WorkspaceQuotaError
from ..tool_context import ToolContext
from ..tool_output import fitting_prefix, output_limit
from ..tool_result import ToolResult

READ_FILE_MAX_LINES = 2000
REPO_SEARCH_TIMEOUT_SECONDS = 15
REPO_SEARCH_MAX_RESULTS = 500
LIST_FILES_MAX_RESULTS = 1000
DEFAULT_EXCLUDE_PARTS = {
    ".git",
    ".harness",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
}
DEFAULT_EXCLUDE_PATHS = {
    ("eval", "results"),
}

# Bound the diff attached to file-change events so a single huge rewrite
# cannot bloat the session JSONL or the transcript.
_DIFF_MAX_LINES = 200


def _change_diff(old: str | None, new: str) -> str:
    """Compact unified diff between two file versions (old=None means new file)."""
    if old is None:
        lines = new.splitlines()
        body = ["+" + line for line in lines]
    else:
        body = [
            line
            for line in difflib.unified_diff(
                old.splitlines(), new.splitlines(), lineterm="", n=1
            )
            if not line.startswith(("--- ", "+++ "))
        ]
    if len(body) > _DIFF_MAX_LINES:
        omitted = len(body) - _DIFF_MAX_LINES
        kept = "\n".join(body[:_DIFF_MAX_LINES])
        return kept + "\n… " + str(omitted) + " more diff lines"
    return "\n".join(body)


def _change_stats(old: str | None, new: str) -> tuple[int, int]:
    """Return actual added/deleted line counts before the display diff is capped."""
    if old is None:
        return len(new.splitlines()), 0
    body = [
        line
        for line in difflib.unified_diff(
            old.splitlines(), new.splitlines(), lineterm="", n=1
        )
        if not line.startswith(("--- ", "+++ "))
    ]
    additions = sum(1 for line in body if line.startswith("+"))
    deletions = sum(1 for line in body if line.startswith("-"))
    return additions, deletions


def _resolve(path: str) -> Path:
    """Resolve a relative path inside the workspace. Prevent escaping."""
    p = Path(config.WORKSPACE, path).resolve()
    ws = Path(config.WORKSPACE).resolve()
    try:
        p.relative_to(ws)
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace: {path}") from exc
    return p


def _workspace_root(tool_context: ToolContext | None = None) -> Path:
    if tool_context is not None:
        return tool_context.workspace.root.resolve()
    return Path(config.WORKSPACE).resolve()


def _resolve_with_context(path: str, tool_context: ToolContext | None = None) -> Path:
    if tool_context is not None:
        return tool_context.workspace.resolve(path)
    return _resolve(path)


def _relative_to_workspace(path: Path, workspace: Path) -> str:
    rel = path.resolve().relative_to(workspace.resolve())
    return "." if str(rel) == "." else rel.as_posix()


def _has_excluded_part(path: Path, workspace: Path, extra_exclude: set[str] | None = None) -> bool:
    try:
        rel = path.resolve().relative_to(workspace.resolve())
    except ValueError:
        return True
    parts = rel.parts
    excluded = DEFAULT_EXCLUDE_PARTS | (extra_exclude or set())
    if any(part in excluded for part in parts):
        return True
    lower_parts = tuple(part.lower() for part in parts)
    return any(_contains_path_parts(lower_parts, blocked) for blocked in DEFAULT_EXCLUDE_PATHS)


def _contains_path_parts(parts: tuple[str, ...], blocked: tuple[str, ...]) -> bool:
    if not blocked or len(parts) < len(blocked):
        return False
    blocked_lower = tuple(part.lower() for part in blocked)
    return any(parts[index:index + len(blocked_lower)] == blocked_lower for index in range(len(parts) - len(blocked_lower) + 1))


def _rg_exclude_globs() -> list[str]:
    globs = [f"!**/{name}/**" for name in sorted(DEFAULT_EXCLUDE_PARTS)]
    globs.append("!eval/results/**")
    globs.append("!**/eval/results/**")
    return globs


def _clamp_int(value: int | None, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def repo_search(
    pattern: str,
    path: str = ".",
    glob: list[str] | None = None,
    case_sensitive: bool = False,
    max_results: int = 100,
    context_lines: int = 0,
    tool_context: ToolContext | None = None,
) -> ToolResult:
    workspace = _workspace_root(tool_context)
    p = _resolve_with_context(path or ".", tool_context)
    if not p.exists():
        return ToolResult(
            tool="repo_search",
            status="failed",
            output=f"[error] Search path not found: {path}",
            error=f"Search path not found: {path}",
            metadata={"path": path, "status_source": "native"},
        )
    if _has_excluded_part(p, workspace):
        return ToolResult(
            tool="repo_search",
            status="failed",
            output=f"[blocked] repo_search does not search internal/generated paths by default: {path}",
            error=f"repo_search blocked for excluded path: {path}",
            metadata={"path": path, "status_source": "permission", "reason": "excluded_path"},
        )
    limit = _clamp_int(max_results, default=100, minimum=1, maximum=REPO_SEARCH_MAX_RESULTS)
    context = _clamp_int(context_lines, default=0, minimum=0, maximum=5)
    search_path = _relative_to_workspace(p, workspace)
    args = [
        "rg",
        "--line-number",
        "--with-filename",
        "--color",
        "never",
    ]
    if not case_sensitive:
        args.append("--ignore-case")
    if context:
        args.extend(["--context", str(context)])
    for item in _rg_exclude_globs():
        args.extend(["--glob", item])
    for item in glob or []:
        if item:
            args.extend(["--glob", str(item)])
    args.extend(["--", pattern, search_path])

    timed_out = threading.Event()
    process = None
    timer = None
    try:
        with tempfile.SpooledTemporaryFile(mode="w+t", max_size=65536, encoding="utf-8", newline="") as selected, tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(args, cwd=workspace, stdout=subprocess.PIPE, stderr=errors,
                                       text=True, encoding="utf-8", errors="replace")
            def expire():
                timed_out.set()
                with contextlib.suppress(OSError):
                    process.kill()
            timer = threading.Timer(REPO_SEARCH_TIMEOUT_SECONDS, expire)
            timer.start()
            returned = 0
            truncated = False
            while fragment := process.stdout.readline(4096):
                if returned >= limit:
                    truncated = True
                    process.kill()
                    break
                selected.write(fragment)
                if fragment.endswith("\n"):
                    returned += 1
            if selected.tell() and returned == 0:
                returned = 1
            process.stdout.close()
            return_code = process.wait()
            timer.cancel()
            if timed_out.is_set():
                return ToolResult(tool="repo_search", status="failed",
                                  output=f"[error] repo_search timed out after {REPO_SEARCH_TIMEOUT_SECONDS}s. Narrow the path or pattern.",
                                  metadata={"status_source": "timeout"})
            failed = not truncated and return_code not in {0, 1}
            if failed:
                selected.write(f"\n[error] rg exited with code {return_code}\n")
                errors.seek(0)
                while chunk := errors.read(65536):
                    selected.write(chunk.decode("utf-8", errors="replace"))
            elif truncated:
                selected.write(f"\n[truncated] Showing first {limit} result lines. Narrow path/pattern or raise max_results.")
            metadata = {"path": path, "explicit_path": search_path, "returned_lines": returned,
                        "truncated": truncated, "status_source": "native"}
            selected.seek(0)
            output = selected.read(65537)
            if len(output) > 65536:
                from ...agent.observations import observation_dir_for
                directory = observation_dir_for(workspace, getattr(tool_context, "session_id", None),
                                                getattr(tool_context, "agent_id", None))
                directory.mkdir(parents=True, exist_ok=True)
                artifact = directory / f"search_{uuid.uuid4().hex}.txt"
                selected.seek(0)
                with artifact.open("x", encoding="utf-8", newline="") as destination:
                    shutil.copyfileobj(selected, destination, length=65536)
                metadata["artifact_path"] = str(artifact)
                output = ""
            return ToolResult(tool="repo_search", status="failed" if failed else "success",
                              output=output or ("" if metadata.get("artifact_path") else "(no matches)"),
                              return_code=return_code if failed else None, metadata=metadata)
    except FileNotFoundError:
        return ToolResult(tool="repo_search", status="failed", output="[error] ripgrep (rg) is not installed or not on PATH.",
                          metadata={"status_source": "native"})
    finally:
        if timer is not None:
            timer.cancel()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if process is not None and process.stdout is not None:
            process.stdout.close()


def read_file(
    path: str,
    start_line: int | None = None,
    max_lines: int | None = None,
    include_line_numbers: bool = False,
    tool_context: ToolContext | None = None,
    start_column: int = 1,
) -> ToolResult:
    p = _resolve_with_context(path, tool_context)
    if not p.is_file():
        return ToolResult(tool="read_file", status="failed", output=f"[error] File not found: {path}",
                          error=f"File not found: {path}", metadata={"path": path, "status_source": "native"})
    references = getattr(tool_context, "tool_output_artifacts", {})
    reference = references.get(p.relative_to(_workspace_root(tool_context)).as_posix())
    if reference and reference.get("sha256") and not reference.get("append_only"):
        from ..output_artifacts import file_digest
        if file_digest(p) != reference["sha256"]:
            return ToolResult(tool="read_file", status="failed", output="[error] Tool output artifact is damaged.",
                              error="Tool output artifact is damaged", metadata={"path": path})
    for name, value, maximum in (("start_line", start_line if start_line is not None else 1, None),
                                 ("start_column", start_column, None),
                                 ("max_lines", max_lines if max_lines is not None else READ_FILE_MAX_LINES, READ_FILE_MAX_LINES)):
        try:
            integer = int(value)
            if integer < 1 or (maximum is not None and integer > maximum):
                raise ValueError
        except (TypeError, ValueError):
            message = f"read_file {name} must be an integer >= 1."
            if maximum is not None:
                message = f"read_file {name} must be <= {maximum} and >= 1."
            return ToolResult(tool="read_file", status="failed", output="[error] " + message,
                              error=message, metadata={"path": path, "status_source": "validation"})
    start, column = int(start_line or 1), int(start_column)
    limit = int(max_lines) if max_lines is not None else READ_FILE_MAX_LINES
    end_line = start + limit - 1
    line, col = 1, 1
    output = ""
    next_read = None
    budget = max(1, output_limit() - 800)
    with p.open("r", encoding="utf-8", errors="replace", newline=None) as stream:
        while fragment := stream.readline(4096):
            fragment_line, fragment_col = line, col
            if fragment.endswith("\n"):
                line, col = line + 1, 1
            else:
                col += len(fragment)
            if fragment_line < start:
                continue
            if fragment_line == start and fragment_col + len(fragment) <= column:
                continue
            if fragment_line > end_line:
                next_read = {"path": path, "start_line": fragment_line, "start_column": fragment_col,
                             "max_lines": limit, "include_line_numbers": include_line_numbers}
                break
            skip = max(0, column - fragment_col) if fragment_line == start else 0
            fragment = fragment[skip:]
            fragment_col += skip
            label = f"{fragment_line}: " if include_line_numbers and (fragment_col == 1 or not output) else ""
            candidate = output + label + fragment
            if count_text_tokens(candidate) <= budget:
                output = candidate
                continue
            prefix = fitting_prefix(candidate, budget)
            consumed = max(0, len(prefix) - len(output) - len(label))
            if consumed == 0 and not output:
                return ToolResult(tool="read_file", status="failed", output="[error] File page cannot fit the output budget.")
            output = prefix if consumed else output
            next_line, next_column = fragment_line, fragment_col + consumed
            next_read = {"path": path, "start_line": next_line, "start_column": next_column,
                         "max_lines": limit, "include_line_numbers": include_line_numbers}
            break
    metadata = {"path": path, "start_line": start, "start_column": column, "max_lines": limit,
                "status_source": "native"}
    if next_read:
        metadata["next_read"] = next_read
    return ToolResult(tool="read_file", status="success", output=output, metadata=metadata)


def read_skill_file(path: str) -> ToolResult:
    """Read a file from the packaged skills catalog (outside workspace)."""
    skills_root = Path(__file__).resolve().parents[2] / "skills"
    catalog_root = (skills_root / "catalog").resolve()
    p = (skills_root / path).resolve()
    try:
        p.relative_to(catalog_root)
    except ValueError:
        return ToolResult(
            tool="read_skill_file",
            status="failed",
            output=f"[error] Path must be inside skills/catalog/: {path}",
            error=f"Path must be inside skills/catalog/: {path}",
            metadata={"path": path, "status_source": "validation"},
        )
    if not p.exists() or not p.is_file():
        return ToolResult(
            tool="read_skill_file",
            status="failed",
            output=f"[error] Skill file not found: {path}",
            error=f"Skill file not found: {path}",
            metadata={"path": path, "status_source": "native"},
        )
    content = p.read_text(encoding="utf-8", errors="replace")
    return ToolResult(
        tool="read_skill_file",
        status="success",
        output=content,
        metadata={
            "path": path,
            "status_source": "native",
            "characters": len(content),
        },
    )


def _quota_result(tool: str, path: str, exc: WorkspaceQuotaError) -> ToolResult:
    return ToolResult(
        tool=tool,
        status="failed",
        output=(
            f"[blocked] {exc.reason}. Stop generating new files, clean up existing "
            "workspace files, or ask the user to raise HARNESS_WORKSPACE_WRITE_QUOTA_MB / "
            "HARNESS_MIN_FREE_DISK_MB."
        ),
        error=exc.reason,
        metadata={
            "path": path,
            "status_source": "resource",
            "resource_kind": getattr(exc, "resource_kind", "workspace_quota"),
            "charged_bytes": exc.charged_bytes,
            "limit_bytes": exc.limit_bytes,
        },
    )


def write_file(
    path: str,
    content: str,
    tool_context: ToolContext | None = None,
) -> ToolResult:
    if not path or not path.strip():
        return ToolResult(
            tool="write_file",
            status="failed",
            output="[error] Empty file path",
            error="Empty file path",
            metadata={"path": path, "status_source": "validation"},
        )
    metadata = {"path": path, "status_source": "native"}
    if tool_context is not None:
        workspace = tool_context.workspace
        try:
            write_result = workspace.write_text(path, content)
        except WorkspaceQuotaError as exc:
            return _quota_result("write_file", path, exc)
        old = write_result.old_content
        rel = write_result.path.relative_to(workspace.root)
        additions, deletions = _change_stats(old, content)
        metadata["file_changes"] = [
            {
                "path": str(rel),
                "operation": "write_file",
                "snapshot_path": str(write_result.snapshot_path) if write_result.snapshot_path else None,
                "diff": _change_diff(old, content),
                "additions": additions,
                "deletions": deletions,
            }
        ]
    else:
        p = _resolve(path)
        old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return ToolResult(
        tool="write_file",
        status="success",
        output=f"Wrote {len(content)} chars to {path}",
        metadata=metadata,
    )


def apply_patch(
    path: str,
    search: str,
    replace: str,
    tool_context: ToolContext | None = None,
) -> ToolResult:
    if not path or not str(path).strip():
        return ToolResult(
            tool="apply_patch",
            status="failed",
            output="[error] Empty file path",
            error="Empty file path",
            metadata={"path": path, "status_source": "validation"},
        )
    if tool_context is not None:
        workspace = tool_context.workspace
        try:
            patch_result = workspace.apply_text_patch(
                path,
                search=search,
                replace=replace,
            )
        except WorkspaceQuotaError as exc:
            return _quota_result("apply_patch", path, exc)
        rel = patch_result.path.relative_to(workspace.root)
        old = patch_result.old_content
        new = patch_result.new_content
        additions, deletions = _change_stats(old, new)
        return ToolResult(
            tool="apply_patch",
            status="success",
            output=f"Patched {path}: replaced {patch_result.replacements} occurrence",
            metadata={
                "path": path,
                "replacements": patch_result.replacements,
                "status_source": "native",
                "file_changes": [
                    {
                        "path": str(rel),
                        "operation": "apply_patch",
                        "snapshot_path": str(patch_result.snapshot_path) if patch_result.snapshot_path else None,
                        "diff": _change_diff(old, new),
                        "additions": additions,
                        "deletions": deletions,
                    }
                ],
            },
        )

    p = _resolve(path)
    if not p.exists():
        return ToolResult(
            tool="apply_patch",
            status="failed",
            output=f"[error] File not found: {path}",
            error=f"File not found: {path}",
            metadata={"path": path, "status_source": "native"},
        )
    original = p.read_text(encoding="utf-8", errors="replace")
    count = original.count(search)
    if count != 1:
        return ToolResult(
            tool="apply_patch",
            status="failed",
            output=f"[error] Patch search text must match exactly once; found {count}",
            error=f"Patch search text must match exactly once; found {count}",
            metadata={"path": path, "status_source": "validation"},
        )
    p.write_text(original.replace(search, replace, 1), encoding="utf-8")
    return ToolResult(
        tool="apply_patch",
        status="success",
        output=f"Patched {path}: replaced 1 occurrence",
        metadata={"path": path, "replacements": 1, "status_source": "native"},
    )


def list_files(
    directory: str = ".",
    depth: int = 2,
    max_results: int = 200,
    include_hidden: bool = False,
    exclude: list[str] | None = None,
    tool_context: ToolContext | None = None,
) -> ToolResult:
    p = _resolve_with_context(directory or ".", tool_context)
    workspace = _workspace_root(tool_context)
    if not p.is_dir():
        return ToolResult(
            tool="list_files",
            status="failed",
            output=f"[error] Not a directory: {directory}",
            error=f"Not a directory: {directory}",
            metadata={"directory": directory, "status_source": "native"},
        )
    limit = _clamp_int(max_results, default=200, minimum=1, maximum=LIST_FILES_MAX_RESULTS)
    max_depth = _clamp_int(depth, default=2, minimum=1, maximum=20)
    extra_exclude = {str(item) for item in (exclude or []) if str(item).strip()}
    entries: list[str] = []
    total_seen = 0

    def should_skip(item: Path) -> bool:
        if not include_hidden and any(part.startswith(".") for part in item.relative_to(workspace).parts):
            return True
        return _has_excluded_part(item, workspace, extra_exclude)

    for root, dirnames, filenames in os.walk(p):
        root_path = Path(root)
        current_depth = len(root_path.relative_to(p).parts)
        include_children = current_depth + 1 <= max_depth
        descend = current_depth + 1 < max_depth
        dirnames[:] = [
            name
            for name in sorted(dirnames)
            if descend and not should_skip(root_path / name)
        ]
        if not include_children:
            dirnames[:] = []
            continue
        for dirname in dirnames:
            item = root_path / dirname
            total_seen += 1
            if len(entries) < limit:
                entries.append(_relative_to_workspace(item, workspace) + "/")
        for filename in sorted(filenames):
            item = root_path / filename
            if should_skip(item):
                continue
            total_seen += 1
            if len(entries) < limit:
                entries.append(_relative_to_workspace(item, workspace))
    if not entries:
        return ToolResult(
            tool="list_files",
            status="success",
            output="(empty)",
            metadata={
                "directory": directory,
                "depth": max_depth,
                "status_source": "native",
                "count": total_seen,
                "returned": 0,
                "truncated": False,
            },
        )
    truncated = total_seen > len(entries)
    output = "\n".join(entries)
    if truncated:
        output += f"\n[truncated] Showing first {len(entries)} entries. Narrow directory or increase max_results."
    return ToolResult(
        tool="list_files",
        status="success",
        output=output,
        metadata={
            "directory": directory,
            "depth": max_depth,
            "status_source": "native",
            "count": total_seen,
            "returned": len(entries),
            "truncated": truncated,
        },
    )
