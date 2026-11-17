"""Workspace filesystem tools."""
from __future__ import annotations

import difflib
from itertools import islice
from pathlib import Path

from ... import config
from ...support.tokens import count_text_tokens
from ...workspace.service import WorkspaceDiskSpaceError
from ..output_artifacts import file_digest
from ..tool_context import ToolContext
from ..tool_output import fitting_prefix, output_limit
from ..tool_result import ToolResult

READ_FILE_MAX_LINES = 2000

# Bound the diff attached to file-change events so a single huge rewrite
# cannot bloat the session JSONL or the transcript.
_DIFF_MAX_LINES = 200


def _is_secret_env_file(path: Path) -> bool:
    """Whether a file is a credential-bearing env file.

    Writes are allowed; only the diff attached to events is redacted so
    secrets do not leak into logs or remote review payloads.
    """
    name = path.name.lower()
    if name == ".env":
        return True
    if not name.startswith(".env."):
        return False
    return not name.endswith((".example", ".template", ".sample", ".default", ".dist"))


def _change_details(path: Path, old: str | None, new: str) -> tuple[str, int, int]:
    """Return a bounded diff and counts for the entire change."""
    if _is_secret_env_file(path):
        return "[redacted .env diff]", 0, 0
    new_lines = new.splitlines()
    if old is None:
        lines = ("+" + line for line in new_lines)
    else:
        lines = islice(difflib.unified_diff(old.splitlines(), new_lines, lineterm="", n=1), 2, None)
    preview = []
    additions = deletions = omitted = 0
    for line in lines:
        additions += line.startswith("+")
        deletions += line.startswith("-")
        if len(preview) < _DIFF_MAX_LINES:
            preview.append(line)
        else:
            omitted += 1
    if omitted:
        preview.append(f"… {omitted} more diff lines")
    return "\n".join(preview), additions, deletions


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


def resolve_read_path(path: str, tool_context: ToolContext | None = None) -> Path:
    if tool_context is None:
        return _resolve(path)
    try:
        return tool_context.workspace.resolve(path)
    except ValueError:
        candidate = Path(path)
        if not candidate.is_absolute():
            raise
        resolved = candidate.resolve()
        if resolved.as_posix() not in tool_context.tool_output_artifacts:
            raise
        return resolved




def read_file(
    path: str,
    start_line: int | None = None,
    max_lines: int | None = None,
    include_line_numbers: bool = False,
    tool_context: ToolContext | None = None,
    start_column: int = 1,
) -> ToolResult:
    p = resolve_read_path(path, tool_context)
    if not p.is_file():
        return ToolResult(tool="read_file", status="failed", output=f"[error] File not found: {path}",
                          error=f"File not found: {path}", metadata={"path": path, "status_source": "native"})
    references = getattr(tool_context, "tool_output_artifacts", {})
    reference = references.get(p.as_posix())
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
    before = p.stat()
    version = None
    if reference and reference.get("sha256") and not reference.get("append_only"):
        if file_digest(p) != reference["sha256"]:
            return ToolResult(tool="read_file", status="failed", output="[error] Tool output artifact is damaged.",
                              error="Tool output artifact is damaged", metadata={"path": path})
    elif tool_context is not None and reference is None:
        version = file_digest(p)
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
    after = p.stat()
    if (before.st_ino, before.st_mtime_ns, before.st_size) != (after.st_ino, after.st_mtime_ns, after.st_size):
        from ...workspace.access import WorkspaceConflict
        raise WorkspaceConflict(f"读取期间文件发生变化，请重新读取：{path}")
    if version is not None and tool_context is not None:
        tool_context.workspace.remember_read_version(p, version)
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


def _disk_space_result(tool: str, path: str, exc: WorkspaceDiskSpaceError) -> ToolResult:
    return ToolResult(
        tool=tool,
        status="failed",
        output=f"[blocked] {exc}. Free disk space before retrying.",
        error=str(exc),
        metadata={
            "path": path,
            "status_source": "resource",
            "resource_kind": "free_disk",
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
        except WorkspaceDiskSpaceError as exc:
            return _disk_space_result("write_file", path, exc)
        old = write_result.old_content
        rel = write_result.path.relative_to(workspace.root)
        redacted = _is_secret_env_file(write_result.path)
        diff, additions, deletions = _change_details(write_result.path, old, content)
        metadata["file_changes"] = [
            {
                "path": str(rel),
                "operation": "write_file",
                "snapshot_path": str(write_result.snapshot_path) if write_result.snapshot_path else None,
                "diff": diff,
                "additions": additions,
                "deletions": deletions,
            }
        ]
        if redacted:
            metadata["diff_redacted"] = True
    else:
        p = _resolve(path)
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
        except WorkspaceDiskSpaceError as exc:
            return _disk_space_result("apply_patch", path, exc)
        rel = patch_result.path.relative_to(workspace.root)
        old = patch_result.old_content
        new = patch_result.new_content
        redacted = _is_secret_env_file(patch_result.path)
        diff, additions, deletions = _change_details(patch_result.path, old, new)
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
                        "diff": diff,
                        "additions": additions,
                        "deletions": deletions,
                    }
                ],
                **({"diff_redacted": True} if redacted else {}),
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
