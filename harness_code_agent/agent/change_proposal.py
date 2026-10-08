"""Persistent isolated-agent change proposals and three-way integration."""
from __future__ import annotations

import difflib
import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..workspace.service import WorkspaceService

_EXCLUDED_NAMES = {
    ".git", ".harness", "__pycache__", ".pytest_cache", ".venv", "venv",
    "node_modules", "jobs",
}
_EXCLUDED_PATHS = {("eval", "results")}


@dataclass(frozen=True)
class Sandbox:
    root: Path
    baseline: Path
    workspace: Path


@dataclass(frozen=True)
class ChangeEntry:
    path: str
    operation: str
    base_sha256: str | None
    result_sha256: str
    base_content: str | None
    result_content: str


@dataclass
class ChangeProposal:
    id: str
    agent_id: str
    entries: tuple[ChangeEntry, ...]
    invalid_reasons: tuple[str, ...] = ()
    status: str = "ready"


@dataclass(frozen=True)
class ConflictFile:
    path: str
    base_content: str
    current_content: str
    worker_content: str
    marked_merge: str


@dataclass
class ConflictSet:
    id: str
    proposal_id: str
    observed_hashes: dict[str, str | None]
    merged_changes: dict[str, str]
    conflicts: dict[str, ConflictFile]
    status: str = "open"


class ChangeProposalStore:
    """Own isolated workspaces and expose validated proposal operations."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sandboxes: dict[str, Sandbox] = {}
        self._proposals: dict[str, ChangeProposal] = {}
        self._conflicts: dict[str, ConflictSet] = {}

    def create_sandbox(self, agent_id: str, source: Path) -> Sandbox:
        root = Path(tempfile.mkdtemp(prefix=f"hca-agent-{agent_id[:8]}-"))
        sandbox = Sandbox(root=root, baseline=root / "baseline", workspace=root / "workspace")
        _copy_workspace(source, sandbox.baseline)
        _copy_workspace(sandbox.baseline, sandbox.workspace)
        with self._lock:
            self._sandboxes[agent_id] = sandbox
        return sandbox

    def sandbox_for(self, agent_id: str) -> Sandbox | None:
        with self._lock:
            return self._sandboxes.get(agent_id)

    def finalize(self, agent_id: str, allowed_paths: list[str]) -> ChangeProposal:
        with self._lock:
            sandbox = self._sandboxes.get(agent_id)
        if sandbox is None:
            raise KeyError(f"agent has no isolated workspace: {agent_id}")
        entries, invalid = _collect_changes(sandbox.baseline, sandbox.workspace, allowed_paths)
        proposal = ChangeProposal(
            id=f"proposal_{uuid.uuid4().hex[:16]}",
            agent_id=agent_id,
            entries=tuple(entries),
            invalid_reasons=tuple(invalid),
            status="invalid" if invalid else "ready",
        )
        with self._lock:
            self._proposals[proposal.id] = proposal
        return proposal

    def get(self, proposal_id: str) -> ChangeProposal:
        with self._lock:
            proposal = self._proposals.get(str(proposal_id))
        if proposal is None:
            raise KeyError(f"unknown proposal: {proposal_id}")
        return proposal

    def snapshot_proposals(self) -> list[dict]:
        """Plain serialisable view for the TUI Work State projection."""

        with self._lock:
            open_conflicts = {
                conflict.proposal_id: conflict
                for conflict in self._conflicts.values()
                if conflict.status == "open"
            }
            snapshots: list[dict] = []
            for proposal in self._proposals.values():
                files = [
                    {
                        "path": entry.path,
                        "operation": entry.operation,
                        **_entry_stats(entry),
                    }
                    for entry in proposal.entries
                ]
                conflict = open_conflicts.get(proposal.id)
                snapshots.append({
                    "id": proposal.id,
                    "agentId": proposal.agent_id,
                    "status": proposal.status,
                    "files": files,
                    "additions": sum(item["additions"] for item in files),
                    "deletions": sum(item["deletions"] for item in files),
                    "invalidReasons": list(proposal.invalid_reasons),
                    "conflict": (
                        {"id": conflict.id, "paths": sorted(conflict.conflicts)}
                        if conflict is not None
                        else None
                    ),
                })
        return snapshots

    def proposal_paths(self, proposal_id: str) -> list[str]:
        return [entry.path for entry in self.get(proposal_id).entries]

    def conflict_paths(self, conflict_id: str) -> list[str]:
        conflict = self._get_conflict(conflict_id)
        return sorted(set(conflict.merged_changes) | set(conflict.conflicts))

    def read_changes(
        self,
        proposal_id: str,
        *,
        path: str = "",
        offset: int = 0,
        limit: int = 12_000,
    ) -> dict:
        proposal = self.get(proposal_id)
        selected = [entry for entry in proposal.entries if not path or entry.path == path]
        if path and not selected:
            raise KeyError(f"proposal does not change path: {path}")
        text = "\n\n".join(_entry_diff(entry) for entry in selected)
        start = max(0, int(offset))
        size = max(1, min(50_000, int(limit)))
        return {
            "proposal_id": proposal.id,
            "status": proposal.status,
            "paths": [entry.path for entry in proposal.entries],
            "invalid_reasons": list(proposal.invalid_reasons),
            "offset": start,
            "next_offset": start + size if start + size < len(text) else None,
            "total_chars": len(text),
            "content": text[start:start + size],
        }

    def apply(self, proposal_id: str, workspace: WorkspaceService) -> dict:
        proposal = self.get(proposal_id)
        if proposal.status != "ready":
            raise ValueError(f"proposal is not ready: {proposal.status}")
        merged: dict[str, str] = {}
        conflicts: dict[str, ConflictFile] = {}
        observed: dict[str, str | None] = {}
        for entry in proposal.entries:
            current_path = workspace.resolve(entry.path)
            current_content = current_path.read_text(encoding="utf-8") if current_path.exists() else None
            current_hash = _sha_text(current_content) if current_content is not None else None
            observed[entry.path] = current_hash
            result, conflict = _merge_entry(entry, current_content)
            if conflict is not None:
                conflicts[entry.path] = conflict
            else:
                merged[entry.path] = result
        if conflicts:
            conflict_set = ConflictSet(
                id=f"conflict_{uuid.uuid4().hex[:16]}",
                proposal_id=proposal.id,
                observed_hashes=observed,
                merged_changes=merged,
                conflicts=conflicts,
            )
            with self._lock:
                self._conflicts[conflict_set.id] = conflict_set
                proposal.status = "conflict"
            return {
                "status": "conflict",
                "proposal_id": proposal.id,
                "conflict_id": conflict_set.id,
                "conflict_paths": sorted(conflicts),
                "message": "No workspace files were changed.",
            }
        results = workspace.write_text_batch(merged)
        proposal.status = "applied"
        return _applied_payload(proposal.id, results, merged, workspace.root)

    def read_conflicts(
        self,
        conflict_id: str,
        *,
        path: str = "",
        offset: int = 0,
        limit: int = 12_000,
    ) -> dict:
        conflict_set = self._get_conflict(conflict_id)
        selected = conflict_set.conflicts
        if path:
            if path not in selected:
                raise KeyError(f"conflict does not contain path: {path}")
            selected = {path: selected[path]}
        blocks = []
        for item in selected.values():
            blocks.append(
                f"### {item.path}\n"
                f"--- BASE ---\n{item.base_content}\n"
                f"--- MAIN ---\n{item.current_content}\n"
                f"--- WORKER ---\n{item.worker_content}\n"
                f"--- MARKED MERGE ---\n{item.marked_merge}"
            )
        text = "\n\n".join(blocks)
        start = max(0, int(offset))
        size = max(1, min(50_000, int(limit)))
        return {
            "conflict_id": conflict_set.id,
            "proposal_id": conflict_set.proposal_id,
            "status": conflict_set.status,
            "paths": sorted(conflict_set.conflicts),
            "offset": start,
            "next_offset": start + size if start + size < len(text) else None,
            "total_chars": len(text),
            "content": text[start:start + size],
        }

    def resolve(self, conflict_id: str, resolutions: dict[str, str], workspace: WorkspaceService) -> dict:
        conflict_set = self._get_conflict(conflict_id)
        if conflict_set.status != "open":
            raise ValueError(f"conflict is not open: {conflict_set.status}")
        expected_paths = set(conflict_set.conflicts)
        supplied_paths = set(resolutions or {})
        if supplied_paths != expected_paths:
            missing = sorted(expected_paths - supplied_paths)
            extra = sorted(supplied_paths - expected_paths)
            raise ValueError(f"resolutions must exactly match conflict paths; missing={missing}, extra={extra}")
        for path, expected_hash in conflict_set.observed_hashes.items():
            current_path = workspace.resolve(path)
            current = current_path.read_text(encoding="utf-8") if current_path.exists() else None
            current_hash = _sha_text(current) if current is not None else None
            if current_hash != expected_hash:
                conflict_set.status = "stale"
                self.get(conflict_set.proposal_id).status = "ready"
                raise ValueError(f"workspace changed after conflict creation: {path}")
        changes = dict(conflict_set.merged_changes)
        changes.update({str(path): str(content) for path, content in resolutions.items()})
        results = workspace.write_text_batch(changes)
        conflict_set.status = "resolved"
        proposal = self.get(conflict_set.proposal_id)
        proposal.status = "applied"
        return _applied_payload(proposal.id, results, changes, workspace.root, conflict_id=conflict_set.id)

    def close_agent(self, agent_id: str, *, discard_changes: bool) -> None:
        with self._lock:
            pending = [p for p in self._proposals.values() if p.agent_id == agent_id and p.status in {"ready", "conflict"}]
            if pending and not discard_changes:
                raise ValueError("agent has an unapplied proposal; pass discard_changes=true to close")
            sandbox = self._sandboxes.pop(agent_id, None)
            proposal_ids = {p.id for p in self._proposals.values() if p.agent_id == agent_id}
            for proposal_id in proposal_ids:
                self._proposals.pop(proposal_id, None)
            for conflict_id, conflict in list(self._conflicts.items()):
                if conflict.proposal_id in proposal_ids:
                    self._conflicts.pop(conflict_id, None)
        if sandbox is not None:
            shutil.rmtree(sandbox.root, ignore_errors=True)

    def close(self) -> None:
        with self._lock:
            sandboxes = list(self._sandboxes.values())
            self._sandboxes.clear()
            self._proposals.clear()
            self._conflicts.clear()
        for sandbox in sandboxes:
            shutil.rmtree(sandbox.root, ignore_errors=True)

    def _get_conflict(self, conflict_id: str) -> ConflictSet:
        with self._lock:
            conflict = self._conflicts.get(str(conflict_id))
        if conflict is None:
            raise KeyError(f"unknown conflict: {conflict_id}")
        return conflict


def _copy_workspace(source: Path, dest: Path) -> None:
    source = source.resolve()

    def ignore(path: str, names: list[str]) -> set[str]:
        current = Path(path).resolve()
        ignored: set[str] = set()
        for name in names:
            item = current / name
            try:
                rel = item.resolve().relative_to(source)
            except ValueError:
                ignored.add(name)
                continue
            if _excluded(rel):
                ignored.add(name)
        return ignored

    shutil.copytree(source, dest, ignore=ignore)


def _collect_changes(baseline: Path, workspace: Path, allowed_paths: list[str]) -> tuple[list[ChangeEntry], list[str]]:
    before = _file_map(baseline)
    after = _file_map(workspace)
    entries: list[ChangeEntry] = []
    invalid: list[str] = []
    for rel in sorted(set(before) | set(after)):
        before_path = before.get(rel)
        after_path = after.get(rel)
        if before_path is not None and after_path is None:
            invalid.append(f"deletion is not supported: {rel}")
            continue
        if after_path is None:
            continue
        before_bytes = before_path.read_bytes() if before_path is not None else None
        after_bytes = after_path.read_bytes()
        if before_bytes == after_bytes:
            continue
        if not _path_allowed(rel, allowed_paths):
            invalid.append(f"changed path is outside allowed_paths: {rel}")
            continue
        try:
            base_content = _normalize_newlines(before_bytes.decode("utf-8")) if before_bytes is not None else None
            result_content = _normalize_newlines(after_bytes.decode("utf-8"))
        except UnicodeDecodeError:
            invalid.append(f"binary or non-UTF-8 change is not supported: {rel}")
            continue
        entries.append(ChangeEntry(
            path=rel,
            operation="modify" if before_bytes is not None else "create",
            base_sha256=_sha_bytes(before_bytes) if before_bytes is not None else None,
            result_sha256=_sha_bytes(after_bytes),
            base_content=base_content,
            result_content=result_content,
        ))
    return entries, invalid


def _file_map(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(root)
            if not _excluded(rel):
                result[rel.as_posix()] = path
    return result


def _excluded(rel: Path) -> bool:
    if any(part in _EXCLUDED_NAMES for part in rel.parts):
        return True
    lowered = tuple(part.lower() for part in rel.parts)
    return any(lowered[:len(blocked)] == blocked for blocked in _EXCLUDED_PATHS)


def _path_allowed(path: str, allowed_paths: list[str]) -> bool:
    candidate = Path(path.replace("\\", "/"))
    for raw in allowed_paths:
        allowed = Path(str(raw).replace("\\", "/"))
        try:
            candidate.relative_to(allowed)
            return True
        except ValueError:
            if candidate == allowed:
                return True
    return False


def _merge_entry(entry: ChangeEntry, current_content: str | None) -> tuple[str, ConflictFile | None]:
    if entry.operation == "create":
        if current_content is None or current_content == entry.result_content:
            return entry.result_content, None
        conflict = ConflictFile(entry.path, "", current_content, entry.result_content, _marked_create_conflict(current_content, entry.result_content))
        return "", conflict
    base = entry.base_content or ""
    if current_content is None:
        conflict = ConflictFile(entry.path, base, "", entry.result_content, _marked_delete_conflict(entry.result_content))
        return "", conflict
    if current_content == base:
        return entry.result_content, None
    if entry.result_content == base or current_content == entry.result_content:
        return current_content, None
    disjoint = _merge_disjoint_text(base, current_content, entry.result_content)
    if disjoint is not None:
        return disjoint, None
    merged, conflicted = _git_merge_text(current_content, base, entry.result_content)
    if not conflicted:
        return merged, None
    return "", ConflictFile(entry.path, base, current_content, entry.result_content, merged)


def _merge_disjoint_text(base: str, current: str, worker: str) -> str | None:
    base_lines = base.splitlines(keepends=True)
    current_edits = _line_edits(base_lines, current.splitlines(keepends=True))
    worker_edits = _line_edits(base_lines, worker.splitlines(keepends=True))
    for current_edit in current_edits:
        for worker_edit in worker_edits:
            if _edits_conflict(current_edit, worker_edit):
                return None
    merged = list(base_lines)
    edits = list(current_edits)
    for edit in worker_edits:
        if edit not in edits:
            edits.append(edit)
    for start, end, replacement in sorted(edits, key=lambda item: (item[0], item[1]), reverse=True):
        merged[start:end] = replacement
    return "".join(merged)


def _line_edits(base_lines: list[str], result_lines: list[str]) -> list[tuple[int, int, list[str]]]:
    matcher = difflib.SequenceMatcher(a=base_lines, b=result_lines, autojunk=False)
    return [
        (base_start, base_end, result_lines[result_start:result_end])
        for tag, base_start, base_end, result_start, result_end in matcher.get_opcodes()
        if tag != "equal"
    ]


def _edits_conflict(left: tuple[int, int, list[str]], right: tuple[int, int, list[str]]) -> bool:
    left_start, left_end, left_replacement = left
    right_start, right_end, right_replacement = right
    if left_start == left_end and right_start == right_end:
        return left_start == right_start and left_replacement != right_replacement
    if left_start == left_end:
        return right_start < left_start < right_end
    if right_start == right_end:
        return left_start < right_start < left_end
    return max(left_start, right_start) < min(left_end, right_end)


def _git_merge_text(current: str, base: str, worker: str) -> tuple[str, bool]:
    with tempfile.TemporaryDirectory(prefix="hca-merge-") as temp_dir:
        root = Path(temp_dir)
        current_path = root / "current"
        base_path = root / "base"
        worker_path = root / "worker"
        current_path.write_text(current, encoding="utf-8")
        base_path.write_text(base, encoding="utf-8")
        worker_path.write_text(worker, encoding="utf-8")
        env = dict(os.environ)
        env.update({"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never"})
        completed = subprocess.run(
            ["git", "merge-file", "-p", "--diff3", str(current_path), str(base_path), str(worker_path)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=30,
            check=False,
        )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(f"git merge-file failed: {completed.stderr.strip()}")
    return completed.stdout, completed.returncode == 1


def _entry_diff(entry: ChangeEntry) -> str:
    before = (entry.base_content or "").splitlines()
    after = entry.result_content.splitlines()
    return "\n".join(difflib.unified_diff(before, after, fromfile=f"a/{entry.path}", tofile=f"b/{entry.path}", lineterm=""))


def _entry_stats(entry: ChangeEntry) -> dict[str, int]:
    additions = deletions = 0
    for line in difflib.unified_diff(
        (entry.base_content or "").splitlines(),
        entry.result_content.splitlines(),
        lineterm="",
    ):
        if line.startswith("+++ ") or line.startswith("--- "):
            continue
        if line.startswith("+"):
            additions += 1
        elif line.startswith("-"):
            deletions += 1
    return {"additions": additions, "deletions": deletions}


def _applied_payload(
    proposal_id: str,
    results,
    changes: dict[str, str],
    workspace_root: Path,
    *,
    conflict_id: str | None = None,
) -> dict:
    relative = {result.path.relative_to(workspace_root).as_posix(): result for result in results}
    payload = {
        "status": "applied",
        "proposal_id": proposal_id,
        "changed_files": list(relative),
        "file_changes": [
            {
                "path": path,
                "operation": "apply_agent_changes",
                "snapshot_path": str(result.snapshot_path) if result.snapshot_path else None,
                "diff": _text_diff(path, result.old_content or "", changes[path]),
            }
            for path, result in relative.items()
        ],
    }
    if conflict_id:
        payload["conflict_id"] = conflict_id
    return payload


def _text_diff(path: str, before: str, after: str) -> str:
    return "\n".join(difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    ))


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_text(value: str) -> str:
    return _sha_bytes(value.encode("utf-8"))


def _normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _marked_create_conflict(current: str, worker: str) -> str:
    return f"<<<<<<< MAIN\n{current}\n||||||| BASE\n\n=======\n{worker}\n>>>>>>> WORKER\n"


def _marked_delete_conflict(worker: str) -> str:
    return f"<<<<<<< MAIN (deleted)\n||||||| BASE\n=======\n{worker}\n>>>>>>> WORKER\n"
