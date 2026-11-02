"""Explicit isolated environments with a stable Local baseline."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .access import WorkspaceAccessCoordinator, WorkspaceConflict
from ..runtime.recovery import RecoveryStore, atomic_json
from ..agent.cancellation import current_token


class WorktreeService:
    def __init__(self, project: Path):
        self.project = project.resolve()
        self.access = WorkspaceAccessCoordinator(self.project)
        self.blobs = RecoveryStore(self.project)

    @staticmethod
    def _git(root, *args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True)
        return result.stdout

    def _files(self, root):
        names = self._git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").decode("utf-8").split("\0")
        result = {}
        for name in sorted(set(filter(None, names))):
            if any(part in {".git", ".harness"} for part in Path(name).parts):
                continue
            path = root / name
            if path.is_symlink():
                raise ValueError(f"Worktree 起点包含符号链接，请先显式处理：{name}")
            if path.is_file():
                result[name] = path.read_bytes()
        return result

    def create(self, session_id: str):
        destination = self.project / ".harness" / "worktrees" / session_id
        with self.access.acquire([self.access.write_claim()], current_token.get()):
            files = self._files(self.project)
            self._git(self.project, "worktree", "add", "--detach", str(destination), "HEAD")
            tracked = self._git(destination, "ls-files", "-z").decode("utf-8").split("\0")
            for name in filter(None, tracked):
                if name not in files:
                    (destination / name).unlink(missing_ok=True)
            for name, data in files.items():
                path = destination / name
                if path.is_symlink():
                    path.unlink()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.resolve().relative_to(destination.resolve())
                shutil.copy2(self.project / name, path)
            baseline = {name: self.blobs._put_blob(data) for name, data in files.items()}
            descriptor = {"root": str(destination), "project": str(self.project), "baseline": baseline}
            atomic_json(self.project / ".harness" / "worktree-environments" / f"{session_id}.json", descriptor)
        return destination

    def apply(self, session_id, *, turn_id="worktree-apply", resolutions=None):
        from ..agent.change_proposal import _git_merge_text
        descriptor = json.loads((self.project / ".harness" / "worktree-environments" / f"{session_id}.json").read_text(encoding="utf-8"))
        root = Path(descriptor["root"])
        work_access = WorkspaceAccessCoordinator(root)
        with self.access.acquire([self.access.write_claim()], current_token.get()), work_access.acquire([work_access.write_claim()], current_token.get()):
            after = self._files(root)
            baseline = {name: self.blobs.blob(digest) for name, digest in descriptor["baseline"].items()}
            changes, originals, conflicts = {}, {}, {}
            for name in sorted(baseline.keys() | after.keys()):
                before, desired = baseline.get(name), after.get(name)
                if before == desired:
                    continue
                path = self.project / name
                current = path.read_bytes() if path.is_file() else None
                originals[name] = current
                if current == desired:
                    continue
                if current != before and not (resolutions is not None and name in resolutions and current == resolutions[name]):
                    if any(data is None or b"\0" in data for data in (current, before, desired)):
                        conflicts[name] = dict(base=before, local=current, task=desired)
                        continue
                    try:
                        merged, conflict = _git_merge_text(current.decode("utf-8"), before.decode("utf-8"), desired.decode("utf-8"))
                    except UnicodeDecodeError:
                        conflicts[name] = dict(base=before, local=current, task=desired)
                        continue
                    if conflict:
                        conflicts[name] = dict(base=before, local=current, task=desired)
                        continue
                    desired = merged.encode("utf-8")
                changes[name] = desired
            if conflicts:
                error = WorkspaceConflict("Local 与 Worktree 均有重叠修改，已保留两边文件：" + ", ".join(conflicts))
                error.files = conflicts
                raise error
            descriptor_path = self.project / ".harness" / "worktree-environments" / f"{session_id}.json"
            records = [(session_id, turn_id, name, originals[name], desired) for name, desired in changes.items()]
            with self.access.transaction(changes, participants=[descriptor_path], change_records=records):
                descriptor["baseline"] = {name: self.blobs._put_blob(data) for name, data in after.items()}
                atomic_json(descriptor_path, descriptor)
            return list(changes)
