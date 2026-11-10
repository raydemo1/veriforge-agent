"""Explicit isolated environments with a stable Local baseline."""
from __future__ import annotations

import ast
import difflib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..runtime.recovery import RecoveryStore, atomic_json
from ..support.cancellation import current_token
from .access import WorkspaceAccessCoordinator, WorkspaceConflict, fingerprint


@dataclass(frozen=True)
class WorktreeResolution:
    base: bytes | None
    local: bytes | None
    result: bytes | None


class WorktreeMergeConflict(WorkspaceConflict):
    def __init__(self, files):
        super().__init__("Local 与 Worktree 存在重叠修改，两边文件已保留：" + ", ".join(files))
        self.files = files


def validate_merge(name, data):
    if data is None:
        return True, "删除文件，需要确认删除是否符合预期"
    try:
        text = data.decode("utf-8")
        if any(line.startswith(("<<<<<<< ", "||||||| ", ">>>>>>> ")) for line in text.splitlines()):
            return False, "仍包含冲突标记"
        suffix = Path(name).suffix.lower()
        if suffix == ".py":
            ast.parse(text, filename=name)
        elif suffix == ".json":
            json.loads(text)
        else:
            return True, "此格式未自动检查；请审阅差异"
        return True, "语法或文件格式检查通过；仍需审阅重叠修改"
    except (UnicodeError, SyntaxError, ValueError) as exc:
        return False, f"格式检查未通过：{exc}"


def prepare_merge_review(directory, files, originals):
    details, approved = [], {}
    for name, versions in files.items():
        initial = originals[name]
        saved = dict(initial, current_local=versions["local"], result=versions["task"])
        for kind, data in saved.items():
            if data is not None:
                path = directory / kind / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        _, check = validate_merge(name, versions["task"])
        details.extend([f"文件：{name}", check])
        for label, source in (("Local", versions["local"]), ("原始 Worktree", initial["task"])):
            try:
                if any(data is not None and b"\0" in data for data in (source, versions["task"])):
                    raise UnicodeError("binary content")
                left = (source or b"").decode("utf-8").splitlines()
                right = (versions["task"] or b"").decode("utf-8").splitlines()
                diff = "\n".join(difflib.unified_diff(left, right, fromfile=f"{label}/{name}", tofile=f"合并结果/{name}", lineterm=""))
                details.append(diff or (f"与{label}相同" if source == versions["task"] else "仅行尾、末尾换行或文件存在状态变化"))
            except UnicodeError:
                details.append(f"{label}：二进制内容，请检查保存的版本文件")
        approved[name] = WorktreeResolution(versions["base"], versions["local"], versions["task"])
    text = "\n\n".join(details)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "review.diff").write_text(text, encoding="utf-8")
    atomic_json(directory / "versions.json", {name: {"original": {kind: fingerprint(data) for kind, data in originals[name].items()},
                                                     "reviewed": {kind: fingerprint(data) for kind, data in versions.items()}}
                                              for name, versions in files.items()})
    return text, approved


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
            for name in files:
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

    def apply(self, session_id, *, turn_id="worktree-apply", approved=None):
        from .merge import git_merge_text
        descriptor = json.loads((self.project / ".harness" / "worktree-environments" / f"{session_id}.json").read_text(encoding="utf-8"))
        root = Path(descriptor["root"])
        work_access = WorkspaceAccessCoordinator(root)
        with self.access.acquire([self.access.write_claim()], current_token.get()), work_access.acquire([work_access.write_claim()], current_token.get()):
            after = self._files(root)
            baseline = {name: self.blobs.blob(digest) for name, digest in descriptor["baseline"].items()}
            for name, resolution in (approved or {}).items():
                path = self.project / name
                current = path.read_bytes() if path.is_file() else None
                if (baseline.get(name), current, after.get(name)) != (resolution.base, resolution.local, resolution.result):
                    raise WorkspaceConflict(f"确认后文件再次发生变化，请重新审阅；文件未被覆盖：{name}")
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
                if current != before and name not in (approved or {}):
                    if any(data is None or b"\0" in data for data in (current, before, desired)):
                        conflicts[name] = {"base": before, "local": current, "task": desired}
                        continue
                    try:
                        merged, conflict = git_merge_text(current.decode("utf-8"), before.decode("utf-8"), desired.decode("utf-8"))
                    except UnicodeDecodeError:
                        conflicts[name] = {"base": before, "local": current, "task": desired}
                        continue
                    if conflict:
                        conflicts[name] = {"base": before, "local": current, "task": desired}
                        continue
                    desired = merged.encode("utf-8")
                    valid, _ = validate_merge(name, desired)
                    if not valid:
                        conflicts[name] = {"base": before, "local": current, "task": after.get(name)}
                        continue
                changes[name] = desired
            if conflicts:
                raise WorktreeMergeConflict(conflicts)
            descriptor_path = self.project / ".harness" / "worktree-environments" / f"{session_id}.json"
            records = [(session_id, turn_id, name, originals[name], desired) for name, desired in changes.items()]
            with self.access.transaction(changes, participants=[descriptor_path], change_records=records):
                descriptor["baseline"] = {name: self.blobs._put_blob(data) for name, data in after.items()}
                atomic_json(descriptor_path, descriptor)
            return list(changes)
