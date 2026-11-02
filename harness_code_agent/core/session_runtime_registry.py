"""Runtime ownership, addressing and replay for one project workbench."""
from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ..sessions.store import SessionStore
from ..workspace.worktrees import WorktreeService


@dataclass
class SessionRuntime:
    id: str
    cwd: Path
    environment: str
    title: str = "新会话"
    visible: bool = True
    status: str = "starting"
    host: object = None


class SessionRuntimeRegistry:
    def __init__(self, project, factory, emit):
        self.project = Path(project).resolve()
        self.store = SessionStore(self.project / ".harness")
        from ..workspace.access import WorkspaceAccessCoordinator
        self.access = WorkspaceAccessCoordinator(self.project)
        self.factory, self.emit = factory, emit
        self._lock = threading.RLock()
        self._records = {}
        self._views = {}
        self.active_id = None
        self._closing = False

    def create(self, *, session_id=None, cwd=None, environment="local"):
        with self._lock:
            if self._closing:
                raise RuntimeError("工作台正在关闭")
            session_id = session_id or self.store._new_session_id()
            if session_id in self._records:
                self.switch(session_id)
                return session_id
            self.access.reserve_runtime(session_id)
            record = SessionRuntime(session_id, Path(cwd or self.project), environment)
            if (self.store.sessions_dir / session_id / "session.json").exists():
                record.title = self.store.read_metadata(session_id).get("title", "新会话")
            self._records[session_id] = record
            self._views[session_id] = {"sequence": 0, "items": [], "snapshot": {}, "interaction": None,
                                       "work": {}, "turnState": "idle", "queueDepth": 0, "commands": []}
            self.active_id = session_id
        try:
            record.host = self.factory(record)
        except BaseException:
            with self._lock:
                self.access.release_runtime(session_id)
                self._records.pop(session_id, None)
                self._views.pop(session_id, None)
            raise
        self.announce()
        return session_id

    def get(self, session_id):
        with self._lock:
            record = self._records.get(session_id)
            if record is None:
                raise ValueError("会话尚未恢复或不存在")
            return record

    def switch(self, session_id):
        with self._lock:
            record = self.get(session_id)
            record.visible = True
            self.active_id = session_id
        self.announce()

    def hide(self, session_id):
        with self._lock:
            self.get(session_id).visible = False
            if self.active_id == session_id:
                self.active_id = next((r.id for r in self._records.values() if r.visible), None)
        self.announce()

    def resume(self, session_id):
        with self._lock:
            if session_id in self._records:
                self.switch(session_id)
                return session_id
        metadata = self.store.read_metadata(session_id)
        return self.create(session_id=session_id, cwd=metadata["cwd"], environment=metadata.get("environment", "local"))

    def fork(self, session_id, *, worktree=False):
        source = self.get(session_id)
        runtime = source.host._require_session()
        fork = self.store.fork(runtime.session.id)
        root = WorktreeService(self.project).create(fork.id) if worktree else source.cwd
        if root != source.cwd:
            from ..runtime.output_artifacts import copy_artifact
            from ..sessions.journal import SessionJournal
            for entry in SessionJournal(fork.journal_path).read():
                references = []
                if entry.kind == "tool_output_artifact":
                    references = [entry.payload]
                elif entry.kind == "message":
                    references = entry.payload.get("metadata", {}).get("tool_output_artifacts", [])
                for reference in references:
                    copy_artifact(source.cwd, root, reference)
        from ..runtime.recovery import atomic_json
        metadata = self.store.read_metadata(fork.id)
        metadata.update(cwd=str(root), environment="worktree" if worktree else source.environment)
        from ..runtime.recovery import RecoveryStore
        from ..workspace.access import WorkspaceAccessCoordinator
        import uuid
        boundary = metadata["history_boundary"]
        base = dict(boundary, id="rp_" + uuid.uuid4().hex, session_id=fork.id,
                    change_cursor=WorkspaceAccessCoordinator(root).cursor(), profile=metadata["profile"],
                    permission_mode=metadata["permission_mode"], model_selection=metadata.get("model_selection", {"model": None, "reasoning_effort": None}))
        base["routing_mode"] = metadata["routing_mode"]
        metadata.update(checkpoint=base, branch_origin=base)
        atomic_json(RecoveryStore(root).root / "points" / f"{base['id']}.json", base)
        atomic_json(fork.metadata_path, metadata)
        return self.create(session_id=fork.id, cwd=root, environment=metadata["environment"])

    def announce(self):
        with self._lock:
            sessions = [dict(id=r.id, title=r.title, environment=r.environment,
                             cwd=str(r.cwd), visible=r.visible, status=r.status,
                             pendingApproval=self._views[r.id]["interaction"] is not None)
                        for r in self._records.values()]
            self.emit({"type": "sessions", "sessions": sessions, "activeSessionId": self.active_id})

    def publish(self, session_id, event):
        with self._lock:
            record = self._records[session_id]
            view = self._views[session_id]
            kind = event["type"]
            if kind == "shutdown":
                record.status = "closed"
                self.announce()
                return
            if kind in {"snapshot", "session_reset"}:
                view["snapshot"] = event["snapshot"]
                if kind == "session_reset":
                    view["items"] = event.get("items", [])
                    view["work"] = {}
            elif kind == "transcript":
                item = event["item"]
                existing = next((i for i, row in enumerate(view["items"]) if row["id"] == item["id"]), None)
                if existing is None:
                    view["items"].append(copy.deepcopy(item))
                else:
                    view["items"][existing] = copy.deepcopy(item)
                if item["kind"] == "user" and record.title == "新会话":
                    record.title = item["body"].splitlines()[0][:32] or "新会话"
            elif kind in {"assistant_delta", "transcript_update"}:
                for item in view["items"]:
                    if item["id"] == event["id"]:
                        item["body"] = item["body"] + event["text"] if kind == "assistant_delta" else event["body"]
                        if event.get("state"):
                            item["state"] = event["state"]
            elif kind == "recovery_available":
                if event["turn"] == 0:
                    view["items"] = [{"id": "recovery-origin", "kind": "status", "title": "会话开始", "body": "",
                                      "turn": 0, "recoveryPointId": event["pointId"]},
                                     *[item for item in view["items"] if item["id"] != "recovery-origin"]]
                for item in view["items"]:
                    if item.get("role") == "group" and item.get("turn") == event["turn"]:
                        item["recoveryPointId"] = event["pointId"]
            elif kind == "interaction":
                view["interaction"] = event
            elif kind == "notice":
                failed = event.get("level") == "error"
                view["items"].append(dict(id=f"notice-{view['sequence'] + 1}", kind="error" if failed else "status",
                                          title="错误" if failed else "提示", body=event["text"], state="failed" if failed else "success"))
            elif kind == "interaction_closed":
                if view["interaction"] and view["interaction"]["id"] == event["id"]:
                    view["interaction"] = None
            elif kind == "turn_state":
                view["turnState"] = event["state"]
                view["queueDepth"] = event.get("queueDepth", 0)
                record.status = event["state"]
            elif kind == "progress":
                record.status = event["status"]
            elif kind == "commands":
                view["commands"] = event["commands"]
            elif kind.endswith("_updated"):
                field = {"plan_updated": "plan", "tasks_updated": "tasks", "agent_run_updated": "agents",
                         "changes_updated": "changes", "verification_updated": "checks", "artifact_updated": "artifacts"}.get(kind)
                if field:
                    view["work"][field] = event[field]
            view["sequence"] += 1
            self.emit(dict(event, sessionId=session_id, sequence=view["sequence"]))
            if kind in {"progress", "turn_state", "interaction", "interaction_closed", "transcript"}:
                self.announce()

    def snapshot(self, session_id):
        with self._lock:
            return copy.deepcopy(self._views[session_id])

    def close(self):
        with self._lock:
            self._closing = True
            records = list(self._records.values())
        errors = []
        lock = threading.Lock()
        def close_one(record):
            try:
                record.host.close()
                self.access.release_runtime(record.id)
            except Exception as exc:
                with lock:
                    errors.append(f"{record.id}: {exc}")
        threads = [threading.Thread(target=close_one, args=(r,), daemon=True) for r in records]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + 15
        for thread in threads:
            thread.join(timeout=max(0, deadline - time.monotonic()))
        if errors or any(t.is_alive() for t in threads):
            raise RuntimeError("部分会话清理未完成，依赖资源已保留：" + "; ".join(errors))
