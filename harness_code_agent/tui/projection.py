"""UI read-model projection: runtime events -> Work State.

The layer is deliberately dumb.  It owns no business rules: proposal
applicability, conflict state and isolation mode all come from the runtime;
this module only shapes the event stream into the six Work State sections the
terminal renders (Plan / Tasks / Agents / Changes / Checks / Artifacts).

Replaying the same journaled events produces the same state, so resume and
live streaming share one code path.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

#: Section key -> UI event type suffix mapping used by the bridge.
SECTION_EVENTS = {
    "plan": "plan_updated",
    "tasks": "tasks_updated",
    "agents": "agent_run_updated",
    "changes": "changes_updated",
    "checks": "verification_updated",
    "artifacts": "artifact_updated",
}

_TERMINAL_AGENT_STATES = {"completed", "failed", "blocked", "interrupted"}
_PLAN_STEP_LIMIT = 12
_CHECK_TOOLS = {"browser_test"}


class WorkStateProjection:
    def __init__(self, cwd: Path) -> None:
        self._cwd = Path(cwd)
        self._plan: dict[str, Any] | None = None
        self._tasks: list[dict[str, Any]] = []
        self._agents: dict[str, dict[str, Any]] = {}
        self._files: dict[str, dict[str, Any]] = {}
        self._proposals: list[dict[str, Any]] = []
        self._checks: dict[str, dict[str, Any]] = {}
        self._artifacts: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Event intake
    # ------------------------------------------------------------------

    def apply_event(self, event: Any) -> set[str]:
        """Apply one journaled runtime event; return changed section keys."""

        data = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        event_type = str(data.get("type") or "")
        payload = data.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        agent = data.get("agent")

        changed: set[str] = set()
        if event_type == "turn_started":
            if self._checks:
                self._checks.clear()
                changed.add("checks")
            # A follow-up turn resumes an unfinished plan.
            if self._plan is not None and self._plan["status"] == "incomplete":
                self._plan["status"] = "executing"
                changed.add("plan")
        elif event_type == "plan_ready":
            self._apply_plan_ready(payload)
            changed.add("plan")
        elif event_type == "profile_switched":
            if self._plan is not None and str(payload.get("reason") or "") == "execute approved plan":
                self._plan["status"] = "executing"
                changed.add("plan")
        elif event_type == "turn_finished":
            if self._plan is not None and self._plan["status"] == "executing":
                # A finished turn is not evidence the plan is done. Only todo
                # alignment marks steps completed; without that evidence the
                # plan stays visibly incomplete (real task outcomes arrive in
                # a later batch). Never force steps to completed.
                steps = self._plan["steps"]
                if steps and all(step["status"] == "completed" for step in steps):
                    self._plan["status"] = "completed"
                else:
                    self._plan["status"] = "incomplete"
                changed.add("plan")
        elif event_type == "tool_call" and str(payload.get("tool") or "") in _CHECK_TOOLS:
            changed |= self._upsert_check({
                "name": "Browser verification",
                "status": "running",
                "detail": str((payload.get("args") or {}).get("url") or ""),
            })
        elif event_type == "tool_result":
            changed |= self._apply_tool_result(payload)
        elif event_type == "file_change" and (agent is None or agent == "main_agent"):
            self._apply_file_change(payload)
            changed.add("changes")
        elif event_type == "agent_spawned":
            self._agents[payload["agent_id"]] = _agent_from_payload(payload)
            changed.add("agents")
        elif event_type == "agent_status":
            changed |= self._apply_agent_status(payload)
        elif event_type == "checks_recorded":
            for check in payload.get("checks") or []:
                if isinstance(check, dict):
                    changed |= self._upsert_check(check)
        return changed

    def merge_proposals(self, proposals: list[dict[str, Any]] | None) -> bool:
        """Merge live ChangeProposal snapshots owned by the coordinator."""

        normalized = list(proposals or [])
        if normalized == self._proposals:
            return False
        self._proposals = normalized
        return True

    # ------------------------------------------------------------------
    # Section snapshots
    # ------------------------------------------------------------------

    def section(self, key: str) -> Any:
        if key == "plan":
            return self._plan
        if key == "tasks":
            return self._snapshot_tasks()
        if key == "agents":
            return list(self._agents.values())
        if key == "changes":
            return self._snapshot_changes()
        if key == "checks":
            return list(self._checks.values())
        if key == "artifacts":
            return list(self._artifacts.values())
        raise KeyError(f"unknown work state section: {key}")

    def changed_sections(self) -> dict[str, Any]:
        return {key: self.section(key) for key in SECTION_EVENTS}

    # ------------------------------------------------------------------
    # Plan / Tasks
    # ------------------------------------------------------------------

    def _apply_plan_ready(self, payload: dict[str, Any]) -> None:
        revision = payload.get("plan_revision")
        path = str(payload.get("plan_path") or "global_plan/current/plan.md")
        markdown = payload.get("plan_markdown")
        if isinstance(markdown, str) and markdown.strip():
            # Journaled snapshot — the only deterministic replay source.
            plan_text = markdown
        else:
            # Backwards compatibility for events recorded before protocol v5.
            plan_text = self._read_plan(path)
        steps = _parse_plan_steps(plan_text)
        self._plan = {
            "status": "ready",
            "revision": int(revision) if isinstance(revision, int) else 0,
            "path": path,
            "steps": steps,
            "completedCount": 0,
            "totalCount": len(steps),
        }

    def _read_plan(self, rel_path: str) -> str:
        try:
            return (self._cwd / rel_path).read_text(encoding="utf-8")
        except OSError:
            return ""

    def _apply_todo(self, metadata: dict[str, Any]) -> set[str]:
        todo_state = metadata.get("todo_state")
        if not isinstance(todo_state, dict):
            return set()
        raw_items = todo_state.get("items")
        if not isinstance(raw_items, list):
            return set()
        tasks: list[dict[str, Any]] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text") or "").strip()
            if not text:
                continue
            tasks.append({
                "id": str(raw.get("id") or ""),
                "text": text,
                "status": str(raw.get("status") or "pending"),
            })
        self._tasks = tasks
        changed = {"tasks"}
        # Todo progress doubles as plan execution progress while executing.
        if self._plan is not None and self._plan["status"] == "executing" and tasks:
            self._align_plan_steps(tasks)
            changed.add("plan")
        return changed

    def _align_plan_steps(self, tasks: list[dict[str, Any]]) -> None:
        steps = self._plan["steps"]
        if not steps or len(tasks) != len(steps):
            return
        for step, task in zip(steps, tasks):
            step["status"] = task["status"]
        completed = sum(1 for step in steps if step["status"] == "completed")
        self._plan["completedCount"] = completed

    def _snapshot_tasks(self) -> dict[str, Any]:
        return {
            "items": self._tasks,
            "completed": sum(1 for item in self._tasks if item["status"] == "completed"),
            "total": len(self._tasks),
        }

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------

    def _apply_agent_status(self, payload: dict[str, Any]) -> set[str]:
        agent_id = str(payload.get("agent_id") or "")
        status = str(payload.get("status") or "")
        if status == "closed":
            if self._agents.pop(agent_id, None) is not None:
                return {"agents"}
            return set()
        existing = self._agents.get(agent_id)
        if existing is None:
            if not agent_id:
                return set()
            self._agents[agent_id] = _agent_from_payload(payload)
        else:
            existing["status"] = status
            existing.update({
                "summary": str(payload.get("summary") or existing["summary"]),
                "error": payload.get("error") or existing["error"],
                "proposalId": payload.get("proposal_id") or existing["proposalId"],
                "durationSeconds": _duration(payload.get("duration_seconds"), existing),
            })
        return {"agents"}

    # ------------------------------------------------------------------
    # Changes
    # ------------------------------------------------------------------

    def _apply_file_change(self, payload: dict[str, Any]) -> None:
        path = str(payload.get("path") or "")
        if not path:
            return
        additions, deletions = _diff_counts(
            str(payload.get("diff") or ""),
            payload.get("additions"),
            payload.get("deletions"),
        )
        self._files[path] = {
            "path": path,
            "operation": str(payload.get("operation") or "modify"),
            "additions": additions,
            "deletions": deletions,
        }

    def _snapshot_changes(self) -> dict[str, Any]:
        workspace = sorted(self._files.values(), key=lambda item: item["path"])
        proposals = [self._decorate_proposal(item) for item in self._proposals]
        proposals = [item for item in proposals if item["status"] != "applied"]
        return {
            "workspace": workspace,
            "additions": sum(item["additions"] for item in workspace),
            "deletions": sum(item["deletions"] for item in workspace),
            "proposals": proposals,
        }

    def _decorate_proposal(self, item: dict[str, Any]) -> dict[str, Any]:
        decorated = dict(item)
        agent = self._agents.get(str(item.get("agentId") or ""))
        decorated["agentName"] = str(agent["name"] if agent else item.get("agentName") or item.get("agentId") or "worker")
        return decorated

    # ------------------------------------------------------------------
    # Checks / artifacts
    # ------------------------------------------------------------------

    def _apply_tool_result(self, payload: dict[str, Any]) -> set[str]:
        tool = str(payload.get("tool") or "")
        changed: set[str] = set()
        metadata = payload.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        if tool == "update_todo" and str(payload.get("status") or "") == "success":
            changed |= self._apply_todo(metadata)
        for check in metadata.get("checks") or []:
            if isinstance(check, dict):
                changed |= self._upsert_check(check)
        for artifact in metadata.get("artifacts") or []:
            if isinstance(artifact, dict) and artifact.get("path"):
                changed |= self._upsert_artifact(artifact)
        running = self._checks.get("browser-verification")
        if tool in _CHECK_TOOLS and (running is None or running["status"] == "running"):
            # Runs without structured evidence still resolve the running row
            # to the tool's objective pass/fail.
            changed |= self._upsert_check({
                "name": "Browser verification",
                "status": "passed" if payload.get("status") == "success" else "failed",
                "detail": str(metadata.get("url") or ""),
            })
        return changed

    def _upsert_check(self, raw: dict[str, Any]) -> set[str]:
        name = str(raw.get("name") or "").strip()
        status = str(raw.get("status") or "").strip()
        if not name or status not in {"passed", "failed", "warning", "running"}:
            return set()
        check_id = _slug(name)
        self._checks[check_id] = {
            "id": check_id,
            "name": name,
            "status": status,
            "detail": str(raw.get("detail") or ""),
        }
        return {"checks"}

    def _upsert_artifact(self, raw: dict[str, Any]) -> set[str]:
        path = str(raw.get("path") or "")
        kind = str(raw.get("kind") or "text")
        artifact = {
            "id": _slug(path),
            "kind": "image" if kind in {"image", "screenshot", "png", "jpg", "jpeg"} else "text",
            "path": path,
            "title": str(raw.get("title") or Path(path).name),
            "detail": str(raw.get("detail") or ""),
        }
        if self._artifacts.get(artifact["id"]) == artifact:
            return set()
        self._artifacts[artifact["id"]] = artifact
        return {"artifacts"}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _agent_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    role = str(payload.get("role") or "explorer")
    return {
        "id": str(payload.get("agent_id") or ""),
        "name": str(payload.get("name") or payload.get("agent_id") or "agent"),
        "role": role,
        "task": str(payload.get("task") or ""),
        "status": str(payload.get("status") or "queued"),
        "isolation": "isolated workspace" if role == "worker" else "read-only",
        "summary": str(payload.get("summary") or ""),
        "error": payload.get("error"),
        "proposalId": payload.get("proposal_id"),
        "durationSeconds": _duration(payload.get("duration_seconds"), None),
    }


def _duration(value: Any, existing: dict[str, Any] | None) -> float | None:
    if isinstance(value, (int, float)):
        return round(float(value), 1)
    return existing.get("durationSeconds") if existing else None


def _diff_counts(diff: str, additions: Any, deletions: Any) -> tuple[int, int]:
    try:
        added = int(additions) if additions is not None else None
        deleted = int(deletions) if deletions is not None else None
    except (TypeError, ValueError):
        added = deleted = None
    if added is not None and deleted is not None:
        return max(0, added), max(0, deleted)
    lines = diff.splitlines()
    added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++ "))
    deleted = sum(1 for line in lines if line.startswith("-") and not line.startswith("--- "))
    return added, deleted


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "item"


_CHECKBOX = re.compile(r"^\s*[-*]\s*\[(?P<done>[ xX])\]\s+(?P<text>.+?)\s*$")
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+(?P<text>\S.*?)\s*$")
_BULLET = re.compile(r"^\s*[-*]\s+(?P<text>\S.*?)\s*$")
_HEADING = re.compile(r"^#{2,4}\s+(?P<text>.+?)\s*#*\s*$")
_HEADING_SKIP = re.compile(
    r"title|summary|overview|background|context|assumption|risk|test|verification|"
    r"标题|概述|总结|背景|上下文|假设|风险|测试|验证",
    re.IGNORECASE,
)


def _parse_plan_steps(markdown: str) -> list[dict[str, str]]:
    lines = markdown.splitlines()

    checkbox = [match for line in lines if (match := _CHECKBOX.match(line))]
    if checkbox:
        return [
            {"text": match.group("text"), "status": "completed" if match.group("done").lower() == "x" else "pending"}
            for match in checkbox[:_PLAN_STEP_LIMIT]
        ]

    numbered = [match for line in lines if (match := _NUMBERED.match(line))]
    if numbered:
        return [{"text": match.group("text"), "status": "pending"} for match in numbered[:_PLAN_STEP_LIMIT]]

    bullets = [
        match for line in lines
        if (match := _BULLET.match(line)) and not _HEADING_SKIP.search(match.group("text"))
    ]
    if bullets:
        return [{"text": match.group("text"), "status": "pending"} for match in bullets[:_PLAN_STEP_LIMIT]]

    headings = [
        match for line in lines
        if (match := _HEADING.match(line)) and not _HEADING_SKIP.search(match.group("text"))
    ]
    return [{"text": match.group("text"), "status": "pending"} for match in headings[:_PLAN_STEP_LIMIT]]
