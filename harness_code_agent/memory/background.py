from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path

from .. import config
from ..agent.providers import ProviderAdapter, client_scope
from .service import MemoryService
from .store import MemoryWriteCommand

log = logging.getLogger("harness")
_worker_lock = threading.Lock()
#: Maps a workspace to a rerun flag while its kicked one-shot worker runs.
#: A kick arriving while the worker is running sets the flag so the worker
#: drains one more pass instead of the kick being lost.
_running_workspaces: dict[Path, bool] = {}


def start_memory_worker(workspace: str | Path) -> None:
    """Kick one non-blocking extraction pass for one workspace.

    The worker is one-shot, not a scheduler: it drains the jobs ready at the
    time of the kick and exits. Failed jobs keep a future ``available_at`` and
    are picked up by a later kick; no timer is involved.
    """
    resolved = Path(workspace).resolve()
    with _worker_lock:
        if resolved in _running_workspaces:
            # Missed-wake-up guard: remember the kick; the running worker
            # re-checks the flag before removing itself.
            _running_workspaces[resolved] = True
            return
        _running_workspaces[resolved] = False
    thread = threading.Thread(target=_run_once, args=(resolved,), daemon=True)
    thread.start()


def _run_once(workspace: Path) -> None:
    try:
        service = MemoryService(workspace)
        store = service.stores["project"]
        store.ensure_initialized()
        while True:
            while True:
                row = _claim_next_job(store)
                if row is False:
                    continue
                if row is None:
                    break
                _process_job(service, store, row)
            with _worker_lock:
                if _running_workspaces.get(workspace):
                    # A kick landed during the drain; run another pass.
                    _running_workspaces[workspace] = False
                else:
                    # Removal and the exit decision share the lock, closing
                    # the window where a kick could see a running worker that
                    # is about to exit.
                    _running_workspaces.pop(workspace, None)
                    return
    finally:
        with _worker_lock:
            _running_workspaces.pop(workspace, None)


def _claim_next_job(store):
    now = time.time()
    with store.connect() as db:
        db.execute(
            "UPDATE extraction_jobs SET status='pending',lease_until=NULL "
            "WHERE status='running' AND lease_until<? AND attempts<3",
            (now,),
        )
        row = db.execute(
            "SELECT * FROM extraction_jobs WHERE status='pending' AND available_at<=? "
            "AND attempts<3 ORDER BY id LIMIT 1",
            (now,),
        ).fetchone()
        if row is None:
            return None
        session_key = "session:" + hashlib.sha256(row["session_id"].encode()).hexdigest()
        if db.execute("SELECT 1 FROM suppressions WHERE key=?", (session_key,)).fetchone():
            db.execute("UPDATE extraction_jobs SET status='suppressed' WHERE id=?", (row["id"],))
            return False
        claimed = db.execute(
            "UPDATE extraction_jobs SET status='running',attempts=attempts+1,lease_until=? "
            "WHERE id=? AND status='pending'",
            (now + 300, row["id"]),
        ).rowcount
        return row if claimed else None


def _process_job(service: MemoryService, store, row) -> None:
    session_key = "session:" + hashlib.sha256(row["session_id"].encode()).hexdigest()
    for scoped_store in service.stores.values():
        scoped_store.ensure_initialized()
        with scoped_store.connect() as db:
            if db.execute("SELECT 1 FROM suppressions WHERE key=?", (session_key,)).fetchone():
                with store.connect() as queue_db:
                    queue_db.execute(
                        "UPDATE extraction_jobs SET status='suppressed',lease_until=NULL WHERE id=?",
                        (row["id"],),
                    )
                return
    try:
        candidates = _extract_candidates(service, Path(row["journal_path"]), row["session_id"])
        for candidate in candidates:
            service.write(candidate)
        with store.connect() as db:
            db.execute("UPDATE extraction_jobs SET status='done',lease_until=NULL,error=NULL WHERE id=?", (row["id"],))
    except Exception as exc:  # noqa: BLE001 - failed jobs are leased and retried later
        log.debug("Memory extraction failed: %s", exc)
        with store.connect() as db:
            db.execute(
                "UPDATE extraction_jobs SET status='pending',lease_until=NULL,available_at=?,error=? WHERE id=?",
                (time.time() + 300, str(exc)[:500], row["id"]),
            )


def _extract_candidates(
    service: MemoryService,
    journal_path: Path,
    session_id: str,
) -> list[MemoryWriteCommand]:
    if not journal_path.is_file():
        return []
    text = journal_path.read_text(encoding="utf-8", errors="replace")[-60_000:]
    existing = [
        {"id": doc.id, "version": doc.version, "scope": doc.scope, "topic": doc.topic, "body": doc.body[:300]}
        for store in service.stores.values()
        for doc in store.list_documents()
    ][:50]
    prompt = {
        "session_id": session_id,
        "existing_memories": existing,
        "journal": text,
        "rules": [
            "Return a JSON array with at most 4 durable memories.",
            "Keep only explicit preferences, project conventions, decisions, or verified reusable lessons.",
            "Exclude one-off state, temporary logs, secrets, hidden reasoning, and likely-to-expire details.",
            "Fields: topic, body, scope(project|user), applicability, source_paths, supersedes(optional), expected_version(optional).",
            "Use supersedes with the old id and expected_version only for an evidence-backed correction of the same fact.",
            "Never merge or supersede memories merely because their paths overlap.",
        ],
    }
    profile = config.resolve_model_profile("fast")
    adapter = ProviderAdapter(profile.provider)
    kwargs = adapter.chat_kwargs(
        profile=profile,
        messages=[
            {"role": "system", "content": "Extract auditable coding-agent memories. Output JSON only."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        max_tokens=1600,
    )
    with client_scope() as client:
        response = client.chat.completions.create(**kwargs)
    raw = str(response.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    values = json.loads(raw)
    if not isinstance(values, list):
        return []
    commands = []
    for value in values[:4]:
        if not isinstance(value, dict):
            continue
        body = str(value.get("body") or "").strip()
        if not body or _looks_secret(body):
            continue
        scope = str(value.get("scope") or "project")
        if scope not in {"project", "user"}:
            scope = "project"
        commands.append(
            MemoryWriteCommand(
                topic=str(value.get("topic") or body[:80]), body=body, scope=scope,
                applicability=str(value.get("applicability") or ""),
                source_sessions=[session_id],
                source_paths=[str(item) for item in value.get("source_paths", []) if isinstance(item, str)],
                supersedes=str(value.get("supersedes") or "") or None,
                expected_version=value.get("expected_version"),
            )
        )
    return commands


def _looks_secret(text: str) -> bool:
    return bool(re.search(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*\S+", text))
