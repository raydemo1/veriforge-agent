from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from .reconciliation import ConsolidationQueue
from .service import MemoryService
from .vectors import BudgetedEmbeddings, VectorIndex

log = logging.getLogger("harness")
_lock = threading.RLock()
_running: dict[tuple, bool] = {}
_timers: dict[tuple, threading.Timer] = {}
_clients: dict[tuple, set[str]] = {}
_active: set[tuple] = set()
_heartbeats: dict[tuple, threading.Timer] = {}


def _key(service):
    store = service.stores["project"]
    return str(store.path), store.namespace


def open_memory_session(workspace, session_id, journal_path):
    service = MemoryService(workspace)
    service.extraction.register(session_id, Path(journal_path))
    with _lock:
        _clients.setdefault(_key(service), set()).add(session_id)
    start_memory_worker(workspace)


def begin_memory_turn(workspace, session_id):
    service = MemoryService(workspace)
    with _lock:
        _active.add((*_key(service), session_id))
    _heartbeat(service, (*_key(service), session_id))


def _heartbeat(service, identity):
    with _lock:
        if identity not in _active:
            return
        service.extraction.set_active(identity[-1])
        timer = threading.Timer(30, _heartbeat, args=(service, identity))
        timer.daemon = True
        previous = _heartbeats.pop(identity, None)
        if previous:
            previous.cancel()
        _heartbeats[identity] = timer
        timer.start()


def finish_memory_turn(workspace, session_id, journal_path=None):
    service = MemoryService(workspace)
    with _lock:
        identity = (*_key(service), session_id)
        _active.discard(identity)
        heartbeat = _heartbeats.pop(identity, None)
        if heartbeat:
            heartbeat.cancel()
    service.extraction.finish_activity(session_id)
    if journal_path is not None:
        service.extraction.enqueue(session_id, Path(journal_path))
    start_memory_worker(workspace)


def close_memory_session(workspace, session_id):
    service = MemoryService(workspace)
    key = _key(service)
    with _lock:
        identity = (*key, session_id)
        _active.discard(identity)
        heartbeat = _heartbeats.pop(identity, None)
        if heartbeat:
            heartbeat.cancel()
        clients = _clients.get(key, set())
        clients.discard(session_id)
        if not clients:
            _clients.pop(key, None)
            timer = _timers.pop(key, None)
            if timer:
                timer.cancel()


def drain_memory_jobs(service, *, max_model_calls=8, max_embedding_requests=16, should_continue=None):
    if not 1 <= max_model_calls <= 16 or not 0 <= max_embedding_requests <= 32:
        raise ValueError("Invalid background memory budget")
    budget = BudgetedEmbeddings(max_embedding_requests)
    def factory(database):
        return VectorIndex(database, embeddings=budget)
    project = service.stores["project"]
    consolidation = ConsolidationQueue(project)
    calls, states = 0, []
    eligible = should_continue or (lambda: True)
    while calls < max_model_calls and eligible():
        extracted = service.extraction.claim()
        if extracted:
            result = service.extraction.process(extracted, max_calls=min(8, max_model_calls - calls), should_continue=eligible)
            calls += result["calls"]
            states.append(result["state"])
        if not eligible():
            break
        task = consolidation.claim() if calls < max_model_calls else None
        if task:
            result = consolidation.process(task, should_continue=eligible)
            calls += result["calls"]
            states.append(result["state"])
        for store in service.stores.values():
            if budget.remaining == 0 or not eligible():
                break
            result = factory(store).sync(max_requests=min(4, budget.remaining), should_continue=eligible)
            if result.get("budget_exhausted"):
                break
        if not extracted and not task:
            break
    service.flush_exports()
    return {"model_calls": calls, "embedding_requests": budget.used, "states": states}


def start_memory_worker(workspace):
    service = MemoryService(workspace)
    _start_service_worker(service)


def _start_service_worker(service):
    key = _key(service)
    with _lock:
        if key not in _clients:
            return
        if key in _running:
            _running[key] = True
            return
        timer = _timers.pop(key, None)
        if timer:
            timer.cancel()
        _running[key] = False
    threading.Thread(target=_run, args=(service, key), daemon=True).start()


def _has_clients(key):
    with _lock:
        return key in _clients


def _run(service, key):
    try:
        while True:
            with _lock:
                if key not in _clients:
                    break
            drain_memory_jobs(service, should_continue=lambda: _has_clients(key))
            with _lock:
                if _running[key]:
                    _running[key] = False
                    continue
                break
    except Exception as exc:  # noqa: BLE001 - durable jobs retain progress; do not log source/provider content
        log.warning("Memory worker paused: %s", type(exc).__name__)
    finally:
        with _lock:
            _running.pop(key, None)
            if key in _clients:
                _schedule(service, key)


def _schedule(service, key):
    store = service.stores["project"]
    now = time.time()
    with store.connect() as db:
        rows = db.execute(
            "SELECT MAX(t.available_at,s.active_until,COALESCE(t.lease_until,0)) AS deadline "
            "FROM extraction_tasks t JOIN extraction_sessions s ON s.namespace=t.namespace AND s.session_id=t.session_id "
            "WHERE t.namespace=? AND s.enabled=1 AND t.status IN ('pending','running') "
            "UNION ALL SELECT MAX(c.available_at,s.active_until,COALESCE(c.lease_until,0)) "
            "FROM consolidation_tasks c JOIN extraction_tasks t ON t.id=c.extraction_task "
            "JOIN extraction_sessions s ON s.namespace=t.namespace AND s.session_id=t.session_id "
            "WHERE c.namespace=? AND s.enabled=1 AND c.status IN ('pending','running') "
            "UNION ALL SELECT o.available_at FROM vector_outbox o JOIN memories m ON m.id=o.memory_id "
            "WHERE m.namespace IN (?, 'user') AND m.status<>'expired' AND o.failures<3",
            (store.namespace, store.namespace, store.namespace),
        ).fetchall()
    deadlines = [row[0] for row in rows if row[0] is not None]
    if deadlines:
        delay = max(30, min(deadlines) - now)
        timer = threading.Timer(delay, _start_service_worker, args=(service,))
        timer.daemon = True
        previous = _timers.pop(key, None)
        if previous:
            previous.cancel()
        _timers[key] = timer
        timer.start()
