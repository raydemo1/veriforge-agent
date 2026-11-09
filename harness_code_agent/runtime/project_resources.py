"""Project-owned execution capacity and cancellable resource waits."""
from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from .. import config

_projects = weakref.WeakValueDictionary()
_projects_lock = threading.Lock()


class ProjectResourceBudget:
    def __init__(self, root):
        self.root = Path(root).resolve()
        agents = max(1, config.MAX_CONCURRENT_AGENTS)
        self.limits = {"llm": agents, "agents": agents, "tools": 8, "network": 2, "mcp_connect": 2}
        self._condition = threading.Condition(threading.RLock())
        self._active = {}
        self._waiting = {}
        self._owners = 0
        self._closed = False
        self.tools = ThreadPoolExecutor(max_workers=self.limits["tools"], thread_name_prefix="project-tools")
        self.agents = ThreadPoolExecutor(max_workers=agents, thread_name_prefix="project-agents")

    @classmethod
    def for_project(cls, root):
        key = os.path.normcase(str(Path(root).resolve()))
        with _projects_lock:
            budget = _projects.get(key)
            if budget is None or budget._closed:
                budget = cls(root)
                _projects[key] = budget
            return budget

    def hold(self):
        with self._condition:
            if self._closed:
                raise RuntimeError("Project execution resources are closed")
            self._owners += 1
        released = False

        def release():
            nonlocal released
            with self._condition:
                if released:
                    return
                released = True
                self._owners -= 1
                close = self._owners == 0
                if close:
                    self._closed = True
            if close:
                self.tools.shutdown(wait=False, cancel_futures=True)
                self.agents.shutdown(wait=False, cancel_futures=True)
        return release

    @contextmanager
    def acquire(self, kind, *, session_id=None, turn_id=None, label="", cancellation_token=None):
        release_owner = self.hold()
        key = uuid.uuid4().hex
        entry = {"resource": kind, "sessionId": session_id, "turnId": turn_id, "label": label,
                     "since": time.time()}
        limit = self.limits.get(kind, 2)
        try:
            with self._condition:
                self._waiting[key] = entry
                while sum(row["resource"] == kind for row in self._active.values()) >= limit:
                    if cancellation_token is not None:
                        cancellation_token.check()
                    self._condition.wait(.05)
                if cancellation_token is not None:
                    cancellation_token.check()
                self._waiting.pop(key)
                self._active[key] = entry
            yield
        finally:
            with self._condition:
                self._waiting.pop(key, None)
                self._active.pop(key, None)
                self._condition.notify_all()
            release_owner()

    @contextmanager
    def waiting(self, kind, *, session_id=None, turn_id=None, label=""):
        """Report a wait for a resource this budget does not itself grant."""
        key = uuid.uuid4().hex
        with self._condition:
            self._waiting[key] = {"resource": kind, "sessionId": session_id, "turnId": turn_id,
                                      "label": label, "since": time.time()}
        try:
            yield
        finally:
            with self._condition:
                self._waiting.pop(key, None)

    def submit(self, kind, function, *args, session_id=None, turn_id=None, label=""):
        release_owner = self.hold()
        key = uuid.uuid4().hex
        entry = {"resource": kind, "sessionId": session_id, "turnId": turn_id, "label": label, "since": time.time()}
        with self._condition:
            self._waiting[key] = entry

        def execute():
            with self._condition:
                self._waiting.pop(key, None)
                self._active[key] = entry
            try:
                return function(*args)
            finally:
                with self._condition:
                    self._active.pop(key, None)
                    self._condition.notify_all()

        def finished(_future):
            with self._condition:
                self._waiting.pop(key, None)
            release_owner()
        try:
            future = (self.agents if kind == "agents" else self.tools).submit(execute)
        except BaseException:
            with self._condition:
                self._waiting.pop(key, None)
            release_owner()
            raise
        future.add_done_callback(finished)
        return future

    @asynccontextmanager
    async def acquire_async(self, kind, *, session_id=None, label=""):
        release_owner = self.hold()
        key = uuid.uuid4().hex
        entry = {"resource": kind, "sessionId": session_id, "turnId": None, "label": label, "since": time.time()}
        try:
            with self._condition:
                self._waiting[key] = entry
            while True:
                with self._condition:
                    if sum(row["resource"] == kind for row in self._active.values()) < self.limits.get(kind, 2):
                        self._waiting.pop(key)
                        self._active[key] = entry
                        break
                await asyncio.sleep(.05)
            yield
        finally:
            with self._condition:
                self._waiting.pop(key, None)
                self._active.pop(key, None)
                self._condition.notify_all()
            release_owner()

    def snapshot(self):
        with self._condition:
            return {"limits": dict(self.limits), "active": [dict(row) for row in self._active.values()],
                        "waiting": [dict(row) for row in self._waiting.values()]}
