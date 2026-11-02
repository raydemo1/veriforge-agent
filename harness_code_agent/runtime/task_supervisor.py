"""Session-visible tracking for tool work that can outlive a cancelled turn."""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, wait


class ToolTaskSupervisor:
    """Track running tool futures behind a small cancellation/cleanup interface."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._futures: dict[Future, dict] = {}
        self._closed = False

    def track(self, future: Future, *, turn_id=None, tool_call_id=None, resources=(), cancellation_token=None, foreground=True) -> None:
        with self._lock:
            if self._closed:
                future.cancel()
                raise RuntimeError("tool task supervisor is closed")
            self._futures[future] = dict(turn_id=turn_id, tool_call_id=tool_call_id,
                                         resources=tuple(resources), cancellation_token=cancellation_token, foreground=foreground)
        future.add_done_callback(self._forget)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return sum(not future.done() for future in self._futures)

    @property
    def pending_foreground_count(self) -> int:
        with self._lock:
            return sum(not future.done() and task["foreground"]
                       for future, task in self._futures.items())

    def close(self, *, timeout: float = 1.0) -> bool:
        with self._lock:
            self._closed = True
            futures = tuple(self._futures)
        self.cancel()
        if futures:
            wait(futures, timeout=max(0.0, timeout))
        return all(future.done() for future in futures)

    def cancel(self, *, turn_id=None) -> None:
        with self._lock:
            tasks = list(self._futures.items())
        for future, task in tasks:
            if turn_id is not None and task["turn_id"] != turn_id:
                continue
            token = task["cancellation_token"]
            if token is not None:
                token.cancel()
            future.cancel()

    def wait_idle(self, *, timeout: float = 1.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self._lock:
                futures = tuple(future for future in self._futures if not future.done())
            if not futures:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            wait(futures, timeout=remaining)

    def _forget(self, future: Future) -> None:
        with self._lock:
            self._futures.pop(future, None)
