"""Cancellation tokens for interrupting agent turns.

Tokens form a tree: a child created via ``parent.create_child()`` (or
``CancellationToken(parent=parent)``) is cancelled together with its parent.
If the parent is already cancelled when the child is created, the child is
born cancelled.  Children must call :meth:`close` when they end so long-lived
parents do not accumulate callbacks to finished turns/calls.
"""
from __future__ import annotations

import threading
from collections.abc import Callable


class CancellationToken:
    """Thread-safe cancellation token passed through the agent loop."""

    def __init__(self, parent: "CancellationToken | None" = None) -> None:
        self._event = threading.Event()
        self._callbacks: list[Callable[[], None]] = []
        self._lock = threading.Lock()
        self._detach_parent: Callable[[], None] | None = None
        if parent is not None:
            # add_callback fires self.cancel immediately when the parent
            # is already cancelled, so the child is born cancelled.
            self._detach_parent = parent.add_callback(self.cancel)

    def create_child(self) -> "CancellationToken":
        return CancellationToken(parent=self)

    def cancel(self) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            callbacks = list(self._callbacks)
            self._callbacks.clear()
        for callback in callbacks:
            callback()

    def add_callback(self, callback: Callable[[], None]) -> Callable[[], None]:
        with self._lock:
            if self._event.is_set():
                run_now = True
            else:
                self._callbacks.append(callback)
                run_now = False
        if run_now:
            callback()

        def remove() -> None:
            with self._lock:
                try:
                    self._callbacks.remove(callback)
                except ValueError:
                    pass

        return remove

    def close(self) -> None:
        """Detach this token from its parent. Safe to call repeatedly."""
        detach = self._detach_parent
        if detach is None:
            return
        self._detach_parent = None
        detach()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        """Raise if cancelled. Call at loop boundaries."""
        if self._event.is_set():
            raise CancelledError("Turn cancelled by user")

    def wait(self, timeout: float | None) -> bool:
        """Wait up to ``timeout`` seconds for cancellation.

        Returns True if the token was cancelled (or already was), False if
        the wait timed out.  Lets retry/backoff loops respond to cancellation
        immediately instead of sleeping through the whole delay.
        """
        return self._event.wait(timeout=timeout)


class CancelledError(Exception):
    pass
