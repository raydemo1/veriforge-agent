"""Resource-aware planning primitives for one assistant tool-call batch."""
from __future__ import annotations

import os
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from ..workspace.paths import is_same_or_child


@dataclass(frozen=True, order=True)
class ResourceClaim:
    domain: str
    key: str = "*"
    scope: str = "exact"  # exact | subtree | global
    access: str = "read"  # read | write


@dataclass(frozen=True)
class CallEffect:
    resources: tuple[ResourceClaim, ...] = ()
    barrier: bool = False
    concurrency_key: str | None = None
    kind: str = "declared"

    @classmethod
    def global_exclusive(cls, *, kind: str = "unknown") -> CallEffect:
        return cls((ResourceClaim("global", "*", "global", "write"),), barrier=True, kind=kind)


@dataclass(frozen=True)
class ToolExecutionPlan:
    """Immutable snapshot shared by scheduling, authorization and execution.

    The shell analysis is computed once during call preparation. The planner
    reads ``effect``; the permission middleware reads ``permission_decision``;
    ``run_bash`` reads ``background`` and the authorization snapshot instead of
    re-analyzing the command.
    """

    effect: CallEffect
    analysis: object | None = None
    permission_decision: object | None = None
    background: bool | None = None


@dataclass
class PlannedCall:
    index: int
    effect: CallEffect
    dependencies: set[int] = field(default_factory=set)


class ExecutionPlanner:
    """Build an ordered conflict graph, then expose its currently-ready calls."""

    def __init__(self, calls: Iterable[tuple[int, CallEffect]]):
        self.calls = [PlannedCall(index, effect) for index, effect in calls]
        for pos, current in enumerate(self.calls):
            for previous in self.calls[:pos]:
                if current.effect.barrier or previous.effect.barrier or effects_conflict(previous.effect, current.effect):
                    current.dependencies.add(previous.index)
            if current.effect.barrier:
                for later in self.calls[pos + 1:]:
                    later.dependencies.add(current.index)

    def ready(self, pending: set[int], completed: set[int]) -> list[int]:
        return [
            call.index
            for call in self.calls
            if call.index in pending and call.dependencies <= completed
        ]


def effects_conflict(left: CallEffect, right: CallEffect) -> bool:
    return any(claims_conflict(a, b) for a in left.resources for b in right.resources)


def claims_conflict(left: ResourceClaim, right: ResourceClaim) -> bool:
    if "global" in {left.domain, right.domain}:
        return left.access == "write" or right.access == "write"
    if left.domain != right.domain or (left.access == right.access == "read"):
        return False
    if "global" in {left.scope, right.scope}:
        return True
    if left.scope == right.scope == "exact":
        return left.key == right.key
    if left.scope == "subtree" and right.scope == "subtree":
        return is_same_or_child(left.key, right.key) or is_same_or_child(right.key, left.key)
    subtree, exact = (left, right) if left.scope == "subtree" else (right, left)
    return is_same_or_child(exact.key, subtree.key)


def normalize_workspace_key(root: str | Path, value: str | Path = ".") -> str:
    root_path = Path(root).resolve()
    raw = Path(value)
    resolved = raw.resolve() if raw.is_absolute() else (root_path / raw).resolve()
    key = os.path.normpath(str(resolved))
    return os.path.normcase(key)


def workspace_claim(root: str | Path, value: str | Path, *, scope: str, access: str) -> ResourceClaim:
    return ResourceClaim("workspace", normalize_workspace_key(root, value), scope, access)


def analyzed_shell_write_claims(
    root: str | Path, command: str, *, analysis=None
) -> tuple[ResourceClaim, ...]:
    """Write claims for exactly the paths a shell command is proven to mutate.

    The sandbox grant may be wider (the workspace itself); the concurrent-write
    lease follows the actual operation, not the capability.
    """
    from .shell_classification import TargetScope, analyze_shell_command

    analysis = analysis or analyze_shell_command(command, root)
    claims: list[ResourceClaim] = []
    for target in analysis.write_targets:
        if target.scope is not TargetScope.WORKSPACE or not target.raw:
            continue
        raw = target.raw.rstrip("/\\")
        claims.append(
            workspace_claim(
                root,
                raw or ".",
                scope="subtree" if _looks_like_directory_target(raw) else "exact",
                access="write",
            )
        )
    return tuple(claims)


def _looks_like_directory_target(raw: str) -> bool:
    tail = raw.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return "." not in tail


class ResourceCoordinator:
    """Session-scoped atomic multi-resource reader/writer coordinator."""

    def __init__(self):
        self._condition = threading.Condition()
        self._active: dict[int, tuple[ResourceClaim, ...]] = {}
        self._next_token = 0

    @contextmanager
    def acquire(self, claims: Iterable[ResourceClaim], cancellation_token=None):
        ordered = tuple(sorted(set(claims)))
        with self._condition:
            while any(
                claims_conflict(wanted, active)
                for active_claims in self._active.values()
                for wanted in ordered
                for active in active_claims
            ):
                if cancellation_token is not None:
                    cancellation_token.check()
                self._condition.wait(timeout=0.05)
            if cancellation_token is not None:
                cancellation_token.check()
            self._next_token += 1
            token = self._next_token
            self._active[token] = ordered
        try:
            yield
        finally:
            with self._condition:
                self._active.pop(token, None)
                self._condition.notify_all()
