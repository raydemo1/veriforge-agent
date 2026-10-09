"""Turn-local verification gate and publication of check evidence."""

from __future__ import annotations

from pathlib import Path

from ..source_files import source_fingerprints
from ..verification import VerificationEngine
from .base import AgentMiddleware


class StaticVerifierMiddleware(AgentMiddleware):
    def __init__(self, workspace_root: str | None = None, workspace=None):
        self._root = Path(workspace_root or getattr(workspace, "root", ".")).resolve()
        self.engine = VerificationEngine()
        self._baseline = source_fingerprints(self._root)
        self._reported_signatures: set[tuple] = set()

    def begin_turn(self, task, messages, runtime_state=None, agent_name=None):
        self._baseline = source_fingerprints(self._root)
        self._reported_signatures.clear()

    def pre_exit(self, messages, runtime_state=None, agent_name=None):
        current = source_fingerprints(self._root)
        files = sorted(
            path
            for path in current.keys() | self._baseline.keys()
            if current.get(path) != self._baseline.get(path)
        )
        if not files:
            return None
        results = self.engine.verify(files, self._root)
        bus = getattr(runtime_state, "event_bus", None)
        if bus is not None and results:
            bus.emit(
                "checks_recorded",
                agent=agent_name,
                payload={"checks": [check.to_dict() for check in results]},
            )
        blocks = [
            f"  {detail}"
            for check in results
            if check.status == "failed"
            for detail in check.details
        ]
        if blocks:
            return (
                "[SYSTEM] LINT CHECK FAILED -- fix these errors before stopping:\n"
                + "\n".join(blocks[:20])
            )
        warns = [
            f"  {detail}"
            for check in results
            if check.status == "warning"
            for detail in check.details
        ]
        signature = tuple(warns[:20])
        if warns and signature not in self._reported_signatures:
            self._reported_signatures.add(signature)
            return (
                "[SYSTEM] Lint warnings (non-blocking):\n"
                + "\n".join(warns[:20])
                + "\nConsider fixing before stopping."
            )
        deferred = [
            f"  {check.name}: {detail}"
            for check in results
            if check.status == "skipped"
            for detail in check.details
            if detail.startswith("Requires project-code execution")
        ]
        signature = tuple(deferred[:20])
        if deferred and signature not in self._reported_signatures:
            self._reported_signatures.add(signature)
            return (
                "[SYSTEM] Verification skipped (non-blocking):\n"
                + "\n".join(deferred[:20])
                + "\nThese checks did not run. Use run_bash if needed, or report them as unverified."
            )
        return None
