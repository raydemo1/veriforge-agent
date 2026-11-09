"""Record static check evidence for source changes made during the turn."""

from __future__ import annotations

from pathlib import Path

from ..source_files import source_fingerprints
from ..verification import VerificationEngine
from .base import AgentMiddleware


class StaticVerifierMiddleware(AgentMiddleware):
    def __init__(self, workspace_root: str | None = None, workspace=None):
        self._root = Path(workspace_root or getattr(workspace, "root", ".")).resolve()
        self._access = getattr(workspace, "access", None)
        self.engine = VerificationEngine()
        self._baseline = source_fingerprints(self._root)

    def begin_turn(self, task, messages, runtime_state=None, agent_name=None):
        self._baseline = source_fingerprints(self._root)

    def after_turn(self, runtime_state=None, agent_name=None) -> None:
        revision = self._access.cursor() if self._access is not None else None
        current = source_fingerprints(self._root)
        files = sorted(
            path
            for path in current.keys() | self._baseline.keys()
            if current.get(path) != self._baseline.get(path)
        )
        if not files:
            return
        results = self.engine.verify(files, self._root)
        stale = bool(results) and (
            current != source_fingerprints(self._root)
            or (self._access is not None and revision != self._access.cursor())
        )
        if runtime_state is not None:
            runtime_state.verification = {
                "checks": [check.to_dict() for check in results],
                "fingerprints": current,
                "files": files,
                "turn_id": getattr(runtime_state, "turn_id", None),
                "sequence": getattr(runtime_state, "observation_sequence", None),
            }
            runtime_state.verification_stale = stale
        bus = getattr(runtime_state, "event_bus", None)
        if bus is not None and results:
            bus.emit(
                "checks_recorded",
                agent=agent_name,
                payload={"checks": [check.to_dict() for check in results], "stale": stale},
            )
