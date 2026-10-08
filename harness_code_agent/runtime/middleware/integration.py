"""Pre-exit gate that keeps unintegrated worker proposals honest.

Workers finish with isolated :class:`ChangeProposal` objects. Integrating
them is the main agent's job (read_agent_changes -> apply_agent_changes ->
resolve conflicts -> verify), never a user approval step. With the manual
Apply/Discard UI gone, the runtime must guarantee the agent cannot simply
stop while a finalized proposal is still pending. This gate injects one
short directive and continues the loop; the agent resolves it by applying,
resolving the conflict (asking the user only for a genuine product
decision), or deliberately discarding via close_agent.
"""
from __future__ import annotations

from .base import AgentMiddleware


class ProposalIntegrationMiddleware(AgentMiddleware):
    """Block turn exit while worker changes are not integrated."""

    def __init__(self, tool_context) -> None:
        self._tool_context = tool_context

    def pre_exit(self, messages: list[dict], runtime_state=None,
                 agent_name: str | None = None) -> str | None:
        coordinator = getattr(self._tool_context, "agent_coordinator", None)
        if coordinator is None:
            return None
        changes = getattr(coordinator, "changes", None)
        if changes is None:
            return None

        ready: list[dict] = []
        conflicts: list[dict] = []
        for snapshot in changes.snapshot_proposals():
            status = snapshot.get("status")
            if status == "ready":
                ready.append(snapshot)
            elif status == "conflict" and snapshot.get("conflict"):
                conflicts.append(snapshot)
        if not ready and not conflicts:
            return None

        lines = [
            "[SYSTEM] UNINTEGRATED WORKER CHANGES -- do not finish yet.",
        ]
        for snapshot in ready:
            paths = [str(item.get("path")) for item in snapshot.get("files", [])]
            lines.append(
                f"- proposal {snapshot.get('id')} (agent {snapshot.get('agentId')}) "
                f"is finalized but not integrated: {', '.join(paths) or 'no files'}"
            )
        if ready:
            lines.append(
                "Review each proposal with read_agent_changes, then integrate it "
                "with apply_agent_changes. Discard deliberately only with "
                "close_agent(agent_id, discard_changes=true) after reviewing."
            )
        for snapshot in conflicts:
            conflict = snapshot.get("conflict") or {}
            lines.append(
                f"- proposal {snapshot.get('id')} has an open merge conflict "
                f"{conflict.get('id')} on: {', '.join(conflict.get('paths') or [])}"
            )
        if conflicts:
            lines.append(
                "Inspect each conflict with read_agent_conflicts (base/current/worker), "
                "then resolve every conflicted path with resolve_agent_conflicts. "
                "If the conflict is genuinely a user or product decision the "
                "requirements do not settle, ask via the question interaction first."
            )
        return "\n".join(lines)
