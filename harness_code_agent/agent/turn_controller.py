"""Single owner for agent-turn continuation and stop decisions."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

log = logging.getLogger("harness")


@dataclass(frozen=True)
class TurnDecision:
    continue_loop: bool
    reason: str


class TurnController:
    """Apply exit gates and record one consistent stop reason."""

    def __init__(self, conversation) -> None:
        self.conversation = conversation

    def check_time_budget(self, *, run_started: float, iteration: int) -> TurnDecision | None:
        conversation = self.conversation
        agent = conversation.agent
        if agent.time_budget is None:
            return None
        elapsed = time.monotonic() - run_started
        if elapsed < agent.time_budget:
            return None
        log.warning(
            "[%s] Time budget exhausted (%.0fs >= %.0fs).",
            agent.name,
            elapsed,
            agent.time_budget,
        )
        conversation.runtime_state.fallback.request_stop(
            reason="time_budget_exhausted",
            limit_type="seconds",
            used=int(elapsed),
            limit=int(agent.time_budget),
            recent_action_summary=conversation.runtime_state.fallback.recent_action_summary,
        )
        conversation.emitter.emit_agent_fallback(conversation.runtime_state.fallback)
        conversation.last_text = conversation._fallback_text()
        conversation.trace.finish("time_budget", iteration)
        return TurnDecision(False, "time_budget")

    def after_no_tool_calls(self, *, iteration: int) -> TurnDecision:
        conversation = self.conversation
        agent = conversation.agent
        if conversation._drain_queued_messages():
            return TurnDecision(True, "queued_message")
        for middleware in agent.middlewares:
            injection = middleware.pre_exit(
                conversation.messages,
                runtime_state=conversation.runtime_state,
                agent_name=agent.name,
            )
            if not injection:
                continue
            conversation._append_message({"role": "user", "content": injection})
            conversation.trace.middleware_inject(
                type(middleware).__name__,
                "pre_exit",
                injection,
            )
            return TurnDecision(True, "pre_exit_gate")
        log.info("[%s] Finished (no more tool calls).", agent.name)
        conversation.trace.finish("no_tool_calls", iteration)
        return TurnDecision(False, "no_tool_calls")

    def after_tool_calls(
        self,
        *,
        finish_reason: str | None,
        iteration: int,
    ) -> TurnDecision:
        conversation = self.conversation
        agent = conversation.agent
        if conversation._drain_queued_messages():
            return TurnDecision(True, "queued_message")
        if finish_reason == "stop":
            log.info("[%s] Finished (stop).", agent.name)
            conversation.trace.finish("stop", iteration)
            return TurnDecision(False, "stop")
        if finish_reason == "length":
            log.warning("[%s] Output truncated (max_tokens hit).", agent.name)
            conversation.trace.error("length_truncated", "max_tokens hit")
            conversation._append_message({
                "role": "user",
                "content": (
                    "[SYSTEM] Your response was truncated (token limit), but your tool calls "
                    "WERE executed successfully. The results are above. "
                    "If you had more tool calls planned, continue with the remaining ones now. "
                    "Do NOT re-run the tools that already executed."
                ),
            })
            return TurnDecision(True, "length_truncated")
        return TurnDecision(True, "tools_executed")
