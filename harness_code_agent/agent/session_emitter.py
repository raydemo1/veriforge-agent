"""Session event emission for one agent conversation.

The emitter owns nothing but the bus handle: every method takes the data it
needs as arguments, so conversation state stays out of this module.
"""
from __future__ import annotations

from typing import Any


class SessionEmitter:
    """Emits structured session events on the conversation's event bus."""

    def __init__(self, event_bus: Any, agent_name: str, provider_name: str):
        self.event_bus = event_bus
        self.agent_name = agent_name
        self.provider_name = provider_name

    def emit_compaction_started(self, *, token_count: int, threshold: int, phase: str) -> None:
        if self.event_bus is None:
            return
        from ..sessions.events import ContextCompactionStartedEvent
        self.event_bus.emit_event(
            ContextCompactionStartedEvent(
                token_count=token_count,
                threshold=threshold,
                forced=False,
                phase=phase,
                agent=self.agent_name,
            ).to_event()
        )

    def emit_compaction_committed(
        self,
        *,
        messages_before: int,
        messages_after: int,
        tokens_saved: int,
        summary_chars: int,
    ) -> None:
        if self.event_bus is None:
            return
        from ..sessions.events import ContextCompactionCommittedEvent
        self.event_bus.emit_event(
            ContextCompactionCommittedEvent(
                summary_chars=summary_chars,
                messages_before=messages_before,
                messages_after=messages_after,
                tokens_saved=tokens_saved,
                agent=self.agent_name,
            ).to_event()
        )

    def emit_context_anxiety_observed(self, *, token_count: int, threshold: int, signal) -> None:
        if self.event_bus is None:
            return
        from ..sessions.events import ContextAnxietyObservedEvent
        self.event_bus.emit_event(
            ContextAnxietyObservedEvent(
                token_count=token_count,
                threshold=threshold,
                score=getattr(signal, "score", 0),
                reasons=list(getattr(signal, "reasons", [])),
                source=getattr(signal, "source", "assistant_recent_messages"),
                agent=self.agent_name,
            ).to_event()
        )

    def emit_llm_request_started(self, call_id: str, *, streamed: bool, model: str) -> None:
        if self.event_bus is None:
            return
        from ..sessions.events import LlmRequestStartedEvent
        self.event_bus.emit_event(
            LlmRequestStartedEvent(
                call_id=call_id,
                provider=self.provider_name,
                model=model,
                streamed=streamed,
                agent=self.agent_name,
            ).to_event()
        )

    def emit_llm_first_token(self, call_id: str, elapsed_ms: int, *, model: str) -> None:
        if self.event_bus is None:
            return
        from ..sessions.events import LlmFirstTokenEvent
        self.event_bus.emit_event(
            LlmFirstTokenEvent(
                call_id=call_id,
                elapsed_ms=elapsed_ms,
                provider=self.provider_name,
                model=model,
                agent=self.agent_name,
            ).to_event()
        )

    def emit_llm_response_finished(
        self,
        call_id: str,
        duration_ms: int,
        *,
        finish_reason: str | None,
        streamed: bool,
        first_token_ms: int | None,
        model: str,
    ) -> None:
        if self.event_bus is None:
            return
        from ..sessions.events import LlmResponseFinishedEvent
        self.event_bus.emit_event(
            LlmResponseFinishedEvent(
                call_id=call_id,
                duration_ms=max(0, int(duration_ms)),
                provider=self.provider_name,
                model=model,
                streamed=streamed,
                finish_reason=finish_reason,
                first_token_ms=first_token_ms,
                agent=self.agent_name,
            ).to_event()
        )

    def emit_agent_fallback(self, fallback) -> None:
        if fallback.fallback_event_emitted or not fallback.stop_requested:
            return
        fallback.fallback_event_emitted = True
        if self.event_bus is None:
            return
        from ..sessions.events import AgentFallbackEvent
        self.event_bus.emit_event(
            AgentFallbackEvent(
                reason=fallback.stop_reason,
                limit_type=fallback.stop_limit_type or None,
                used=fallback.stop_used,
                limit=fallback.stop_limit,
                last_tool=fallback.stop_last_tool or None,
                fingerprint_hash=fallback.stop_fingerprint_hash or None,
                recent_action_summary=fallback.recent_action_summary,
                agent=self.agent_name,
            ).to_event()
        )
