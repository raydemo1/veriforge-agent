from __future__ import annotations

import inspect
import logging
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .. import config
from ..runtime.execution_planner import (
    CallEffect,
    ExecutionPlanner,
    acquire_concurrency,
)
from ..runtime.tool_call_validation import ToolCall, validate_tool_call
from ..runtime.tool_failures import FailureMode, ToolFailure, failure_batch_key
from ..runtime.tool_registry import tool_schemas_for_profile
from ..runtime.tool_result import ToolResult
from ..runtime.tool_runner import (
    _registry_for_context,
    emit_tool_call_started,
    execute_tool_result,
    finalize_executed_tool_result,
    finalize_intercepted_tool_result,
)
from .cancellation import CancellationToken, CancelledError

log = logging.getLogger("harness")


MAX_WORKERS = 8


@dataclass
class PreparedToolCall:
    index: int
    tool_call_id: str
    name: str
    args: dict
    effect: CallEffect
    raw: Any
    blocked_result: ToolResult | None = None
    emit_events: bool = True
    permission_decision: Any = None


@lru_cache(maxsize=None)
def _middleware_accepts_decision(middleware_type) -> bool:
    try:
        params = inspect.signature(middleware_type.before_tool).parameters
    except (TypeError, ValueError):
        return True
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return "permission_decision" in params


@dataclass
class ExecutedToolCall:
    prepared: PreparedToolCall
    result: ToolResult
    error: Exception | None = None
    stop_after_tool_loop: bool = False
    intercepted: bool = False


@dataclass
class ExecutionGroup:
    calls: list[PreparedToolCall]
    parallel: bool = True


class ToolExecutor:

    def __init__(self, conversation, cancellation_token=None):
        self.conversation = conversation
        self.agent = conversation.agent
        self.runtime_state = conversation.runtime_state
        self.cancellation_token = cancellation_token
        self._tool_calls: list = []
        self._executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self._deferred_user_messages: list[str] = []
        self._middleware_activity: dict[str, dict[str, Any]] = {}

    def execute(self, tool_calls: list) -> bool:
        self._tool_calls = list(tool_calls or [])
        self.conversation.compaction_gate.begin_tool_call()
        try:
            prepared, stop = self._prepare_calls(tool_calls)
            if stop:
                return True
            planner = ExecutionPlanner((call.index, call.effect) for call in prepared)
            by_index = {call.index: call for call in prepared}
            pending = set(by_index)
            completed: set[int] = set()
            buffered: dict[int, ExecutedToolCall] = {}
            next_record = 0
            while pending:
                self.conversation._check_cancelled(self.cancellation_token)
                ready_indexes = planner.ready(pending, completed)
                if not ready_indexes:
                    raise RuntimeError("Tool execution planner produced a dependency cycle")
                safe_indexes = [index for index in ready_indexes if not self._requires_approval(by_index[index])]
                selected = safe_indexes or [ready_indexes[0]]
                executed = self._execute_group(ExecutionGroup([by_index[index] for index in selected]))
                stop_after_group = False
                for item in executed:
                    buffered[item.prepared.index] = item
                    pending.discard(item.prepared.index)
                    completed.add(item.prepared.index)
                    if item.stop_after_tool_loop or self.runtime_state.fallback.stop_requested:
                        stop_after_group = True
                while next_record in buffered:
                    self._record_executed_result(buffered.pop(next_record))
                    next_record += 1
                if stop_after_group or self.runtime_state.fallback.stop_requested:
                    reason = self.runtime_state.fallback.stop_reason
                    for index in sorted(pending):
                        call = by_index[index]
                        output = f"[blocked] Agent fallback triggered ({reason}); tool was not executed."
                        buffered[index] = ExecutedToolCall(
                            call,
                            ToolResult(
                                tool=call.name,
                                status="failed",
                                output=output,
                                error=output.removeprefix("[blocked] "),
                                metadata={"status_source": "budget", "fallback_reason": reason},
                            ),
                            stop_after_tool_loop=True,
                            intercepted=True,
                        )
                    pending.clear()
                    while next_record in buffered:
                        self._record_executed_result(buffered.pop(next_record))
                        next_record += 1
                    self.conversation.emitter.emit_agent_fallback(self.runtime_state.fallback)
                    self.conversation.last_text = self.conversation._fallback_text()
                    self._flush_deferred_user_messages()
                    return True
            self._flush_deferred_user_messages()
            return False
        except CancelledError:
            # The assistant message with tool_calls is already in history;
            # answer every call the API still expects so the next request
            # is not rejected with a 400.
            self._repair_orphaned_tool_calls()
            raise
        finally:
            self.conversation.compaction_gate.end_tool_call()
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _repair_orphaned_tool_calls(self) -> None:
        answered = {
            message.get("tool_call_id")
            for message in self.conversation.messages
            if message.get("role") == "tool"
        }
        orphans: list[ToolCall] = []
        for index, tc in enumerate(self._tool_calls):
            call = ToolCall.from_raw(tc, index)
            if call.tool_call_id not in answered:
                orphans.append(call)
        if not orphans:
            return
        log.info("[%s] Answering %d cancelled tool calls", self.agent.name, len(orphans))
        self.conversation.trace.error("cancelled_tool_calls", f"{len(orphans)} unanswered")
        for call in orphans:
            self.conversation._append_message({
                "role": "tool",
                "tool_call_id": call.tool_call_id,
                "content": "[cancelled] Execution cancelled before this tool produced a result.",
            })

    def _prepare_calls(self, tool_calls: list) -> tuple[list[PreparedToolCall], bool]:
        prepared: list[PreparedToolCall] = []
        for index, tc in enumerate(tool_calls):
            call = ToolCall.from_raw(tc, index)
            fn_name = call.name or "<unknown>"
            registry = _registry_for_context(self.agent.tool_context)
            validation = validate_tool_call(call, registry, self.agent.tool_context)
            if validation.error is not None:
                log.warning(
                    "[%s] Rejected tool call %s (%s): %s",
                    self.agent.name,
                    fn_name,
                    validation.error.kind,
                    validation.error.message,
                )
                self.conversation.trace.error(
                    validation.error.kind,
                    f"{fn_name}: {validation.error.message}",
                )
                prepared.append(
                    PreparedToolCall(
                        index=index,
                        tool_call_id=call.tool_call_id,
                        name=fn_name,
                        args=validation.arguments,
                        effect=CallEffect.global_exclusive(kind="blocked"),
                        raw=tc,
                        blocked_result=validation.error.to_result(fn_name),
                    )
                )
                continue

            fn_args = validation.arguments
            effect, blocked = self._classify_call(fn_name, fn_args)
            if self.agent.allowed_tool_names is not None and fn_name not in self.agent.allowed_tool_names:
                output = f"[blocked] Tool '{fn_name}' is not available to this agent profile."
                blocked = ToolResult(
                    tool=fn_name,
                    status="failed",
                    output=output,
                    error=output.removeprefix("[blocked] "),
                    metadata={"status_source": "permission"},
                )
                effect = CallEffect.global_exclusive(kind="blocked")
                self.conversation.trace.middleware_inject("ToolSchemaGuard", "before_tool", output)
            prepared.append(
                PreparedToolCall(
                    index=index,
                    tool_call_id=tc["id"],
                    name=fn_name,
                    args=fn_args,
                    effect=effect,
                    raw=tc,
                    blocked_result=blocked,
                    permission_decision=self._permission_decision(fn_name, fn_args, registry),
                )
            )
        return prepared, False

    def _classify_call(self, name: str, args: dict) -> tuple[CallEffect, ToolResult | None]:
        registry = _registry_for_context(self.agent.tool_context)
        return registry.effect_for(name, args, self.agent.tool_context), None

    def _permission_decision(self, name: str, args: dict, registry):
        context = self.agent.tool_context
        if context is None:
            return None
        workspace = getattr(context, "workspace", None)
        root = str(workspace.root) if workspace is not None and getattr(workspace, "root", None) is not None else None
        return context.permission_policy.decide_tool_call(
            name,
            args,
            tool_permission=registry.permission_for(name),
            workspace_root=root,
        )

    def _requires_approval(self, prepared: PreparedToolCall) -> bool:
        if prepared.blocked_result is not None:
            return False
        if prepared.permission_decision is None:
            return False
        return prepared.permission_decision.action == "ask"

    def _execute_group(self, group: ExecutionGroup) -> list[ExecutedToolCall]:
        ready: list[PreparedToolCall] = []
        executed: list[ExecutedToolCall] = []
        for prepared in group.calls:
            self.conversation._check_cancelled(self.cancellation_token)
            if prepared.emit_events and not self.conversation._record_tool_call_budget(prepared.name, prepared.args):
                output = (
                    f"[blocked] Agent fallback triggered ({self.runtime_state.fallback.stop_reason}); "
                    "tool was not executed."
                )
                executed.append(
                    ExecutedToolCall(
                        prepared,
                        ToolResult(
                            tool=prepared.name,
                            status="failed",
                            output=output,
                            error=output.removeprefix("[blocked] "),
                            metadata={
                                "status_source": "budget",
                                "fallback_reason": self.runtime_state.fallback.stop_reason,
                            },
                        ),
                        stop_after_tool_loop=True,
                        intercepted=True,
                    )
                )
                break
            if prepared.blocked_result is not None:
                executed.append(ExecutedToolCall(prepared, prepared.blocked_result, intercepted=True))
                continue
            blocked = self._run_before_tool(prepared)
            if blocked is not None:
                executed.append(ExecutedToolCall(prepared, blocked, intercepted=True))
                if self.runtime_state.fallback.stop_requested:
                    break
                continue
            allowed_started = time.perf_counter()
            for mw in self.agent.middlewares:
                mw.on_tool_allowed(
                    prepared.name,
                    prepared.args,
                    self.conversation.messages,
                    runtime_state=self.runtime_state,
                    agent_name=self.agent.name,
                )
            activity = self._middleware_activity.setdefault(
                prepared.tool_call_id,
                _new_middleware_activity(),
            )
            activity["hooks"] += len(self.agent.middlewares)
            activity["duration_ms"] += (
                time.perf_counter() - allowed_started
            ) * 1000
            ready.append(prepared)

        if not ready:
            return executed
        futures: dict[Future, PreparedToolCall] = {}
        child_tokens: dict[Future, CancellationToken] = {}
        pending: set[Future] = set()
        try:
            for prepared in ready:
                if prepared.emit_events:
                    emit_tool_call_started(
                        name=prepared.name,
                        arguments=prepared.args,
                        tool_context=self.agent.tool_context,
                        agent_name=self.agent.name,
                    )
                child_token = self._make_child_token()
                future = self._executor.submit(self._execute_one, prepared, child_token)
                if child_token is not None:
                    child_tokens[future] = child_token
                    # Single lifecycle owner for the parent callback: the
                    # detach runs whether the call succeeds, raises, or its
                    # future is cancelled — so abnormal exits cannot leak
                    # callbacks onto the long-lived turn token.
                    future.add_done_callback(
                        lambda _f, token=child_token: token.close()
                    )
                if self.agent.tool_context is not None:
                    self.agent.tool_context.tool_tasks.track(future)
                futures[future] = prepared
                pending.add(future)
            # No generic per-call deadline: a tool runs until it finishes or
            # the turn token is cancelled.  Timeout is a capability of the
            # concrete backend (run_bash kills its process tree, HTTP/MCP
            # clients have their own limits), not a universal tool semantic.
            while pending:
                self.conversation._check_cancelled(self.cancellation_token)
                done, pending = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                for future in done:
                    prepared = futures[future]
                    # The future's done callback detaches the child token.
                    child_tokens.pop(future, None)
                    try:
                        executed.append(future.result())
                    except Exception as exc:
                        executed.append(
                            ExecutedToolCall(
                                prepared,
                                ToolResult(
                                    tool=prepared.name,
                                    status="failed",
                                    output=f"[error] {type(exc).__name__}: {exc}",
                                    error=f"{type(exc).__name__}: {exc}",
                                    metadata={"status_source": "exception"},
                                ),
                                error=exc,
                            )
                        )
            self.conversation._check_cancelled(self.cancellation_token)
        except Exception:
            # Cooperative cancellation: Python cannot kill a running thread,
            # so started calls unwind themselves when they observe their
            # (already-cancelled-via-parent, or explicitly cancelled below)
            # child token.  Detach still happens via each future's done
            # callback once the call settles.
            for future in pending:
                token = child_tokens.get(future)
                if token is not None:
                    token.cancel()
                future.cancel()
            if pending:
                wait(pending, timeout=0.25)
            raise
        return executed

    def _make_child_token(self) -> CancellationToken | None:
        """Per-call child token: parent turn cancellation propagates down."""
        parent = self.cancellation_token
        if parent is None:
            return None
        return parent.create_child()

    def _run_before_tool(self, prepared: PreparedToolCall) -> ToolResult | None:
        activity = self._middleware_activity.setdefault(
            prepared.tool_call_id,
            _new_middleware_activity(),
        )
        started = time.perf_counter()
        for mw in self.agent.middlewares:
            activity["hooks"] += 1
            if _middleware_accepts_decision(type(mw)):
                blocked = mw.before_tool(
                    prepared.name,
                    prepared.args,
                    self.conversation.messages,
                    runtime_state=self.runtime_state,
                    agent_name=self.agent.name,
                    permission_decision=prepared.permission_decision,
                )
            else:
                # Backward compatibility for custom middlewares written
                # against the older before_tool signature.
                blocked = mw.before_tool(
                    prepared.name,
                    prepared.args,
                    self.conversation.messages,
                    self.runtime_state,
                    self.agent.name,
                )
            if not blocked:
                continue
            activity["outcome"] = "blocked"
            activity["sources"].append(type(mw).__name__)
            activity["duration_ms"] += (time.perf_counter() - started) * 1000
            blocked_text = blocked.to_text() if isinstance(blocked, ToolResult) else str(blocked)
            self.conversation.trace.middleware_inject(type(mw).__name__, "before_tool", blocked_text)
            if isinstance(blocked, ToolResult):
                return blocked
            return ToolResult(
                tool=prepared.name,
                status="failed",
                output=blocked_text,
                error=blocked_text,
                metadata={"status_source": "approval" if blocked_text.startswith("[approval_denied]") else "permission"},
            )
        activity["duration_ms"] += (time.perf_counter() - started) * 1000
        return None

    def _execute_one(self, prepared: PreparedToolCall, cancellation_token=None) -> ExecutedToolCall:
        context = self.agent.tool_context
        resource_guard = context.resource_coordinator.acquire(prepared.effect.resources) if context is not None else nullcontext()
        with acquire_concurrency(prepared.effect.concurrency_key), resource_guard:
            return self._execute_one_unlimited(prepared, cancellation_token)

    def _execute_one_unlimited(
        self, prepared: PreparedToolCall, cancellation_token=None
    ) -> ExecutedToolCall:
        tool_result = execute_tool_result(
            prepared.name,
            prepared.args,
            runtime_state=self.runtime_state,
            agent_name=self.agent.name,
            tool_context=self.agent.tool_context,
            emit_events=False,
            cancellation_token=cancellation_token,
        )
        return ExecutedToolCall(prepared, tool_result)

    def _record_executed_result(self, item: ExecutedToolCall) -> None:
        prepared = item.prepared
        tool_result = item.result

        # Normalize every outcome into the canonical failure model *before*
        # finalization so events, messages and middleware all see it.
        failure: ToolFailure | None = None
        if tool_result.status == "failed":
            failure = ToolFailure.from_result(
                tool_call_id=prepared.tool_call_id,
                tool_name=prepared.name,
                result=tool_result,
                intercepted=item.intercepted,
                tool_args=prepared.args,
            )
            failure = self.runtime_state.failures.observe(
                failure,
                batch_key=failure_batch_key(self.conversation.messages),
            )
            tool_result = failure.stamp_metadata(tool_result)
            item.result = tool_result
        elif tool_result.status == "success":
            self.runtime_state.failures.observe_success(prepared.name)

        if item.intercepted:
            if prepared.emit_events:
                tool_result = finalize_intercepted_tool_result(
                    tool_result,
                    arguments=prepared.args,
                    tool_context=self.agent.tool_context,
                    agent_name=self.agent.name,
                )
            result = tool_result.to_text()
            self.conversation.trace.tool_call(prepared.name, prepared.args, result)
            self.conversation._append_message({
                "role": "tool",
                "tool_call_id": prepared.tool_call_id,
                "content": result,
            })
            if failure is not None:
                self._dispatch_tool_failure(prepared, failure)
            self._emit_middleware_activity(prepared, fallback_outcome="blocked")
            return

        tool_result = finalize_executed_tool_result(
            tool_result,
            arguments=prepared.args,
            tool_context=self.agent.tool_context,
            agent_name=self.agent.name,
            emit_call=False,
        )
        self._reveal_tool_schemas_from_result(tool_result)
        result = tool_result.to_text()
        log.debug("[%s] tool result: %s", self.agent.name, result[:200])
        self.conversation.trace.tool_call(prepared.name, prepared.args, result)
        observation = self.conversation.observation_store.create(
            tool=prepared.name,
            args=prepared.args,
            result=tool_result,
            fact_tracker=self.conversation.fact_tracker,
        )
        self.conversation._append_message({
            "role": "tool",
            "tool_call_id": prepared.tool_call_id,
            "content": self.conversation.observation_store.observed_message(observation, tool_result),
        })
        invalidation = self.conversation.fact_tracker.apply_mutation(
            tool=prepared.name,
            args=prepared.args,
            result=tool_result,
            observations=self.conversation.observation_store.observations,
            exclude_ids={observation.id},
        )
        if invalidation:
            self._deferred_user_messages.append(invalidation)

        post_started = time.perf_counter()
        activity = self._middleware_activity.setdefault(
            prepared.tool_call_id,
            _new_middleware_activity(),
        )
        for mw in self.agent.middlewares:
            activity["hooks"] += 1
            inject = mw.post_tool(
                prepared.name,
                prepared.args,
                tool_result,
                self.conversation.messages,
                runtime_state=self.runtime_state,
                agent_name=self.agent.name,
            )
            if inject:
                activity["outcome"] = "guided"
                activity["sources"].append(type(mw).__name__)
                self._deferred_user_messages.append(inject)
                self.conversation.trace.middleware_inject(type(mw).__name__, "post_tool", inject)
        activity["duration_ms"] += (time.perf_counter() - post_started) * 1000
        if failure is not None:
            self._dispatch_tool_failure(prepared, failure)
        self._emit_middleware_activity(prepared)

    def _dispatch_tool_failure(self, prepared: PreparedToolCall, failure: ToolFailure) -> None:
        """Run the on_tool_failure chain and execute the first FailureAction.

        Middleware only decides; the executor performs every side effect
        (guidance injection / fallback stop), so policy stays replay-free.
        Every decision is recorded in the activity stats, trace JSONL and as
        a ``tool_failure_decision`` session event.
        """
        activity = self._middleware_activity.setdefault(
            prepared.tool_call_id,
            _new_middleware_activity(),
        )
        for mw in self.agent.middlewares:
            activity["hooks"] += 1
            action = mw.on_tool_failure(
                failure,
                self.conversation.messages,
                runtime_state=self.runtime_state,
                agent_name=self.agent.name,
            )
            if action is None:
                continue

            source_name = type(mw).__name__
            activity["sources"].append(source_name)
            if action.mode == FailureMode.STOP:
                activity["outcome"] = "stopped"
            elif activity["outcome"] == "passed":
                # A silent regeneration request still changes what happens
                # next; keep it distinguishable from a plain pass-through.
                activity["outcome"] = "guided" if action.message else "regenerated"

            if action.message:
                self._deferred_user_messages.append(action.message)
                self.conversation.trace.middleware_inject(
                    source_name, "on_tool_failure", action.message
                )
            self.conversation.trace.tool_failure_policy(
                source=source_name,
                tool=failure.tool_name,
                tool_call_id=failure.tool_call_id,
                kind=failure.kind,
                category=failure.category,
                visibility=failure.visibility,
                attempt=failure.attempt,
                mode=action.mode,
                intercepted=failure.intercepted,
                turn_failure_count=self.runtime_state.failures.turn_failure_count,
                stop_reason=action.stop_reason,
            )
            event_bus = getattr(self.agent.tool_context, "event_bus", None)
            if event_bus is not None:
                event_bus.emit(
                    "tool_failure_decision",
                    agent=self.agent.name,
                    payload={
                        "tool": failure.tool_name,
                        "tool_call_id": failure.tool_call_id,
                        "kind": failure.kind,
                        "category": failure.category,
                        "phase": failure.phase,
                        "visibility": failure.visibility,
                        "attempt": failure.attempt,
                        "mode": action.mode,
                        "stop_reason": action.stop_reason,
                        "intercepted": failure.intercepted,
                        "turn_failure_count": self.runtime_state.failures.turn_failure_count,
                        "source": source_name,
                    },
                )
            if action.mode == FailureMode.STOP:
                self.runtime_state.fallback.request_stop(
                    reason=action.stop_reason or "tool_failure_policy",
                    limit_type=action.stop_limit_type or "failure_policy",
                    used=failure.attempt,
                    limit=action.stop_limit,
                    last_tool=failure.tool_name,
                )
            return

    def _emit_middleware_activity(
        self,
        prepared: PreparedToolCall,
        *,
        fallback_outcome: str = "passed",
    ) -> None:
        activity = self._middleware_activity.pop(
            prepared.tool_call_id,
            _new_middleware_activity(),
        )
        outcome = str(activity.get("outcome") or fallback_outcome)
        if outcome == "passed" and fallback_outcome != "passed":
            outcome = fallback_outcome
        event_bus = getattr(self.agent.tool_context, "event_bus", None)
        if event_bus is None:
            return
        sources = list(dict.fromkeys(str(item) for item in activity["sources"]))
        event_bus.emit(
            "middleware_activity",
            agent=self.agent.name,
            payload={
                "tool": prepared.name,
                "tool_call_id": prepared.tool_call_id,
                "hooks": int(activity["hooks"]),
                "duration_ms": round(float(activity["duration_ms"]), 1),
                "outcome": outcome,
                "sources": sources,
            },
        )

    def _flush_deferred_user_messages(self) -> None:
        while self._deferred_user_messages:
            content = self._deferred_user_messages.pop(0)
            self.conversation._append_message({"role": "user", "content": content})

    def _reveal_tool_schemas_from_result(self, tool_result: ToolResult) -> None:
        if tool_result.tool != "tool_search":
            return
        raw_names = tool_result.metadata.get("revealed_tool_names")
        if not isinstance(raw_names, list) or self.agent.tool_context is None:
            return
        context = self.agent.tool_context
        registry = context.tool_registry
        if registry is None:
            return
        revealed = context.revealed_tool_names
        new_names = {
            str(name)
            for name in raw_names
            if isinstance(name, str)
            and name not in revealed
            and registry.get(name) is not None
            and registry.disclosure_for(name) == "deferred"
        }
        if not new_names:
            return

        allowed_permissions = context.allowed_tool_permissions
        revealed_schemas = tool_schemas_for_profile(
            allowed_permissions=allowed_permissions,
            include_names=new_names,
            exclude_names=context.blocked_tool_names,
            registry=registry,
            disclosure={"deferred"},
        )
        if not revealed_schemas:
            return

        current_schemas = list(self.agent.tool_schemas or [])
        current_names = {
            schema.get("function", {}).get("name")
            for schema in current_schemas
            if isinstance(schema, dict)
        }
        additions = [
            schema
            for schema in revealed_schemas
            if schema.get("function", {}).get("name") not in current_names
        ]
        if not additions:
            revealed.update(new_names)
            return
        revealed.update(new_names)
        self.agent.update_tool_schemas(current_schemas + additions)


def _new_middleware_activity() -> dict[str, Any]:
    return {
        "hooks": 0,
        "duration_ms": 0.0,
        "outcome": "passed",
        "sources": [],
    }
