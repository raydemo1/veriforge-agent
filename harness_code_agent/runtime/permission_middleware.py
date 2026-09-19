"""PermissionMiddleware — permission/approval orchestration as a before_tool middleware.

This middleware is the single point of permission enforcement in the agent loop.

Design:
  - :class:`ShellCommandAnalyzer` (inside PermissionPolicy): facts only
  - PermissionPolicy: tool + args + facts → allow / ask / deny
  - ApprovalProvider: user interaction adapter
  - PermissionMiddleware: orchestrates the prepared decision + provider

The decision is computed once during call preparation and passed in as
``permission_decision``; the middleware never re-runs the policy. When invoked
standalone (without a prepared decision) it computes one itself.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .shell_classification import ShellEffect, TargetScope
from .approvals import ApprovalRequest
from .middleware import AgentMiddleware

if TYPE_CHECKING:
    from .tool_context import ToolContext

log = logging.getLogger("harness")

_BLOCKED_MESSAGE = (
    "[blocked] 工具未执行：该命令被安全黑名单拦截，可能造成不可恢复的数据或系统破坏。"
    "请向用户明确说明未执行，并建议先备份或改用可恢复方案。"
)
_EXTERNAL_WRITE_MESSAGE = (
    "[blocked] 工具未执行：该命令会修改工作区之外的环境"
    "（全局安装、工作区外路径、推送远端等），当前权限模式不允许。"
    "请向用户明确说明未执行，并在获得明确授权后再试。"
)
_SYSTEM_WRITE_MESSAGE = (
    "[blocked] 工具未执行：该命令会写入系统目录或系统配置，当前权限模式不允许。"
    "请向用户明确说明未执行。"
)
_WORKSPACE_READ_ONLY_MESSAGE = (
    "[blocked] 工具未执行：当前为只读权限模式，不能通过 shell 修改工作区。"
    "请向用户明确说明未执行。"
)
_SHELL_DENY_FALLBACK = (
    "[blocked] 工具未执行：当前权限模式不允许此命令。请向用户明确说明未执行，"
    "并说明需要用户如何授权。"
)


class PermissionMiddleware(AgentMiddleware):
    """Enforce the prepared permission decision and drive the approval flow."""

    def __init__(self, tool_context: ToolContext, tool_registry):
        self._ctx = tool_context
        self._registry = tool_registry

    def before_tool(
        self,
        tool_name: str,
        tool_args: dict,
        messages: list[dict],
        runtime_state=None,
        agent_name: str | None = None,
        permission_decision=None,
    ) -> str | None:
        decision = permission_decision or self._decide(tool_name, tool_args)

        # --- deny ---
        if not decision.allowed and not decision.requires_approval:
            log.info(
                "PermissionMiddleware: blocked %s (risk=%s, reason=%s)",
                tool_name, decision.risk, decision.reason,
            )
            message = self._deny_message(tool_name, decision)
            if message is None:
                return f"[blocked] {decision.reason}"
            return message

        # --- ask ---
        if decision.requires_approval:
            redacted_args = _redact_tool_args(tool_args)
            approval_request = ApprovalRequest(
                tool_name=tool_name,
                args=redacted_args,
                risk=decision.risk,
                reason=decision.reason,
                agent_name=agent_name,
                session_id=self._ctx.session_id,
            )
            self._ctx.event_bus.emit(
                "approval_requested",
                agent=agent_name,
                payload={
                    "tool": tool_name,
                    "risk": decision.risk,
                    "reason": decision.reason,
                    "args": redacted_args,
                },
            )

            approval_result = self._ctx.approval_provider.request(approval_request)

            self._ctx.event_bus.emit(
                "approval_decided",
                agent=agent_name,
                payload={
                    "tool": tool_name,
                    "approved": approval_result.approved,
                    "reason": approval_result.reason,
                    "metadata": approval_result.metadata,
                },
            )

            if not approval_result.approved:
                log.info(
                    "PermissionMiddleware: user denied %s (reason=%s)",
                    tool_name, approval_result.reason,
                )
                return (
                    f"[approval_denied] 工具未执行：该操作未获得审批（{approval_result.reason}）。"
                    "请向用户明确说明未执行，不要假设文件或外部状态已经改变；如需继续，请先确认操作范围并准备备份。"
                )

        return None

    def _decide(self, tool_name: str, tool_args: dict):
        registry = self._ctx.tool_registry or self._registry
        workspace_root = self._workspace_root()
        return self._ctx.permission_policy.decide_tool_call(
            tool_name,
            tool_args,
            tool_permission=registry.permission_for(tool_name),
            workspace_root=workspace_root,
        )

    def _workspace_root(self) -> str | None:
        workspace = getattr(self._ctx, "workspace", None)
        root = getattr(workspace, "root", None)
        return str(root) if root is not None else None

    def _deny_message(self, tool_name: str, decision) -> str | None:
        if tool_name != "run_bash":
            return None
        if decision.risk == "shell_blocked":
            return _BLOCKED_MESSAGE
        analysis = getattr(decision, "analysis", None)
        effects = getattr(analysis, "effects", frozenset())
        scopes = {t.scope for t in getattr(analysis, "targets", ())}
        if TargetScope.SYSTEM in scopes and (
            effects & {ShellEffect.WRITE, ShellEffect.DELETE}
        ):
            return _SYSTEM_WRITE_MESSAGE
        if (
            TargetScope.EXTERNAL in scopes
            or ShellEffect.GIT_MUTATION in effects and ShellEffect.NETWORK in effects
        ):
            return _EXTERNAL_WRITE_MESSAGE
        if (
            effects & {ShellEffect.WRITE, ShellEffect.DELETE}
            and self._is_read_only_mode()
        ):
            return _WORKSPACE_READ_ONLY_MESSAGE
        if decision.risk == "shell_risky":
            return _SHELL_DENY_FALLBACK
        return None

    def _is_read_only_mode(self) -> bool:
        return getattr(self._ctx.permission_policy, "mode", "") == "read-only"


def _redact_tool_args(arguments: dict) -> dict:
    """Redact large argument values for display/logging."""
    redacted = dict(arguments or {})
    if "content" in redacted:
        redacted["content"] = f"[{len(str(redacted['content']))} chars]"
    return redacted
