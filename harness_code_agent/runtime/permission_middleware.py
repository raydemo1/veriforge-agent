"""PermissionMiddleware — permission/approval orchestration as a before_tool middleware.

This middleware is the single point of permission enforcement in the agent loop.

Design:
  - ShellCommandAnalyzer (inside PermissionPolicy): facts only
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

from .approvals import ApprovalRequest, ApprovalResult, redact_for_reviewer, redact_text
from .middleware import AgentMiddleware
from .shell_classification import ShellEffect, TargetScope
from .tool_result import ToolResult

if TYPE_CHECKING:
    from .tool_context import ToolContext

log = logging.getLogger("harness")

TASK_SUMMARY_CHARS = 240

_BLOCKED_MESSAGE = (
    "[blocked] 工具未执行：该命令被安全黑名单拦截，可能造成不可恢复的数据或系统破坏。"
    "请向用户明确说明未执行，并建议先备份或改用可恢复方案。"
)
_EXTERNAL_WRITE_MESSAGE = (
    "[blocked] 工具未执行：该命令会修改工作区之外的环境"
    "（全局安装、工作区外路径、推送远端等），当前沙箱不支持通过审批扩大可写范围。"
    "请向用户明确说明未执行，并由用户自行执行或切换到完全访问模式。"
)
_SYSTEM_WRITE_MESSAGE = (
    "[blocked] 工具未执行：该命令会写入系统目录或系统配置，属于不可审批的禁止操作。"
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
_HUMAN_ONLY_MESSAGE = (
    "[approval_denied] 工具未执行：该操作不可由自动审批放行，且当前没有可用的人工审批入口。"
    "请向用户明确说明未执行，并请用户自行执行或开启交互会话后重试。"
)
def _decided_by(result: ApprovalResult) -> str:
    """Who made this call: a person, the reviewer, or nobody (gate unavailable)."""
    declared = str(result.metadata.get("decided_by") or "").strip()
    if declared in {"human", "reviewer", "unavailable"}:
        return declared
    # A provider that does not label itself is treated as the interactive
    # surface: only the reviewer path labels its results.
    return "human"


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
    ) -> ToolResult | None:
        decision = permission_decision or self._decide(tool_name, tool_args)

        # --- deny ---
        if not decision.allowed and not decision.requires_approval:
            log.info(
                "PermissionMiddleware: blocked %s (risk=%s, reason=%s)",
                tool_name, decision.risk, decision.reason,
            )
            message = self._deny_message(tool_name, decision) or (
                f"[blocked] {decision.reason}"
            )
            return ToolResult(
                tool=tool_name,
                status="failed",
                output=message,
                error=message,
                metadata={
                    "status_source": "permission",
                    "risk": decision.risk,
                    "reason": decision.reason,
                },
            )

        # --- ask ---
        if decision.requires_approval:
            request = self._approval_request(
                tool_name, tool_args, decision, messages, agent_name
            )
            self._ctx.event_bus.emit(
                "approval_requested",
                agent=agent_name,
                payload={
                    "tool": tool_name,
                    "risk": decision.risk,
                    "reason": decision.reason,
                    "args": request.args,
                    "write_dirs": list(request.write_dirs),
                    "uncertainty": request.uncertainty,
                    "requires_human": request.requires_human,
                    "sandbox_permissions": request.sandbox_permissions,
                    "additional_permissions": request.additional_permissions,
                    "justification": request.justification,
                    "network_access": request.network_access,
                },
            )

            approval_result, decided_by = self._request_approval(request)

            self._ctx.event_bus.emit(
                "approval_decided",
                agent=agent_name,
                payload={
                    "tool": tool_name,
                    "approved": approval_result.approved,
                    "reason": approval_result.reason,
                    "decided_by": decided_by,
                    "metadata": approval_result.metadata,
                },
            )

            if not approval_result.approved:
                log.info(
                    "PermissionMiddleware: denied %s (decided_by=%s, reason=%s)",
                    tool_name, decided_by, approval_result.reason,
                )
                # Only a person saying "no" ends the turn. An unmet automatic
                # gate is a result the agent may react to, except that an
                # explicit reviewer denial must not invite a rewritten retry.
                human_denied = decided_by == "human"
                failure = str(approval_result.metadata.get("failure") or "")
                if human_denied:
                    message = "[approval_denied] 用户已拒绝该操作。"
                elif failure == "no_human_provider":
                    message = _HUMAN_ONLY_MESSAGE
                elif failure == "denied":
                    message = (
                        f"[approval_denied] 自动审批拒绝了该操作：{approval_result.reason}。"
                        "不要改写命令重试同一操作；请向用户说明未执行。"
                    )
                elif decided_by == "unavailable":
                    message = (
                        "[approval_denied] 当前没有可用的审批入口，该操作未执行。"
                        "请向用户说明，并请用户开启交互会话或自行执行。"
                    )
                else:
                    message = (
                        f"[approval_denied] 自动审批未能完成（{failure or 'unknown'}）："
                        f"{approval_result.reason}。该操作未执行；可以稍后重试同一请求。"
                    )
                return ToolResult(
                    tool=tool_name,
                    status="failed",
                    output=message,
                    error=message,
                    metadata={
                        "status_source": "approval" if human_denied else "approval_auto",
                        "reason": approval_result.reason,
                        "decided_by": decided_by,
                        "failure": approval_result.metadata.get("failure"),
                    },
                )

        return None

    # ------------------------------------------------------------------
    # Approval plumbing
    # ------------------------------------------------------------------

    def _approval_request(
        self,
        tool_name: str,
        tool_args: dict,
        decision,
        messages: list[dict],
        agent_name: str | None,
    ) -> ApprovalRequest:
        redacted_args, unsafe = redact_for_reviewer(tool_args)
        analysis = getattr(decision, "analysis", None)
        uncertainty = getattr(analysis, "uncertainty", None)
        write_dirs = tuple(
            str(item) for item in (tool_args.get("write_paths") or ())
        ) if isinstance(tool_args.get("write_paths"), (list, tuple)) else ()
        return ApprovalRequest(
            tool_name=tool_name,
            args=redacted_args,
            risk=decision.risk,
            reason=decision.reason,
            agent_name=agent_name,
            session_id=self._ctx.session_id,
            task_summary=_task_summary(messages),
            cwd=self._workspace_root(),
            write_dirs=write_dirs,
            uncertainty=uncertainty,
            # Content that cannot be reduced to a safe prompt (unredactable
            # credentials) never reaches the reviewer.
            requires_human=bool(getattr(decision, "requires_human", False)) or unsafe,
            sandbox_permissions=str(
                tool_args.get("sandbox_permissions") or "use_default"
            ),
            additional_permissions=(
                tool_args.get("additional_permissions")
                if isinstance(tool_args.get("additional_permissions"), dict)
                else {}
            ),
            justification=redact_text(str(tool_args.get("justification") or "")) or None,
            network_access=bool(getattr(decision, "network_access", False)),
        )

    def _request_approval(self, request: ApprovalRequest) -> tuple[ApprovalResult, str]:
        """Ask the mode's gate, and report who actually decided."""
        result = self._ctx.approval_provider.request(request)
        return result, _decided_by(result)

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
        if tool_name != "bash":
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
            if decision.reason:
                return f"{_SHELL_DENY_FALLBACK} 原因：{decision.reason}"
            return _SHELL_DENY_FALLBACK
        return None

    def _is_read_only_mode(self) -> bool:
        return getattr(self._ctx.permission_policy, "mode", "") == "read-only"


def _task_summary(messages: list[dict] | None) -> str | None:
    """The latest user request, as short untrusted context for the reviewer."""
    for message in reversed(list(messages or ())):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(
                str(part.get("text") or "") for part in content if isinstance(part, dict)
            )
        text = redact_text(" ".join(str(content or "").split()))
        if text:
            return text[:TASK_SUMMARY_CHARS]
    return None
