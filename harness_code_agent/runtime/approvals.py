from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ApprovalRequest:
    tool_name: str
    args: dict
    risk: str
    reason: str
    agent_name: str | None = None
    session_id: str | None = None


@dataclass
class ApprovalResult:
    approved: bool
    reason: str
    metadata: dict = field(default_factory=dict)


class ApprovalProvider(Protocol):
    def request(self, request: ApprovalRequest) -> ApprovalResult:
        ...


class NoApprovalProvider:
    """Safe default for non-interactive tests and batch runs."""

    def request(self, request: ApprovalRequest) -> ApprovalResult:
        return ApprovalResult(False, "no approval provider configured")


class StaticApprovalProvider:
    """Deterministic provider for tests and scripted runs."""

    def __init__(self, *, approved: bool, reason: str = ""):
        self.approved = approved
        self.reason = reason or ("approved" if approved else "denied")

    def request(self, request: ApprovalRequest) -> ApprovalResult:
        return ApprovalResult(self.approved, self.reason)


class LlmAutoApprovalProvider:
    """Use the configured fast model to decide approval requests."""

    MIN_CONFIDENCE = 0.75

    def request(self, request: ApprovalRequest) -> ApprovalResult:
        from .. import config
        from ..support.llm import ProviderAdapter, client_scope

        profile = config.resolve_model_profile("fast")
        metadata = {
            "approval_source": "llm_auto",
            "model": profile.model,
            "provider": profile.provider,
        }
        try:
            adapter = ProviderAdapter(profile.provider)
            with client_scope() as client:
                response = client.chat.completions.create(**adapter.chat_kwargs(
                    profile=profile,
                    messages=_approval_messages(request),
                    max_tokens=300,
                ))
            raw = response.choices[0].message.content or ""
            data = _parse_json_object(raw)
            approved = bool(data.get("approved"))
            confidence = float(data.get("confidence", 0.0))
            reason = str(data.get("reason") or "").strip() or "no reason provided"
            metadata.update({
                "confidence": confidence,
                "raw_reason": reason,
            })
            if approved and confidence >= self.MIN_CONFIDENCE:
                return ApprovalResult(True, f"llm-auto approved: {reason}", metadata)
            if approved:
                return ApprovalResult(False, f"llm-auto confidence too low: {confidence:.2f}", metadata)
            return ApprovalResult(False, f"llm-auto denied: {reason}", metadata)
        except Exception as exc:
            metadata["error"] = str(exc)
            return ApprovalResult(False, "llm-auto approval failed", metadata)


class ConsoleApprovalProvider:
    """Prompt on stdin/stdout when an interactive terminal is available."""

    def request(self, request: ApprovalRequest) -> ApprovalResult:
        if not sys.stdin.isatty():
            return ApprovalResult(False, "approval requires an interactive terminal")
        print("\nApproval required for tool call:")
        print(f"  tool: {request.tool_name}")
        print(f"  risk: {request.risk}")
        print(f"  reason: {request.reason}")
        if request.tool_name == "run_bash":
            print(f"  command: {request.args.get('command', '')}")
        else:
            print(f"  args: {_summarize_args(request.args)}")
        answer = input("Approve this action? [y/N] ").strip().lower()
        if answer in {"y", "yes"}:
            return ApprovalResult(True, "approved by user")
        return ApprovalResult(False, "denied by user")


def _summarize_args(args: dict) -> dict:
    summary = dict(args or {})
    if "content" in summary:
        summary["content"] = f"[{len(str(summary['content']))} chars]"
    return summary


def _approval_messages(request: ApprovalRequest) -> list[dict]:
    payload = {
        "tool_name": request.tool_name,
        "args": _summarize_args(request.args),
        "risk": request.risk,
        "permission_reason": request.reason,
        "agent_name": request.agent_name,
        "session_id": request.session_id,
    }
    return [
        {
            "role": "system",
            "content": (
                "You are an automatic approval gate for a local coding agent. "
                "Decide whether the proposed tool call is reasonably safe and necessary "
                "for a coding task. Approve only when the action is specific, bounded, "
                "and not likely to destroy unrelated user data or expose secrets. "
                "Return only compact JSON with keys: approved, confidence, reason. "
                "confidence must be a number from 0 to 1."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def _parse_json_object(raw: str) -> dict:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise TypeError("approval response must be a JSON object")
    return data
