"""Deterministic approval/question providers for tests."""
from __future__ import annotations

from harness_code_agent.runtime.approvals import ApprovalRequest, ApprovalResult
from harness_code_agent.runtime.questions import (
    QuestionRequest,
    QuestionResult,
    question_result_from_option,
)


class StaticApprovalProvider:
    def __init__(self, *, approved: bool, reason: str = ""):
        self.approved = approved
        self.reason = reason or ("approved" if approved else "denied")

    def request(self, request: ApprovalRequest) -> ApprovalResult:
        return ApprovalResult(self.approved, self.reason)


class StaticQuestionProvider:
    def __init__(self, *, index: int = 0, custom_text: str = "", reason: str = "selected by static provider"):
        self.index = index
        self.custom_text = custom_text
        self.reason = reason

    def ask(self, request: QuestionRequest) -> QuestionResult:
        if self.index < 0 or self.index >= len(request.options):
            return QuestionResult(cancelled=True, reason=f"invalid static question option index: {self.index}")
        return question_result_from_option(
            request,
            self.index,
            custom_text=self.custom_text,
            metadata={"ui": "static"},
            reason=self.reason,
        )
