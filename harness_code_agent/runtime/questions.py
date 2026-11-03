from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Protocol

OTHER_LABELS = {"other", "其他"}


@dataclass(frozen=True)
class QuestionOption:
    label: str
    value: str = ""
    description: str = ""
    is_other: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "value": self.value or self.label,
            "description": self.description,
            "is_other": self.is_other,
        }


@dataclass(frozen=True)
class QuestionRequest:
    question: str
    options: list[QuestionOption]
    agent_name: str | None = None
    session_id: str | None = None
    details: str = ""


@dataclass
class QuestionResult:
    selected_index: int | None = None
    label: str = ""
    value: str = ""
    is_other: bool = False
    custom_text: str = ""
    cancelled: bool = False
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_index": self.selected_index,
            "label": self.label,
            "value": self.value,
            "is_other": self.is_other,
            "custom_text": self.custom_text,
            "cancelled": self.cancelled,
            "reason": self.reason,
            "metadata": self.metadata,
        }


class QuestionProvider(Protocol):
    def ask(self, request: QuestionRequest) -> QuestionResult:
        ...


class NoQuestionProvider:
    """Safe default for non-interactive tests and batch runs."""

    def ask(self, request: QuestionRequest) -> QuestionResult:
        return QuestionResult(cancelled=True, reason="no question provider configured")


class StaticQuestionProvider:
    """Deterministic provider for tests and scripted runs."""

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


class ConsoleQuestionProvider:
    """Ask a question on stdin/stdout when an interactive terminal is available."""

    def ask(self, request: QuestionRequest) -> QuestionResult:
        if not sys.stdin.isatty():
            return QuestionResult(cancelled=True, reason="question requires an interactive terminal")

        print()
        print(request.question)
        if request.details:
            print(request.details)
        for index, option in enumerate(request.options, start=1):
            suffix = f" - {option.description}" if option.description else ""
            print(f"  {index}. {option.label}{suffix}")

        while True:
            answer = input("Choose an option number: ").strip()
            try:
                selected_index = int(answer) - 1
            except ValueError:
                print("Please enter a number.")
                continue
            if 0 <= selected_index < len(request.options):
                break
            print("Choice is out of range.")

        option = request.options[selected_index]
        custom_text = ""
        if option.is_other:
            custom_text = input(f"{option.label}: ").strip()
        return question_result_from_option(
            request,
            selected_index,
            custom_text=custom_text,
            metadata={"ui": "console"},
            reason="selected by user",
        )


def normalize_question_options(raw_options: list[Any] | None, *, other_label: str = "其他") -> list[QuestionOption]:
    options: list[QuestionOption] = []
    other_option: QuestionOption | None = None
    for raw in raw_options or []:
        option = _coerce_question_option(raw)
        if option is None:
            continue
        if option.is_other or _is_other_label(option.label):
            other_option = QuestionOption(
                label=option.label or other_label,
                value=option.value or option.label or other_label,
                description=option.description,
                is_other=True,
            )
        else:
            options.append(option)

    if other_option is None:
        other_option = QuestionOption(label=other_label, value=other_label, is_other=True)
    options.append(other_option)
    return options


def question_result_from_option(
    request: QuestionRequest,
    selected_index: int,
    *,
    custom_text: str = "",
    metadata: dict[str, Any] | None = None,
    reason: str = "selected by user",
) -> QuestionResult:
    option = request.options[selected_index]
    return QuestionResult(
        selected_index=selected_index,
        label=option.label,
        value=option.value or option.label,
        is_other=option.is_other,
        custom_text=custom_text if option.is_other else "",
        cancelled=False,
        reason=reason,
        metadata=metadata or {},
    )


def _coerce_question_option(raw: Any) -> QuestionOption | None:
    if isinstance(raw, QuestionOption):
        label = raw.label.strip()
        if not label:
            return None
        return QuestionOption(
            label=label,
            value=(raw.value or label).strip(),
            description=raw.description.strip(),
            is_other=raw.is_other,
        )
    if isinstance(raw, str):
        label = raw.strip()
        if not label:
            return None
        return QuestionOption(label=label, value=label, is_other=_is_other_label(label))
    if isinstance(raw, dict):
        label = str(raw.get("label") or raw.get("title") or raw.get("value") or "").strip()
        if not label:
            return None
        value = str(raw.get("value") or label).strip()
        description = str(raw.get("description") or "").strip()
        is_other = bool(raw.get("is_other") or raw.get("other") or _is_other_label(label))
        return QuestionOption(label=label, value=value, description=description, is_other=is_other)
    return None


def _is_other_label(label: str) -> bool:
    return label.strip().lower() in OTHER_LABELS
