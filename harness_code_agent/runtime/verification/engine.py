"""Objective verification evidence shared by language providers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Literal["passed", "failed", "warning", "skipped"]
    details: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": "\n".join(self.details)[:2_000],
        }


class VerificationProvider(Protocol):
    def supports(self, files: list[str], workspace: Path) -> bool: ...
    def verify(self, files: list[str], workspace: Path) -> list[CheckResult]: ...


class VerificationEngine:
    def __init__(self, providers: list[VerificationProvider] | None = None):
        if providers is None:
            from .languages import GoProvider, RustProvider, TypeScriptProvider
            from .python import PythonProvider

            providers = [
                PythonProvider(),
                TypeScriptProvider(),
                GoProvider(),
                RustProvider(),
            ]
        self.providers = providers

    def verify(self, files: list[str], workspace: Path) -> list[CheckResult]:
        return [
            result
            for provider in self.providers
            if provider.supports(files, workspace)
            for result in provider.verify(files, workspace)
        ]
