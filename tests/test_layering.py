"""Architecture guard: the package dependency graph stays acyclic.

The runtime used to import in every direction at once (memory -> agent,
sessions -> agent, workspace -> agent), because shared primitives lived in
``agent``. They now live in ``support`` and the two boundary packages import
only down. These tests keep that true, so the next convenience import fails
loudly instead of quietly restoring the cycle.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "harness_code_agent"

#: First-party names: package directories plus root-level modules.
FIRST_PARTY = {p.name for p in PACKAGE.iterdir() if p.is_dir()} | {
    p.stem for p in PACKAGE.glob("*.py")
}

#: Packages that must never import ``agent`` (the orchestration layer).
LOWER_LAYERS = ("memory", "sessions", "workspace")


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(PACKAGE).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imported_packages(path: Path) -> set[str]:
    """Absolute package names imported by ``path`` (relative imports resolved)."""
    dotted = _module_name(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    packages: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = dotted.split(".") if path.name == "__init__.py" else dotted.split(".")[:-1]
                up = node.level - 1
                if up:
                    base = base[: len(base) - up]
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            if target:
                packages.add(target.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("harness_code_agent."):
                    packages.add(alias.name[len("harness_code_agent.") :].split(".")[0])
    return packages & FIRST_PARTY


def _sources(package: str) -> list[Path]:
    return sorted((PACKAGE / package).rglob("*.py"))


def test_support_is_a_leaf():
    allowed = {"support", "provider_resolution", "config"}
    offenders = {
        path.relative_to(PACKAGE).as_posix(): sorted(_imported_packages(path) - allowed)
        for path in _sources("support")
        if _imported_packages(path) - allowed
    }
    assert offenders == {}, f"support must not import other layers: {offenders}"


@pytest.mark.parametrize("package", LOWER_LAYERS)
def test_lower_layers_do_not_import_the_agent(package: str):
    offenders = {
        path.relative_to(PACKAGE).as_posix(): sorted(_imported_packages(path) & {"agent"})
        for path in _sources(package)
        if "agent" in _imported_packages(path)
    }
    assert offenders == {}, f"{package} must not import agent: {offenders}"


def test_tool_context_only_names_the_coordinator_for_typing():
    """runtime/tool_context.py is the single allowed upward reference: a
    TYPE_CHECKING annotation, which is not part of the runtime graph."""
    source = (PACKAGE / "runtime" / "tool_context.py").read_text(encoding="utf-8")
    for line in source.splitlines():
        if "agent.coordinator" in line and line.strip().startswith("from"):
            assert line.startswith("    from "), (
                "tool_context may only import AgentCoordinator for type checking"
            )
