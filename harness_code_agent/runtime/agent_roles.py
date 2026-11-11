"""Subagent role contract shared by the coordinator, the policy and the schema.

A role is a runtime contract, not a prompt style.  The coordinator validates
the name, the permission policy derives its ceiling from it, the tool schema
advertises it to the model, and the coordinator builds its instructions from
it.  All four read this module, so a new role cannot be half-added.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentRole:
    """One delegated role and the facts every consumer needs about it."""

    name: str
    #: Cannot modify the shared workspace at all.
    read_only: bool
    #: May propose edits inside its own isolated copy of the workspace.
    can_write: bool
    #: How the role is described to the subagent itself.
    instruction: str

    def __post_init__(self) -> None:
        if self.read_only and self.can_write:
            raise ValueError(f"{self.name}: a read-only role cannot also write")


AGENT_ROLE_DEFINITIONS: tuple[AgentRole, ...] = (
    AgentRole(
        name="explorer",
        read_only=True,
        can_write=False,
        instruction="Investigate code and return evidence. Do not modify files.",
    ),
    AgentRole(
        name="test_designer",
        read_only=True,
        can_write=False,
        instruction="Design focused tests and failure cases. Do not modify files.",
    ),
    AgentRole(
        name="reviewer",
        read_only=True,
        can_write=False,
        instruction="Review for correctness, regressions, security, and missing tests. Do not modify files.",
    ),
    AgentRole(
        name="verifier",
        read_only=True,
        can_write=False,
        instruction="Run focused verification and interpret results. Do not modify source files.",
    ),
    AgentRole(
        name="worker",
        read_only=False,
        can_write=True,
        instruction="Implement the assigned change in your isolated workspace and verify it.",
    ),
)

#: Role names in declaration order; the tool schema advertises this order.
AGENT_ROLES: tuple[str, ...] = tuple(role.name for role in AGENT_ROLE_DEFINITIONS)
AGENT_ROLE_NAMES = frozenset(AGENT_ROLES)
READ_ONLY_ROLE_NAMES = frozenset(role.name for role in AGENT_ROLE_DEFINITIONS if role.read_only)
WRITING_ROLE_NAMES = frozenset(role.name for role in AGENT_ROLE_DEFINITIONS if role.can_write)

_BY_NAME = {role.name: role for role in AGENT_ROLE_DEFINITIONS}


def role_definition(name: str) -> AgentRole:
    """Return the declared role, rejecting names outside the contract."""
    try:
        return _BY_NAME[str(name).strip().lower()]
    except KeyError:
        raise ValueError(f"unknown agent role: {name}") from None
