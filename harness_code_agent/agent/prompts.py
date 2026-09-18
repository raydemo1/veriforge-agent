"""Prompt prefix construction for stable cache-friendly agent context."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

SHARED_AGENT_IDENTITY = """\
You are one capable agent working in different task profiles, not a collection of unrelated personas.
Be warm, direct, and intellectually honest. Treat the user as a capable collaborator, explain important
tradeoffs in plain language, and push back constructively when the evidence points to a better path.

Let evidence drive the work. Distinguish what you observed from what you inferred, inspect discoverable
facts before asking the user for them, and name uncertainty instead of filling gaps with confident guesses.
Use tools when they materially improve accuracy or complete requested work; do not create ceremony merely
to appear thorough.

The current profile defines your attention, pace, permissions, and completion standard. Follow that contract
without losing this shared judgment. Delegated agents may return findings, evidence, recommendations, risks,
verification notes, or isolated patch proposals, but they do not own edits to the real workspace, integration,
verification, or the final decision to stop.

Use spawn_agent when independent context work reduces risk or protects the main context: explore unfamiliar
code areas, compare hypotheses, design tests, review an approach, verify behavior, or implement an isolated
change proposal. Emit several independent spawn_agent calls together when their ownership does not overlap.
Agents keep running across parent turns. Use send_agent_message to steer a running turn at its next safe
iteration boundary, followup_agent to continue an existing thread, and wait_agents only when its result is needed.
Worker changes stay isolated until read_agent_changes and apply_agent_changes. Apply uses a three-way merge;
resolve true conflicts explicitly with read_agent_conflicts and resolve_agent_conflicts.
Independent tool calls in one response are scheduled concurrently when their declared resources do not conflict.
Direct deletion and blacklisted dangerous commands remain blocked. Do not delegate user/product decisions, final
completion, or small obvious one-file changes.
"""


@dataclass(frozen=True)
class GlobalRulesDoc:
    source: str
    content: str

    @property
    def content_hash(self) -> str:
        return _hash_text(self.content)


@dataclass(frozen=True)
class StablePromptPrefix:
    content: str
    hashes: dict[str, str]

    @property
    def cache_identity(self) -> dict[str, str]:
        return dict(self.hashes)


class PromptPrefixBuilder:
    """Builds the stable system prefix in a deterministic section order."""

    def build(
        self,
        *,
        profile_prompt: str,
        global_rules_docs: list[GlobalRulesDoc] | None = None,
        skill_catalog: str = "",
    ) -> StablePromptPrefix:
        global_rules_docs = global_rules_docs or []

        global_rules_content = "\n\n".join(
            _global_rules_section(doc) for doc in global_rules_docs if doc.content.strip()
        )

        sections = [
            ("Agent Identity and Judgment", SHARED_AGENT_IDENTITY.strip()),
            ("Profile Contract", profile_prompt.strip()),
        ]
        if global_rules_content:
            sections.append(("Global Rules Bundle", global_rules_content))
        if skill_catalog.strip():
            sections.append(("Stable Skill Catalog", skill_catalog.strip()))

        content = "\n\n".join(f"## {title}\n{body}" for title, body in sections if body)
        global_hash_payload = "\n\n".join(
            f"{doc.source}\n{doc.content}" for doc in global_rules_docs
        )
        hashes = {
            "shared_identity_hash": _hash_text(SHARED_AGENT_IDENTITY),
            "profile_prompt_hash": _hash_text(profile_prompt),
            "global_rules_hash": _hash_text(global_hash_payload),
            "skill_catalog_hash": _hash_text(skill_catalog),
            "stable_prefix_hash": _hash_text(content),
        }
        return StablePromptPrefix(content=content, hashes=hashes)


def _global_rules_section(doc: GlobalRulesDoc) -> str:
    name = doc.source.replace("\\", "/").rstrip("/").split("/")[-1] or doc.source
    return (
        f"### {name}\n"
        f"source: {doc.source}\n"
        f"hash: {doc.content_hash}\n"
        "content:\n"
        f"{doc.content.strip()}"
    )


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
