"""Skill discovery and catalog generation with progressive disclosure.

L1 lists every skill's name, description, path, and invocation permission.
L2 (SKILL.md) and L3 (files referenced from it) stay on disk until the agent
loads them through read_skill_file.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("harness")

SKILLS_DIR = Path(__file__).resolve().parent / "catalog"

_SKILL_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_COMMAND_LINE = re.compile(r"/([a-z0-9]+(?:-[a-z0-9]+)*)(?:[ \t\r\n]+(.*))?", re.DOTALL)


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    path: str
    source_path: Path
    disable_model_invocation: bool = False
    argument_hint: str = ""

    def as_catalog_item(self) -> dict[str, str | bool]:
        return {
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "disable_model_invocation": self.disable_model_invocation,
            "argument_hint": self.argument_hint,
        }


@dataclass(frozen=True)
class SkillInvocation:
    name: str
    arguments: str
    path: str
    prompt: str


class SkillRegistry:
    """Discover skills and render the layered-disclosure catalog."""

    def __init__(self, skills_dir: Path | str | None = None):
        self.skills_dir = Path(skills_dir) if skills_dir else SKILLS_DIR
        self.skills: list[SkillMetadata] = []
        self._by_name: dict[str, SkillMetadata] = {}
        self._fingerprint: tuple[tuple[str, int, int], ...] = ()
        self._discover()

    @property
    def model_catalog(self) -> list[dict[str, str | bool]]:
        return [
            skill.as_catalog_item()
            for skill in self.skills
            if not skill.disable_model_invocation
        ]

    @property
    def user_only(self) -> list[dict[str, str | bool]]:
        return [
            skill.as_catalog_item()
            for skill in self.skills
            if skill.disable_model_invocation
        ]

    def search(self, name: str) -> SkillMetadata | None:
        return self._by_name.get(name.strip().lower())

    def refresh_if_changed(self) -> bool:
        """Re-discover only when the catalog files actually changed."""
        fingerprint = self._fingerprint_of(self.skills_dir)
        if fingerprint == self._fingerprint:
            return False
        self._discover()
        return True

    def catalog_listing(self) -> str:
        """Render the whole L1 catalog, split by invocation permission."""
        if not self.skills:
            return ""
        blocks = ["Model-invoked:", *_listing_lines(self.model_catalog)]
        blocks.extend(["", "User-invoked only:", *_listing_lines(self.user_only)])
        return "\n".join(blocks)

    def build_user_invocation(self, line: str) -> SkillInvocation | None:
        """Expand an exact `/name <arguments>` command into an agent instruction."""
        match = _COMMAND_LINE.fullmatch(line.strip())
        if match is None:
            return None
        skill = self._by_name.get(match.group(1).lower())
        if skill is None:
            return None
        arguments = (match.group(2) or "").strip()
        body = skill.source_path.read_text(encoding="utf-8", errors="replace")
        prompt = "\n".join(
            [
                f"User explicitly invoked `/{skill.name}`.",
                f"Skill source: `{skill.path}`",
                f"Arguments: {arguments or '(none)'}",
                ("Follow the skill instructions below for this turn. "
                "Do not treat the frontmatter description as model-routing guidance."),
                "",
                "<invoked-skill>",
                body,
                "</invoked-skill>",
            ]
        )
        return SkillInvocation(
            name=skill.name,
            arguments=arguments,
            path=skill.path,
            prompt=prompt,
        )

    def _discover(self) -> None:
        skills: list[SkillMetadata] = []
        by_name: dict[str, SkillMetadata] = {}
        if self.skills_dir.is_dir():
            for skill_file in sorted(self.skills_dir.rglob("SKILL.md")):
                skill = _parse_skill(skill_file, self.skills_dir)
                key = skill.name.lower()
                if key in by_name:
                    raise ValueError(
                        f"Duplicate skill name '{skill.name}': "
                        f"{by_name[key].source_path} and {skill_file}"
                    )
                skills.append(skill)
                by_name[key] = skill
                log.debug("Discovered skill: %s", skill.name)
        else:
            log.info("No skills directory found at %s", self.skills_dir)
        self.skills = skills
        self._by_name = by_name
        self._fingerprint = self._fingerprint_of(self.skills_dir)

    def _fingerprint_of(self, root: Path) -> tuple[tuple[str, int, int], ...]:
        entries: list[tuple[str, int, int]] = []
        if not root.is_dir():
            return ()
        for path in sorted(root.rglob("SKILL.md")):
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append((path.relative_to(root).as_posix(), stat.st_mtime_ns, stat.st_size))
        return tuple(entries)


def _parse_skill(skill_file: Path, skills_dir: Path) -> SkillMetadata:
    meta = _parse_frontmatter(skill_file)
    if meta is None:
        raise ValueError(f"Skill is missing valid frontmatter: {skill_file}")
    name = str(meta.get("name", "")).strip()
    if not name:
        raise ValueError(f"Skill has no name: {skill_file}")
    if not _SKILL_NAME.fullmatch(name):
        raise ValueError(f"Invalid skill name '{name}' in {skill_file}")
    if skill_file.parent.name != name:
        raise ValueError(f"Skill name '{name}' must match folder '{skill_file.parent.name}'")
    description = str(meta.get("description", "")).strip()
    if not description:
        raise ValueError(f"Skill '{name}' has no description: {skill_file}")
    relative = Path(skills_dir.name) / skill_file.relative_to(skills_dir)
    return SkillMetadata(
        name=name,
        description=description,
        path=relative.as_posix(),
        source_path=skill_file,
        disable_model_invocation=_parse_bool(meta.get("disable-model-invocation", False)),
        argument_hint=str(meta.get("argument-hint", "")).strip(),
    )


def _listing_lines(items: list[dict[str, str | bool]]) -> list[str]:
    if not items:
        return ["- (none)"]
    lines: list[str] = []
    for item in items:
        lines.append(f"- **{item['name']}**: {item['description']}")
        lines.append(f"  Path: `{item['path']}`")
    return lines


def _parse_frontmatter(path: Path) -> dict[str, str] | None:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return None
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if key in meta:
            raise ValueError(f"Duplicate frontmatter key '{key}' in {path}")
        meta[key] = _strip_scalar_quotes(_strip_yaml_comment(value.strip()))
    return meta


def _strip_scalar_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _strip_yaml_comment(value: str) -> str:
    """Strip a trailing YAML inline comment, respecting quoted strings."""
    in_quote = False
    quote_char: str | None = None
    for i, ch in enumerate(value):
        if ch in ("'", '"') and (i == 0 or value[i - 1] != "\\"):
            if not in_quote:
                in_quote = True
                quote_char = ch
            elif ch == quote_char:
                in_quote = False
        elif ch == "#" and not in_quote:
            return value[:i].rstrip()
    return value


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "1", "on"}:
        return True
    if normalized in {"false", "no", "0", "off", ""}:
        return False
    raise ValueError(f"Invalid boolean frontmatter value: {value}")
