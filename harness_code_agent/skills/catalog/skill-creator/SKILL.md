---
name: skill-creator
description: Create, update, or trim repository skills. Use when a capability gap needs a new or revised skill.
argument-hint: "<skill name or capability>"
disable-model-invocation: true
---

# Skill Creator

Create the smallest repository skill that adds reusable judgment or procedure. This workflow edits only `harness_code_agent/skills/catalog/`; global Codex and agent skill directories are out of scope.

## 1. Establish the need

Read the request, search the current catalog, and inspect adjacent skills. Prefer updating or composing an existing skill when the proposed capability overlaps. Do not create a skill for one-off facts, generic advice, or behavior already enforced by code or tests.

Choose the invocation permission deliberately:

- **Model-invoked:** omit `disable-model-invocation`; the concise description enters the L1 catalog and the model may select the skill when a task makes it relevant.
- **User-invoked:** set `disable-model-invocation: true`; the skill keeps its catalog entry but the model may only load it after an explicit user request through `/name` or by naming it.

Write the description as one function sentence followed by `Use when <condition>.`

## 2. Design the skill

Use a lowercase kebab-case directory whose name exactly matches frontmatter `name`. Provide a specific `description`; add `argument-hint` only when arguments are useful.

Keep `SKILL.md` focused on decisions and procedures that change behavior. Move substantial templates, examples, or reference material into sibling files and point to them explicitly. Compose another catalog skill through `read_skill_file("catalog/<name>/SKILL.md")`; do not refer to Claude's Skill tool or assume a global installation.

Do not add provider-specific agent metadata. Do not copy an upstream skill blindly: retain only the parts compatible with VeriForge's tools, authorization rules, and repository workflow.

## 3. Implement and validate

After editing:

1. Instantiate `SkillRegistry` and confirm discovery succeeds with no duplicate or invalid metadata.
2. Confirm the skill is listed under the intended catalog group and that `/name` resolves.
3. Load its body and required sibling files through `read_skill_file`.
4. Add or update focused catalog tests, including a negative or boundary case when behavior changed.
5. Search for stale names and incompatible invocation instructions.
6. Run the focused test suite and `git diff --check`.

Report the invocation surface, files changed, validation evidence, and any intentionally excluded upstream behavior.
