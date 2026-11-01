# Skill system

VeriForge loads skills in three disclosure layers:

1. **L1 catalog** — every skill's name, description, and path, rendered into the stable system prompt by `PromptPrefixBuilder`.
2. **L2 body** — the selected `SKILL.md`, loaded on demand with `read_skill_file`.
3. **L3 references** — sibling templates, references, and scripts named by the loaded `SKILL.md`, loaded through the same tool only when its pointer fires.

Two invocation permissions share that mechanism:

- **Model-invoked skills** omit `disable-model-invocation`; the model may select them when a task makes them relevant.
- **User-invoked skills** set `disable-model-invocation: true`; they stay in the catalog with name, description, and path, but the model may not select them on its own. The user opens them with `/name` or by naming the skill in a request.

Calling semantics:

- `/name` is deterministic. Every skill has a Slash command, its `SKILL.md` is injected into the current turn, and no routing happens.
- Natural language gets no programmatic matching. The model reads L1 and decides for itself, including whether the user explicitly asked for a user-invoked skill.
- `disable-model-invocation` constrains model behavior through the L1 routing policy. It is not a runtime permission gate.

The routing policy lives in `agent/prompts.py` (`SKILL_ROUTING_POLICY`); `skills/registry.py` only discovers skills, parses metadata, and renders the catalog listing.

The catalog is rebuilt only when `SKILL.md` files on disk actually change, never by a file watcher or background scan.

The engineering and productivity foundation is a reviewed fork of [mattpocock/skills](https://github.com/mattpocock/skills), synchronized from `main` at commit `5b15a47f2d7150f545fbcacbfe381787fc0230dc` (August 21, 2026). It is maintained locally and updated selectively rather than synchronized automatically.

Local adaptations:

- every skill is reachable as a VeriForge dynamic slash command, while model-invocation remains a separate permission;
- model-invoked skills are composed through `read_skill_file`, not slash commands;
- `workflows` and `setup-workflows` replace author-branded names;
- upstream agent metadata is excluded because VeriForge owns registration and invocation;
- `to-spec`, `to-tickets`, and `implement` are optional, scope-sensitive workflow stages rather than a mandatory document chain;
- `skill-creator` and `find-skills` are explicit user workflows adapted to this repository catalog;
- excluded upstream workflows include `prototype`, `wayfinder`, `wizard`, `teach`, and other workflows that do not match this product's execution model;
- VeriForge-specific core skills cover guarded execution, frontend design, and frontend diagnosis.
