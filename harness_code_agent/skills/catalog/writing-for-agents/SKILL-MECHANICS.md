# Skill mechanics

The skill-specific branch of [`writing-for-agents`](SKILL.md): what changes when the document is a skill (frontmatter, the invocation choice, and router skills). Everything else about writing it is the universal reference in `SKILL.md`.

## Invocation

Two choices, trading the two loads:

- A **model-invoked** skill keeps a model-facing `description`, so the agent can select it autonomously and other skills can reach it. You can still type its name: model-invocation always _includes_ user reach; a description only ever adds agent selection, never removes the human's. Mechanics: omit `disable-model-invocation`, and write a description carrying the trigger branches (the pointer-writing rules in `SKILL.md` apply in full).
- A **user-invoked** skill sets `disable-model-invocation: true`. Its name, description, and path still sit in the catalog, because the model must be able to recognize an explicit request for it; what it may not do is select the skill on its own. Mechanics: set the flag, and write a description that states the skill's purpose plainly so an explicit user request is still matched to it.

Pick user-invocation when autonomous selection would be wrong: a workflow the human should choose, a destructive step, or a sequence whose timing is the user's to decide.

Descriptions of shared reference stay in the catalog whichever permission owns the skill. Two user-invoked skills that both need the same reference can each point at it; the pointer wording in each `SKILL.md` decides when it loads.

## Splitting by invocation

The invocation cut of splitting (the sequence cut lives in `SKILL.md`): split off a model-invoked skill when you have a distinct leading word that should trigger it on its own (a trigger word you actually use in your prompts), or another skill must reach it. You pay context load for the new always-loaded description, so that independent reach has to be worth it.

## Router skills

When user-invoked skills multiply past what you can remember, that piled-up cognitive load is cured by a **router skill**: one user-invoked skill that names the others and when to reach for each, so the human has one skill to remember instead of many. It can only hint, never select them: a user-invoked skill's routing policy keeps the model from choosing its targets on its own.
