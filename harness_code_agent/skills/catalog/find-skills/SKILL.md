---
name: find-skills
description: Audit the current skill catalog and recommend a few external skills for capability gaps. Use when a capability or workflow gap needs an outside skill.
argument-hint: "<capability or workflow gap>"
disable-model-invocation: true
---

# Find Skills

Find candidates for this repository catalog, not for the user's global Codex installation. Discovery never authorizes installation.

## 1. Audit before searching

Translate the request into concrete triggers, required tools, and expected outputs. Search `harness_code_agent/skills/catalog/` for existing coverage and identify whether the gap is missing capability, weak instructions, or poor routing. Recommend updating an existing skill when that is simpler.

## 2. Discover candidates

Search current primary sources such as the upstream repository, the author's documentation, skills.sh, and GitHub. Prefer maintained repositories with readable source, clear provenance, compatible licensing, and recent substantive activity.

Evaluate the complete candidate, including scripts, references, tool assumptions, external calls, permissions, hidden setup, and overlap with the catalog. Popularity and install counts are signals, not proof of quality or compatibility.

## 3. Recommend, then wait

Present at most three candidates. For each, state:

- what gap it covers;
- source and maintenance evidence;
- required tools or permissions;
- overlap and incompatibilities;
- whether to adapt, merge into an existing skill, or reject.

Make one recommendation and explain the smallest useful subset. Do not clone into the catalog, install globally, or execute third-party scripts until the user explicitly approves the candidate and scope.

## 4. Add an approved candidate

After approval, load `catalog/skill-creator/SKILL.md` with `read_skill_file` and adapt the selected material to VeriForge. Preserve license notices where required, record the upstream commit in the catalog README, exclude unsupported metadata, and validate the resulting registry and tests.
