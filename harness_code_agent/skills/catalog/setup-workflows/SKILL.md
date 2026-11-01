---
name: setup-workflows
description: Configure the issue tracker, triage labels, and domain document conventions. Use when a repository has no workflow contract yet.
disable-model-invocation: true
---

# Setup Workflows

Configure the repository contract consumed by `/triage`, `/to-spec`, `/to-tickets`, `domain-modeling`, `diagnosing-bugs`, and `tdd`.

## 1. Inspect

Read:

- repository remotes and hosting provider;
- applicable `AGENTS.md` or `CLAUDE.md`;
- existing `CONTEXT.md`, `CONTEXT-MAP.md`, and ADR directories;
- `docs/agents/` and `.scratch/`;
- existing issue labels when the configured tracker is accessible.

Done when existing conventions and missing decisions are distinguishable.

## 2. Resolve three decisions

Ask one decision at a time and recommend the detected default.

### Issue tracker

Choose:

- GitHub;
- GitLab;
- local Markdown under `.scratch/<feature>/issues/`;
- another tracker described by the user.

For GitHub or GitLab, also decide whether external PRs/MRs are a triage surface. Default: no.

### Triage labels

Map these canonical roles to real tracker labels:

- `needs-triage`
- `needs-info`
- `ready-for-agent`
- `ready-for-human`
- `wontfix`

Reuse existing labels rather than creating synonyms.

### Domain docs

Choose:

- single context: root `CONTEXT.md` and `docs/adr/`;
- multiple contexts: root `CONTEXT-MAP.md` pointing to per-context files.

Done when all three choices have explicit answers.

## 3. Confirm the draft

Show the proposed instruction block plus:

- `docs/agents/issue-tracker.md`
- `docs/agents/triage-labels.md`
- `docs/agents/domain.md`

Do not write until the user confirms.

## 4. Write

Update the applicable instruction file without duplicating an existing `## Agent skills` section. If neither `AGENTS.md` nor `CLAUDE.md` exists, ask which one to create.

Load the relevant references with `read_skill_file`:

- `catalog/setup-workflows/issue-tracker-github.md`
- `catalog/setup-workflows/issue-tracker-gitlab.md`
- `catalog/setup-workflows/issue-tracker-local.md`
- `catalog/setup-workflows/triage-labels.md`
- `catalog/setup-workflows/domain.md`

For another tracker, record executable read/list/create/comment/label/close operations in plain prose.

Done when the instruction file links all three generated docs and every configured operation matches the selected tracker.

## 5. Verify

Read the written files back. If external tracker operations are configured, run a read-only status or list command. Report missing CLIs or authentication without mutating the tracker.

Setup is complete when all workflow consumers can discover the tracker, label mapping, and domain layout without asking again.
