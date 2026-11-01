---
name: workflows
description: Route a situation to the right workflow. Use when the goal is clear but the matching command is not.
argument-hint: "<situation>"
disable-model-invocation: true
---

# Workflows

Use this router when you know the situation but not which `/name` workflow fits.

## Build or change something

1. Run `/setup-workflows` once if issue-tracker and domain-doc conventions are not configured.
2. Use `/grill-with-docs` when the user wants codebase-aware decisions stress-tested. Use `/grill-me` when repository context is unnecessary.
3. Use `/to-spec` when settled decisions need a durable specification. Skip it for a clear, small request.
4. Use `/to-tickets` when work spans multiple sessions, can run in parallel, or needs explicit blocking edges. A small spec can go directly to `/implement`.
5. Use `/implement` for one user-scoped request, spec slice, or agent-ready ticket.
6. After a non-trivial change, request a code review or let the model load `code-review`; tiny changes only need a focused self-review.

## Incoming work

- Use `/triage` for raw issues or external pull requests.
- Do not triage tickets produced by `/to-tickets`; they are already agent-ready.

## Codebase health

- Use `/improve-codebase-architecture` to find deepening opportunities and choose one for a later implementation flow.

## Context boundaries

- Use `/handoff` when a fresh session needs the current context.
- Use built-in `/compact` only at a phase boundary when continuing in the same session.

## Skill maintenance

- Use `/skill-creator` to create, update, or prune a repository skill.
- Use `/find-skills` to audit the current catalog and discover a missing skill before adding one.
- The model loads `writing-for-agents` when it needs guidance on concise agent instructions.

Done when the user has one recommended next command and understands why it fits.
