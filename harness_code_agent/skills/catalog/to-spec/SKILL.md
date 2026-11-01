---
name: to-spec
description: Turn settled conversation and repository context into a reviewable spec. Use when a request is understood and needs a written specification.
argument-hint: "<request, conversation, or source reference>"
disable-model-invocation: true
---

# To Spec

Synthesize what is already known. Ask only when a missing decision would materially change scope, safety, or acceptance.

## Process

1. Read the request and any referenced discussion, issue, document, or code. Explore the repository only as far as needed to distinguish current behavior from the requested outcome. Use existing domain vocabulary and respect relevant ADRs.

2. Identify observable acceptance checks and the highest practical test seams. Prefer existing public seams; propose new seams only when necessary.

3. Draft a specification proportional to the work. Include only sections that add decision value:

- problem and desired outcome;
- scope and explicit non-goals;
- observable acceptance criteria;
- user stories when they clarify actors or outcomes;
- implementation and testing decisions already settled;
- risks, open questions, and dependencies.

Avoid volatile file paths and code snippets unless they encode a durable contract more precisely than prose.

4. Show the draft and call out assumptions. Do not publish to an external tracker until the user confirms. If local persistence was requested, save it using existing repository conventions.

5. When external publication is explicitly requested, use the configured tracker contract. If none exists, explain that `/setup-workflows` configures it instead of guessing or creating labels.

Done when the specification preserves the decisions needed by a fresh implementation session without turning implementation logs into product requirements.
