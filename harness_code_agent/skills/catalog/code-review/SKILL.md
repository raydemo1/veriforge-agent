---
name: code-review
description: Review code changes for correctness, repository standards, and spec conformance. Use when reviewing a branch, PR, diff, uncommitted change, or file range.
---

Review the change scope the user supplies. A commit, branch, tag, or merge-base is one valid scope; staged, unstaged, untracked, or path-scoped changes are equally valid.

Use two axes when both have evidence:

- **Standards**: does the code conform to this repo's documented coding standards?
- **Spec**: does the code faithfully implement the originating issue / spec?

For a medium or large change, run the axes as independent parallel sub-agents when delegation is available and authorized. Review a small localized diff directly; extra agents should improve independence or coverage, not add ceremony.

Issue-tracker configuration is optional. Use it when present, but never require `/setup-workflows` merely to review code.

## Process

### 1. Pin the review scope

Use the scope the user named. If they ask to review current or recent work without a fixed point, inspect `git status` and review the narrowest evident scope: `git diff HEAD` for tracked working-tree changes, plus relevant untracked files. Path arguments further narrow the scope.

For a supplied fixed point, capture `git diff <fixed-point>...HEAD` and `git log <fixed-point>..HEAD --oneline`. For working-tree review, capture the commands and file list that cover staged, unstaged, and relevant untracked changes. If two plausible scopes would materially change the review, ask once; otherwise state the chosen scope and proceed.

Confirm any supplied ref resolves and that the chosen scope is non-empty. A bad ref or empty scope should fail here, not inside a reviewer.

### 2. Identify the spec source

Look for an originating spec in this order:

1. Issue references in the commit messages (`#123`, `Closes #45`, GitLab `!67`, etc.), fetched via the workflow in `docs/agents/issue-tracker.md`.
2. A path the user passed as an argument.
3. A spec file under `docs/`, `specs/`, or `.scratch/` matching the branch name or feature.
4. A requirement or acceptance criteria stated directly in the conversation.

If no spec exists, skip the Spec axis and say so. Ask for a spec only when the user explicitly requests conformance review and the missing source prevents it.

### 3. Identify the standards sources

Anything in the repo that documents how code should be written, such as `CODING_STANDARDS.md` or `CONTRIBUTING.md`.

On top of whatever the repo documents, the Standards axis always carries the **smell baseline** below: a fixed set of Fowler code smells (_Refactoring_, ch.3) that applies even when a repo documents nothing. Two rules bind it:

- **The repo overrides.** A documented repo standard always wins; where it endorses something the baseline would flag, suppress the smell.
- **Always a judgement call.** Each smell is a labelled heuristic ("possible Feature Envy"), never a hard violation. Like any standard here, skip anything tooling already enforces.

Each smell reads *what it is* → *how to fix*; match it against the diff:

- **Mysterious Name**: a function, variable, or type whose name doesn't reveal what it does or holds. → rename it; if no honest name comes, the design's murky.
- **Duplicated Code**: the same logic shape appears in more than one hunk or file in the change. → extract the shared shape, call it from both.
- **Feature Envy**: a method that reaches into another object's data more than its own. → move the method onto the data it envies.
- **Data Clumps**: the same few fields or params keep travelling together (a type wanting to be born). → bundle them into one type, pass that.
- **Primitive Obsession**: a primitive or string standing in for a domain concept that deserves its own type. → give the concept its own small type.
- **Repeated Switches**: the same `switch`/`if`-cascade on the same type recurs across the change. → replace with polymorphism, or one map both sites share.
- **Shotgun Surgery**: one logical change forces scattered edits across many files in the diff. → gather what changes together into one module.
- **Divergent Change**: one file or module is edited for several unrelated reasons. → split so each module changes for one reason.
- **Speculative Generality**: abstraction, parameters, or hooks added for needs the spec doesn't have. → delete it; inline back until a real need shows.
- **Message Chains**: long `a.b().c().d()` navigation the caller shouldn't depend on. → hide the walk behind one method on the first object.
- **Middle Man**: a class or function that mostly just delegates onward. → cut it, call the real target direct.
- **Refused Bequest**: a subclass or implementer that ignores or overrides most of what it inherits. → drop the inheritance, use composition.

### 4. Choose review execution

For small changes, apply the relevant checks directly. For medium or large changes, spawn the Standards and Spec reviewers in parallel when both axes apply; if the spec is absent, run only Standards/correctness. Preserve independent evidence between axes.

When using sub-agents:

**Standards sub-agent prompt** should include:

- The full diff command and commit list.
- The list of standards-source files you found in step 3, **plus the smell baseline from step 3** pasted in full (the sub-agent has no other access to it).
- The brief: "Report, per file/hunk where relevant, (a) every place the diff violates a documented standard: cite the standard (file + the rule); and (b) any baseline smell you spot: name it and quote the hunk. Distinguish hard violations from judgement calls: documented-standard breaches can be hard, but baseline smells are always judgement calls, and a documented repo standard overrides the baseline. Skip anything tooling enforces. Under 400 words."

**Spec sub-agent prompt** should include:

- The diff command and commit list.
- The path or fetched contents of the spec.
- The brief: "Report: (a) requirements the spec asked for that are missing or partial; (b) behaviour in the diff that wasn't asked for (scope creep); (c) requirements that look implemented but where the implementation looks wrong. Quote the spec line for each finding. Under 400 words."

If the spec is missing, skip the Spec sub-agent and note this in the final report.

### 5. Aggregate

Lead with actionable findings ordered by severity, with file and line references where possible. Label each finding as Standards, Spec, or both so the evidence axes remain visible without forcing the user to read two repetitive reports.

End with a concise scope and verification summary. If there are no actionable findings, say so and name any residual testing or evidence gaps.

## Why two axes

A change can pass one axis and fail the other:

- Code that follows every standard but implements the wrong thing → **Standards pass, Spec fail.**
- Code that does exactly what the issue asked but breaks the project's conventions → **Spec pass, Standards fail.**

Reporting them separately stops one axis from masking the other.
