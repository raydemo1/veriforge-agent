---
name: implement
description: Implement a clearly scoped request, spec, or ticket. Use when the scope is settled and code changes should be made.
argument-hint: "<request, spec path, or ticket>"
disable-model-invocation: true
---

# Implement

Implement exactly the requested scope.

1. Read the user request, any referenced spec or ticket, and the applicable repository instructions.
2. State the observable target, exclusions, and verification approach. Resolve only decisions that materially block implementation.
3. Load `catalog/tdd/SKILL.md` with `read_skill_file` when test-first work is requested or justified by risk. Load `catalog/vibe-execution-guard/SKILL.md` when scope or risk needs a hard boundary.
4. Implement the smallest vertical slice through the real public interface.
5. Run focused checks during the change. Run broader checks when the change is broad, repository rules require them, or they are affordable and relevant.
6. For non-trivial, high-risk, or explicitly requested work, load `catalog/code-review/SKILL.md` and review the resulting diff. Otherwise perform a focused self-review.
7. Report changed behavior, verification evidence, and remaining issues. Commit or push only when the user requested it.

Done when the requested slice is observable through its public interface, its acceptance checks pass, and unrelated scope is untouched.
