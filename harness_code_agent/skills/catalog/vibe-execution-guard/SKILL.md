---
name: vibe-execution-guard
description: Boundary-first execution for high-risk or tightly scoped changes. Use when the request limits exactly what may change, or touches auth, secrets, payments, privacy, destructive data, migrations, incidents, or repeatedly broken fixes.
---

# Guarded Execution

Lock the boundary, prove the cause, make the smallest reversible change.

## 1. Lock

State:

- target behavior;
- explicit exclusions;
- affected data or users;
- rollback point;
- minimum meaningful verification.

Ask only when a missing answer changes safety or user-visible behavior.

Done when an out-of-scope edit can be recognized before it happens.

## 2. Inspect

Read the real call path, nearby tests, configuration, and existing safety controls. Search before introducing helpers or policies.

For a bug, load `diagnosing-bugs` and establish a tight red-capable reproduction before theorizing.

Done when the evidence identifies the boundary and the failure mechanism, or the missing evidence is reported as a blocker.

## 3. Patch

Change only the smallest cause supported by evidence. Do not combine cleanup or architecture work with the guarded patch.

If the necessary scope expands, stop and obtain approval for the new boundary.

Done when every changed line supports the target behavior, its regression check, or required rollback safety.

## 4. Verify

Run the focused regression or integration check, then the strongest cheap check for the affected boundary. Remove temporary diagnostics.

Report:

```md
Target:
Locked scope:
Risk and rollback:
Verification:
Remaining risk:
```

Guarded execution is complete only when the target passes, exclusions remain untouched, rollback is still available, and unverified risk is explicit.
