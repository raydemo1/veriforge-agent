---
name: frontend-debugging
description: Browser-specific probes for frontend bugs. Use when a bug depends on rendering, CSS, browser state, timing, hydration, or UI automation.
---

# Frontend Diagnosis

`diagnosing-bugs` owns the overall loop. This skill supplies browser-specific probes.

## Classify the failing boundary

Use the tight red loop to distinguish:

- state or derived data;
- rendering or hydration;
- layout, CSS, or animation;
- browser API or event behavior;
- network timing;
- automation setup.

Move pure calculations and state transitions into deterministic tests. Keep the real browser in the loop for visual, focus, layout, pointer, and platform behavior.

## Probe one prediction

Prefer:

- DOM and computed-style assertions;
- console and network capture;
- fixed clocks, seeded randomness, and explicit animation frames;
- representative viewport and device-pixel-ratio checks;
- screenshots only when pixels are the actual contract.

Temporary browser state such as `window.__debug` must be development-only and removed before completion.

## Common signals

- Flicker: inspect transitions or animations on frequently updated elements.
- Stale UI: inspect closure capture, effect dependencies, memoization, cache keys, and component keys.
- Drift: run many deterministic frames or updates and compare against a known result.
- Browser-only failure: inspect unsupported APIs, focus, pointer events, measurements, viewport, and real network ordering.

Done when the reported UI symptom is reproduced, classified, caught by a stable check at the correct boundary, and no temporary diagnostics remain.
