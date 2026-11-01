---
name: frontend-design
description: Design and implement a distinctive web interface. Use when creating or substantially restyling a page, component, dashboard, landing page, or visual artifact.
license: Complete terms in LICENSE.txt
---

# Frontend Design

Create a working interface with one deliberate visual thesis, not a collection of familiar AI defaults.

## 1. Read the product

Establish:

- audience and task;
- required states and interactions;
- framework and repository constraints;
- accessibility and responsive expectations;
- existing visual language worth preserving.

Inspect the application before asking discoverable questions.

Done when the interface purpose and non-negotiable behavior are explicit.

## 2. Choose a thesis

Select one coherent direction—editorial, utilitarian, playful, austere, tactile, archival, cinematic, or another justified concept. Define the memorable device that expresses it.

Avoid default gradients, interchangeable card grids, excessive pills, generic hero layouts, and decoration without product meaning.

Done when typography, color, spacing, motion, and composition can all be judged against the same thesis.

## 3. Build the real interface

- Use the existing design system and dependencies when they fit.
- Preserve the product's information hierarchy.
- Implement real states and interactions, including loading, empty, error, focus, and disabled states where applicable.
- Use semantic controls, labels, keyboard access, visible focus, readable contrast, and reduced-motion behavior.
- Make responsive behavior intentional rather than merely stacked.
- Keep visual complexity proportional to the thesis: expressive designs need disciplined implementation; restrained designs need exact spacing and typography.

Do not add dependencies unless the chosen behavior cannot reasonably be built with the existing stack.

Done when the primary workflow is functional at desktop and mobile widths.

## 4. Verify the experience

Run the build/static checks and inspect the rendered page in a browser. Exercise representative clicks, typing, keyboard navigation, responsive widths, console output, and network failures.

Fix visible regressions rather than describing them away.

Done when the interface works, the thesis is visible without explanation, and browser verification covers its primary workflow.
