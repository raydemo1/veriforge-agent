import { describe, expect, test } from "bun:test";
import { initialWorkState, type WorkState } from "../protocol.ts";
import { visibleSegments, workSegments } from "./strip.ts";

function work(overrides: Partial<WorkState>): WorkState {
  return { ...initialWorkState, ...overrides };
}

describe("work strip", () => {
  test("empty work state renders no segments", () => {
    expect(workSegments(initialWorkState)).toEqual([]);
  });

  test("plan status drives text and tone", () => {
    const ready = workSegments(work({
      plan: { status: "ready", revision: 1, path: "plan.md", steps: [], completedCount: 0, totalCount: 4 },
    }));
    expect(ready[0]).toMatchObject({ key: "plan", text: "4 步", tone: "warning" });

    const executing = workSegments(work({
      plan: { status: "executing", revision: 1, path: "plan.md", steps: [], completedCount: 2, totalCount: 6 },
    }));
    expect(executing[0]).toMatchObject({ text: "2/6", tone: "accent" });

    const completed = workSegments(work({
      plan: { status: "completed", revision: 1, path: "plan.md", steps: [], completedCount: 3, totalCount: 3 },
    }));
    expect(completed[0]).toMatchObject({ text: "3/3", tone: "success" });

    const incomplete = workSegments(work({
      plan: { status: "incomplete", revision: 1, path: "plan.md", steps: [], completedCount: 2, totalCount: 6 },
    }));
    expect(incomplete[0]).toMatchObject({ text: "2/6", tone: "warning" });
  });

  test("checks tone follows the worst status", () => {
    const passed = workSegments(work({
      checks: [{ id: "a", name: "A", status: "passed", detail: "" }],
    }));
    expect(passed.find((s) => s.key === "checks")).toMatchObject({ tone: "success", text: "1/1" });

    const running = workSegments(work({
      checks: [
        { id: "a", name: "A", status: "passed", detail: "" },
        { id: "b", name: "B", status: "running", detail: "" },
      ],
    }));
    expect(running.find((s) => s.key === "checks")).toMatchObject({ tone: "accent", text: "1/2" });

    const failed = workSegments(work({
      checks: [
        { id: "a", name: "A", status: "passed", detail: "" },
        { id: "b", name: "B", status: "failed", detail: "" },
      ],
    }));
    expect(failed.find((s) => s.key === "checks")).toMatchObject({ tone: "error" });
  });

  test("change count merges workspace files and proposal files", () => {
    const segments = workSegments(work({
      changes: {
        workspace: [{ path: "a.py", operation: "modify", additions: 1, deletions: 1 }],
        additions: 1,
        deletions: 1,
        proposals: [{
          id: "p1", agentId: "x", agentName: "w", status: "ready",
          files: [
            { path: "b.py", operation: "create", additions: 1, deletions: 0 },
            { path: "c.py", operation: "create", additions: 1, deletions: 0 },
          ],
          additions: 2, deletions: 0, invalidReasons: [], conflict: null,
        }],
      },
    }));
    expect(segments.find((s) => s.key === "changes")).toMatchObject({ text: "3 变更" });
  });

  test("skipped checks never look like passed evidence", () => {
    const skipped = { id: "b", name: "Ruff", status: "skipped" as const, detail: "missing" };
    expect(workSegments(work({ checks: [skipped] }))).toContainEqual({ key: "checks", label: "检查", text: "已跳过", tone: "muted" });
    const mixed = workSegments(work({ checks: [{ id: "a", name: "Python", status: "passed", detail: "" }, skipped] }));
    expect(mixed.find((s) => s.key === "checks")).toMatchObject({ text: "1/1 · 1 跳过", tone: "muted" });
  });

  test("narrow widths hide segments before wrapping", () => {
    const wide = work({
      plan: { status: "ready", revision: 1, path: "plan.md", steps: [], completedCount: 0, totalCount: 1 },
      agents: [{
        id: "a", name: "w", role: "worker", task: "", status: "running",
        isolation: "isolated workspace", summary: "", error: null, proposalId: null, durationSeconds: null,
      }],
      checks: [{ id: "a", name: "A", status: "running", detail: "" }],
      artifacts: [{ id: "x", kind: "image", path: "x.png", title: "x.png", detail: "" }],
    });
    const segments = workSegments(wide);
    expect(visibleSegments(segments, 100).map((s) => s.key)).toEqual(["plan", "agents", "checks", "artifacts"]);
    expect(visibleSegments(segments, 80).map((s) => s.key)).toEqual(["plan", "checks"]);
    expect(visibleSegments(segments, 40).map((s) => s.key)).toEqual(["plan", "checks"]);
  });
});
