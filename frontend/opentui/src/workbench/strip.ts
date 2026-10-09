import type { WorkState } from "../protocol.ts";

export type WorkbenchTab = "plan" | "tasks" | "changes" | "agents" | "checks" | "artifacts";
export type StripTone = "accent" | "success" | "warning" | "error" | "muted";
export type StripSegment = {
  key: WorkbenchTab;
  /** Full label, e.g. "计划"; dropped on narrow terminals. */
  label: string;
  /** Value, e.g. "4/6" or "4 变更". */
  text: string;
  tone: StripTone;
};

export function workSegments(work: WorkState): StripSegment[] {
  const segments: StripSegment[] = [];

  if (work.plan) {
    const total = work.plan.totalCount;
    if (work.plan.status === "ready") {
      segments.push({ key: "plan", label: "计划", text: total ? `${total} 步` : "待执行", tone: "warning" });
    } else if (work.plan.status === "executing") {
      segments.push({ key: "plan", label: "计划", text: total ? `${work.plan.completedCount}/${total}` : "执行中", tone: "accent" });
    } else if (work.plan.status === "incomplete") {
      segments.push({ key: "plan", label: "计划", text: total ? `${work.plan.completedCount}/${total}` : "未完成", tone: "warning" });
    } else {
      segments.push({ key: "plan", label: "计划", text: total ? `${total}/${total}` : "已完成", tone: "success" });
    }
  }

  const activeAgents = work.agents.filter((agent) =>
    agent.status === "running" || agent.status === "queued" || agent.status === "blocked",
  );
  if (activeAgents.length) {
    const tone: StripTone = activeAgents.some((agent) => agent.status === "blocked")
      ? "warning"
      : "accent";
    segments.push({ key: "agents", label: "", text: `${activeAgents.length} 代理`, tone });
  }

  const workspaceCount = work.changes.workspace.length;
  const proposalCount = work.changes.proposals.reduce((sum, item) => sum + item.files.length, 0);
  const changeCount = workspaceCount + proposalCount;
  if (changeCount) {
    segments.push({ key: "changes", label: "", text: `${changeCount} 变更`, tone: "muted" });
  }

  if (work.checks.length) {
    const passed = work.checks.filter((check) => check.status === "passed").length;
    const running = work.checks.some((check) => check.status === "running");
    const failed = work.checks.some((check) => check.status === "failed");
    const warned = work.checks.some((check) => check.status === "warning");
    const skipped = work.checks.filter((check) => check.status === "skipped").length;
    const tone: StripTone = failed ? "error" : warned ? "warning" : running ? "accent" : skipped ? "muted" : "success";
    const text = skipped === work.checks.length ? "已跳过" : `${passed}/${work.checks.length - skipped}${skipped ? ` · ${skipped} 跳过` : ""}`;
    segments.push({ key: "checks", label: "检查", text, tone });
  }

  if (work.artifacts.length) {
    segments.push({ key: "artifacts", label: "", text: `${work.artifacts.length} 产物`, tone: "muted" });
  }

  return segments;
}

/** Hide-before-wrap: drop segments as the terminal gets narrower. */
export function visibleSegments(segments: StripSegment[], width: number): StripSegment[] {
  if (width >= 96) return segments;
  if (width >= 70) return segments.filter((segment) => segment.key !== "agents" && segment.key !== "artifacts");
  // Narrow: keep plan/changes/checks, labels dropped at render time.
  return segments.filter((segment) => segment.key === "plan" || segment.key === "changes" || segment.key === "checks");
}
