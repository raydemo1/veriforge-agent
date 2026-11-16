import { useMemo } from "react";
import { useKeyboard } from "@opentui/react";
import type {
  Artifact,
  Check,
  PlanState,
  TasksState,
  WorkAgent,
  WorkFile,
  WorkState,
} from "../protocol.ts";
import type { Theme } from "../theme.ts";
import type { WorkbenchTab } from "./strip.ts";

const TABS: ReadonlyArray<{ key: WorkbenchTab; label: string }> = [
  { key: "plan", label: "计划" },
  { key: "tasks", label: "任务" },
  { key: "changes", label: "变更" },
  { key: "agents", label: "代理" },
  { key: "checks", label: "检查" },
  { key: "artifacts", label: "产物" },
];

const AGENT_ROLE_LABELS: Record<string, string> = {
  explorer: "探索者",
  test_designer: "测试设计",
  reviewer: "审查者",
  verifier: "验证者",
  worker: "执行者",
};

const AGENT_STATUS: Record<string, { label: string; tone: "accent" | "success" | "error" | "warning" | "muted" }> = {
  queued: { label: "等待中", tone: "muted" },
  running: { label: "运行中", tone: "accent" },
  completed: { label: "完成", tone: "success" },
  failed: { label: "失败", tone: "error" },
  blocked: { label: "阻塞", tone: "warning" },
  interrupted: { label: "已中断", tone: "muted" },
};

/** Cell columns the string occupies (CJK/fullwidth glyphs are double-width). */
function displayWidth(text: string): number {
  let width = 0;
  for (const char of text) {
    const code = char.codePointAt(0) ?? 0;
    width += (
      (code >= 0x1100 && code <= 0x115f)
      || (code >= 0x2e80 && code <= 0x303e)
      || (code >= 0x3041 && code <= 0x33ff)
      || (code >= 0x3400 && code <= 0x4dbf)
      || (code >= 0x4e00 && code <= 0x9fff)
      || (code >= 0xa000 && code <= 0xa4cf)
      || (code >= 0xac00 && code <= 0xd7a3)
      || (code >= 0xf900 && code <= 0xfaff)
      || (code >= 0xfe30 && code <= 0xfe4f)
      || (code >= 0xff00 && code <= 0xff60)
      || (code >= 0xffe0 && code <= 0xffe6)
    ) ? 2 : 1;
  }
  return width;
}

function toneColor(theme: Theme, tone: "accent" | "success" | "error" | "warning" | "muted"): string {
  if (tone === "success") return theme.success;
  if (tone === "warning") return theme.warning;
  if (tone === "error") return theme.error;
  if (tone === "accent") return theme.accent;
  return theme.subtle;
}

// ---------------------------------------------------------------------------
// Small building blocks
// ---------------------------------------------------------------------------

function EmptyState({ text, theme }: { text: string; theme: Theme }) {
  return <text fg={theme.subtle}>{text}</text>;
}

function FileStats({ file, theme }: { file: WorkFile; theme: Theme }) {
  return (
    <box style={{ flexDirection: "row", gap: 1 }}>
      <text fg={theme.subtle}>{file.path}</text>
      <text fg={theme.diffAdd}>{file.additions ? `+${file.additions}` : ""}</text>
      <text fg={theme.diffDelete}>{file.deletions ? `-${file.deletions}` : ""}</text>
    </box>
  );
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

function PlanView({ plan, theme }: { plan: PlanState; theme: Theme }) {
  if (!plan) return <EmptyState text="暂无计划" theme={theme} />;
  const statusLabel = plan.status === "ready"
    ? "就绪"
    : plan.status === "executing"
      ? "执行中"
      : plan.status === "incomplete"
        ? "未完成"
        : "已完成";
  const statusTone = plan.status === "ready"
    ? "warning"
    : plan.status === "executing"
      ? "accent"
      : plan.status === "incomplete"
        ? "warning"
        : "success";
  return (
    <box style={{ flexDirection: "column", gap: 1 }}>
      <text fg={toneColor(theme, statusTone)}>{`计划 rev ${plan.revision} · ${statusLabel}`}</text>
      <text fg={theme.subtle}>{plan.path}</text>
      {plan.status === "ready" ? (
        <text fg={theme.muted}>回复「继续」执行，或直接发送修改意见</text>
      ) : null}
      {plan.steps.length ? (
        <box style={{ flexDirection: "column" }}>
          {plan.steps.map((step, index) => (
            <text
              key={index}
              fg={step.status === "completed" ? theme.success : step.status === "in_progress" ? theme.accent : theme.muted}
            >
              {`${step.status === "completed" ? "✓" : step.status === "in_progress" ? "›" : "·"} ${step.text}`}
            </text>
          ))}
        </box>
      ) : <EmptyState text="计划中没有可解析的步骤" theme={theme} />}
    </box>
  );
}

function TasksView({ tasks, theme }: { tasks: TasksState; theme: Theme }) {
  if (!tasks.items.length) return <EmptyState text="暂无任务" theme={theme} />;
  return (
    <box style={{ flexDirection: "column", gap: 1 }}>
      <text fg={theme.subtle}>{`${tasks.completed}/${tasks.total} 完成`}</text>
      <box style={{ flexDirection: "column" }}>
        {tasks.items.map((item) => {
          const marker = item.status === "completed" ? "✓" : item.status === "in_progress" ? "›" : item.status === "cancelled" ? "~" : "·";
          const color = item.status === "completed"
            ? theme.success
            : item.status === "in_progress"
              ? theme.accent
              : item.status === "cancelled"
                ? theme.subtle
                : theme.muted;
          return (
            <text key={item.id || item.text} fg={color}>
              {`${marker} ${item.text}`}
            </text>
          );
        })}
      </box>
    </box>
  );
}

function ChangesView({ changes, theme }: {
  changes: WorkState["changes"];
  theme: Theme;
}) {
  if (!changes.workspace.length) {
    return <EmptyState text="暂无变更" theme={theme} />;
  }

  return (
    <box style={{ flexDirection: "column", gap: 1 }}>
      <box style={{ flexDirection: "column" }}>
        <text fg={theme.muted}>{`工作区 · ${changes.workspace.length} 文件 · +${changes.additions} -${changes.deletions}`}</text>
        {changes.workspace.map((file) => <FileStats key={file.path} file={file} theme={theme} />)}
      </box>
    </box>
  );
}

function AgentsView({ agents, theme }: { agents: WorkAgent[]; theme: Theme }) {
  if (!agents.length) return <EmptyState text="暂无代理活动" theme={theme} />;
  return (
    <box style={{ flexDirection: "column", gap: 2 }}>
      {agents.map((agent) => {
        const status = AGENT_STATUS[agent.status] ?? AGENT_STATUS.queued;
        const marker = agent.status === "completed"
          ? "✓"
          : agent.status === "failed"
            ? "×"
            : agent.status === "blocked"
              ? "!"
              : agent.status === "running"
                ? "›"
                : "·";
        return (
          <box key={agent.id} style={{ flexDirection: "column", gap: 1 }}>
            <box style={{ flexDirection: "row", justifyContent: "space-between" }}>
              <text fg={theme.text}>{`${marker} ${agent.name}`}</text>
              <text fg={toneColor(theme, status.tone)}>
                {agent.status === "running" ? `${status.label}…` : status.label}
              </text>
            </box>
            {agent.task ? <text fg={theme.muted} style={{ paddingLeft: 2 }}>{agent.task}</text> : null}
            {agent.summary ? <text fg={theme.subtle} style={{ paddingLeft: 2 }}>{agent.summary}</text> : null}
            {agent.error ? <text fg={theme.error} style={{ paddingLeft: 2 }}>{`× ${agent.error}`}</text> : null}
            <text fg={theme.subtle} style={{ paddingLeft: 2 }}>
              {`${AGENT_ROLE_LABELS[agent.role] ?? agent.role} · ${agent.isolation}${agent.durationSeconds != null ? ` · ${agent.durationSeconds}s` : ""}`}
            </text>
          </box>
        );
      })}
    </box>
  );
}

function ChecksView({ checks, theme }: { checks: Check[]; theme: Theme }) {
  if (!checks.length) return <EmptyState text="暂无检查" theme={theme} />;
  return (
    <box style={{ flexDirection: "column" }}>
      {checks.map((check) => {
        const marker = check.status === "passed"
          ? "✓"
          : check.status === "failed"
            ? "×"
            : check.status === "warning"
              ? "!"
              : check.status === "skipped" ? "–" : "›";
        const color = check.status === "passed"
          ? theme.success
          : check.status === "failed"
            ? theme.error
            : check.status === "warning"
              ? theme.warning
              : check.status === "skipped" ? theme.subtle : theme.accent;
        return (
          <box key={check.id} style={{ flexDirection: "row", gap: 1 }}>
            <text fg={color}>{marker}</text>
            <text fg={theme.text}>{check.name}</text>
            <text fg={theme.subtle}>{check.status === "running" ? "运行中…" : check.status === "skipped" ? `已跳过 · ${check.detail}` : check.detail}</text>
          </box>
        );
      })}
    </box>
  );
}

function ArtifactsView({ artifacts, theme }: { artifacts: Artifact[]; theme: Theme }) {
  if (!artifacts.length) return <EmptyState text="暂无产物" theme={theme} />;
  return (
    <box style={{ flexDirection: "column", gap: 1 }}>
      {artifacts.map((artifact) => (
        <box key={artifact.id} style={{ flexDirection: "column" }}>
          <box style={{ flexDirection: "row", gap: 1 }}>
            <text fg={theme.accent}>{artifact.kind === "image" ? "+" : "·"}</text>
            <text fg={theme.text}>{`[${artifact.kind === "image" ? "图片" : "文本"}] ${artifact.title}`}</text>
          </box>
          {artifact.detail ? <text fg={theme.muted}>{artifact.detail}</text> : null}
          <text fg={theme.subtle}>{artifact.path}</text>
        </box>
      ))}
    </box>
  );
}

// ---------------------------------------------------------------------------
// Overlay shell
// ---------------------------------------------------------------------------

function tabBadge(tab: WorkbenchTab, work: WorkState): string {
  switch (tab) {
    case "plan":
      return work.plan
        ? (work.plan.status === "ready" ? String(work.plan.totalCount) : `${work.plan.completedCount}/${work.plan.totalCount}`)
        : "";
    case "tasks":
      return work.tasks.total ? `${work.tasks.completed}/${work.tasks.total}` : "";
    case "changes": {
      return work.changes.workspace.length ? String(work.changes.workspace.length) : "";
    }
    case "agents":
      return work.agents.length ? String(work.agents.length) : "";
    case "checks":
      return work.checks.length
        ? `${work.checks.filter((check) => check.status === "passed").length}/${work.checks.length}`
        : "";
    case "artifacts":
      return work.artifacts.length ? String(work.artifacts.length) : "";
  }
}

export function Workbench({ work, theme, terminalWidth, terminalHeight, initialTab, onTabChange, onClose }: {
  work: WorkState;
  theme: Theme;
  terminalWidth: number;
  terminalHeight: number;
  initialTab: WorkbenchTab;
  onTabChange?: (tab: WorkbenchTab) => void;
  onClose: () => void;
}) {
  const tab = initialTab;
  const setTab = (value: WorkbenchTab) => onTabChange?.(value);

  useKeyboard((key) => {
    if (key.name === "escape") {
      key.preventDefault();
      onClose();
      return;
    }
    if (key.name === "left" || key.name === "right") {
      key.preventDefault();
      const index = TABS.findIndex((item) => item.key === tab);
      const next = key.name === "left"
        ? TABS[(index - 1 + TABS.length) % TABS.length]
        : TABS[(index + 1) % TABS.length];
      setTab(next.key);
      return;
    }
    const digit = Number(key.name);
    if (digit >= 1 && digit <= TABS.length) {
      key.preventDefault();
      setTab(TABS[digit - 1].key);
    }
  });

  const modalWidth = Math.min(110, Math.max(36, terminalWidth - 6));
  const modalHeight = Math.min(24, Math.max(10, terminalHeight - 2));

  // Narrow terminals: window the tab row, keeping the selected tab visible.
  // Widths are cell columns: CJK labels occupy two columns each.
  const innerWidth = modalWidth - 2;
  const tabWidths = TABS.map((item) => {
    const badge = tabBadge(item.key, work);
    return displayWidth(item.label) + (badge ? 1 + badge.length : 0) + 3; // padding 2 + gap 1
  });
  const selectedIndex = TABS.findIndex((item) => item.key === tab);
  const buildShown = (offset: number) => {
    const items: Array<{ item: typeof TABS[number]; index: number }> = [];
    // Reserve columns for the ‹ / › window markers.
    let budget = innerWidth - (offset > 0 ? 2 : 0) - 2;
    for (let index = offset; index < TABS.length; index += 1) {
      if (budget - tabWidths[index] < 0) break;
      budget -= tabWidths[index];
      items.push({ item: TABS[index], index });
    }
    return items;
  };
  let shown = buildShown(0);
  let tabOffsetCalc = 0;
  // Keep the selected tab inside the windowed row (e.g. digit jump on narrow terminals).
  if (!shown.some(({ index }) => index === selectedIndex)) {
    tabOffsetCalc = selectedIndex;
    shown = buildShown(tabOffsetCalc);
  }
  const tabOffset = tabOffsetCalc;
  const moreAfter = tabOffset + shown.length < TABS.length;

  const view = useMemo(() => {
    switch (tab) {
      case "plan":
        return <PlanView plan={work.plan} theme={theme} />;
      case "tasks":
        return <TasksView tasks={work.tasks} theme={theme} />;
      case "changes":
        return <ChangesView changes={work.changes} theme={theme} />;
      case "agents":
        return <AgentsView agents={work.agents} theme={theme} />;
      case "checks":
        return <ChecksView checks={work.checks} theme={theme} />;
      case "artifacts":
        return <ArtifactsView artifacts={work.artifacts} theme={theme} />;
      default:
        return null;
    }
  }, [tab, work, theme]);

  return (
    <box
      position="absolute"
      top={0}
      left={0}
      width="100%"
      height="100%"
      zIndex={25}
      onMouseDown={onClose}
      style={{ width: "100%", height: "100%" }}
    >
      <box
        position="absolute"
        top={Math.max(0, Math.floor((terminalHeight - modalHeight) / 2))}
        left={Math.max(0, Math.floor((terminalWidth - modalWidth) / 2))}
        border
        borderStyle="rounded"
        borderColor={theme.border}
        onMouseDown={(event) => event.stopPropagation()}
        style={{ width: modalWidth, height: modalHeight, flexDirection: "column", backgroundColor: theme.surface }}
      >
        <box style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between", height: 1, paddingLeft: 1, paddingRight: 1, backgroundColor: theme.surfaceRaised }}>
          <text fg={theme.accent}><strong>工作台</strong></text>
          {innerWidth >= 42 ? <text fg={theme.subtle}>← → 切换 · 1-6 直达 · Esc 关闭</text> : <text fg={theme.subtle}>Esc</text>}
        </box>
        <box style={{ flexDirection: "row", height: 1, paddingLeft: 1, gap: 1 }}>
          {tabOffset > 0 ? <text fg={theme.subtle}>‹</text> : null}
          {shown.map(({ item }) => {
            const active = item.key === tab;
            const badge = tabBadge(item.key, work);
            return (
              <box
                key={item.key}
                focusable
                onMouseDown={() => setTab(item.key)}
                style={{ flexDirection: "row", paddingLeft: 1, paddingRight: 1, backgroundColor: active ? theme.accentSoft : theme.surface }}
              >
                <text fg={active ? theme.accent : theme.muted}>
                  {`${item.label}${badge ? ` ${badge}` : ""}`}
                </text>
              </box>
            );
          })}
          {moreAfter ? <text fg={theme.subtle}>›</text> : null}
        </box>
        <scrollbox style={{ flexGrow: 1, flexShrink: 1, minHeight: 0, paddingLeft: 1, paddingRight: 1, paddingTop: 1 }}>
          {view}
        </scrollbox>
      </box>
    </box>
  );
}
