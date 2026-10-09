import { useMemo, useState } from "react";
import { useKeyboard } from "@opentui/react";
import type {
  ActionResult,
  Artifact,
  Check,
  PlanState,
  TasksState,
  WorkAgent,
  WorkFile,
  WorkProposal,
  WorkState,
  WorkbenchParams,
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

function WButton({ label, color, theme, disabled = false, busy = false, onInvoke }: {
  label: string;
  color: string;
  theme: Theme;
  disabled?: boolean;
  busy?: boolean;
  onInvoke: () => void;
}) {
  const [hover, setHover] = useState(false);
  const active = hover && !disabled;
  return (
    <box
      focusable={!disabled}
      onMouseOver={() => setHover(true)}
      onMouseOut={() => setHover(false)}
      onMouseDown={() => { if (!disabled && !busy) onInvoke(); }}
      onKeyDown={(key) => {
        if (!disabled && !busy && (key.name === "return" || key.name === "kpenter" || key.name === "space")) onInvoke();
      }}
      style={{
        flexDirection: "row",
        paddingLeft: 1,
        paddingRight: 1,
        backgroundColor: disabled ? theme.surface : active ? theme.surfaceSelected : theme.surfaceRaised,
      }}
    >
      <text fg={disabled ? theme.subtle : color}>{busy ? "› …" : label}</text>
    </box>
  );
}

function FileStats({ file, theme, marker, indent }: {
  file: WorkFile;
  theme: Theme;
  marker?: string;
  indent?: boolean;
}) {
  return (
    <box style={{ flexDirection: "row", gap: 1, paddingLeft: indent ? 2 : 0 }}>
      {marker ? <text fg={theme.subtle}>{marker}</text> : null}
      <text fg={theme.subtle}>{file.path}</text>
      <text fg={theme.diffAdd}>{file.additions ? `+${file.additions}` : ""}</text>
      <text fg={theme.diffDelete}>{file.deletions ? `-${file.deletions}` : ""}</text>
    </box>
  );
}

function DiffText({ content, theme }: { content: string; theme: Theme }) {
  const allLines = content.split("\n");
  const lines = allLines.slice(0, 300);
  return (
    <box style={{ flexDirection: "column", paddingLeft: 2 }}>
      {lines.map((line, index) => {
        let color = theme.subtle;
        if (line.startsWith("@@")) color = theme.accent;
        else if (line.startsWith("+") && !line.startsWith("+++")) color = theme.diffAdd;
        else if (line.startsWith("-") && !line.startsWith("---")) color = theme.diffDelete;
        return <text key={index} fg={color}>{line || " "}</text>;
      })}
      {allLines.length > 300 ? <text fg={theme.subtle}>… 其余内容已省略</text> : null}
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

type ExpandedDiff = { kind: "proposal" | "conflict"; content: string };

function ProposalCard({ proposal, theme, busy, expanded, onToggleReview, onToggleConflict }: {
  proposal: WorkProposal;
  theme: Theme;
  busy: string | null;
  expanded: ExpandedDiff | undefined;
  onToggleReview: () => void;
  onToggleConflict: () => void;
}) {
  // Worker-centric observation language: the main agent owns integration,
  // so a live proposal waits for the main agent, never for user approval.
  const status = proposal.status === "ready"
    ? { text: "待主代理整合", tone: "accent" as const }
    : proposal.status === "conflict"
      ? { text: "冲突", tone: "warning" as const }
      : { text: "无效", tone: "error" as const };

  return (
    <box style={{ flexDirection: "column", gap: 1 }}>
      <box style={{ flexDirection: "row", justifyContent: "space-between" }}>
        <text fg={theme.text}>{proposal.agentName}</text>
        <text fg={toneColor(theme, status.tone)}>{status.text}</text>
      </box>
      <text fg={theme.subtle} style={{ paddingLeft: 2 }}>
        {`${proposal.files.length} 文件 · +${proposal.additions} -${proposal.deletions}`}
      </text>
      {proposal.files.map((file) => <FileStats key={file.path} file={file} theme={theme} marker="~" indent />)}
      {proposal.invalidReasons.length ? (
        <box style={{ flexDirection: "column", paddingLeft: 2 }}>
          {proposal.invalidReasons.map((reason, index) => (
            <text key={index} fg={theme.error}>{`! ${reason}`}</text>
          ))}
        </box>
      ) : null}
      <box style={{ flexDirection: "row", gap: 1, paddingLeft: 2 }}>
        <WButton
          label={expanded?.kind === "proposal" ? "收起差异" : "查看差异"}
          color={theme.muted}
          theme={theme}
          busy={busy === "read_proposal"}
          onInvoke={onToggleReview}
        />
        {proposal.conflict ? (
          <WButton
            label={expanded?.kind === "conflict" ? "收起冲突" : "查看冲突"}
            color={theme.warning}
            theme={theme}
            busy={busy === "read_conflict"}
            onInvoke={onToggleConflict}
          />
        ) : null}
      </box>
      {expanded ? <DiffText content={expanded.content} theme={theme} /> : null}
    </box>
  );
}

function ChangesView({ changes, theme, onWorkbench }: {
  changes: WorkState["changes"];
  theme: Theme;
  onWorkbench: (params: WorkbenchParams) => Promise<ActionResult | undefined>;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Record<string, ExpandedDiff>>({});

  if (!changes.workspace.length && !changes.proposals.length) {
    return <EmptyState text="暂无变更" theme={theme} />;
  }

  const toggleDiff = async (proposal: WorkProposal, op: WorkbenchParams, label: string, kind: "proposal" | "conflict") => {
    if (expanded[proposal.id]?.kind === kind) {
      setExpanded((current) => {
        const next = { ...current };
        delete next[proposal.id];
        return next;
      });
      return;
    }
    setBusy(label);
    try {
      const result = await onWorkbench(op);
      if (result?.content) {
        setExpanded((current) => ({ ...current, [proposal.id]: { kind, content: result.content ?? "" } }));
      }
    } finally {
      setBusy(null);
    }
  };

  return (
    <box style={{ flexDirection: "column", gap: 1 }}>
      {changes.workspace.length ? (
        <box style={{ flexDirection: "column" }}>
          <text fg={theme.muted}>{`工作区 · ${changes.workspace.length} 文件 · +${changes.additions} -${changes.deletions}`}</text>
          {changes.workspace.map((file) => <FileStats key={file.path} file={file} theme={theme} />)}
        </box>
      ) : null}
      {changes.proposals.length ? (
        <box style={{ flexDirection: "column", gap: 2 }}>
          <text fg={theme.muted}>{`执行者 · ${changes.proposals.length}`}</text>
          {changes.proposals.map((proposal) => (
            <ProposalCard
              key={proposal.id}
              proposal={proposal}
              theme={theme}
              busy={busy}
              expanded={expanded[proposal.id]}
              onToggleReview={() => void toggleDiff(proposal, { op: "read_proposal", proposalId: proposal.id }, "read_proposal", "proposal")}
              onToggleConflict={() => {
                if (!proposal.conflict) return;
                void toggleDiff(
                  proposal,
                  { op: "read_conflict", conflictId: proposal.conflict.id },
                  "read_conflict",
                  "conflict",
                );
              }}
            />
          ))}
        </box>
      ) : null}
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
      const count = work.changes.workspace.length
        + work.changes.proposals.reduce((sum, item) => sum + item.files.length, 0);
      return count ? String(count) : "";
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

export function Workbench({ work, theme, terminalWidth, terminalHeight, initialTab, onClose, onWorkbench }: {
  work: WorkState;
  theme: Theme;
  terminalWidth: number;
  terminalHeight: number;
  initialTab: WorkbenchTab;
  onClose: () => void;
  onWorkbench: (params: WorkbenchParams) => Promise<ActionResult | undefined>;
}) {
  const [tab, setTab] = useState<WorkbenchTab>(initialTab);

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
        return <ChangesView changes={work.changes} theme={theme} onWorkbench={onWorkbench} />;
      case "agents":
        return <AgentsView agents={work.agents} theme={theme} />;
      case "checks":
        return <ChecksView checks={work.checks} theme={theme} />;
      case "artifacts":
        return <ArtifactsView artifacts={work.artifacts} theme={theme} />;
      default:
        return null;
    }
    // onWorkbench identity changes each render; the view only needs
    // recomputation when the data or selected tab does.
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
