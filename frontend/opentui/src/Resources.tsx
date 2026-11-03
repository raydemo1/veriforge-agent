import React from "react";
import type { ProjectResourcesSnapshot } from "./protocol.ts";
import type { Theme } from "./theme.ts";

export function Resources({ snapshot, theme, height, onSelect }: {
  snapshot: ProjectResourcesSnapshot | null; theme: Theme; height: number; onSelect: (id: string) => void;
}) {
  if (!snapshot) return null;
  const { sessions, budget } = snapshot;
  const llm = budget.active.filter((entry) => entry.resource === "llm").length;
  const totals = [
    `${sessions.filter((session) => session.state === "active").length} 活动`,
    `${sessions.filter((session) => session.turn).length} 执行中`,
    `${sessions.reduce((sum, session) => sum + session.jobs.length, 0)} 后台`,
    `${sessions.reduce((sum, session) => sum + session.approvals, 0)} 待审批`,
    `LLM ${llm}/${budget.limits.llm}`,
  ];
  return <box border={["bottom"]} borderColor={theme.border}
    style={{ flexDirection: "column", flexShrink: 0, maxHeight: Math.max(5, Math.floor(height / 3)), paddingLeft: 2, paddingRight: 2, paddingBottom: 1 }}>
    <box style={{ flexDirection: "row", flexWrap: "wrap", gap: 2 }}>
      {totals.map((total) => <text key={total} fg={theme.muted}>{total}</text>)}
    </box>
    <scrollbox style={{ maxHeight: Math.max(2, Math.floor(height / 3) - 3), marginTop: 1 }}>
      {sessions.map((session) => {
        const status = session.waitingReason ?? (session.approvals ? "等待审批" : session.state === "active" ? "运行中" : session.state === "warm" ? "就绪" : "休眠");
        const tone = session.waitingReason || session.approvals ? theme.warning : session.state === "active" ? theme.accent : theme.subtle;
        const details = [session.agents ? `${session.agents} 子代理` : "", session.queue ? `${session.queue} 排队` : ""].filter(Boolean);
        return <box key={session.id} style={{ flexDirection: "column", marginBottom: 1 }}>
          <box focusable onMouseDown={() => onSelect(session.id)}
            onKeyDown={(key) => { if (key.name === "return") onSelect(session.id); }}
            style={{ flexDirection: "row", justifyContent: "space-between", gap: 2 }}>
            <text fg={theme.text} style={{ flexShrink: 1 }}><span fg={tone}>● </span>{session.title}</text>
            <text fg={tone}>{[...details, status].join(" · ")}</text>
          </box>
          {session.jobs.map((job) => <text key={job.id} fg={theme.muted} style={{ paddingLeft: 2 }}>
            {`↳ ${job.pid}  ${job.status === "stopping" ? "停止中  " : ""}${job.rssMiB} MB · CPU ${job.cpuSeconds}s  ${job.command.split(/\r?\n/)[0]}`}
          </text>)}
          {session.lastError ? <text fg={theme.error} style={{ paddingLeft: 2 }}>{session.lastError}</text> : null}
        </box>;
      })}
    </scrollbox>
  </box>;
}
