import React, { useEffect, useRef, useState } from "react";
import { stringWidth } from "bun";
import type { BoxRenderable, ScrollBoxRenderable } from "@opentui/core";
import type { SessionDescriptor } from "./protocol.ts";
import type { Theme } from "./theme.ts";

function fit(text: string, columns: number) {
  if (stringWidth(text) <= columns) return text;
  let result = "";
  for (const character of text) {
    if (stringWidth(result + character) > columns - 1) break;
    result += character;
  }
  return result + "…";
}

function Action({ children, onClick, theme, accent = false }: {
  children: string; onClick: () => void; theme: Theme; accent?: boolean;
}) {
  const [hover, setHover] = useState(false);
  const node = useRef<BoxRenderable | null>(null);
  return <box ref={node} focusable onMouseOver={() => setHover(true)} onMouseOut={() => setHover(false)}
    onMouseDown={() => { onClick(); setTimeout(() => node.current?.blur(), 0); }} onKeyDown={(key) => { if (key.name === "return") onClick(); }}
    style={{ paddingLeft: 1, paddingRight: 1, backgroundColor: hover ? theme.surfaceSelected : theme.background }}>
    <text fg={accent ? theme.accent : hover ? theme.text : theme.muted}>{children}</text>
  </box>;
}

export function SessionTabs({ sessions, activeId, unread, width, theme, onSelect, onNew, onHide,
  onSwitcher, onWorktree, onFork, onApply }: {
  sessions: SessionDescriptor[]; activeId: string; unread: Record<string, number>; width: number; theme: Theme;
  onSelect: (id: string) => void; onNew: () => void; onHide: (id: string) => void;
  onSwitcher: () => void; onWorktree: () => void; onFork: () => void; onApply?: () => void;
}) {
  const scroll = useRef<ScrollBoxRenderable | null>(null);
  const narrow = width < 76;
  const active = sessions.find((item) => item.id === activeId);
  const pending = sessions.filter((item) => item.pendingApproval).length;
  useEffect(() => { scroll.current?.scrollChildIntoView(`session-${activeId}`); }, [activeId, sessions.length, width]);
  const path = active?.cwd.split(/[\\/]/).filter(Boolean).at(-1) ?? "工作台";
  return <box style={{ flexDirection: "column", flexShrink: 0, backgroundColor: theme.background }}>
    <box style={{ flexDirection: "row", justifyContent: "space-between", paddingLeft: 2, paddingRight: 1, paddingTop: 1 }}>
      <text><span fg={theme.accent}><strong>VeriForge</strong></span><span fg={theme.subtle}>{`  /  ${fit(path, narrow ? 16 : 28)}`}</span></text>
      <box style={{ flexDirection: "row" }}>
        {onApply ? <Action theme={theme} accent onClick={onApply}>应用到 Local</Action> : null}
        {!narrow ? <>
          <Action theme={theme} onClick={onFork}>分支</Action>
          <Action theme={theme} onClick={onWorktree}>隔离工作区</Action>
        </> : null}
        <Action theme={theme} onClick={onSwitcher}>{pending ? (narrow ? `◆ ${pending} ▾` : `会话 ${pending} 待审批 ▾`) : "会话 ▾"}</Action>
        <Action theme={theme} accent onClick={onNew}>＋ 新建</Action>
      </box>
    </box>
    <scrollbox ref={scroll} scrollX scrollY={false} horizontalScrollbarOptions={{ visible: false }} verticalScrollbarOptions={{ visible: false }} style={{ height: 2, flexShrink: 0, paddingLeft: 1, paddingRight: 1 }}>
      <box style={{ flexDirection: "row", gap: 1, height: 2 }}>
        {(narrow && active ? [active] : sessions).map((session) => {
          const selected = session.id === activeId;
          const busy = ["running", "queued", "cancelling", "starting"].includes(session.status);
          const tone = session.pendingApproval ? theme.warning : session.status === "failed" ? theme.error : busy ? theme.accent : theme.subtle;
          const title = fit(session.title, narrow ? Math.max(12, width - 24) : 20);
          const count = (unread[session.id] ?? 0) > 0 ? ` ${unread[session.id]}` : "";
          const tabWidth = stringWidth(title + count) + 4 + (selected ? 2 : 0) + (session.environment === "worktree" ? 2 : 0);
          return <box key={session.id} id={`session-${session.id}`} focusable
            onMouseDown={() => onSelect(session.id)}
            onKeyDown={(key) => { if (key.name === "return") onSelect(session.id); }}
            border={["bottom"]} borderColor={selected ? theme.accent : theme.border}
            style={{ flexDirection: "row", alignItems: "flex-start", height: 2, flexShrink: 0, width: tabWidth, paddingLeft: 1, paddingRight: 1,
              backgroundColor: selected ? theme.surfaceSelected : theme.background }}>
            <text fg={tone} style={{ height: 1, flexShrink: 0 }}>{session.pendingApproval ? "◆ " : busy ? "● " : "· "}</text>
            <text fg={selected ? theme.text : theme.muted} style={{ height: 1, flexShrink: 0 }}><strong>{title}</strong></text>
            {session.environment === "worktree" ? <text fg={theme.subtle} style={{ height: 1, flexShrink: 0 }}> ↗</text> : null}
            {count ? <text fg={theme.warning} style={{ height: 1, flexShrink: 0 }}>{count}</text> : null}
            {selected ? <box focusable onMouseDown={(event) => { event.stopPropagation(); onHide(session.id); }}
              onKeyDown={(key) => { if (key.name === "return") { key.stopPropagation(); onHide(session.id); } }}
              style={{ paddingLeft: 1, flexShrink: 0 }}><text fg={theme.subtle}>×</text></box> : null}
          </box>;
        })}
      </box>
    </scrollbox>
  </box>;
}
