import React, { useEffect, useMemo, useReducer, useRef, useState } from "react";
import type { ReactNode } from "react";
import { stringWidth } from "bun";
import { CliRenderEvents, SyntaxStyle } from "@opentui/core";
import type { BoxRenderable, ClipboardReadResult, InputRenderable, ScrollBoxRenderable, TextareaRenderable } from "@opentui/core";
import { useKeyboard, useRenderer, useTerminalDimensions } from "@opentui/react";
import { resolveIcons } from "./icons.ts";
import type { IconSet } from "./icons.ts";
import { formatUserError } from "./errors.ts";
import { clipboardFilePaths, formatBytes } from "./attachment-utils.ts";
import type { ActionName, ActionResult, AttachmentItem, CommandItem, IconPreference, Interaction, PanelOption, PanelSpec, SubmitResult, ThemePreference, TranscriptItem, TurnSubmission, UiEvent } from "./protocol.ts";
import { initialState, reduceEvent, withOptimisticUserMessage } from "./state.ts";
import { resolveTheme } from "./theme.ts";
import type { Theme, ThemeMode } from "./theme.ts";

type AppProps = {
  events?: AsyncIterable<UiEvent>;
  onSubmit?: (submission: TurnSubmission) => Promise<SubmitResult>;
  onCancel?: () => void;
  onExit?: () => void;
  onAction?: (name: ActionName, params?: Record<string, unknown>) => Promise<ActionResult>;
  onResolveInteraction?: (id: string, result: Record<string, unknown>) => Promise<void>;
  onPickFiles?: (inputMode: "text" | "multimodal") => Promise<string[]>;
  onReadClipboard?: () => Promise<ClipboardReadResult>;
  onCopyText?: (text: string) => Promise<boolean>;
  initialTask?: string;
  themePreference?: ThemePreference;
  iconPreference?: IconPreference;
};

function isEnterKey(name: string): boolean {
  return name === "return" || name === "kpenter" || name === "enter";
}

function commandMatches(command: CommandItem, query: string): boolean {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return true;
  const haystack = `${command.name} ${command.description}`.toLowerCase();
  if (haystack.startsWith(normalized) || command.name.toLowerCase().startsWith(normalized)) return true;
  let cursor = 0;
  for (const char of normalized) {
    cursor = haystack.indexOf(char, cursor);
    if (cursor < 0) return false;
    cursor += 1;
  }
  return true;
}

const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

function Spinner({ color }: { color: string }) {
  const [frame, setFrame] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setFrame((value) => (value + 1) % SPINNER_FRAMES.length), 180);
    return () => clearInterval(timer);
  }, []);
  return <text fg={color}>{SPINNER_FRAMES[frame]}</text>;
}

function markerFor(item: TranscriptItem, icons: IconSet): string {
  if (item.state === "success") return icons.success;
  if (item.state === "failed") return icons.failure;
  if (item.state === "pending") return icons.pending;
  if (item.kind === "file") return icons.file;
  if (item.kind === "tool") return icons.tool;
  if (item.kind === "profile") return icons.profile;
  if (item.kind === "thought") return icons.question;
  return icons.running;
}

function toneFor(item: TranscriptItem, theme: Theme): string {
  if (item.state === "failed" || item.kind === "error") return theme.error;
  if (item.state === "success") return theme.success;
  if (item.state === "pending" || item.kind === "plan") return theme.warning;
  return theme.accent;
}

const MARKDOWN_SYNTAX_STYLE = SyntaxStyle.fromStyles({
  default: {},
  "markup.heading": { bold: true },
  "markup.strong": { bold: true },
  "markup.italic": { italic: true },
  "markup.strikethrough": { dim: true },
  "markup.raw": { bold: true },
  "markup.link.label": { underline: true },
  "markup.link.url": { underline: true },
});

function MarkdownText({ text, color, streaming = false }: { text: string; color: string; streaming?: boolean }) {
  return (
    <markdown
      content={text}
      syntaxStyle={MARKDOWN_SYNTAX_STYLE}
      fg={color}
      conceal
      concealCode
      streaming={streaming}
      internalBlockMode="coalesced"
      style={{ flexGrow: 0, flexShrink: 1 }}
    />
  );
}

const MAX_DIFF_LINES = 50;
type FileDiffLine = { kind: "add" | "delete" | "context" | "hunk"; content: string };

function parseFileDiff(diff: string): FileDiffLine[] {
  return diff.split(/\r?\n/).reduce<FileDiffLine[]>((lines, rawLine, index, allLines) => {
    if (index === allLines.length - 1 && rawLine === "") return lines;
    // Keep hunk headers (they locate the change); drop only file headers.
    if (rawLine.startsWith("@@")) return [...lines, { kind: "hunk", content: rawLine }];
    if (rawLine.startsWith("--- ") || rawLine.startsWith("+++ ")) return lines;
    if (rawLine.startsWith("… ") && rawLine.endsWith(" more diff lines")) return lines;
    if (rawLine.startsWith("+")) return [...lines, { kind: "add", content: rawLine.slice(1) }];
    if (rawLine.startsWith("-")) return [...lines, { kind: "delete", content: rawLine.slice(1) }];
    if (rawLine.startsWith(" ")) return [...lines, { kind: "context", content: rawLine.slice(1) }];
    return [...lines, { kind: "context", content: rawLine }];
  }, []);
}

function FileDiffTitle({ title, theme, icons }: { title: string; theme: Theme; icons: IconSet }) {
  const match = title.match(/^(.*?)(?:  \+(\d+)  -(\d+))$/);
  const label = match?.[1] ?? title;
  return (
    <text>
      <span fg={theme.accent}>{icons.file}</span>
      <span fg={theme.text}>{` ${label}`}</span>
      {match ? <span fg={theme.diffAdd}>{`  +${match[2]}`}</span> : null}
      {match ? <span fg={theme.diffDelete}>{`  -${match[3]}`}</span> : null}
    </text>
  );
}

function FileDiffBlock({ item, theme, icons }: { item: TranscriptItem; theme: Theme; icons: IconSet }) {
  const [expanded, setExpanded] = useState(false);
  const toggleFocus = useKeyboardFocusVisible<BoxRenderable>();
  const lines = useMemo(() => parseFileDiff(item.body), [item.body]);
  const visibleLines = expanded ? lines : lines.slice(0, MAX_DIFF_LINES);
  const remaining = Math.max(0, lines.length - visibleLines.length);
  const toggle = () => setExpanded((value) => !value);
  return (
    <box border borderStyle="rounded" borderColor={theme.border} style={{ flexDirection: "column", maxWidth: 110, marginLeft: item.parentId ? 2 : 0, paddingLeft: 1, paddingRight: 1, paddingTop: 1, paddingBottom: 1, backgroundColor: theme.surfaceRaised }}>
      <FileDiffTitle title={item.title} theme={theme} icons={icons} />
      {visibleLines.length ? (
        <box style={{ flexDirection: "column", marginTop: 1, backgroundColor: theme.surface }}>
          {visibleLines.map((line, index) => {
            if (line.kind === "hunk") {
              return (
                <box key={`${index}-hunk`} style={{ flexDirection: "row", width: "100%", backgroundColor: theme.surface }}>
                  <text fg={theme.subtle}>{line.content}</text>
                </box>
              );
            }
            const added = line.kind === "add";
            const deleted = line.kind === "delete";
            const backgroundColor = added ? theme.diffAddBackground : deleted ? theme.diffDeleteBackground : theme.surface;
            const markerColor = added ? theme.diffAdd : deleted ? theme.diffDelete : theme.subtle;
            const marker = added ? "+" : deleted ? "-" : " ";
            return (
              <box key={`${index}-${line.kind}`} style={{ flexDirection: "row", width: "100%", backgroundColor }}>
                <text fg={markerColor}>{`${marker} `}</text>
                <text fg={theme.text}>{line.content}</text>
              </box>
            );
          })}
        </box>
      ) : null}
      {lines.length > MAX_DIFF_LINES ? (
        <box
          ref={toggleFocus.ref}
          focusable
          onMouseDown={() => { toggle(); setTimeout(() => toggleFocus.ref.current?.blur(), 0); }}
          onKeyDown={(key) => { if (isEnterKey(key.name) || key.name === "space") { key.preventDefault(); toggle(); } }}
          style={{ flexDirection: "row", justifyContent: "space-between", marginTop: 1, paddingLeft: 1, paddingRight: 1, backgroundColor: toggleFocus.focused ? theme.surfaceSelected : theme.surfaceRaised }}
        >
          <text fg={theme.muted}>{expanded ? "收起" : `余 ${remaining} 行`}</text>
          <text fg={theme.accent}>{expanded ? "⌃" : "▸"}</text>
        </box>
      ) : null}
    </box>
  );
}

function GroupChildRow({ item, theme, icons }: { item: TranscriptItem; theme: Theme; icons: IconSet }) {
  if (item.kind === "assistant") {
    return (
      <box style={{ flexDirection: "column", maxWidth: 110, marginLeft: 2 }}>
        <MarkdownText text={item.body} color={theme.text} streaming={item.state === "running"} />
      </box>
    );
  }
  if (item.kind === "file") return <FileDiffBlock item={item} theme={theme} icons={icons} />;
  if (item.kind === "agent") {
    const directed = item.direction !== undefined;
    const running = item.state === "running";
    const markerColor = directed ? theme.accent : toneFor(item, theme);
    const stackBody = directed || item.state === "failed" || item.body.includes("\n");
    return (
      <box style={{ flexDirection: "column", maxWidth: 110, marginLeft: 2 }}>
        <box style={{ flexDirection: "row" }}>
          {directed
            ? <text fg={markerColor}>{item.direction === "in" ? "←" : "→"}</text>
            : running
              ? <Spinner color={markerColor} />
              : <text fg={markerColor}>{markerFor(item, icons)}</text>}
          <text fg={markerColor}>{` ${item.title}${!stackBody && item.body ? `  · ${item.body}` : ""}`}</text>
        </box>
        {stackBody && item.body ? <text fg={theme.muted} style={{ marginLeft: 2 }}>{item.body}</text> : null}
      </box>
    );
  }
  const tone = toneFor(item, theme);
  const running = item.state === "running";
  const stackBody = item.state === "failed" || item.body.includes("\n");
  return (
    <box style={{ flexDirection: "column", maxWidth: 110, marginLeft: 2 }}>
      <box style={{ flexDirection: "row" }}>
        {running
          ? <Spinner color={tone} />
          : <text fg={tone}>{markerFor(item, icons)}</text>}
        <text fg={tone}>{` ${item.title}${!stackBody && item.body ? `  ${item.body}` : ""}`}</text>
      </box>
      {stackBody && item.body ? <text fg={theme.muted} style={{ marginLeft: 2 }}>{item.body}</text> : null}
    </box>
  );
}

function isCollapsibleRow(item: TranscriptItem): boolean {
  return (item.kind === "tool" || item.kind === "thought") && item.state === "success";
}

// Collapsed summaries read as natural sentences ("已阅读 2 个文件") instead of
// raw counts; thoughts are deliberately left out of the summary.
const TOOL_SUMMARY_PHRASES: Record<string, (count: number) => string> = {
  "文件已阅读": (n) => `已阅读 ${n} 个文件`,
  "文件已写入": (n) => `已写入 ${n} 个文件`,
  "文件已修改": (n) => `已修改 ${n} 个文件`,
  "文件已删除": (n) => `已删除 ${n} 个文件`,
  "文件已重命名": (n) => `已重命名 ${n} 个文件`,
  "目录已查看": (n) => `已查看目录 ${n} 次`,
  "文件已搜索": (n) => `已搜索文件 ${n} 次`,
  "代码已搜索": (n) => `已搜索代码 ${n} 次`,
  "命令已执行": (n) => `已执行命令 ${n} 次`,
  "网页已搜索": (n) => `已搜索网页 ${n} 次`,
  "网页已阅读": (n) => `已阅读网页 ${n} 个`,
  "子代理已启动": (n) => `已启动子代理 ${n} 个`,
};

function summarizeToolRows(rows: TranscriptItem[]): string {
  const counts = new Map<string, number>();
  let unmapped = 0;
  const keys = Object.keys(TOOL_SUMMARY_PHRASES);
  for (const row of rows) {
    if (row.kind === "thought") continue;
    // Result titles carry a detail suffix ("文件已阅读  README.md"), so
    // match the label as a prefix instead of an exact key.
    const key = keys.find((k) => row.title === k || row.title.startsWith(`${k}  `));
    if (key) counts.set(key, (counts.get(key) ?? 0) + 1);
    else unmapped += 1;
  }
  const parts = [...counts.entries()].map(([title, count]) => TOOL_SUMMARY_PHRASES[title](count));
  // Unknown tools are aggregated without naming them.
  if (unmapped > 0) parts.push(`执行了工具 ${unmapped} 次`);
  return parts.join("，");
}

function ToolGroupSummary({ rows, theme, icons }: { rows: TranscriptItem[]; theme: Theme; icons: IconSet }) {
  const [expanded, setExpanded] = useState(false);
  const toggleFocus = useKeyboardFocusVisible<BoxRenderable>();
  if (!rows.length) return null;
  return (
    <box style={{ flexDirection: "column", maxWidth: 110, marginLeft: 2 }}>
      <box
        ref={toggleFocus.ref}
        focusable
        onMouseDown={() => { setExpanded((value) => !value); setTimeout(() => toggleFocus.ref.current?.blur(), 0); }}
        onKeyDown={(key) => { if (isEnterKey(key.name) || key.name === "space") { key.preventDefault(); setExpanded((value) => !value); } }}
        style={{ flexDirection: "row", paddingLeft: 1, paddingRight: 1, backgroundColor: toggleFocus.focused ? theme.surfaceSelected : undefined }}
      >
        <text fg={theme.accent}>{expanded ? "▾" : "▸"}</text>
        <text fg={theme.muted}>{` ${summarizeToolRows(rows)}`}</text>
      </box>
      {expanded ? rows.map((row) => <GroupChildRow key={row.id} item={row} theme={theme} icons={icons} />) : null}
    </box>
  );
}

function AssistantGroupBlock({ item, rows, theme, icons }: { item: TranscriptItem; rows: TranscriptItem[]; theme: Theme; icons: IconSet }) {
  // Collapse only *consecutive* runs of successful tool/thought rows; a
  // failure or text block ends a run, preserving the real execution order.
  const rendered: ReactNode[] = [];
  let cursor = 0;
  while (cursor < rows.length) {
    if (isCollapsibleRow(rows[cursor])) {
      const start = cursor;
      while (cursor < rows.length && isCollapsibleRow(rows[cursor])) cursor += 1;
      rendered.push(
        <ToolGroupSummary key={`collapsed-${rows[start].id}`} rows={rows.slice(start, cursor)} theme={theme} icons={icons} />
      );
    } else {
      rendered.push(<GroupChildRow key={rows[cursor].id} item={rows[cursor]} theme={theme} icons={icons} />);
      cursor += 1;
    }
  }
  return (
    <box style={{ flexDirection: "column", maxWidth: 96, gap: 1 }}>
      {/* Thinking indicator: spinner under the title while the group is
          running, so the turn opens with visible motion instead of a bare
          heading that sits silent until the first token. */}
      <box style={{ flexDirection: "row", gap: 1 }}>
        <text><span fg={theme.accent}><strong>{icons.assistant}</strong></span><span fg={theme.text}><strong> 助手</strong></span></text>
        {item.state === "running" ? <Spinner color={theme.accent} /> : null}
        {item.state === "running" ? <text fg={theme.muted}>思考中</text> : null}
      </box>
      {rendered}
    </box>
  );
}

function Transcript({ items, theme, icons, narrow }: { items: TranscriptItem[]; theme: Theme; icons: IconSet; narrow: boolean }) {
  const childrenByGroup = new Map<string, TranscriptItem[]>();
  for (const item of items) {
    if (item.parentId) {
      const list = childrenByGroup.get(item.parentId);
      if (list) list.push(item);
      else childrenByGroup.set(item.parentId, [item]);
    }
  }
  return (
    <scrollbox stickyScroll focused style={{ height: 1, flexGrow: 1, flexShrink: 1, minHeight: 0, paddingLeft: narrow ? 1 : 2, paddingRight: narrow ? 1 : 2 }}>
      <box style={{ flexDirection: "column", gap: 1, paddingTop: 1, paddingBottom: 1 }}>
        {items.map((item) => {
          if (item.kind === "assistant" && item.role === "group") {
            return <AssistantGroupBlock key={item.id} item={item} rows={childrenByGroup.get(item.id) ?? []} theme={theme} icons={icons} />;
          }
          if (item.parentId && childrenByGroup.has(item.parentId)) return null;
          if (item.kind === "user" || item.kind === "assistant") {
            const assistant = item.kind === "assistant";
            return (
              <box key={item.id} style={{ flexDirection: "column", maxWidth: 96 }}>
                <text>
                  <span fg={theme.accent}><strong>{assistant ? icons.assistant : icons.prompt}</strong></span>
                  <span fg={theme.text}><strong>{` ${assistant ? "助手" : "你"}`}</strong></span>
                </text>
                {assistant ? <MarkdownText text={item.body} color={theme.text} streaming={item.state === "running"} /> : <text fg={theme.text}>{item.body}</text>}
              </box>
            );
          }
          if (item.kind === "file") return <FileDiffBlock key={item.id} item={item} theme={theme} icons={icons} />;
          if (item.kind === "agent") {
            const directed = item.direction !== undefined;
            const running = item.state === "running";
            const markerColor = directed ? theme.accent : toneFor(item, theme);
            const stackBody = directed || item.state === "failed" || item.body.includes("\n");
            return (
              <box key={item.id} style={{ flexDirection: "column", maxWidth: 110 }}>
                <box style={{ flexDirection: "row" }}>
                  {directed
                    ? <text fg={markerColor}>{item.direction === "in" ? "←" : "→"}</text>
                    : running
                      ? <Spinner color={markerColor} />
                      : <text fg={markerColor}>{markerFor(item, icons)}</text>}
                  <text fg={markerColor}>{` ${item.title}${!stackBody && item.body ? `  · ${item.body}` : ""}`}</text>
                </box>
                {stackBody && item.body ? <text fg={theme.muted} style={{ marginLeft: 2 }}>{item.body}</text> : null}
              </box>
            );
          }
          if (item.id === "welcome" && item.kind === "status") {
            // Startup/welcome row: bare copy when ready, spinner while loading,
            // failure marker only when it actually failed. Scoped to the
            // welcome row — other status blocks carry meaningful titles.
            const failed = item.state === "failed";
            const ready = item.state === "success";
            const tone = failed ? theme.error : ready ? theme.success : theme.accent;
            return (
              <box key={item.id} style={{ flexDirection: "column", maxWidth: 110 }}>
                <box style={{ flexDirection: "row" }}>
                  {item.state === "running" ? <Spinner color={tone} /> : failed ? <text fg={tone}>{icons.failure}</text> : null}
                  <text fg={tone}>{ready ? item.body : ` ${failed ? item.title : item.body}`}</text>
                </box>
                {failed && item.body ? <text fg={theme.muted} style={{ marginLeft: 2 }}>{item.body}</text> : null}
              </box>
            );
          }
          const genericTone = toneFor(item, theme);
          const genericRunning = item.state === "running";
          const stackGenericBody = item.state === "failed" || item.body.includes("\n");
          return (
            <box key={item.id} style={{ flexDirection: "column", maxWidth: 110, marginLeft: item.parentId ? 2 : 0 }}>
              <box style={{ flexDirection: "row" }}>
                {genericRunning
                  ? <Spinner color={genericTone} />
                  : <text fg={genericTone}>{markerFor(item, icons)}</text>}
                <text fg={genericTone}>{` ${item.title}${!stackGenericBody && item.body ? `  ${item.body}` : ""}`}</text>
              </box>
              {stackGenericBody && item.body ? <text fg={theme.muted} style={{ marginLeft: 2 }}>{item.body}</text> : null}
            </box>
          );
        })}
      </box>
    </scrollbox>
  );
}

function useKeyboardFocusVisible<T extends BoxRenderable>() {
  const [focused, setFocused] = useState(false);
  const ref = useRef<T | null>(null);
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const on = () => setFocused(true);
    const off = () => setFocused(false);
    node.on("focused", on);
    node.on("blurred", off);
    return () => { node.off("focused", on); node.off("blurred", off); };
  }, []);
  return { ref, focused };
}

function ActionButton({ icon, label, theme, color, activeColor, bold = false, defaultBg, onInvoke }: {
  icon?: string;
  label?: string;
  theme: Theme;
  color: string;
  activeColor?: string;
  bold?: boolean;
  defaultBg: string;
  onInvoke: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  const { ref, focused } = useKeyboardFocusVisible<BoxRenderable>();
  const active = hovered || focused;
  const content = icon && label ? `${icon} ${label}` : (icon ?? label ?? "");
  return (
    <box
      ref={ref}
      focusable
      onMouseOver={() => setHovered(true)}
      onMouseOut={() => setHovered(false)}
      onMouseDown={() => {
        onInvoke();
        // OpenTUI focuses the box only after this handler returns, so the
        // blur has to run on the next tick. Mouse activation must not leave
        // keyboard focus behind; keyboard activation keeps focus as expected.
        setTimeout(() => ref.current?.blur(), 0);
      }}
      onKeyDown={(key) => { if (isEnterKey(key.name) || key.name === "space") onInvoke(); }}
      style={{ flexDirection: "row", paddingLeft: 1, paddingRight: 1, backgroundColor: active ? theme.surfaceSelected : defaultBg }}
    >
      <text fg={active ? (activeColor ?? color) : color}>
        {bold ? <strong>{content}</strong> : content}
      </text>
    </box>
  );
}

function HeaderAction({ icon, label, theme, onInvoke }: { icon: string; label: string; theme: Theme; onInvoke: () => void }) {
  return <ActionButton icon={icon} label={label} theme={theme} color={theme.subtle} activeColor={theme.text} defaultBg={theme.background} onInvoke={onInvoke} />;
}

function Header({ cwd, theme, icons, compact, narrow, onHistory, onNew }: { cwd: string; theme: Theme; icons: IconSet; compact: boolean; narrow: boolean; onHistory: () => void; onNew: () => void }) {
  const label = compact ? (cwd.replace(/\\/g, "/").split("/").filter(Boolean).at(-1) ?? cwd) : cwd;
  return (
    <box style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between", height: 1, flexShrink: 0, paddingLeft: narrow ? 1 : 2, paddingRight: 1 }}>
      <text fg={theme.subtle}>{label}</text>
      <box style={{ flexDirection: "row", gap: 1 }}>
        <HeaderAction icon={icons.session} label={narrow ? "" : "历史"} theme={theme} onInvoke={onHistory} />
        <HeaderAction icon={icons.newSession} label={narrow ? "" : "新会话"} theme={theme} onInvoke={onNew} />
      </box>
    </box>
  );
}

function panelIcon(kind: PanelSpec["kind"], icons: IconSet): string {
  if (kind === "sessions") return icons.session;
  if (kind === "observe") return icons.observe;
  if (kind === "permission") return icons.approval;
  if (kind === "checkpoint") return icons.checkpoint;
  if (kind === "model") return icons.assistant;
  return icons.profile;
}

type PanelAnchor = "history" | "profile" | "permission" | "model" | "effort" | "top-right";

// Profile/permission buttons sit in the composer's left group; their panels
// anchor to the bottom-left like the buttons they belong to.
const LEFT_PANEL_ANCHORS: ReadonlySet<PanelAnchor> = new Set(["profile", "permission"]);

// Model/effort buttons sit in the composer's right group, so their panels
// pop from the right screen edge like the buttons they belong to.
const RIGHT_PANEL_ANCHORS: ReadonlySet<PanelAnchor> = new Set(["model", "effort"]);

function defaultPanelAnchor(panel: PanelSpec): PanelAnchor {
  if (panel.kind === "sessions") return "history";
  if (panel.kind === "profile") return "profile";
  if (panel.kind === "permission") return "permission";
  if (panel.kind === "model") return "model";
  if (panel.kind === "effort") return "effort";
  return "top-right";
}

function PanelOptionRow({ option, active, busy, theme, icons, onSelect }: { option: PanelOption; active: boolean; busy: boolean; theme: Theme; icons: IconSet; onSelect: () => void }) {
  const [hovered, setHovered] = useState(false);
  const tone = busy
    ? theme.warning
    : option.tone === "danger"
      ? theme.error
      : option.tone === "warning"
        ? theme.warning
        : active
          ? theme.focus
          : theme.text;
  const backgroundColor = active ? theme.accentSoft : hovered ? theme.surfaceSelected : theme.surface;
  return (
    <box
      onMouseOver={() => setHovered(true)}
      onMouseOut={() => setHovered(false)}
      onMouseDown={onSelect}
      style={{ flexDirection: "column", paddingLeft: 1, paddingRight: 1, backgroundColor }}
    >
      {busy ? (
        <box style={{ flexDirection: "row" }}>
          <Spinner color={tone} />
          <text fg={tone}>{` ${option.label}  处理中…`}</text>
        </box>
      ) : (
        <text>
          <span fg={active ? theme.focus : theme.subtle}>{active ? icons.prompt : " "}</span>
          <span fg={option.selected ? theme.success : theme.subtle}>{` ${option.selected ? icons.success : " "}`}</span>
          <span fg={tone}>{` ${option.label}`}</span>
        </text>
      )}
      {option.description && !busy ? (
        <text fg={theme.subtle}>
          {`     ${option.description}`}
          {option.badge ? <span fg={theme.accent}>{` · ${option.badge}`}</span> : null}
        </text>
      ) : null}
    </box>
  );
}

function PanelView({ panel, textWidth, theme, icons, onSelect, busyId }: { panel: PanelSpec; textWidth: number; theme: Theme; icons: IconSet; onSelect: (id: string) => void; busyId: string | null }) {
  const [query, setQuery] = useState("");
  const scrollRef = useRef<ScrollBoxRenderable | null>(null);
  const options = useMemo(() => (panel.options ?? []).filter((item) => `${item.label} ${item.description ?? ""}`.toLowerCase().includes(query.toLowerCase())), [panel.options, query]);
  const currentOptionId = panel.options?.find((item) => item.selected || item.tone === "success")?.id;
  const initialSelected = Math.max(0, (panel.options ?? []).findIndex((item) => item.id === currentOptionId));
  const [selected, setSelected] = useState(initialSelected);
  useEffect(() => {
    const currentIndex = options.findIndex((item) => item.id === currentOptionId);
    setSelected(currentIndex >= 0 ? currentIndex : 0);
  }, [currentOptionId, panel.kind, panel.options]);
  useEffect(() => setSelected((value) => Math.min(value, Math.max(0, options.length - 1))), [options.length]);
  useEffect(() => scrollRef.current?.scrollChildIntoView(`panel-option-${selected}`), [selected]);
  useKeyboard((key) => {
    if (key.name === "up" && options.length) setSelected((value) => (value - 1 + options.length) % options.length);
    else if (key.name === "down" && options.length) setSelected((value) => (value + 1) % options.length);
    else if (key.name === "pageup") setSelected((value) => Math.max(0, value - 8));
    else if (key.name === "pagedown") setSelected((value) => Math.min(options.length - 1, value + 8));
    else if (isEnterKey(key.name) && options[selected] && !busyId) onSelect(options[selected].id);
  });
  const footer = options.length
    ? "↑↓ 选择 · Enter 确认 · Esc 关闭"
    : panel.searchable
      ? "输入搜索 · Esc 关闭"
      : "Esc 关闭";
  // ScrollBoxRenderable stretches to maxHeight with short content and fails
  // to paint at tiny heights, so only use one when scrolling is actually
  // needed; otherwise render a plain box and the panel fits its content.
  const optionsMaxHeight = panel.searchable ? 10 : 12;
  const optionContentRows = options.reduce((sum, option) => sum + 1 + (option.description ? 1 : 0), 0);
  const optionsScrollNeeded = optionContentRows > optionsMaxHeight;
  const bodyMaxHeight = options.length ? 5 : 8;
  const bodyTextWidth = Math.max(8, textWidth - 4);
  const bodyAllRows = panel.body
    ? Math.max(1, Math.ceil(stringWidth(panel.body) / bodyTextWidth))
    : 0;
  const bodyRows = Math.min(bodyAllRows, bodyMaxHeight);
  const bodyScrollNeeded = bodyAllRows > bodyMaxHeight;
  const optionRows = options.map((option, index) => (
    <box id={`panel-option-${index}`} key={option.id} style={{ flexDirection: "column" }}>
      <PanelOptionRow
        option={option}
        active={index === selected}
        busy={option.id === busyId}
        theme={theme}
        icons={icons}
        onSelect={() => { if (!busyId) onSelect(option.id); }}
      />
    </box>
  ));
  return (
    <box style={{ flexDirection: "column", width: "100%", maxHeight: "100%", minHeight: 3, backgroundColor: theme.surface }}>
      <box style={{ flexDirection: "row", height: 1, paddingLeft: 1, paddingRight: 1, backgroundColor: theme.surfaceRaised }}>
        <text fg={theme.accent}><strong>{`${panelIcon(panel.kind, icons)} ${panel.title}`}</strong></text>
      </box>
      {panel.searchable ? <input value={query} placeholder={`${icons.search} 搜索会话…`} focused onInput={setQuery} style={{ paddingLeft: 1, paddingRight: 1, backgroundColor: theme.surface, textColor: theme.text, cursorColor: theme.accent, placeholderColor:theme.muted }} /> : null}
      {bodyRows ? (
        bodyScrollNeeded ? (
          <scrollbox style={{ height: bodyMaxHeight, paddingLeft: 1, paddingRight: 1 }}><text fg={theme.muted}>{panel.body}</text></scrollbox>
        ) : (
          <box style={{ paddingLeft: 1, paddingRight: 1 }}><text fg={theme.muted}>{panel.body}</text></box>
        )
      ) : null}
      {options.length ? (
        optionsScrollNeeded ? (
          <scrollbox ref={scrollRef} style={{ height: optionsMaxHeight, paddingTop: 1, paddingBottom: 1 }}>
            {optionRows}
          </scrollbox>
        ) : (
          <box style={{ paddingTop: 1, paddingBottom: 1 }}>{optionRows}</box>
        )
      ) : null}
      <text fg={theme.subtle} style={{ paddingLeft: 1 }}>{footer}</text>
    </box>
  );
}

function PanelOverlay({ panel, anchor, theme, icons, terminalWidth, terminalHeight, onClose, onSelect, busyId }: {
  panel: PanelSpec;
  anchor: PanelAnchor;
  theme: Theme;
  icons: IconSet;
  terminalWidth: number;
  terminalHeight: number;
  onClose: () => void;
  onSelect: (id: string) => void;
  busyId: string | null;
}) {
  useKeyboard((key) => {
    if (key.name !== "escape") return;
    key.preventDefault();
    onClose();
  });
  const modalWidth = Math.min(60, Math.max(30, terminalWidth - 12));
  const modalHeight = Math.min(18, Math.max(7, terminalHeight - 4));
  const footerLeft = LEFT_PANEL_ANCHORS.has(anchor) ? 2 : Math.max(2, Math.min(12, terminalWidth - modalWidth - 2));
  const placement = anchor === "history"
    ? { top: 1, right: 2 }
    : anchor === "top-right"
      ? { top: 2, right: 2 }
      : RIGHT_PANEL_ANCHORS.has(anchor)
        ? { bottom: 2, right: 2 }
        : { bottom: 2, left: footerLeft };
  return (
    <box
      position="absolute"
      top={0}
      left={0}
      width="100%"
      height="100%"
      zIndex={20}
      onMouseDown={onClose}
      style={{ width: "100%", height: "100%" }}
    >
      {/* No dimming mask: OpenTUI's opacity compositing wipes the covered
          cells' text and bleeds blended colors into their backgrounds. The
          background content stays fully visible; clicking it closes the panel. */}
      <box
        position="absolute"
        {...placement}
        border
        borderStyle="rounded"
        borderColor={theme.border}
        onMouseDown={(event) => event.stopPropagation()}
        style={{ width: modalWidth, maxWidth: "100%", maxHeight: modalHeight, flexShrink: 1, backgroundColor: theme.surface, zIndex: 21 }}
      >
        <PanelView key={panel.kind} panel={panel} textWidth={modalWidth - 2} theme={theme} icons={icons} onSelect={onSelect} busyId={busyId} />
      </box>
    </box>
  );
}

function InteractionOptionRow({ index, label, description, active, deny, theme, icons, onPick }: {
  index: number;
  label: string;
  description: string;
  active: boolean;
  deny: boolean;
  theme: Theme;
  icons: IconSet;
  onPick: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  const tone = active ? theme.focus : deny ? theme.error : theme.text;
  const backgroundColor = active ? theme.accentSoft : hovered ? theme.surfaceSelected : theme.surfaceRaised;
  return (
    <box
      focusable
      onMouseOver={() => setHovered(true)}
      onMouseOut={() => setHovered(false)}
      onMouseDown={onPick}
      style={{ flexDirection: "column", backgroundColor, paddingLeft: 1, paddingRight: 1 }}
    >
      <text fg={tone}>{`${active ? icons.prompt : " "} [${index + 1}] ${label}`}</text>
      {description ? <text fg={theme.subtle}>{`    ${description}`}</text> : null}
    </box>
  );
}

// Fixed approval reasons come from permissions.py; translate the known ones
// instead of exposing English internals. Unknown reasons pass through.
const APPROVAL_REASON_LABELS: Record<string, string> = {
  "shell command mutates external paths": "将修改工作区外的文件",
  "shell command writes inside the workspace": "将修改工作区文件",
  "shell command mutates git remote state": "将修改远程 Git 状态",
  "shell executes a program whose effects cannot be determined": "将运行无法完全判断影响的程序",
};

function InteractionView({ interaction, theme, icons, narrow, onResolve }: { interaction: Interaction; theme: Theme; icons: IconSet; narrow: boolean; onResolve: (result: Record<string, unknown>) => void }) {
  const [selected, setSelected] = useState(0);
  const selectedRef = useRef(0);
  const otherRef = useRef<InputRenderable | null>(null);
  const approvalOptions = interaction.kind === "approval"
    ? [{ label: "仅本次允许", value: "approve" }, ...(interaction.payload.persistAvailable ? [{ label: "本项目后续允许", value: "persist" }] : []), { label: "拒绝", value: "deny" }]
    : [];
  const questionOptions = interaction.kind === "question" ? interaction.payload.options : [];
  const options = interaction.kind === "approval" ? approvalOptions : questionOptions.map((option, index) => ({ ...option, label: option.label, value: String(index) }));
  const otherSelected = interaction.kind === "question" && Boolean(questionOptions[selected]?.is_other);
  const select = (index: number) => {
    selectedRef.current = index;
    setSelected(index);
  };
  const resolveCurrent = () => {
    if (interaction.kind === "approval") onResolve({ decision: approvalOptions[selectedRef.current].value });
    else onResolve({ selectedIndex: selectedRef.current, customText: otherRef.current?.value ?? "" });
  };
  useKeyboard((key) => {
    // Up/down always move selection, including while the "other" input is open,
    // so the custom branch can never become a navigation trap.
    if (key.name === "up" || (!otherSelected && key.name === "left")) {
      key.preventDefault();
      select((selectedRef.current - 1 + options.length) % options.length);
      return;
    }
    if (key.name === "down" || (!otherSelected && key.name === "right")) {
      key.preventDefault();
      select((selectedRef.current + 1) % options.length);
      return;
    }
    if (key.name === "escape") {
      key.preventDefault();
      onResolve(interaction.kind === "approval" ? { decision: "deny" } : { cancelled: true });
    }
    else if (isEnterKey(key.name)) {
      key.preventDefault();
      resolveCurrent();
    }
    else if (/^[1-9]$/.test(key.name)) {
      const index = Number(key.name) - 1;
      if (index < options.length) {
        key.preventDefault();
        if (index === selectedRef.current) resolveCurrent();
        else select(index);
      }
    }
  });
  const isBash = interaction.kind === "approval" && interaction.payload.toolName === "run_bash";
  const commandText = isBash ? String(interaction.payload.args?.command ?? "").trim() : "";
  const argsText = interaction.kind === "approval" && !isBash ? JSON.stringify(interaction.payload.args, null, 2) : "";
  const argsHeight = Math.min(Math.max(2, argsText.split("\n").length), 6);
  const commandHeight = Math.min(Math.max(1, commandText.split("\n").length), 6);
  const reasonText = interaction.kind === "approval"
    ? (APPROVAL_REASON_LABELS[interaction.payload.reason] ?? interaction.payload.reason)
    : "";
  return (
    <box border borderStyle="rounded" borderColor={interaction.kind === "approval" ? theme.warning : theme.accent} style={{ flexDirection: "column", flexShrink: 0, marginLeft: narrow ? 1 : 2, marginRight: narrow ? 1 : 2, paddingLeft: 1, paddingRight: 1, paddingTop: 1, paddingBottom: 1, backgroundColor: theme.surfaceRaised }}>
      <text fg={interaction.kind === "approval" ? theme.warning : theme.accent}><strong>{interaction.kind === "approval" ? "需要确认" : interaction.payload.question}</strong></text>
      {interaction.kind === "approval" ? (
        <box style={{ flexDirection: "column", flexShrink: 0, marginTop: 1 }}>
          {isBash ? (
            <>
              <text fg={theme.subtle}>执行命令</text>
              <scrollbox style={{ height: commandHeight, backgroundColor: theme.surface }}>
                <text fg={theme.text}>{commandText}</text>
              </scrollbox>
            </>
          ) : (
            <>
              <text fg={theme.warning}>{interaction.payload.toolName}</text>
              <text fg={theme.subtle}>参数</text>
              <scrollbox style={{ height: argsHeight, backgroundColor: theme.surface }}>
                <text fg={theme.muted}>{argsText}</text>
              </scrollbox>
            </>
          )}
          <text fg={theme.muted}>{reasonText}</text>
        </box>
      ) : null}
      <box style={{ flexDirection: "column", paddingTop: 1 }}>
        {options.map((option, index) => (
          <InteractionOptionRow
            key={`${option.value}-${index}`}
            index={index}
            label={option.label}
            description={interaction.kind === "question" ? (questionOptions[index]?.description ?? "") : ""}
            active={index === selected}
            deny={interaction.kind === "approval" && option.value === "deny"}
            theme={theme}
            icons={icons}
            onPick={() => { if (index === selectedRef.current) resolveCurrent(); else select(index); }}
          />
        ))}
      </box>
      {otherSelected ? <input ref={otherRef} placeholder="其他说明…" focused style={{ backgroundColor: theme.surface, textColor: theme.text, cursorColor: theme.accent, placeholderColor: theme.muted }} /> : null}
      <text fg={theme.subtle}>{`↑↓/点击 选择 · Enter 确认 · Esc ${interaction.kind === "approval" ? "拒绝" : "取消"}`}</text>
    </box>
  );
}

type Completion = { id: string; label: string; description: string; insert: string; kind: "command" | "file" | "session" };
type CompletionSection = { title: string; items: Completion[] };

const COMMAND_LABEL_WIDTH = 30;

function truncateText(text: string, maxWidth: number): string {
  if (stringWidth(text) <= maxWidth) return text;
  if (maxWidth <= 1) return "…";
  let result = "";
  for (const char of text) {
    if (stringWidth(result + char) > maxWidth - 1) break;
    result += char;
  }
  return `${result.trimEnd()}…`;
}

function StopAction({ icon, theme, onStop }: { icon: string; theme: Theme; onStop: () => void }) {
  return <ActionButton icon={icon} theme={theme} color={theme.error} bold defaultBg={theme.surface} onInvoke={onStop} />;
}

const BRAILLE_ROW_BITS = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]];
const RING_POINTS = [
  { x: 1, y: 0 }, { x: 2, y: 0 }, { x: 3, y: 1 }, { x: 3, y: 2 },
  { x: 2, y: 3 }, { x: 1, y: 3 }, { x: 0, y: 2 }, { x: 0, y: 1 },
];

function brailleRing(litCount: number): [string, string] {
  const build = (offset: number) => {
    let bits = 0;
    RING_POINTS.forEach((point, index) => {
      if (point.x < offset || point.x >= offset + 2 || index >= litCount) return;
      bits |= BRAILLE_ROW_BITS[point.y][point.x - offset];
    });
    return String.fromCharCode(0x2800 + bits);
  };
  return [build(0), build(2)];
}

function formatTokenK(value: number): string {
  if (value >= 1000) {
    const k = value / 1000;
    return `${k >= 100 ? Math.round(k) : Math.round(k * 10) / 10}K`;
  }
  return String(value);
}

function ContextGauge({ percent, tokens, windowTokens, theme }: { percent: number; tokens: number; windowTokens: number; theme: Theme }) {
  const remainingPct = Math.max(0, Math.min(100, percent));
  const litCount = Math.max(1, Math.round((remainingPct / 100) * RING_POINTS.length));
  const [left, right] = brailleRing(litCount);
  const used = 100 - remainingPct;
  const color = used < 60 ? theme.success : used < 85 ? theme.warning : theme.error;
  const [active, setActive] = useState(false);
  const ref = useRef<BoxRenderable | null>(null);
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const show = () => setActive(true);
    const hide = () => setActive(false);
    node.on("focused", show);
    node.on("blurred", hide);
    return () => { node.off("focused", show); node.off("blurred", hide); };
  }, []);
  const remainingTokens = Math.max(0, windowTokens - tokens);
  const details = windowTokens > 0
    ? [`上下文剩余 ${formatTokenK(remainingTokens)} / ${formatTokenK(windowTokens)}`, `已使用 ${used}%`]
    : [`已使用 ${used}%`];
  // Absolute boxes mis-measure intrinsic width against these CJK lines and
  // collapse to ~2 columns, so derive it explicitly.
  const detailWidth = Math.max(...details.map((line) => stringWidth(line))) + 2;
  const popoverWidth = detailWidth + 2;
  // Expand left by default; if that would leave the screen on the left side,
  // expand right instead.
  const expandRight = (ref.current?.x ?? 0) + 6 - popoverWidth < 0;
  return (
    <box
      ref={ref}
      focusable
      onMouseOver={() => setActive(true)}
      onMouseOut={() => setActive(false)}
      style={{ paddingLeft: 1, paddingRight: 1 }}
    >
      <text>
        <span fg={color}>{`${left}${right}`}</span>
        <span fg={theme.subtle}>{` ${remainingPct}%`}</span>
      </text>
      {active ? (
        <box
          position="absolute"
          {...(expandRight ? { left: 0 } : { right: 2 })}
          bottom={3}
          zIndex={40}
          border
          borderStyle="rounded"
          borderColor={theme.border}
          style={{ flexDirection: "column", width: detailWidth + 2, backgroundColor: theme.surfaceRaised, paddingLeft: 1, paddingRight: 1 }}
        >
          {details.map((line) => <text key={line} fg={theme.text}>{line}</text>)}
        </box>
      ) : null}
    </box>
  );
}

function AttachAction({ theme, onAddFiles }: { theme: Theme; onAddFiles: () => void }) {
  return <ActionButton icon="＋" theme={theme} color={theme.accent} bold defaultBg={theme.surface} onInvoke={onAddFiles} />;
}

function CandidateRow({ rowId, item, active, compact, labelWidth, descriptionWidth, theme, onPick }: {
  rowId: string;
  item: Completion;
  active: boolean;
  compact: boolean;
  labelWidth: number;
  descriptionWidth: number;
  theme: Theme;
  onPick: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  const backgroundColor = active ? theme.accentSoft : hovered ? theme.surfaceSelected : theme.surfaceRaised;
  return (
    <box
      id={rowId}
      onMouseOver={() => setHovered(true)}
      onMouseOut={() => setHovered(false)}
      onMouseDown={onPick}
      style={{ flexDirection: "row", backgroundColor }}
    >
      <box style={{ width: labelWidth, flexDirection: "row", flexShrink: 0, justifyContent: "flex-start", paddingLeft: 1, paddingRight: 1 }}>
        <text fg={active ? theme.focus : theme.accent}>{truncateText(item.label, labelWidth - 2)}</text>
      </box>
      {compact ? null : (
        <text fg={active ? theme.text : theme.subtle} wrapMode="none" truncate>
          {truncateText(item.description, descriptionWidth)}
        </text>
      )}
    </box>
  );
}

function CompletionOverlay({ theme, title, footer, bottomOffset, maxHeight, narrow, onClose, children }: { theme: Theme; title: string; footer: string; bottomOffset: number; maxHeight: number; narrow: boolean; onClose: () => void; children: React.ReactNode }) {
  const gutter = narrow ? 1 : 2;
  return (
    <box
      position="absolute"
      top={0}
      left={0}
      width="100%"
      height="100%"
      zIndex={30}
      onMouseDown={onClose}
      style={{ width: "100%", height: "100%" }}
    >
      <box
        position="absolute"
        left={gutter}
        right={gutter}
        bottom={bottomOffset}
        zIndex={31}
        border
        borderStyle="rounded"
        borderColor={theme.border}
        onMouseDown={(event) => event.stopPropagation()}
        style={{ flexDirection: "column", maxHeight, backgroundColor: theme.surfaceRaised, paddingLeft: 1, paddingRight: 1 }}
      >
        <text fg={theme.accent}><strong>{title}</strong></text>
        {children}
        <text fg={theme.subtle}>{footer}</text>
      </box>
    </box>
  );
}

function Composer({ value, onChange, onSubmit, onCancel, running, stopping, queueDepth, commands, theme, icons, compact, narrow, terminalWidth, terminalHeight, disabled, sessionReady, onAction, attachments, onAddFiles, onStagePaths, onPaste, onRemoveAttachment, snapshot, onOpenPanel }: { value: string; onChange: (value: string) => void; onSubmit: () => void; onCancel: () => void; running: boolean; stopping: boolean; queueDepth: number; commands: CommandItem[]; theme: Theme; icons: IconSet; compact: boolean; narrow: boolean; terminalWidth: number; terminalHeight: number; disabled: boolean; sessionReady: boolean; onAction?: AppProps["onAction"]; attachments: AttachmentItem[]; onAddFiles: () => void; onStagePaths: (paths: string[], source: "mention" | "clipboard") => Promise<boolean>; onPaste: (editor: TextareaRenderable | null) => void; onRemoveAttachment: (id: string) => void; snapshot: typeof initialState.snapshot; onOpenPanel: (panel: "profile" | "permission" | "model" | "effort") => void }) {
  const [selected, setSelected] = useState(0);
  const [mentions, setMentions] = useState<Completion[]>([]);
  const [dismissedCompletionValue, setDismissedCompletionValue] = useState<string | null>(null);
  const editorRef = useRef<TextareaRenderable | null>(null);
  const paletteRef = useRef<ScrollBoxRenderable | null>(null);
  const slashQuery = value.trimStart().startsWith("/") && !value.includes(" ") ? value.trim() : "";
  const mentionMatch = value.match(/(?:^|\s)@([^\s]*)$/);
  const mentionQuery = mentionMatch?.[1];
  useEffect(() => {
    let active = true;
    if (!mentionMatch || !onAction || !sessionReady) { setMentions([]); return; }
    const timer = setTimeout(() => {
      void onAction("complete_mention", { prefix: mentionQuery ?? "" }).then((result) => {
        if (active) setMentions((result.candidates ?? []).map((item) => ({ id: item.insertText, label: item.display, description: item.description, insert: `@${item.insertText} `, kind: item.kind })));
      }).catch(() => { if (active) setMentions([]); });
    }, 120);
    return () => { active = false; clearTimeout(timer); };
  }, [mentionQuery, onAction, sessionReady]);
  const candidates = useMemo<Completion[]>(() => {
    if (slashQuery) return commands.filter((command) => commandMatches(command, slashQuery)).map((command) => ({ id: command.name, label: command.name, description: command.description, insert: `${command.name} `, kind: "command" }));
    return mentionMatch ? mentions : [];
  }, [commands, slashQuery, mentionQuery, mentions]);
  const commandMode = Boolean(slashQuery);
  const candidateLabelWidth = commandMode ? COMMAND_LABEL_WIDTH + 2 : Math.min(42, Math.max(18, Math.floor(terminalWidth * 0.55)));
  const descriptionWidth = Math.max(1, terminalWidth - candidateLabelWidth - 8);
  const mentionSections = useMemo<CompletionSection[]>(() => {
    if (commandMode) return [];
    const sessions = candidates.filter((item) => item.kind === "session");
    const files = candidates.filter((item) => item.kind === "file");
    return [
      sessions.length ? { title: "历史会话", items: sessions } : null,
      files.length ? { title: "当前工作区文件", items: files } : null,
    ].filter((section): section is CompletionSection => section !== null);
  }, [candidates, commandMode]);
  const paletteOpen = !disabled && candidates.length > 0 && Boolean(slashQuery || mentionMatch) && dismissedCompletionValue !== value;
  useEffect(() => {
    if (dismissedCompletionValue !== null && dismissedCompletionValue !== value) {
      setDismissedCompletionValue(null);
    }
  }, [dismissedCompletionValue, value]);
  useEffect(() => setSelected((value) => Math.min(value, Math.max(0, candidates.length - 1))), [candidates.length, slashQuery, mentionQuery]);
  useEffect(() => { if (paletteOpen) paletteRef.current?.scrollChildIntoView(`completion-${selected}`); }, [paletteOpen, selected]);
  useEffect(() => { if (editorRef.current && editorRef.current.plainText !== value) editorRef.current.setText(value); }, [value]);
  const applyCompletion = (completion: Completion) => {
    if (completion.kind === "file") {
      const path = completion.insert.replace(/^@file:/, "").trim();
      void onStagePaths([path], "mention").then((accepted) => {
        if (!accepted) return;
        const next = mentionMatch ? value.slice(0, value.lastIndexOf("@")) : value;
        editorRef.current?.replaceText(next);
        onChange(next);
      });
      return;
    }
    const replacement = completion.kind === "command" ? completion.insert : `@${completion.insert.replace(/^@/, "")}`;
    const next = mentionMatch ? value.slice(0, value.lastIndexOf("@")) + replacement : replacement;
    editorRef.current?.replaceText(next);
    onChange(next);
  };
  useKeyboard((key) => {
    if (disabled) return;
    if (key.ctrl && key.name === "v") {
      key.preventDefault();
      onPaste(editorRef.current);
      return;
    }
    if (paletteOpen) {
      let handled = true;
      if (key.name === "up") setSelected((value) => (value - 1 + candidates.length) % candidates.length);
      else if (key.name === "down") setSelected((value) => (value + 1) % candidates.length);
      else if (key.name === "pageup") setSelected((value) => Math.max(0, value - 8));
      else if (key.name === "pagedown") setSelected((value) => Math.min(candidates.length - 1, value + 8));
      else if (key.name === "home") setSelected(0);
      else if (key.name === "end") setSelected(candidates.length - 1);
      else if (key.name === "tab" || isEnterKey(key.name)) applyCompletion(candidates[selected]);
      else handled = false;
      if (handled) key.preventDefault();
    } else if (key.name === "escape") {
      key.preventDefault();
      onCancel();
    }
  });
  const renderCandidate = (item: Completion, index: number) => (
    <CandidateRow
      key={item.id}
      rowId={`completion-${index}`}
      item={item}
      active={index === selected}
      compact={compact}
      labelWidth={candidateLabelWidth}
      descriptionWidth={descriptionWidth}
      theme={theme}
      onPick={() => applyCompletion(item)}
    />
  );
  const composerBottomOffset = narrow ? 11 : 8;
  // The model button shares the right-side row with the gauge and effort
  // selector; a fixed truncation made the row wider than the screen in
  // compact/narrow windows, so the group shrank and left a gap on the right.
  const modelTextBudget = Math.max(8, narrow ? terminalWidth - 26 : terminalWidth - 50);
  const modelText = truncateText(compact ? shortModelLabel(snapshot.model) : snapshot.model, modelTextBudget);
  const paletteMaxHeight = Math.max(5, Math.min(14, terminalHeight - composerBottomOffset - 1));
  const paletteBodyRows = Math.min(
    candidates.length + (commandMode ? 0 : mentionSections.length),
    Math.max(2, paletteMaxHeight - 3)
  );
  return (
    <box style={{ flexDirection: "column", flexShrink: 0, paddingLeft: narrow ? 1 : 2, paddingRight: narrow ? 1 : 2 }}>
      {paletteOpen ? (
        <CompletionOverlay
          theme={theme}
          title={commandMode ? "命令" : "添加上下文"}
          footer={` ${selected + 1}/${candidates.length}  ↑↓ 选择  Enter/Tab 使用  点击外层关闭`}
          bottomOffset={composerBottomOffset}
          maxHeight={paletteMaxHeight}
          narrow={narrow}
          onClose={() => setDismissedCompletionValue(value)}
        >
          <scrollbox ref={paletteRef} style={{ height: paletteBodyRows, backgroundColor: theme.surfaceRaised }}>
            {commandMode
              ? candidates.map(renderCandidate)
              : mentionSections.map((section) => (
                <React.Fragment key={section.title}>
                  <text fg={theme.muted}><strong>{section.title}</strong></text>
                  {section.items.map((item) => renderCandidate(item, candidates.indexOf(item)))}
                </React.Fragment>
              ))}
          </scrollbox>
        </CompletionOverlay>
      ) : null}
      <box border borderStyle="rounded" borderColor={theme.border} style={{ flexDirection: "column", backgroundColor: theme.surface, paddingTop: 1, paddingBottom: 1 }}>
        <box style={{ flexDirection: "row", paddingLeft: 1 }}>
          <text fg={theme.accent}><strong>{`${icons.prompt} `}</strong></text>
          <textarea
            ref={editorRef}
            initialValue={value}
            placeholder="从这里开始吧，/ 可查看命令，@ 可添加上下文…"
            keyBindings={[
              { name: "return", action: "submit" },
              { name: "kpenter", action: "submit" },
              { name: "linefeed", action: "submit" },
              { name: "return", shift: true, action: "newline" },
              { name: "kpenter", shift: true, action: "newline" },
              { name: "linefeed", shift: true, action: "newline" },
            ]}
            onContentChange={() => onChange(editorRef.current?.plainText ?? "")}
            onSubmit={() => { if (!paletteOpen) onSubmit(); }}
            focused={!disabled}
            style={{ flexGrow: 1, minHeight: 2, maxHeight: 6, backgroundColor: theme.surface, textColor: theme.text, cursorColor: theme.accent, placeholderColor: theme.muted }}
          />
        </box>
        <box style={{ flexDirection: narrow ? "column" : "row", alignItems: narrow ? "flex-start" : "center", justifyContent: "space-between", paddingLeft: 1, paddingRight: 1, marginTop: 1 }}>
          <box style={{ flexDirection: "row", alignItems: "center", flexShrink: 1, minWidth: 0 }}>
            <AttachAction theme={theme} onAddFiles={onAddFiles} />
            <ToolbarAction label={`${snapshot.routingMode === "auto" ? "自动" : profileLabel(snapshot.profile)} ▾`} theme={theme} tone={theme.text} onInvoke={() => onOpenPanel("profile")} />
            <ToolbarAction label={`${compact ? "审批" : permissionLabel(snapshot.permissionMode)} ▾`} theme={theme} tone={snapshot.permissionMode === "danger-full-access" ? theme.error : theme.text} onInvoke={() => onOpenPanel("permission")} />
          </box>
          <box style={{ flexDirection: "row", alignItems: "center", flexGrow: narrow ? 1 : 0, flexShrink: 1, minWidth: 0, justifyContent: narrow ? "space-between" : "flex-start", width: narrow ? "100%" : undefined, marginLeft: narrow ? 0 : 2 }}>
            <ContextGauge percent={snapshot.contextPercent} tokens={snapshot.contextTokens ?? 0} windowTokens={snapshot.contextWindowTokens ?? 0} theme={theme} />
            <ToolbarAction label={`${icons.assistant} ${modelText} ▾`} theme={theme} tone={theme.text} onInvoke={() => onOpenPanel("model")} />
            <ToolbarAction label={`${effortLabel(snapshot.reasoningEffort)} ▾`} theme={theme} tone={theme.text} onInvoke={() => onOpenPanel("effort")} />
            {queueDepth ? <text fg={theme.subtle}>{` 已排队 ${queueDepth}`}</text> : null}
            {stopping ? <text fg={theme.subtle}> 停止中…</text> : running ? <StopAction icon={icons.stop} theme={theme} onStop={onCancel} /> : null}
          </box>
        </box>
      </box>
      {attachments.length ? (
        <box style={{ flexDirection: "column", alignItems: "flex-start" }}>
          {attachments.map((attachment) => (
            <box key={attachment.id} style={{ flexDirection: "row", paddingLeft: 1, gap: 1 }}>
              <text fg={theme.subtle}>{`${icons.file} ${attachment.name}  ${formatBytes(attachment.size)}`}</text>
              <ActionButton icon="×" theme={theme} color={theme.error} defaultBg={theme.surface} onInvoke={() => onRemoveAttachment(attachment.id)} />
            </box>
          ))}
        </box>
      ) : null}
    </box>
  );
}

function profileLabel(profile: string): string {
  return ({
    general: "通用",
    "coding-agent": "编码",
    plan: "规划",
    "app-builder": "应用构建",
    review: "审查",
  } as Record<string, string>)[profile] ?? profile;
}

function permissionLabel(permissionMode: string): string {
  return permissionMode === "workspace-write" ? "请求批准" : permissionMode === "llm-auto" ? "替我审批" : permissionMode === "danger-full-access" ? "完全访问" : permissionMode;
}

function shortModelLabel(model: string): string {
  const base = model.includes("/") ? (model.split("/").at(-1) ?? model) : model;
  const parts = base.split("-");
  const rest = parts.length > 1 ? parts.slice(1) : parts;
  return rest.map((part) => (/^\d/.test(part) ? part.toUpperCase() : `${part[0]?.toUpperCase() ?? ""}${part.slice(1)}`)).join(" ");
}

function effortLabel(effort: string | null | undefined): string {
  return ({ low: "低", high: "高", max: "最大" } as Record<string, string>)[effort ?? "high"] ?? (effort || "高");
}

function ToolbarAction({ label, theme, tone, onInvoke }: { label: string; theme: Theme; tone: string; onInvoke: () => void }) {
  return <ActionButton label={label} theme={theme} color={tone} defaultBg={theme.surface} onInvoke={onInvoke} />;
}

export function App({ events, onSubmit, onCancel, onExit, onAction, onResolveInteraction, onPickFiles, onReadClipboard, onCopyText, initialTask, themePreference = "auto", iconPreference = "auto" }: AppProps) {
  const [state, dispatch] = useReducer(reduceEvent, initialState);
  const [draft, setDraft] = useState("");
  const [attachments, setAttachments] = useState<AttachmentItem[]>([]);
  const [externalPaths, setExternalPaths] = useState<string[] | null>(null);
  const [composerVersion, setComposerVersion] = useState(0);
  const [panel, setPanel] = useState<PanelSpec | null>(null);
  const [panelBusyId, setPanelBusyId] = useState<string | null>(null);
  const [panelAnchor, setPanelAnchor] = useState<PanelAnchor>("top-right");
  const [detectedTheme, setDetectedTheme] = useState<ThemeMode | null>(null);
  const renderer = useRenderer();
  const { width, height } = useTerminalDimensions();
  const theme = resolveTheme(themePreference, detectedTheme);
  const icons = resolveIcons(iconPreference);
  const compact = width < 96;
  const narrow = width < 70;
  const stopping = state.turnState === "cancelling";
  const running = state.turnState === "running" || state.turnState === "queued" || stopping;

  useEffect(() => {
    let active = true;
    void renderer.waitForThemeMode(120).then((mode) => { if (active && mode) setDetectedTheme(mode); });
    const handler = (mode: ThemeMode) => setDetectedTheme(mode);
    renderer.on(CliRenderEvents.THEME_MODE, handler);
    return () => { active = false; renderer.off(CliRenderEvents.THEME_MODE, handler); };
  }, [renderer]);
  useEffect(() => {
    if (!events) return;
    let active = true;
    void (async () => {
      for await (const event of events) {
        if (!active) return;
        if (event.type === "panel") {
          setPanel(event.panel);
          setPanelAnchor(defaultPanelAnchor(event.panel));
        }
        else {
          if (event.type === "interaction") setPanel(null);
          if (event.type === "session_reset") {
            setPanel(null);
            setDraft("");
            setAttachments([]);
            setExternalPaths(null);
            setComposerVersion((version) => version + 1);
          }
          dispatch(event);
        }
      }
    })();
    return () => { active = false; };
  }, [events]);

  const invoke = async (name: ActionName, params?: Record<string, unknown>, anchor?: PanelAnchor) => {
    if (!onAction) return;
    try {
      const result = await onAction(name, params);
      if (result.panel) {
        setPanel(result.panel);
        setPanelAnchor(anchor ?? defaultPanelAnchor(result.panel));
      }
      else if (name === "new_session") {
        setPanel(null);
        setDraft("");
        setComposerVersion((version) => version + 1);
      }
      if (result.message) dispatch({ type: "notice", text: result.message });
    } catch (error) {
      dispatch({ type: "notice", level: "error", text: formatUserError(error) });
    }
  };

  useKeyboard((key) => {
    if (key.ctrl && key.name === "c") {
      key.preventDefault();
      const selection = renderer.getSelection();
      if (selection) {
        const text = selection.getSelectedText();
        if (text && onCopyText) {
          void onCopyText(text).then((copied) => {
            if (!copied) dispatch({ type: "notice", level: "error", text: "复制失败，请重试。" });
          }).catch(() => dispatch({ type: "notice", level: "error", text: "复制失败，请重试。" }));
        }
        return;
      }
      if (stopping) return;
      if (running) onCancel?.();
      else if (draft.length > 0) setDraft("");
      else onExit?.();
    }
    else if (key.ctrl && key.name === "o") { key.preventDefault(); void invoke("open_panel", { panel: "observe" }); }
    else if (key.ctrl && key.name === "p") { key.preventDefault(); void invoke("open_panel", { panel: "permission" }); }
  });

  useEffect(() => {
    const text = initialTask?.trim();
    if (!text) return;
    void onSubmit?.({ text, attachmentIds: [] }).then((result) => {
      if (result.accepted) dispatch({ type: "transcript", item: { id: `user-${Date.now()}`, kind: "user", title: "你", body: text } });
    }).catch((error) => dispatch({ type: "notice", level: "error", text: formatUserError(error) }));
  }, [initialTask, onSubmit]);

  const stagePaths = async (paths: string[], source: "picker" | "mention" | "clipboard"): Promise<boolean> => {
    if (!onAction || !paths.length) return false;
    try {
      const result = await onAction("stage_attachments", { paths, source });
      setAttachments((current) => {
        const merged = new Map(current.map((item) => [item.id, item]));
        for (const item of result.attachments ?? []) merged.set(item.id, item);
        return [...merged.values()];
      });
      return true;
    } catch (error) {
      dispatch({ type: "notice", level: "error", text: formatUserError(error) });
      return false;
    }
  };
  const addFiles = () => {
    if (!onPickFiles) return;
    void onPickFiles(state.snapshot.inputMode ?? "text").then((paths) => stagePaths(paths, "picker")).catch((error) => {
      dispatch({ type: "notice", level: "error", text: formatUserError(error) });
    });
  };
  const paste = (editor: TextareaRenderable | null) => {
    if (!onReadClipboard || !onAction) return;
    void onReadClipboard().then(async (result) => {
      if (result.status !== "read") return;
      const { mimeType, bytes } = result.representation;
      if (mimeType.startsWith("image/")) {
        const response = await onAction("stage_attachments", { clipboard: { dataBase64: Buffer.from(bytes).toString("base64"), mimeType, name: `clipboard.${mimeType.split("/")[1] === "jpeg" ? "jpg" : mimeType.split("/")[1]}` } });
        setAttachments((current) => {
          const merged = new Map(current.map((item) => [item.id, item]));
          for (const item of response.attachments ?? []) merged.set(item.id, item);
          return [...merged.values()];
        });
        return;
      }
      const text = new TextDecoder().decode(bytes);
      const paths = clipboardFilePaths(text);
      if (mimeType === "text/uri-list") {
        await stagePaths(paths, "clipboard");
        return;
      }
      editor?.insertText(text);
      setDraft(editor?.plainText ?? `${draft}${text}`);
    }).catch((error) => dispatch({ type: "notice", level: "error", text: formatUserError(error) }));
  };
  const removeAttachment = (id: string) => {
    void onAction?.("remove_attachment", { attachmentId: id }).then(() => setAttachments((current) => current.filter((item) => item.id !== id))).catch((error) => {
      dispatch({ type: "notice", level: "error", text: formatUserError(error) });
    });
  };
  const submitWithAuthorization = async (authorizedPaths: string[] = []) => {
    const text = draft.trim();
    if (!text && !attachments.length) return;
    if (state.snapshot.status === "failed") {
      dispatch({ type: "notice", level: "error", text: "会话未就绪，暂时无法提交。" });
      return;
    }
    const panelCommands: Record<string, string> = { "/checkpoint": "checkpoint", "/mcp": "mcp", "/observe": "observe" };
    if (!attachments.length && panelCommands[text]) { setDraft(""); void invoke("open_panel", { panel: panelCommands[text] }); return; }
    if (!attachments.length && (text === "/compact" || text === "/fork")) { setDraft(""); void invoke("panel_action", { panel: "command", action: text.slice(1) }); return; }
    if (!onSubmit) return;
    try {
      const result = await onSubmit({ text, attachmentIds: attachments.map((item) => item.id), authorizedPaths });
      if (!result.accepted) {
        setExternalPaths(result.confirmation?.paths ?? null);
        return;
      }
      const submittedAttachments = result.attachments ?? attachments;
      const summary = submittedAttachments.map((item) => `[${item.kind}] ${item.name} (${formatBytes(item.size)})`).join("\n");
      const body = [text, summary].filter(Boolean).join("\n");
      dispatch({ type: "transcript", item: withOptimisticUserMessage(state, body).items.at(-1)! });
      setDraft("");
      setAttachments([]);
      setExternalPaths(null);
    } catch (error) {
      dispatch({ type: "notice", level: "error", text: formatUserError(error) });
    }
  };
  const submit = () => { void submitWithAuthorization(); };
  const selectPanel = (id: string) => {
    if (!panel) return;
    void (async () => {
      if (!onAction) return;
      setPanelBusyId(id);
      try {
        const result = await onAction("panel_action", { panel: panel.kind, action: id });
        if (result.panel) setPanel(result.panel);
        else setPanel(null);
        if (result.message) dispatch({ type: "notice", text: result.message });
      } catch (error) { dispatch({ type: "notice", level: "error", text: formatUserError(error) }); }
      finally { setPanelBusyId(null); }
    })();
  };
  const resolveInteraction = (result: Record<string, unknown>) => {
    const interaction = state.interaction;
    if (!interaction) return;
    const request = onResolveInteraction?.(interaction.id, result);
    if (!request) return;
    void request.catch((error) => {
      dispatch({ type: "notice", level: "error", text: formatUserError(error) });
      dispatch({ type: "interaction_closed", id: interaction.id });
    });
  };

  return (
    <box style={{ flexDirection: "column", width: "100%", height: "100%", backgroundColor: theme.background }}>
      <Header cwd={state.snapshot.cwd} theme={theme} icons={icons} compact={compact} narrow={narrow} onHistory={() => void invoke("open_sessions", undefined, "history")} onNew={() => void invoke("new_session")} />
      <Transcript items={state.items} theme={theme} icons={icons} narrow={narrow} />
      {state.interaction ? <InteractionView interaction={state.interaction} theme={theme} icons={icons} narrow={narrow} onResolve={resolveInteraction} /> : externalPaths ? (
        <box border borderStyle="rounded" borderColor={theme.warning} style={{ flexDirection: "column", marginLeft: narrow ? 1 : 2, marginRight: narrow ? 1 : 2, paddingLeft: 1, paddingRight: 1 }}>
          <text fg={theme.warning}><strong>允许读取工作区外文件？</strong></text>
          {externalPaths.map((path) => <text key={path} fg={theme.text}>{path}</text>)}
          <box style={{ flexDirection: "row", gap: 1 }}>
            <ActionButton label="允许本次读取" theme={theme} color={theme.success} defaultBg={theme.surfaceRaised} onInvoke={() => void submitWithAuthorization(externalPaths)} />
            <ActionButton label="拒绝" theme={theme} color={theme.error} defaultBg={theme.surfaceRaised} onInvoke={() => setExternalPaths(null)} />
          </box>
        </box>
      ) : <Composer key={composerVersion} value={draft} onChange={setDraft} onSubmit={submit} onCancel={onCancel ?? (() => undefined)} running={running} stopping={stopping} queueDepth={state.queueDepth} commands={state.commands} theme={theme} icons={icons} compact={compact} narrow={narrow} terminalWidth={width} terminalHeight={height} disabled={stopping} sessionReady={Boolean(state.snapshot.sessionId) && state.snapshot.status !== "failed"} onAction={onAction} attachments={attachments} onAddFiles={addFiles} onStagePaths={stagePaths} onPaste={paste} onRemoveAttachment={removeAttachment} snapshot={state.snapshot} onOpenPanel={(panel) => void invoke("open_panel", { panel }, panel)} />}
      {panel && !state.interaction ? <PanelOverlay panel={panel} anchor={panelAnchor} theme={theme} icons={icons} terminalWidth={width} terminalHeight={height} onClose={() => setPanel(null)} onSelect={selectPanel} busyId={panelBusyId} /> : null}
    </box>
  );
}
