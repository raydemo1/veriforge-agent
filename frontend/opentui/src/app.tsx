import React, { useEffect, useMemo, useReducer, useRef, useState } from "react";
import { stringWidth } from "bun";
import { CliRenderEvents, SyntaxStyle } from "@opentui/core";
import type { BoxRenderable, ClipboardReadResult, ScrollBoxRenderable, TextareaRenderable } from "@opentui/core";
import { useKeyboard, useRenderer, useTerminalDimensions } from "@opentui/react";
import { resolveIcons } from "./icons.ts";
import type { IconSet } from "./icons.ts";
import { formatUserError } from "./errors.ts";
import { clipboardFilePaths, formatBytes } from "./attachment-utils.ts";
import type { ActionName, ActionResult, AttachmentItem, CommandItem, IconPreference, Interaction, PanelSpec, SubmitResult, ThemePreference, TranscriptItem, TurnSubmission, UiEvent } from "./protocol.ts";
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

function markerFor(item: TranscriptItem, icons: IconSet, spinnerFrame = 0): string {
  if (item.state === "running" && (item.kind === "tool" || item.kind === "thought")) return SPINNER_FRAMES[spinnerFrame];
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
type FileDiffLine = { kind: "add" | "delete" | "context"; content: string };

function parseFileDiff(diff: string): FileDiffLine[] {
  return diff.split(/\r?\n/).reduce<FileDiffLine[]>((lines, rawLine, index, allLines) => {
    if (index === allLines.length - 1 && rawLine === "") return lines;
    if (rawLine.startsWith("@@") || rawLine.startsWith("--- ") || rawLine.startsWith("+++ ")) return lines;
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
      <span fg={theme.diffAdd}>{`${icons.file} ${label}`}</span>
      {match ? <span fg={theme.diffAdd}>{`  +${match[2]}`}</span> : null}
      {match ? <span fg={theme.diffDelete}>{`  -${match[3]}`}</span> : null}
    </text>
  );
}

function FileDiffBlock({ item, theme, icons }: { item: TranscriptItem; theme: Theme; icons: IconSet }) {
  const [expanded, setExpanded] = useState(false);
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
          focusable
          onMouseDown={toggle}
          onKeyDown={(key) => { if (isEnterKey(key.name) || key.name === "space") { key.preventDefault(); toggle(); } }}
          style={{ flexDirection: "row", justifyContent: "space-between", marginTop: 1, paddingLeft: 1, paddingRight: 1, backgroundColor: theme.surfaceRaised }}
        >
          <text fg={theme.muted}>{expanded ? "收起" : `余 ${remaining} 行`}</text>
          <text fg={theme.accent}>{expanded ? "⌃" : "▸"}</text>
        </box>
      ) : null}
    </box>
  );
}

function Transcript({ items, theme, icons }: { items: TranscriptItem[]; theme: Theme; icons: IconSet }) {
  const hasRunningTool = items.some((item) => item.kind === "tool" && item.state === "running");
  const hasRunningThought = items.some((item) => item.kind === "thought" && item.state === "running");
  const hasRunningAgent = items.some((item) => item.kind === "agent" && item.state === "running");
  const [spinnerFrame, setSpinnerFrame] = useState(0);
  useEffect(() => {
    setSpinnerFrame(0);
    if (!hasRunningTool && !hasRunningThought && !hasRunningAgent) return;
    const timer = setInterval(() => setSpinnerFrame((value) => (value + 1) % SPINNER_FRAMES.length), 180);
    return () => clearInterval(timer);
  }, [hasRunningTool, hasRunningThought, hasRunningAgent]);

  return (
    <scrollbox stickyScroll focused style={{ height: 1, flexGrow: 1, flexShrink: 1, minHeight: 0, paddingLeft: 2, paddingRight: 2 }}>
      <box style={{ flexDirection: "column", gap: 1, paddingTop: 1, paddingBottom: 1 }}>
        {items.map((item) => {
          if (item.kind === "assistant" && item.role === "group") {
            return (
              <box key={item.id} style={{ flexDirection: "column", maxWidth: 96 }}>
                <text fg={theme.success}><strong>{`${icons.assistant} 助手`}</strong></text>
              </box>
            );
          }
          if (item.kind === "assistant" && item.parentId) {
            return (
              <box key={item.id} style={{ flexDirection: "column", maxWidth: 110, marginLeft: 2 }}>
                <MarkdownText text={item.body} color={theme.text} streaming={item.state === "running"} />
              </box>
            );
          }
          if (item.kind === "user" || item.kind === "assistant") {
            const assistant = item.kind === "assistant";
            return (
              <box key={item.id} style={{ flexDirection: "column", maxWidth: 96 }}>
                <text fg={assistant ? theme.success : theme.accent}>
                  <strong>{`${assistant ? icons.assistant : icons.prompt} ${assistant ? "助手" : "你"}`}</strong>
                </text>
                {assistant ? <MarkdownText text={item.body} color={theme.text} streaming={item.state === "running"} /> : <text fg={theme.text}>{item.body}</text>}
              </box>
            );
          }
          if (item.kind === "file") return <FileDiffBlock key={item.id} item={item} theme={theme} icons={icons} />;
          if (item.kind === "agent") {
            const directed = item.direction !== undefined;
            const marker = directed
              ? (item.direction === "in" ? "←" : "→")
              : markerFor(item, icons, spinnerFrame);
            const markerColor = directed ? theme.accent : toneFor(item, theme);
            const bodyNextLine = directed || item.state === "failed";
            return (
              <box key={item.id} style={{ flexDirection: "column", maxWidth: 110 }}>
                <text fg={markerColor}>{`${marker} ${item.title}${!bodyNextLine && item.body ? `  · ${item.body}` : ""}`}</text>
                {bodyNextLine && item.body ? <text fg={theme.muted} style={{ marginLeft: 2 }}>{item.body}</text> : null}
              </box>
            );
          }
          return (
            <box key={item.id} style={{ flexDirection: "column", maxWidth: 110, marginLeft: item.parentId ? 2 : 0 }}>
              <text fg={toneFor(item, theme)}>{`  ${markerFor(item, icons, spinnerFrame)} ${item.title}${item.body ? `  ${item.body}` : ""}`}</text>
            </box>
          );
        })}
      </box>
    </scrollbox>
  );
}

function HeaderAction({ icon, label, theme, onInvoke }: { icon: string; label: string; theme: Theme; onInvoke: () => void }) {
  const [hovered, setHovered] = useState(false);
  const color = hovered ? theme.text : theme.subtle;
  return (
    <box
      focusable
      onMouseOver={() => setHovered(true)}
      onMouseOut={() => setHovered(false)}
      onMouseDown={onInvoke}
      onKeyDown={(key) => { if (isEnterKey(key.name) || key.name === "space") onInvoke(); }}
      style={{ flexDirection: "row", paddingLeft: 1, paddingRight: 1, backgroundColor: hovered ? theme.surfaceSelected : theme.background }}
    >
      <text fg={color}>{icon}</text>
      <text fg={color}>{` ${label}`}</text>
    </box>
  );
}

function Header({ cwd, theme, icons, compact, onHistory, onNew }: { cwd: string; theme: Theme; icons: IconSet; compact: boolean; onHistory: () => void; onNew: () => void }) {
  const label = compact ? (cwd.replace(/\\/g, "/").split("/").filter(Boolean).at(-1) ?? cwd) : cwd;
  return (
    <box style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between", height: 1, flexShrink: 0, paddingLeft: 2, paddingRight: 1 }}>
      <text fg={theme.subtle}>{label}</text>
      <box style={{ flexDirection: "row", gap: 1 }}>
        <HeaderAction icon={icons.session} label="历史" theme={theme} onInvoke={onHistory} />
        <HeaderAction icon={icons.newSession} label="新会话" theme={theme} onInvoke={onNew} />
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

const BOTTOM_PANEL_ANCHORS: ReadonlySet<PanelAnchor> = new Set(["profile", "permission", "model", "effort"]);

function defaultPanelAnchor(panel: PanelSpec): PanelAnchor {
  if (panel.kind === "sessions") return "history";
  if (panel.kind === "profile") return "profile";
  if (panel.kind === "permission") return "permission";
  if (panel.kind === "model") return "model";
  if (panel.kind === "effort") return "effort";
  return "top-right";
}

function PanelView({ panel, theme, icons, onSelect, busyId }: { panel: PanelSpec; theme: Theme; icons: IconSet; onSelect: (id: string) => void; busyId: string | null }) {
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
  return (
    <box style={{ flexDirection: "column", width: "100%", maxHeight: "100%", minHeight: 3, backgroundColor: theme.surface }}>
      <box style={{ flexDirection: "row", height: 1, paddingLeft: 1, paddingRight: 1, backgroundColor: theme.surfaceRaised }}>
        <text fg={theme.accent}><strong>{`${panelIcon(panel.kind, icons)} ${panel.title}`}</strong></text>
      </box>
      {panel.searchable ? <input value={query} placeholder={`${icons.search} 搜索会话…`} focused onInput={setQuery} style={{ paddingLeft: 1, paddingRight: 1, backgroundColor: theme.surface, textColor: theme.text, cursorColor: theme.accent, placeholderColor: theme.muted }} /> : null}
      {panel.body ? <scrollbox style={{ maxHeight: options.length ? 5 : 8, flexShrink: 1, paddingLeft: 1, paddingRight: 1 }}><text fg={theme.muted}>{panel.body}</text></scrollbox> : null}
      {options.length ? (
        <scrollbox ref={scrollRef} style={{ maxHeight: panel.searchable ? 10 : 12, flexShrink: 1, minHeight: 1, paddingTop: 1, paddingBottom: 1 }}>
          {options.map((option, index) => {
            const active = index === selected;
            const busy = option.id === busyId;
            const tone = busy ? theme.warning : option.tone === "danger" ? theme.error : option.tone === "warning" ? theme.warning : option.tone === "success" ? theme.success : active ? theme.focus : theme.text;
            return (
              <box id={`panel-option-${index}`} key={option.id} onMouseDown={() => { if (!busyId) onSelect(option.id); }} style={{ flexDirection: "column", paddingLeft: 1, paddingRight: 1, backgroundColor: active ? theme.surfaceSelected : theme.surface }}>
                <text fg={tone}>{`${busy ? SPINNER_FRAMES[0] : active ? icons.prompt : " "} ${option.label}${busy ? "  处理中…" : ""}`}</text>
                {option.description && !busy ? <text fg={theme.subtle}>{`    ${option.description}`}</text> : null}
              </box>
            );
          })}
        </scrollbox>
      ) : null}
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
  const footerLeft = BOTTOM_PANEL_ANCHORS.has(anchor) ? 2 : Math.max(2, Math.min(12, terminalWidth - modalWidth - 2));
  const placement = anchor === "history"
    ? { top: 1, right: 2 }
    : anchor === "top-right"
      ? { top: 2, right: 2 }
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
      <box position="absolute" top={0} left={0} width="100%" height="100%" style={{ backgroundColor: theme.background, opacity: 0.35 }} />
      <box
        position="absolute"
        {...placement}
        border
        borderStyle="rounded"
        borderColor={theme.border}
        onMouseDown={(event) => event.stopPropagation()}
        style={{ width: modalWidth, maxWidth: "100%", maxHeight: modalHeight, flexShrink: 1, backgroundColor: theme.surface, zIndex: 21 }}
      >
        <PanelView key={panel.kind} panel={panel} theme={theme} icons={icons} onSelect={onSelect} busyId={busyId} />
      </box>
    </box>
  );
}

function InteractionView({ interaction, theme, icons, onResolve }: { interaction: Interaction; theme: Theme; icons: IconSet; onResolve: (result: Record<string, unknown>) => void }) {
  const [selected, setSelected] = useState(0);
  const selectedRef = useRef(0);
  const [customText, setCustomText] = useState("");
  const approvalOptions = interaction.kind === "approval"
    ? [{ label: "仅本次允许", value: "approve" }, ...(interaction.payload.persistAvailable ? [{ label: "以后允许这类命令", value: "persist" }] : []), { label: "拒绝", value: "deny" }]
    : [];
  const questionOptions = interaction.kind === "question" ? interaction.payload.options : [];
  const options = interaction.kind === "approval" ? approvalOptions : questionOptions.map((option, index) => ({ ...option, label: option.label, value: String(index) }));
  const otherSelected = interaction.kind === "question" && Boolean(questionOptions[selected]?.is_other);
  const select = (index: number) => {
    selectedRef.current = index;
    setSelected(index);
  };
  useKeyboard((key) => {
    if (otherSelected) {
      if (key.name === "escape") onResolve({ cancelled: true });
      else if (isEnterKey(key.name)) onResolve({ selectedIndex: selectedRef.current, customText });
      return;
    }
    if (key.name === "escape") onResolve(interaction.kind === "approval" ? { decision: "deny" } : { cancelled: true });
    else if (key.name === "left" || key.name === "up") select((selectedRef.current - 1 + options.length) % options.length);
    else if (key.name === "right" || key.name === "down") select((selectedRef.current + 1) % options.length);
    else if (isEnterKey(key.name)) {
      if (interaction.kind === "approval") onResolve({ decision: approvalOptions[selectedRef.current].value });
      else onResolve({ selectedIndex: selectedRef.current, customText });
    }
    else if (/^[1-9]$/.test(key.name)) {
      const index = Number(key.name) - 1;
      if (index < options.length) {
        if (index === selectedRef.current) {
          if (interaction.kind === "approval") onResolve({ decision: approvalOptions[selectedRef.current].value });
          else onResolve({ selectedIndex: selectedRef.current, customText });
        } else {
          select(index);
        }
      }
    }
  });
  return (
    <box border borderStyle="rounded" borderColor={interaction.kind === "approval" ? theme.warning : theme.accent} style={{ flexDirection: "column", flexShrink: 0, marginLeft: 2, marginRight: 2, paddingLeft: 1, paddingRight: 1, paddingTop: 1, paddingBottom: 1, backgroundColor: theme.surfaceRaised }}>
      <text fg={interaction.kind === "approval" ? theme.warning : theme.accent}><strong>{interaction.kind === "approval" ? "需要确认" : interaction.payload.question}</strong></text>
      {interaction.kind === "approval" ? <text fg={theme.muted}>{`${interaction.payload.toolName}  ${interaction.payload.risk}\n${interaction.payload.reason}\n${JSON.stringify(interaction.payload.args, null, 2)}`}</text> : null}
      <box style={{ flexDirection: "column", paddingTop: 1 }}>
        {options.map((option, index) => <text key={`${option.value}-${index}`} fg={index === selected ? theme.focus : (interaction.kind === "approval" && option.value === "deny" ? theme.error : theme.text)}>{`${index === selected ? icons.prompt : " "} [${index + 1}] ${option.label}${interaction.kind === "question" && questionOptions[index]?.description ? ` — ${questionOptions[index].description}` : ""}`}</text>)}
      </box>
      {otherSelected ? <input value={customText} placeholder="其他说明…" focused onInput={setCustomText} style={{ backgroundColor: theme.surface, textColor: theme.text, cursorColor: theme.accent, placeholderColor: theme.muted }} /> : null}
    </box>
  );
}

type Completion = { id: string; label: string; description: string; insert: string; kind: "command" | "file" | "session" };
type CompletionSection = { title: string; items: Completion[] };

const COMMAND_LABEL_WIDTH = 30;

function truncateLabel(label: string, maxWidth: number): string {
  if (label.length <= maxWidth) return label;
  return `${label.slice(0, Math.max(0, maxWidth - 1))}…`;
}

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
  const [hovered, setHovered] = useState(false);
  return (
    <box
      focusable
      onMouseOver={() => setHovered(true)}
      onMouseOut={() => setHovered(false)}
      onMouseDown={onStop}
      onKeyDown={(key) => { if (isEnterKey(key.name) || key.name === "space") onStop(); }}
      style={{ paddingLeft: 1, paddingRight: 1, backgroundColor: hovered ? theme.surfaceSelected : theme.surface }}
    >
      <text fg={theme.error}><strong>{icon}</strong></text>
    </box>
  );
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
          right={2}
          bottom={3}
          zIndex={40}
          border
          borderStyle="rounded"
          borderColor={theme.border}
          style={{ flexDirection: "column", backgroundColor: theme.surfaceRaised, paddingLeft: 1, paddingRight: 1 }}
        >
          {details.map((line) => <text key={line} fg={theme.text}>{line}</text>)}
        </box>
      ) : null}
    </box>
  );
}

function AttachAction({ theme, onAddFiles }: { theme: Theme; onAddFiles: () => void }) {
  const [hovered, setHovered] = useState(false);
  return (
    <box
      focusable
      onMouseOver={() => setHovered(true)}
      onMouseOut={() => setHovered(false)}
      onMouseDown={onAddFiles}
      onKeyDown={(key) => { if (isEnterKey(key.name) || key.name === "space") onAddFiles(); }}
      style={{ flexDirection: "row", paddingLeft: 1, paddingRight: 1, backgroundColor: hovered ? theme.surfaceSelected : theme.surface }}
    >
      <text fg={theme.accent}><strong>＋</strong></text>
    </box>
  );
}

function CompletionOverlay({ theme, title, footer, onClose, children }: { theme: Theme; title: string; footer: string; onClose: () => void; children: React.ReactNode }) {
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
      <box position="absolute" top={0} left={0} width="100%" height="100%" style={{ backgroundColor: theme.background, opacity: 0.35 }} />
      <box
        position="absolute"
        left={2}
        right={2}
        bottom={2}
        zIndex={31}
        border
        borderStyle="rounded"
        borderColor={theme.border}
        onMouseDown={(event) => event.stopPropagation()}
        style={{ flexDirection: "column", maxHeight: 14, backgroundColor: theme.surfaceRaised, paddingLeft: 1, paddingRight: 1 }}
      >
        <text fg={theme.accent}><strong>{title}</strong></text>
        {children}
        <text fg={theme.subtle}>{footer}</text>
      </box>
    </box>
  );
}

function Composer({ value, onChange, onSubmit, onCancel, running, stopping, queueDepth, commands, theme, icons, compact, terminalWidth, disabled, sessionReady, onAction, attachments, onAddFiles, onStagePaths, onPaste, onRemoveAttachment, snapshot, onOpenPanel }: { value: string; onChange: (value: string) => void; onSubmit: () => void; onCancel: () => void; running: boolean; stopping: boolean; queueDepth: number; commands: CommandItem[]; theme: Theme; icons: IconSet; compact: boolean; terminalWidth: number; disabled: boolean; sessionReady: boolean; onAction?: AppProps["onAction"]; attachments: AttachmentItem[]; onAddFiles: () => void; onStagePaths: (paths: string[], source: "mention" | "clipboard") => Promise<boolean>; onPaste: (editor: TextareaRenderable | null) => void; onRemoveAttachment: (id: string) => void; snapshot: typeof initialState.snapshot; onOpenPanel: (panel: "profile" | "permission" | "model" | "effort") => void }) {
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
    <box id={`completion-${index}`} key={item.id} onMouseDown={() => applyCompletion(item)} style={{ flexDirection: "row", backgroundColor: index === selected ? theme.surfaceSelected : theme.surfaceRaised }}>
      <box style={{ width: candidateLabelWidth, flexDirection: "row", flexShrink: 0, justifyContent: "flex-start", paddingLeft: 1, paddingRight: 1 }}>
        <text fg={index === selected ? theme.focus : theme.accent}>{truncateLabel(item.label, candidateLabelWidth - 2)}</text>
      </box>
      {compact ? null : (
        <text fg={index === selected ? theme.text : theme.subtle} wrapMode="none" truncate>
          {truncateText(item.description, descriptionWidth)}
        </text>
      )}
    </box>
  );
  return (
    <box style={{ flexDirection: "column", flexShrink: 0, paddingLeft: 2, paddingRight: 2 }}>
      {paletteOpen ? (
        <CompletionOverlay
          theme={theme}
          title={commandMode ? "命令" : "添加上下文"}
          footer={` ${selected + 1}/${candidates.length}  ↑↓ 选择  Enter/Tab 使用  点击外层关闭`}
          onClose={() => setDismissedCompletionValue(value)}
        >
          <scrollbox ref={paletteRef} style={{ height: Math.min(candidates.length + (commandMode ? 0 : mentionSections.length), 10), backgroundColor: theme.surfaceRaised }}>
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
        <box style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between", paddingLeft: 1, paddingRight: 1 }}>
          <box style={{ flexDirection: "row", alignItems: "center", flexShrink: 1, minWidth: 0 }}>
            <AttachAction theme={theme} onAddFiles={onAddFiles} />
            <ToolbarAction label={`${snapshot.routingMode === "auto" ? "自动" : profileLabel(snapshot.profile)} ▾`} theme={theme} tone={theme.text} onInvoke={() => onOpenPanel("profile")} />
            <ToolbarAction label={`${compact ? "审批" : permissionLabel(snapshot.permissionMode)} ▾`} theme={theme} tone={snapshot.permissionMode === "danger-full-access" ? theme.error : theme.text} onInvoke={() => onOpenPanel("permission")} />
          </box>
          <box style={{ flexDirection: "row", alignItems: "center", flexShrink: 0, marginLeft: 2 }}>
            <ContextGauge percent={snapshot.contextPercent} tokens={snapshot.contextTokens ?? 0} windowTokens={snapshot.contextWindowTokens ?? 0} theme={theme} />
            <ToolbarAction label={`${icons.assistant} ${compact ? shortModelLabel(snapshot.model) : truncateText(snapshot.model, 26)} ▾`} theme={theme} tone={theme.text} onInvoke={() => onOpenPanel("model")} />
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
              <box focusable onMouseDown={() => onRemoveAttachment(attachment.id)} onKeyDown={(key) => { if (isEnterKey(key.name) || key.name === "space") onRemoveAttachment(attachment.id); }}>
                <text fg={theme.error}>×</text>
              </box>
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
  const [hovered, setHovered] = useState(false);
  return (
    <box
      focusable
      onMouseOver={() => setHovered(true)}
      onMouseOut={() => setHovered(false)}
      onMouseDown={onInvoke}
      onKeyDown={(key) => { if (isEnterKey(key.name) || key.name === "space") onInvoke(); }}
      style={{ paddingLeft: 1, paddingRight: 1, backgroundColor: hovered ? theme.surfaceSelected : theme.surface }}
    >
      <text fg={tone}><strong>{label}</strong></text>
    </box>
  );
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
      <Header cwd={state.snapshot.cwd} theme={theme} icons={icons} compact={compact} onHistory={() => void invoke("open_sessions", undefined, "history")} onNew={() => void invoke("new_session")} />
      <Transcript items={state.items} theme={theme} icons={icons} />
      {state.interaction ? <InteractionView interaction={state.interaction} theme={theme} icons={icons} onResolve={resolveInteraction} /> : panel ? null : externalPaths ? (
        <box border borderStyle="rounded" borderColor={theme.warning} style={{ flexDirection: "column", marginLeft: 2, marginRight: 2, paddingLeft: 1, paddingRight: 1 }}>
          <text fg={theme.warning}><strong>允许读取工作区外文件？</strong></text>
          {externalPaths.map((path) => <text key={path} fg={theme.text}>{path}</text>)}
          <box style={{ flexDirection: "row", gap: 2 }}>
            <box focusable onMouseDown={() => void submitWithAuthorization(externalPaths)} onKeyDown={(key) => { if (isEnterKey(key.name)) void submitWithAuthorization(externalPaths); }}><text fg={theme.success}>允许本次读取</text></box>
            <box focusable onMouseDown={() => setExternalPaths(null)} onKeyDown={(key) => { if (isEnterKey(key.name)) setExternalPaths(null); }}><text fg={theme.error}>拒绝</text></box>
          </box>
        </box>
      ) : <Composer key={composerVersion} value={draft} onChange={setDraft} onSubmit={submit} onCancel={onCancel ?? (() => undefined)} running={running} stopping={stopping} queueDepth={state.queueDepth} commands={state.commands} theme={theme} icons={icons} compact={compact} terminalWidth={width} disabled={stopping} sessionReady={state.snapshot.status === "ready"} onAction={onAction} attachments={attachments} onAddFiles={addFiles} onStagePaths={stagePaths} onPaste={paste} onRemoveAttachment={removeAttachment} snapshot={state.snapshot} onOpenPanel={(panel) => void invoke("open_panel", { panel }, panel)} />}
      {panel && !state.interaction ? <PanelOverlay panel={panel} anchor={panelAnchor} theme={theme} icons={icons} terminalWidth={width} terminalHeight={height} onClose={() => setPanel(null)} onSelect={selectPanel} busyId={panelBusyId} /> : null}
    </box>
  );
}
