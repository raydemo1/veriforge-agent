import type { CommandItem, Interaction, Snapshot, TranscriptItem, SessionEvent, UiEvent, WorkState, SessionDescriptor, AttachmentItem, PanelSpec } from "./protocol.ts";
import { DEFAULT_COMMANDS, initialWorkState } from "./protocol.ts";
import { formatUserError } from "./errors.ts";

export const initialSnapshot: Snapshot = {
  profile: "general", permissionMode: "workspace-write", model: "starting", reasoningEffort: "high", provider: "auto",
  contextPercent: 100, status: "starting", cwd: "工作区", inputMode: "text",
};
export type SessionViewState = {
  snapshot: Snapshot; items: TranscriptItem[]; commands: CommandItem[];
  turnState: "idle" | "running" | "queued" | "cancelling" | "cancelled"; queueDepth: number; interaction: Interaction | null;
  work: WorkState;
};
export const initialSessionState: SessionViewState = {
  snapshot: initialSnapshot,
  items: [{ id: "welcome", kind: "status", title: "VeriForge", body: "准备工作区…", state: "running" }],
  commands: DEFAULT_COMMANDS, turnState: "idle", queueDepth: 0, interaction: null,
  work: initialWorkState,
};

const STARTUP_COPY: Record<string, string> = {
  starting: "正在准备 Python 会话…",
  "checking workspace": "正在检查工作区…",
  "loading skills": "正在加载技能…",
  "connecting tools": "正在连接工具…",
  "connecting external tools": "正在连接外部工具…",
};
const READY_HINT = "今天想造点什么？";
// "ready" = core session usable; "external tools ready" = background MCP
// warm-up also finished, i.e. the whole startup is done.
const READY_STATUSES = new Set(["ready", "external tools ready"]);

export function reduceSessionEvent(state: SessionViewState, event: SessionEvent): SessionViewState {
  if (event.type === "snapshot") return { ...state, snapshot: event.snapshot };
  if (event.type === "session_reset") return { ...state, snapshot: event.snapshot, items: event.items ?? [], turnState: "idle", queueDepth: 0, interaction: null, work: initialWorkState };
  if (event.type === "commands") return { ...state, commands: event.commands };
  if (event.type === "progress") {
    const isReady = READY_STATUSES.has(event.status);
    const welcomeState = isReady
      ? "success"
      : event.status === "failed"
        ? "failed"
        : "running";
    const welcomeBody = isReady
      ? READY_HINT
      : event.status === "failed"
        ? (event.detail || "会话启动失败")
        : STARTUP_COPY[event.status] || event.detail || event.status;
    return {
      ...state, snapshot: { ...state.snapshot, status: event.status },
      items: state.items.map((item) => item.id === "welcome" ? {
        ...item,
        title: event.status === "failed" ? "会话启动失败" : item.title,
        body: welcomeBody,
        state: welcomeState,
      } : item),
    };
  }
  if (event.type === "turn_state") return { ...state, turnState: event.state, queueDepth: event.queueDepth ?? (event.state === "queued" ? state.queueDepth + 1 : 0) };
  if (event.type === "transcript") {
    const existingIndex = state.items.findIndex((item) => item.id === event.item.id);
    if (existingIndex < 0) {
      // The welcome row is an empty-state placeholder: drop it as soon as
      // real content (anything other than notices) enters the transcript.
      const realContent = event.item.kind !== "status" && event.item.kind !== "error";
      const items = realContent ? state.items.filter((item) => item.id !== "welcome") : state.items;
      return { ...state, items: [...items, event.item] };
    }
    return {
      ...state,
      items: state.items.map((item, index) => index === existingIndex ? event.item : item),
    };
  }
  if (event.type === "transcript_update") return { ...state, items: state.items.map((item) => item.id === event.id ? { ...item, body: event.body, state: event.state ?? item.state } : item) };
  if (event.type === "recovery_available") {
    if (event.turn === 0) return { ...state, items: [{ id: "recovery-origin", kind: "status", title: "会话开始", body: "", turn: 0, recoveryPointId: event.pointId }, ...state.items.filter((item) => item.id !== "recovery-origin")] };
    return { ...state, items: state.items.map((item) => item.role === "group" && item.turn === event.turn ? { ...item, recoveryPointId: event.pointId } : item) };
  }
  if (event.type === "assistant_delta") return { ...state, items: state.items.map((item) => item.id === event.id ? { ...item, body: item.body + event.text } : item) };
  if (event.type === "notice") return {
    ...state,
    items: [...state.items, { id: `notice-${state.items.length}-${Date.now()}`, kind: event.level === "error" ? "error" : "status", title: event.level === "error" ? "错误" : "提示", body: event.level === "error" ? formatUserError(event.text) : event.text, state: event.level === "error" ? "failed" : "success" }],
  };
  if (event.type === "interaction") return { ...state, interaction: event };
  if (event.type === "interaction_closed" && state.interaction?.id === event.id) return { ...state, interaction: null };
  if (event.type === "plan_updated") return { ...state, work: { ...state.work, plan: event.plan } };
  if (event.type === "tasks_updated") return { ...state, work: { ...state.work, tasks: event.tasks } };
  if (event.type === "agent_run_updated") return { ...state, work: { ...state.work, agents: event.agents } };
  if (event.type === "changes_updated") return { ...state, work: { ...state.work, changes: event.changes } };
  if (event.type === "verification_updated") return { ...state, work: { ...state.work, checks: event.checks } };
  if (event.type === "artifact_updated") return { ...state, work: { ...state.work, artifacts: event.artifacts } };
  return state;
}

export type SessionUI = {
  draft: string; attachments: AttachmentItem[]; workbenchOpen: boolean;
  workbenchTab: "plan" | "tasks" | "agents" | "changes" | "checks" | "artifacts";
  scrollTop: number | null; panel: PanelSpec | null;
};
export const initialSessionUI: SessionUI = {
  draft: "", attachments: [], workbenchOpen: false, workbenchTab: "plan", scrollTop: null, panel: null,
};
export type SessionState = {
  view: SessionViewState; ui: SessionUI; sequence: number; unreadCount: number;
};
export type AppState = {
  activeSessionId: string | null; sessionOrder: string[];
  descriptors: Record<string, SessionDescriptor>; sessions: Record<string, SessionState>;
};
export const initialState: AppState = { activeSessionId: null, sessionOrder: [], descriptors: {}, sessions: {} };
export type AppAction = UiEvent | {
  type: "session_ui"; sessionId: string; field: keyof SessionUI;
  value: SessionUI[keyof SessionUI] | ((current: any) => any);
} | { type: "local_event"; sessionId: string; event: SessionEvent };
const emptySession = (): SessionState => ({ view: structuredClone(initialSessionState), ui: structuredClone(initialSessionUI), sequence: 0, unreadCount: 0 });
export function reduceEvent(state: AppState, action: AppAction): AppState {
  if (action.type === "sessions") {
    const sessions = { ...state.sessions };
    for (const descriptor of action.sessions) sessions[descriptor.id] ??= emptySession();
    if (action.activeSessionId && sessions[action.activeSessionId])
      sessions[action.activeSessionId] = { ...sessions[action.activeSessionId]!, unreadCount: 0 };
    return { ...state, sessions, activeSessionId: action.activeSessionId,
      sessionOrder: action.sessions.filter((s) => s.visible).map((s) => s.id),
      descriptors: Object.fromEntries(action.sessions.map((s) => [s.id, s])) };
  }
  if (!("sessionId" in action)) return state;
  const sessionId = action.sessionId;
  const session = state.sessions[sessionId] ?? emptySession();
  let updated: SessionState;
  if (action.type === "session_ui") {
    const value = typeof action.value === "function" ? action.value(session.ui[action.field]) : action.value;
    updated = { ...session, ui: { ...session.ui, [action.field]: value } };
  } else if (action.type === "local_event") {
    updated = { ...session, view: reduceSessionEvent(session.view, action.event) };
  } else if (action.type === "session_view") {
    if (action.view.sequence < session.sequence) return state;
    updated = { ...session, sequence: action.view.sequence,
      view: { ...session.view, ...action.view, snapshot: { ...initialSnapshot, ...action.view.snapshot }, work: { ...initialWorkState, ...action.view.work } } };
  } else {
    if (action.sequence <= session.sequence) return state;
    const unread = state.activeSessionId !== sessionId &&
      (action.type === "interaction" || (action.type === "turn_state" && action.state === "idle"));
    updated = { ...session, sequence: action.sequence, view: reduceSessionEvent(session.view, action),
      unreadCount: session.unreadCount + (unread ? 1 : 0),
      ui: action.type === "panel" ? { ...session.ui, panel: action.panel }
        : action.type === "interaction" ? { ...session.ui, panel: null } : session.ui };
  }
  return { ...state, sessions: { ...state.sessions, [sessionId]: updated } };
}
