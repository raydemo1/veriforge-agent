import type { CommandItem, Interaction, Snapshot, TranscriptItem, UiEvent } from "./protocol.ts";
import { DEFAULT_COMMANDS } from "./protocol.ts";
import { formatUserError } from "./errors.ts";

export const initialSnapshot: Snapshot = {
  profile: "general", permissionMode: "workspace-write", model: "starting", reasoningEffort: "high", provider: "auto",
  contextPercent: 100, status: "starting", cwd: "工作区", inputMode: "text",
};
export type AppState = {
  snapshot: Snapshot; items: TranscriptItem[]; commands: CommandItem[];
  turnState: "idle" | "running" | "queued" | "cancelling" | "cancelled"; queueDepth: number; interaction: Interaction | null;
};
export const initialState: AppState = {
  snapshot: initialSnapshot,
  items: [{ id: "welcome", kind: "status", title: "VeriForge", body: "准备工作区…", state: "running" }],
  commands: DEFAULT_COMMANDS, turnState: "idle", queueDepth: 0, interaction: null,
};

const STARTUP_COPY: Record<string, string> = {
  starting: "正在准备 Python 会话…",
  "checking workspace": "正在检查工作区…",
  "loading skills": "正在加载技能…",
  "connecting tools": "正在连接工具…",
};
const READY_HINT = "输入任务开始，/ 查看命令，@ 添加上下文";

export function reduceEvent(state: AppState, event: UiEvent): AppState {
  if (event.type === "snapshot") return { ...state, snapshot: event.snapshot };
  if (event.type === "session_reset") return { ...state, snapshot: event.snapshot, items: event.items ?? [], turnState: "idle", queueDepth: 0, interaction: null };
  if (event.type === "commands") return { ...state, commands: event.commands };
  if (event.type === "progress") {
    const welcomeState = event.status === "ready"
      ? "success"
      : event.status === "failed"
        ? "failed"
        : "running";
    const welcomeBody = event.status === "ready"
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
    if (existingIndex < 0) return { ...state, items: [...state.items, event.item] };
    return {
      ...state,
      items: state.items.map((item, index) => index === existingIndex ? event.item : item),
    };
  }
  if (event.type === "transcript_update") return { ...state, items: state.items.map((item) => item.id === event.id ? { ...item, body: event.body, state: event.state ?? item.state } : item) };
  if (event.type === "assistant_delta") return { ...state, items: state.items.map((item) => item.id === event.id ? { ...item, body: item.body + event.text } : item) };
  if (event.type === "notice") return {
    ...state,
    items: [...state.items, { id: `notice-${state.items.length}-${Date.now()}`, kind: event.level === "error" ? "error" : "status", title: event.level === "error" ? "错误" : "提示", body: event.level === "error" ? formatUserError(event.text) : event.text, state: event.level === "error" ? "failed" : "success" }],
  };
  if (event.type === "interaction") return { ...state, interaction: event };
  if (event.type === "interaction_closed" && state.interaction?.id === event.id) return { ...state, interaction: null };
  return state;
}

export function withOptimisticUserMessage(state: AppState, text: string): AppState {
  return { ...state, items: [...state.items, { id: `user-${Date.now()}`, kind: "user", title: "你", body: text }] };
}
