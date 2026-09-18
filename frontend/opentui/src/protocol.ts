export type ThemePreference = "auto" | "dark" | "light";
export type IconPreference = "auto" | "nerd" | "unicode";

export const UI_PROTOCOL_VERSION = 4;

export type Snapshot = {
  profile: string;
  permissionMode: string;
  model: string;
  reasoningEffort?: string | null;
  provider: string;
  contextPercent: number;
  status: string;
  cwd: string;
  sessionId?: string;
  routingMode?: "auto" | "pinned";
  dirtyCount?: number;
  inputMode?: "text" | "multimodal";
};

export type AttachmentItem = {
  id: string;
  name: string;
  path: string;
  source: "clipboard" | "picker" | "mention" | "path";
  mimeType: string;
  size: number;
  sha256: string;
  kind: "text" | "docx" | "image" | "pdf";
  cached: boolean;
};

export type TurnSubmission = { text: string; attachmentIds: string[]; authorizedPaths?: string[] };
export type SubmitResult = {
  accepted: boolean;
  attachments?: AttachmentItem[];
  confirmation?: { kind: "external_paths"; paths: string[] };
};

export type CommandItem = { name: string; description: string; category?: string };
export type TranscriptItem = {
  id: string;
  kind: "user" | "assistant" | "tool" | "status" | "plan" | "todo" | "error" | "file" | "thought" | "profile";
  title: string;
  body: string;
  state?: "running" | "success" | "failed" | "pending" | "changed";
  role?: "group" | "message";
  parentId?: string;
};

export type ApprovalInteraction = {
  type: "interaction";
  id: string;
  kind: "approval";
  payload: { toolName: string; args: Record<string, unknown>; risk: string; reason: string; persistAvailable: boolean };
};
export type QuestionInteraction = {
  type: "interaction";
  id: string;
  kind: "question";
  payload: { question: string; options: Array<{ label: string; value: string; description: string; is_other: boolean }> };
};
export type Interaction = ApprovalInteraction | QuestionInteraction;

export type PanelOption = { id: string; label: string; description?: string; tone?: "default" | "success" | "warning" | "danger"; selected?: boolean };
export type PanelSpec = {
  kind: "sessions" | "profile" | "permission" | "model" | "effort" | "checkpoint" | "mcp" | "observe" | "help";
  title: string;
  body?: string;
  options?: PanelOption[];
  searchable?: boolean;
};

export type ActionName = "open_sessions" | "new_session" | "open_panel" | "panel_action" | "toggle_permission" | "complete_mention" | "stage_attachments" | "remove_attachment";
export type ActionResult = {
  ok: boolean;
  message?: string;
  panel?: PanelSpec;
  candidates?: Array<{ insertText: string; display: string; description: string; kind: "file" | "session" }>;
  attachments?: AttachmentItem[];
};

export type UiEvent =
  | { type: "snapshot"; snapshot: Snapshot }
  | { type: "session_reset"; snapshot: Snapshot; items?: TranscriptItem[] }
  | { type: "transcript"; item: TranscriptItem }
  | { type: "transcript_update"; id: string; body: string; state?: TranscriptItem["state"] }
  | { type: "assistant_delta"; id: string; text: string }
  | { type: "commands"; commands: CommandItem[] }
  | { type: "progress"; status: string; detail: string }
  | { type: "notice"; text: string; level?: "info" | "warning" | "error" }
  | { type: "turn_state"; state: "idle" | "running" | "queued" | "cancelling" | "cancelled"; queueDepth?: number }
  | { type: "panel"; panel: PanelSpec }
  | Interaction
  | { type: "interaction_closed"; id: string }
  | { type: "shutdown"; reason?: string };

export type BridgeRequest = {
  type: "request";
  id: string;
  method: "initialize" | "submit" | "cancel" | "action" | "resolve_interaction" | "shutdown";
  params?: Record<string, unknown>;
};
export type BridgeMessage =
  | { type: "response"; id: string; ok: true; result?: unknown }
  | { type: "response"; id: string; ok: false; error: string }
  | { type: "event"; event: UiEvent };

export const DEFAULT_COMMANDS: CommandItem[] = [
  { name: "/checkpoint", category: "工作流", description: "打开检查点管理" },
  { name: "/mcp", category: "工作流", description: "打开 MCP 服务与工具管理" },
  { name: "/compact", category: "工作流", description: "压缩当前对话上下文" },
  { name: "/fork", category: "会话", description: "从当前会话创建并进入分支" },
  { name: "/observe", category: "会话", description: "打开当前项目的运行观察" },
];
