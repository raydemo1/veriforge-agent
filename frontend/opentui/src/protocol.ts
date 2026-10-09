export type ThemePreference = "auto" | "dark" | "light";
export type IconPreference = "auto" | "nerd" | "unicode";

export const UI_PROTOCOL_VERSION = 6;

export type Snapshot = {
  profile: string;
  permissionMode: string;
  model: string;
  reasoningEffort?: string | null;
  provider: string;
  contextPercent: number;
  contextTokens?: number;
  contextWindowTokens?: number;
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
};

export type CommandItem = { name: string; description: string; category?: string };
export type TranscriptItem = {
  id: string;
  kind: "user" | "assistant" | "tool" | "status" | "plan" | "error" | "file" | "thought" | "profile" | "agent";
  title: string;
  body: string;
  state?: "running" | "success" | "failed" | "pending" | "changed";
  role?: "group" | "message";
  parentId?: string;
  turn?: number;
  recoveryPointId?: string;
  direction?: "in" | "out";
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

export type PanelOption = { id: string; label: string; description?: string; badge?: string; tone?: "default" | "success" | "warning" | "danger"; selected?: boolean };
export type PanelSpec = {
  kind: "sessions" | "profile" | "permission" | "model" | "effort" | "mcp" | "observe" | "help";
  title: string;
  body?: string;
  options?: PanelOption[];
  searchable?: boolean;
};

// ---------------------------------------------------------------------------
// Work State (protocol v5): structured read model for the Work Strip/Workbench
// ---------------------------------------------------------------------------

export type PlanStep = { text: string; status: "pending" | "in_progress" | "completed" };
export type PlanState = {
  status: "ready" | "executing" | "completed" | "incomplete";
  revision: number;
  path: string;
  steps: PlanStep[];
  completedCount: number;
  totalCount: number;
} | null;

export type TaskItem = {
  id: string;
  text: string;
  status: "pending" | "in_progress" | "completed" | "cancelled";
};
export type TasksState = { items: TaskItem[]; completed: number; total: number };

export type WorkAgent = {
  id: string;
  name: string;
  role: string;
  task: string;
  status: "queued" | "running" | "completed" | "failed" | "blocked" | "interrupted";
  isolation: "read-only" | "isolated workspace";
  summary: string;
  error: string | null;
  proposalId: string | null;
  durationSeconds: number | null;
};

export type WorkFile = { path: string; operation: string; additions: number; deletions: number };
export type WorkProposal = {
  id: string;
  agentId: string;
  agentName: string;
  status: "ready" | "invalid" | "conflict" | "applied";
  files: WorkFile[];
  additions: number;
  deletions: number;
  invalidReasons: string[];
  conflict: { id: string; paths: string[] } | null;
};
export type ChangesState = {
  workspace: WorkFile[];
  additions: number;
  deletions: number;
  proposals: WorkProposal[];
};

export type Check = {
  id: string;
  name: string;
  status: "passed" | "failed" | "warning" | "running" | "skipped";
  detail: string;
};
export type Artifact = {
  id: string;
  kind: "image" | "text";
  path: string;
  title: string;
  detail: string;
};

export type WorkState = {
  plan: PlanState;
  tasks: TasksState;
  agents: WorkAgent[];
  changes: ChangesState;
  checks: Check[];
  artifacts: Artifact[];
};

export const initialWorkState: WorkState = {
  plan: null,
  tasks: { items: [], completed: 0, total: 0 },
  agents: [],
  changes: { workspace: [], additions: 0, deletions: 0, proposals: [] },
  checks: [],
  artifacts: [],
};

export type ActionName =
  | "rewind"
  | "open_sessions"
  | "new_session"
  | "open_panel"
  | "panel_action"
  | "toggle_permission"
  | "complete_mention"
  | "stage_attachments"
  | "remove_attachment"
  | "workbench_action";
export type ActionResult = {
  ok: boolean;
  message?: string;
  panel?: PanelSpec;
  candidates?: Array<{ insertText: string; display: string; description: string; kind: "file" | "session" }>;
  attachments?: AttachmentItem[];
  queued?: boolean;
  content?: string;
  paths?: string[];
  totalChars?: number;
  status?: string;
  conflictId?: string;
  conflictPaths?: string[];
  changedFiles?: string[];
};

export type WorkbenchOp =
  | "read_proposal"
  | "read_conflict";
export type WorkbenchParams =
  | { op: "read_proposal"; proposalId: string }
  | { op: "read_conflict"; conflictId: string };

export type UiEvent =
  | { type: "snapshot"; snapshot: Snapshot }
  | { type: "session_reset"; snapshot: Snapshot; items?: TranscriptItem[] }
  | { type: "transcript"; item: TranscriptItem }
  | { type: "transcript_update"; id: string; body: string; state?: TranscriptItem["state"] }
  | { type: "recovery_available"; turn: number; pointId: string }
  | { type: "assistant_delta"; id: string; text: string }
  | { type: "commands"; commands: CommandItem[] }
  | { type: "progress"; status: string; detail: string }
  | { type: "notice"; text: string; level?: "info" | "warning" | "error" }
  | { type: "turn_state"; state: "idle" | "running" | "queued" | "cancelling" | "cancelled"; queueDepth?: number }
  | { type: "panel"; panel: PanelSpec }
  | { type: "plan_updated"; plan: PlanState }
  | { type: "tasks_updated"; tasks: TasksState }
  | { type: "agent_run_updated"; agents: WorkAgent[] }
  | { type: "changes_updated"; changes: ChangesState }
  | { type: "verification_updated"; checks: Check[] }
  | { type: "artifact_updated"; artifacts: Artifact[] }
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
  { name: "/mcp", category: "工作流", description: "打开 MCP 服务与工具管理" },
  { name: "/compact", category: "工作流", description: "压缩当前对话上下文" },
  { name: "/context", category: "会话", description: "查看上下文预算组成" },
  { name: "/memory", category: "会话", description: "查看与管理长期记忆" },
  { name: "/fork", category: "会话", description: "从当前会话创建并进入分支" },
  { name: "/observe", category: "会话", description: "打开当前项目的运行观察" },
];
