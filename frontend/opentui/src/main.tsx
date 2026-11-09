import React from "react";
import { CliRenderEvents, createCliRenderer, createHostClipboard } from "@opentui/core";
import { createRoot } from "@opentui/react";
import { App } from "./app.tsx";
import { formatUserError } from "./errors.ts";
import { ActionName, ActionResult, BridgeMessage, BridgeRequest, DEFAULT_COMMANDS, IconPreference, SubmitResult, ThemePreference, TurnSubmission, UI_PROTOCOL_VERSION, UiEvent, SessionEvent } from "./protocol.ts";

function flag(name: string): boolean {
  return process.argv.includes(name);
}

function option(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  if (index < 0) return undefined;
  return process.argv[index + 1];
}

function mockSessionEvents(): AsyncIterable<SessionEvent> {
  return {
    async *[Symbol.asyncIterator]() {
      yield {
        type: "snapshot",
        snapshot: {
          permissionMode: "workspace-write",
          model: "deepseek-v4-flash",
          provider: "deepseek",
          contextPercent: 92,
          status: "ready",
          cwd: "veriforge-agent",
          sessionId: "preview-session",
          dirtyCount: 1,
          inputMode: "multimodal",
        },
      };
      yield { type: "progress", status: "ready", detail: "界面就绪" };
      yield {
        type: "commands",
        commands: [
          ...DEFAULT_COMMANDS,
          { name: "/handoff", category: "Skills", description: "整理当前上下文并生成交接文档" },
          { name: "/implement", category: "Skills", description: "按计划执行实现任务" },
          { name: "/triage", category: "Skills", description: "整理问题并生成可执行简报" },
          { name: "/workflows", category: "Skills", description: "查看当前可用工作流" },
        ],
      };
      yield { type: "turn_state", state: "idle" };
      yield {
        type: "plan_updated",
        plan: {
          status: "executing",
          revision: 1,
          path: "global_plan/current/plan.md",
          steps: [
            { text: "Protocol v5：新增 Work State 事件", status: "completed" },
            { text: "实现 Python 投影层", status: "completed" },
            { text: "实现 Workbench Overlay", status: "in_progress" },
            { text: "Changes 视图收敛为纯展示", status: "pending" },
            { text: "窄屏适配与视觉收尾", status: "pending" },
            { text: "补齐单测与端到端验证", status: "pending" },
          ],
          completedCount: 2,
          totalCount: 6,
        },
      };
      yield {
        type: "tasks_updated",
        tasks: {
          items: [
            { id: "t1", text: "Protocol v5：新增 Work State 事件", status: "completed" },
            { id: "t2", text: "实现 Python 投影层", status: "completed" },
            { id: "t3", text: "实现 Workbench Overlay", status: "in_progress" },
            { id: "t4", text: "Changes 视图收敛为纯展示", status: "pending" },
          ],
          completed: 2,
          total: 4,
        },
      };
      yield {
        type: "agent_run_updated",
        agents: [
          {
            id: "agent-1", name: "worker-1", role: "worker", task: "实现 Workbench 变更视图",
            status: "running", isolation: "isolated workspace", summary: "", error: null,
            proposalId: "proposal_demo1", durationSeconds: 18,
          },
        ],
      };
      yield {
        type: "changes_updated",
        changes: {
          workspace: [{ path: "harness_code_agent/tui/protocol.py", operation: "modify", additions: 8, deletions: 2 }],
          additions: 8,
          deletions: 2,
          proposals: [
            {
              id: "proposal_demo1", agentId: "agent-1", agentName: "worker-1", status: "ready",
              files: [{ path: "frontend/opentui/src/workbench/Workbench.tsx", operation: "create", additions: 420, deletions: 0 }],
              additions: 420, deletions: 0, invalidReasons: [], conflict: null,
            },
          ],
        },
      };
      yield {
        type: "verification_updated",
        checks: [
          { id: "python-syntax", name: "Python syntax", status: "passed", detail: "4 files parsed" },
          { id: "browser-verification", name: "Browser verification", status: "running", detail: "" },
          { id: "ruff-lint", name: "Ruff lint", status: "failed", detail: "[F811] app.py:12: redefinition" },
        ],
      };
      yield {
        type: "artifact_updated",
        artifacts: [
          { id: "browser-png", kind: "image", path: ".harness/artifacts/browser.png", title: "browser.png", detail: "http://localhost:5173" },
        ],
      };
      yield { type: "notice", text: "OpenTUI 已就绪：输入 / 查看命令" };
    },
  };
}

const previewSessions = [
  { id: "preview-session", title: "重构执行生命周期", environment: "local" as "local" | "worktree", cwd: "veriforge-agent", visible: true, status: "idle", pendingApproval: false },
  { id: "preview-mcp", title: "整理 MCP 注册", environment: "local" as "local" | "worktree", cwd: "veriforge-agent", visible: true, status: "idle", pendingApproval: false },
  { id: "preview-ui", title: "工作台交互", environment: "worktree" as "local" | "worktree", cwd: "veriforge-agent", visible: true, status: "idle", pendingApproval: false },
];
let previewActive = previewSessions[0]!.id;
const previewUpdates: UiEvent[] = [];
const previewSequences: Record<string, number> = {};
let wakePreview = () => {};
const previewEvent = (sessionId: string, event: SessionEvent): UiEvent => ({ ...event, sessionId, sequence: previewSequences[sessionId] = (previewSequences[sessionId] ?? 0) + 1 });
function updatePreview(event: UiEvent) { previewUpdates.push(event); wakePreview(); }
function announcePreview() { updatePreview({ type: "sessions", sessions: structuredClone(previewSessions), activeSessionId: previewActive }); }
function mockEvents(): AsyncIterable<UiEvent> {
  return { async *[Symbol.asyncIterator]() {
    yield { type: "sessions", sessions: previewSessions, activeSessionId: previewActive };
    for (const session of previewSessions) {
      for await (const event of mockSessionEvents()) {
        yield previewEvent(session.id, event.type === "snapshot" ? { ...event, snapshot: { ...event.snapshot, sessionId: session.id } } : event);
      }
    }
    while (true) {
      if (!previewUpdates.length) await new Promise<void>((resolve) => { wakePreview = resolve; });
      while (previewUpdates.length) yield previewUpdates.shift()!;
    }
  } };
}

type BridgeClient = {
  events: AsyncIterable<UiEvent>;
  closed: Promise<void>;
  submit: (sessionId: string, submission: TurnSubmission) => Promise<SubmitResult>;
  cancel: (sessionId: string) => void;
  action: (sessionId: string, name: ActionName, params?: Record<string, unknown>) => Promise<ActionResult>;
  resolveInteraction: (sessionId: string, id: string, result: Record<string, unknown>) => Promise<void>;
  shutdown: () => void;
};

function terminateBridgeTree(child: ReturnType<typeof Bun.spawn>): void {
  if (process.platform === "win32") {
    Bun.spawnSync(["taskkill", "/PID", String(child.pid), "/T", "/F"], { stdout: "ignore", stderr: "ignore" });
    return;
  }
  try {
    process.kill(-child.pid, "SIGTERM");
  } catch {
    child.kill();
  }
}

function createBridgeClient(): BridgeClient {
  const python = option("--python") ?? "python";
  const cwd = option("--cwd") ?? process.cwd();
  const args = [
    python,
    "-m",
    "harness_code_agent.tui_bridge",
    "--cwd",
    cwd,
  ];

  const child = Bun.spawn(args, {
    stdin: "pipe",
    stdout: "pipe",
    stderr: "inherit",
    detached: process.platform !== "win32",
  });
  let requestCounter = 0;
  let ended = false;
  let graceful = false;
  let buffer = "";
  const pending = new Map<string, { resolve: (value: unknown) => void; reject: (reason: unknown) => void }>();
  const eventQueue: UiEvent[] = [];
  const eventWaiters: Array<(event: UiEvent | null) => void> = [];
  let resolveClosed: () => void = () => undefined;
  const closed = new Promise<void>((resolve) => { resolveClosed = resolve; });
  const enqueue = (event: UiEvent) => {
    const waiter = eventWaiters.shift();
    if (waiter) waiter(event);
    else eventQueue.push(event);
  };

  const consumeMessage = (message: BridgeMessage) => {
    if (message.type === "event") {
      enqueue(message.event);
      return;
    }
    const waiter = pending.get(message.id);
    if (!waiter) return;
    pending.delete(message.id);
    if (message.ok) waiter.resolve(message.result);
    else waiter.reject(new Error(message.error));
  };

  const readLoop = async () => {
    if (!child.stdout) return;
    const decoder = new TextDecoder();
    try {
      for await (const chunk of child.stdout) {
        buffer += decoder.decode(chunk, { stream: true });
        let newline = buffer.indexOf("\n");
        while (newline >= 0) {
          const line = buffer.slice(0, newline).trim();
          buffer = buffer.slice(newline + 1);
          if (line) {
            try {
              consumeMessage(JSON.parse(line) as BridgeMessage);
            } catch (error) {
              console.error(`桥接消息解析失败：${formatUserError(error)}`);
            }
          }
          newline = buffer.indexOf("\n");
        }
      }
    } finally {
      ended = true;
      if (!graceful) console.error("与 Python 桥接的连接已断开。");
      resolveClosed();
    }
  };
  void readLoop();

  const request = (method: BridgeRequest["method"], params?: Record<string, unknown>, sessionId?: string) => {
    if (ended || !child.stdin) return Promise.reject(new Error("OpenTUI bridge is closed"));
    const id = `req-${++requestCounter}`;
    const message: BridgeRequest = { type: "request", id, method, params, sessionId };
    child.stdin.write(`${JSON.stringify(message)}\n`);
    child.stdin.flush();
    return new Promise<unknown>((resolve, reject) => pending.set(id, { resolve, reject }));
  };

  void request("initialize", { protocolVersion: UI_PROTOCOL_VERSION }).catch((error) => {
    console.error(`工作台启动失败：${formatUserError(error)}`);
  });

  const events: AsyncIterable<UiEvent> = {
    async *[Symbol.asyncIterator]() {
      while (true) {
        if (eventQueue.length) {
          yield eventQueue.shift() as UiEvent;
          continue;
        }
        if (ended) return;
        const event = await new Promise<UiEvent | null>((resolve) => eventWaiters.push(resolve));
        if (event === null) return;
        yield event;
      }
    },
  };

  const fire = (method: BridgeRequest["method"], params?: Record<string, unknown>) => {
    void request(method, params).catch((error) => console.error(`操作失败：${formatUserError(error)}`));
  };
  const shutdown = () => {
    if (!ended) {
      graceful = true;
      fire("shutdown");
      const timer = setTimeout(() => {
        if (!ended) { console.error("部分资源尚未退出，正在关闭工作台进程及其子进程。"); terminateBridgeTree(child); }
      }, 20000);
      void child.exited.finally(() => clearTimeout(timer));
    }
  };
  process.once("exit", () => {
    if (!ended) terminateBridgeTree(child);
  });

  return {
    events,
    closed,
    submit: (sessionId, submission) => request("submit", submission, sessionId).then((result) => result as SubmitResult),
    cancel: (sessionId) => { void request("cancel", {}, sessionId).catch(console.error); },
    action: (sessionId, name, params) => request("action", { name, params }, sessionId).then((result) => result as ActionResult),
    resolveInteraction: (sessionId, id, result) => request("resolve_interaction", { id, result }, sessionId).then(() => undefined),
    shutdown,
  };
}

async function openWindowsFilePicker(inputMode: "text" | "multimodal"): Promise<string[]> {
  if (process.platform !== "win32") throw new Error("当前版本的文件选择器仅支持 Windows");
  const filter = inputMode === "multimodal"
    ? "支持的文件|*.txt;*.md;*.json;*.yaml;*.yml;*.toml;*.xml;*.csv;*.tsv;*.py;*.js;*.jsx;*.ts;*.tsx;*.java;*.go;*.rs;*.c;*.cpp;*.h;*.hpp;*.cs;*.docx;*.png;*.jpg;*.jpeg;*.webp;*.gif;*.pdf|所有文件|*.*"
    : "文本、代码、PDF 和 Word|*.txt;*.md;*.json;*.yaml;*.yml;*.toml;*.xml;*.csv;*.tsv;*.py;*.js;*.jsx;*.ts;*.tsx;*.java;*.go;*.rs;*.c;*.cpp;*.h;*.hpp;*.cs;*.pdf;*.docx|所有文件|*.*";
  const script = [
    "Add-Type -AssemblyName System.Windows.Forms",
    "$dialog = New-Object System.Windows.Forms.OpenFileDialog",
    "$dialog.Multiselect = $true",
    `$dialog.Filter = '${filter.replaceAll("'", "''")}'`,
    "$dialog.CheckFileExists = $true",
    "$result = $dialog.ShowDialog()",
    "if ($result -eq [System.Windows.Forms.DialogResult]::OK) { $dialog.FileNames | ConvertTo-Json -Compress } else { '[]' }",
  ].join("\r\n");
  const encoded = Buffer.from(script, "utf16le").toString("base64");
  const proc = Bun.spawn(["pwsh", "-NoProfile", "-Sta", "-EncodedCommand", encoded], { stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr, exitCode] = await Promise.all([new Response(proc.stdout).text(), new Response(proc.stderr).text(), proc.exited]);
  if (exitCode !== 0) throw new Error(stderr.trim() || "无法打开文件选择器");
  const parsed = JSON.parse(stdout.trim() || "[]") as string[] | string;
  return Array.isArray(parsed) ? parsed : parsed ? [parsed] : [];
}

function mockAction(name: ActionName, params?: Record<string, unknown>, sessionId = previewActive): Promise<ActionResult> {
  if (["new_session", "fork_session", "fork_worktree"].includes(name)) {
    const id = `preview-${Date.now()}`;
    previewSessions.push({ ...previewSessions[0]!, id, title: "新会话", environment: name === "fork_worktree" ? "worktree" : "local" });
    previewActive = id; announcePreview();
    updatePreview(previewEvent(id, { type: "snapshot", snapshot: { sessionId: id, permissionMode: "workspace-write", model: "deepseek-flash", provider: "deepseek", contextPercent: 100, status: "ready", cwd: "veriforge-agent" } }));
    updatePreview(previewEvent(id, { type: "progress", status: "ready", detail: "" }));
    return Promise.resolve({ ok: true, sessionId: id });
  }
  if (name === "switch_session") {
    const target = previewSessions.find((item) => item.id === params?.sessionId);
    if (target) { target.visible = true; previewActive = target.id; announcePreview(); }
    return Promise.resolve({ ok: Boolean(target) });
  }
  if (name === "hide_session") {
    const target = previewSessions.find((item) => item.id === (params?.sessionId ?? sessionId));
    if (target) target.visible = false;
    if (previewActive === target?.id) previewActive = previewSessions.find((item) => item.visible)?.id ?? "";
    if (!previewActive) return mockAction("new_session");
    announcePreview(); return Promise.resolve({ ok: true });
  }
  if (name === "panel_action" && params?.panel === "sessions") return mockAction("switch_session", { sessionId: params.action });
  if (name === "open_sessions") return Promise.resolve({ ok: true, panel: { kind: "sessions", title: "会话", searchable: true, options: previewSessions.map((item) => ({ id: item.id, label: item.title, selected: item.id === previewActive })) } });
  if (name === "open_panel") {
    const panel = String(params?.panel ?? "help");
    if (panel === "permission") return Promise.resolve({ ok: true, panel: {
      kind: "permission",
      title: "审批模式",
      options: [
        { id: "workspace-write", label: "请求批准", description: "编辑外部文件和使用互联网时始终询问", selected: true },
        { id: "llm-auto", label: "替我审批", description: "仅对检测到的风险操作请求批准" },
        { id: "danger-full-access", label: "完全访问权限", description: "不受限制地访问互联网和电脑上的任何文件", tone: "danger" },
      ],
    } });
    if (panel === "observe") return Promise.resolve({ ok: true, panel: { kind: "observe", title: "运行观察  当前会话", body: "当前会话  状态：就绪\n令牌：0\n工具：0", options: [{ id: "observe:project", label: "切换到项目概览" }] } });

    if (panel === "mcp") return Promise.resolve({ ok: true, panel: { kind: "mcp", title: "MCP 管理", body: "已连接 0 个服务  已注册 0 个工具", options: [{ id: "reload", label: "重新加载全部 MCP 服务" }] } });
    return Promise.resolve({ ok: true, panel: { kind: "help", title: "快捷键与命令", body: "Enter 提交  Shift+Enter 换行  Ctrl+C 关闭工作台  Ctrl+O 运行观察  Ctrl+P 打开审批模式" } });
  }
  if (name === "complete_mention") return Promise.resolve({ ok: true, candidates: [
    { insertText: "session:session-1", display: "重构命令栏滚动与焦点", description: "历史会话", kind: "session" },
    { insertText: "file:README.md", display: "README.md", description: "当前工作区文件", kind: "file" },
  ] });
  if (name === "workbench_action") {
    const op = String(params?.op ?? "");
    if (op === "read_proposal" || op === "read_conflict") {
      return Promise.resolve({
        ok: true,
        content: "@@ -1,3 +1,4 @@\n context\n+added line\n-removed line\n more context",
        paths: ["frontend/opentui/src/workbench/Workbench.tsx"],
        totalChars: 86,
      });
    }
  }
  return Promise.resolve({ ok: true, message: "操作已完成" });
}

async function main() {
  const noAltScreen = flag("--no-alt-screen");
  const mock = flag("--mock");
  const bridge = flag("--bridge");
  const client = bridge ? createBridgeClient() : undefined;
  const initialTask = option("--first-task");
  const themePreference = (option("--theme") ?? "auto") as ThemePreference;
  const iconPreference = (option("--icons") ?? "auto") as IconPreference;
  const renderer = await createCliRenderer({
    exitOnCtrlC: false,
    clearOnShutdown: true,
    screenMode: noAltScreen ? "main-screen" : "alternate-screen",
  });
  const hostClipboard = createHostClipboard({ maxReadBytes: 20 * 1024 * 1024 });
  renderer.on(CliRenderEvents.FOCUS, () => renderer.requestRender());
  let rendererDestroyed = false;
  const destroyRenderer = () => {
    if (rendererDestroyed) return;
    rendererDestroyed = true;
    renderer.destroy();
  };
  let closing = false;
  const exitWorkbench = () => {
    if (closing) return;
    closing = true;
    client?.shutdown();
    void hostClipboard.dispose();
    destroyRenderer();
  };
  process.once("SIGINT", exitWorkbench);
  process.once("SIGTERM", exitWorkbench);
  process.once("SIGHUP", exitWorkbench);
  createRoot(renderer).render(
    <App
      events={client?.events ?? (mock ? mockEvents() : undefined)}
      initialTask={initialTask}
      onSubmit={client?.submit ?? (async (id, submission) => { updatePreview(previewEvent(id, { type: "transcript", item: { id: `user-${Date.now()}`, kind: "user", title: "你", body: submission.text } })); return { accepted: true }; })}
      onCancel={client?.cancel ?? (() => { if (mock) console.error("mock cancel"); })}
      onAction={client?.action ?? (mock ? ((id, name, params) => mockAction(name, params, id)) : undefined)}
      onResolveInteraction={client?.resolveInteraction}
      onPickFiles={openWindowsFilePicker}
      onReadClipboard={() => hostClipboard.read({ preferredTypes: ["image/png", "image/jpeg", "image/webp", "image/gif", "text/uri-list", "text/plain"] })}
      onCopyText={async (text) => (await hostClipboard.writeText(text, { selection: "clipboard" })).status === "written"}
      themePreference={themePreference}
      iconPreference={iconPreference}
      onExit={exitWorkbench}
    />,
  );
  void client?.closed.then(destroyRenderer);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
