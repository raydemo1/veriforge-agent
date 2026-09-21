import React from "react";
import { CliRenderEvents, createCliRenderer, createHostClipboard } from "@opentui/core";
import { createRoot } from "@opentui/react";
import { App } from "./app.tsx";
import { formatUserError } from "./errors.ts";
import { ActionName, ActionResult, BridgeMessage, BridgeRequest, DEFAULT_COMMANDS, IconPreference, SubmitResult, ThemePreference, TurnSubmission, UI_PROTOCOL_VERSION, UiEvent } from "./protocol.ts";

function flag(name: string): boolean {
  return process.argv.includes(name);
}

function option(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  if (index < 0) return undefined;
  return process.argv[index + 1];
}

function mockEvents(): AsyncIterable<UiEvent> {
  return {
    async *[Symbol.asyncIterator]() {
      yield {
        type: "snapshot",
        snapshot: {
          profile: "coding-agent",
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
      yield { type: "notice", text: "OpenTUI 已就绪：输入 / 查看命令" };
    },
  };
}

type BridgeClient = {
  events: AsyncIterable<UiEvent>;
  closed: Promise<void>;
  submit: (submission: TurnSubmission) => Promise<SubmitResult>;
  cancel: () => void;
  action: (name: ActionName, params?: Record<string, unknown>) => Promise<ActionResult>;
  resolveInteraction: (id: string, result: Record<string, unknown>) => Promise<void>;
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
  const profile = option("--profile") ?? "general";
  const args = [
    python,
    "-m",
    "harness_code_agent.tui_bridge",
    "--cwd",
    cwd,
    "--profile",
    profile,
  ];
  if (flag("--profile-explicit")) args.push("--profile-explicit");

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
  let readySettled = false;
  let resolveReady: () => void = () => undefined;
  let rejectReady: (reason: unknown) => void = () => undefined;
  let sessionReady = Promise.resolve();
  const resetReady = () => {
    readySettled = false;
    sessionReady = new Promise<void>((resolve, reject) => {
      resolveReady = resolve;
      rejectReady = reject;
    });
    void sessionReady.catch(() => undefined);
  };
  resetReady();
  const markReady = () => {
    if (readySettled) return;
    readySettled = true;
    resolveReady();
  };
  const markReadyFailed = (reason: unknown) => {
    if (readySettled) return;
    readySettled = true;
    rejectReady(reason);
  };

  const enqueue = (event: UiEvent) => {
    const waiter = eventWaiters.shift();
    if (waiter) waiter(event);
    else eventQueue.push(event);
  };

  const consumeMessage = (message: BridgeMessage) => {
    if (message.type === "event") {
      if (message.event.type === "progress" && message.event.status === "starting" && readySettled) resetReady();
      if (message.event.type === "progress" && message.event.status === "ready") markReady();
      if (message.event.type === "progress" && message.event.status === "failed") markReadyFailed(new Error(message.event.detail));
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
      markReadyFailed(new Error("OpenTUI bridge exited before the session became ready"));
      // Surface the bridge exit inside the UI instead of only tearing down.
      if (!graceful) {
        enqueue({
          type: "notice",
          level: "error",
          text: readySettled
            ? "与 Python 桥接的连接已断开，当前暂时无法提交任务。"
            : "会话启动失败，请检查 Python 环境与依赖后重试。",
        });
      }
      for (const waiter of pending.values()) waiter.reject(new Error("OpenTUI bridge exited"));
      pending.clear();
      while (eventWaiters.length) eventWaiters.shift()?.(null);
      resolveClosed();
    }
  };
  void readLoop();

  const request = (method: BridgeRequest["method"], params?: Record<string, unknown>) => {
    if (ended || !child.stdin) return Promise.reject(new Error("OpenTUI bridge is closed"));
    const id = `req-${++requestCounter}`;
    const message: BridgeRequest = { type: "request", id, method, params };
    child.stdin.write(`${JSON.stringify(message)}\n`);
    child.stdin.flush();
    return new Promise<unknown>((resolve, reject) => pending.set(id, { resolve, reject }));
  };

  void request("initialize", { protocolVersion: UI_PROTOCOL_VERSION }).then((result) => {
    const status = (result as { status?: string } | undefined)?.status;
    if (status === "ready") markReady();
  }).catch((error) => {
    markReadyFailed(error);
    enqueue({ type: "notice", level: "error", text: `会话启动失败：${formatUserError(error)}` });
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
    }
  };
  process.once("exit", () => {
    if (!ended) terminateBridgeTree(child);
  });

  return {
    events,
    closed,
    submit: (submission) => sessionReady.then(() => request("submit", submission)).then((result) => result as SubmitResult),
    cancel: () => fire("cancel"),
    action: (name, params) => sessionReady.then(() => request("action", { name, params })).then((result) => result as ActionResult),
    resolveInteraction: (id, result) => request("resolve_interaction", { id, result }).then(() => undefined),
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

function mockAction(name: ActionName, params?: Record<string, unknown>): Promise<ActionResult> {
  if (name === "open_sessions") return Promise.resolve({ ok: true, panel: { kind: "sessions", title: "历史会话", searchable: true, options: [
    { id: "session-1", label: "重构命令栏滚动与焦点", description: "coding-agent  刚刚" },
    { id: "session-2", label: "检查 MCP 工具加载", description: "general  2 小时前" },
  ] } });
  if (name === "open_panel") {
    const panel = String(params?.panel ?? "help");
    if (panel === "profile") return Promise.resolve({ ok: true, panel: {
      kind: "profile",
      title: "工作模式",
      options: [
        { id: "auto", label: "自动路由", description: "根据当前任务选择工作模式" },
        { id: "general", label: "通用", description: "适合一般问答、分析和轻量任务" },
        { id: "coding-agent", label: "工作区可写", description: "允许模型直接修改工作区文件", selected: true },
      ],
    } });
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
    if (panel === "checkpoint") return Promise.resolve({ ok: true, panel: { kind: "checkpoint", title: "检查点", body: "当前：自动开启  每 5 轮", options: [{ id: "create", label: "立即创建检查点" }, { id: "auto_on", label: "开启自动检查点" }, { id: "auto_off", label: "关闭自动检查点" }] } });
    if (panel === "mcp") return Promise.resolve({ ok: true, panel: { kind: "mcp", title: "MCP 管理", body: "已连接 0 个服务  已注册 0 个工具", options: [{ id: "reload", label: "重新加载全部 MCP 服务" }] } });
    return Promise.resolve({ ok: true, panel: { kind: "help", title: "快捷键与命令", body: "Enter 提交  Shift+Enter 换行  Ctrl+C 取消/退出  Ctrl+O 运行观察  Ctrl+P 打开审批模式" } });
  }
  if (name === "complete_mention") return Promise.resolve({ ok: true, candidates: [
    { insertText: "session:session-1", display: "重构命令栏滚动与焦点", description: "历史会话", kind: "session" },
    { insertText: "file:README.md", display: "README.md", description: "当前工作区文件", kind: "file" },
  ] });
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
  createRoot(renderer).render(
    <App
      events={client?.events ?? (mock ? mockEvents() : undefined)}
      initialTask={initialTask}
      onSubmit={client?.submit ?? (async (submission) => { if (mock) console.error(`mock submit: ${submission.text}`); return { accepted: true }; })}
      onCancel={client?.cancel ?? (() => { if (mock) console.error("mock cancel"); })}
      onAction={client?.action ?? (mock ? mockAction : undefined)}
      onResolveInteraction={client?.resolveInteraction}
      onPickFiles={openWindowsFilePicker}
      onReadClipboard={() => hostClipboard.read({ preferredTypes: ["image/png", "image/jpeg", "image/webp", "image/gif", "text/uri-list", "text/plain"] })}
      onCopyText={async (text) => (await hostClipboard.writeText(text, { selection: "clipboard" })).status === "written"}
      themePreference={themePreference}
      iconPreference={iconPreference}
      onExit={() => {
        client?.shutdown();
        void hostClipboard.dispose();
        destroyRenderer();
      }}
    />,
  );
  void client?.closed.then(destroyRenderer);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
