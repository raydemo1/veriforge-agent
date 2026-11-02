import { act } from "react";
import { afterEach, describe, expect, test } from "bun:test";
import { testRender } from "@opentui/react/test-utils";
import type { TestRendererSetup } from "@opentui/core/testing";
import type { Renderable } from "@opentui/core";
import { App as RealApp } from "./app.tsx";
import type { ActionName, UiEvent, SessionEvent } from "./protocol.ts";
import { initialSessionState as initialState } from "./state.ts";

function App(props: React.ComponentProps<typeof RealApp>) {
  const stream = eventStream();
  return <RealApp {...props} events={props.events ?? stream.events} />;
}

type AnyNode = Renderable & { _childrenInLayoutOrder?: AnyNode[] };

function findFocused(node: AnyNode): AnyNode | null {
  if (node.focused) return node;
  for (const child of node._childrenInLayoutOrder ?? []) {
    const found = findFocused(child);
    if (found) return found;
  }
  return null;
}

function eventStream() {
  let sequence = 2;
  const queued: UiEvent[] = [{ type: "sessions", activeSessionId: "test-session", sessions: [{
    id: "test-session", title: "当前任务", cwd: "workspace", environment: "local", visible: true, status: "ready", pendingApproval: false,
  }] }, { type: "snapshot", sessionId: "test-session", sequence: 1, snapshot: { ...initialState.snapshot, sessionId: "test-session", status: "ready" } },
  { type: "progress", sessionId: "test-session", sequence: 2, status: "ready", detail: "" }];
  let waiters: Array<() => void> = [];
  return {
    events: {
      [Symbol.asyncIterator]() {
        return {
          async next() {
            while (queued.length === 0) {
              await new Promise<void>((resolve) => waiters.push(resolve));
            }
            return { value: queued.shift()!, done: false };
          },
        };
      },
    },
    push(event: SessionEvent) {
      queued.push({ ...event, sessionId: "test-session", sequence: ++sequence });
      const current = waiters;
      waiters = [];
      current.forEach((resolve) => resolve());
    },
  };
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

describe("OpenTUI app interactions", () => {
  let setup: TestRendererSetup;

  afterEach(() => {
    setup?.renderer.destroy();
  });

  test.each([50, 100])("rewind appears only on hover and keeps layout stable at W=%d", async (width) => {
    const stream = eventStream();
    const actions: Array<{ name: ActionName; params?: Record<string, unknown> }> = [];
    setup = await testRender(<App events={stream.events} onAction={async (_sessionId, name, params) => { actions.push({ name, params }); return { ok: true }; }} />, { width, height: 30 });
    stream.push({ type: "transcript", item: { id: "user-one", kind: "user", title: "你", body: "调整认证逻辑" } });
    stream.push({ type: "transcript", item: { id: "group-one", kind: "assistant", title: "助手", body: "", role: "group", state: "success", turn: 1 } });
    stream.push({ type: "transcript", item: { id: "answer-one", kind: "assistant", title: "助手", body: "认证逻辑已调整", parentId: "group-one", state: "success" } });
    stream.push({ type: "recovery_available", turn: 1, pointId: "point-one" });
    await sleep(300);
    const initial = await setup.waitForFrame((frame) => frame.includes("认证逻辑已调整"));
    expect(initial).not.toContain("回撤到此处");
    expect(initial.split("\n")[0]).not.toContain("回撤");
    const answerRow = initial.split("\n").findIndex((row) => row.includes("认证逻辑已调整"));
    const userRow = initial.split("\n").findIndex((row) => row.includes("调整认证逻辑"));
    await setup.mockMouse.moveTo(5, userRow);
    await sleep(50);
    const hovered = await setup.waitForFrame((frame) => frame.includes("回撤到此处"));
    expect(hovered.split("\n").findIndex((row) => row.includes("认证逻辑已调整"))).toBe(answerRow);
    const buttonRow = hovered.split("\n").findIndex((row) => row.includes("回撤到此处"));
    const buttonColumn = hovered.split("\n")[buttonRow].indexOf("回撤到此处");
    await setup.mockMouse.moveTo(buttonColumn + 1, buttonRow);
    await sleep(30);
    expect(setup.captureCharFrame()).toContain("回撤到此处");
    await setup.mockMouse.click(buttonColumn + 1, buttonRow);
    await sleep(30);
    expect(actions).toEqual([{ name: "rewind", params: { pointId: "point-one" } }]);
    await setup.mockMouse.moveTo(0, 0);
    await sleep(100);
    await setup.waitForFrame((frame) => !frame.includes("回撤到此处"));
  });

  test("keyboard focus exposes rewind and running turns disable it", async () => {
    const stream = eventStream();
    const actions: ActionName[] = [];
    setup = await testRender(<App events={stream.events} onAction={async (_sessionId, name) => { actions.push(name); return { ok: true }; }} />, { width: 100, height: 30 });
    stream.push({ type: "transcript", item: { id: "group-keyboard", kind: "assistant", title: "助手", body: "", role: "group", state: "success", turn: 1, recoveryPointId: "point-keyboard" } });
    await setup.renderOnce();
    await sleep(50);
    const node = setup.renderer.root.findDescendantById("recovery-turn-group-keyboard")!;
    node.focus();
    await sleep(100);
    await setup.waitForFrame((frame) => frame.includes("回撤到此处"));
    setup.mockInput.pressEnter();
    await sleep(30);
    expect(actions).toEqual(["rewind"]);
    stream.push({ type: "turn_state", state: "running" });
    await sleep(50);
    await setup.waitForFrame((frame) => !frame.includes("回撤到此处"));
    setup.mockInput.pressEnter();
    await sleep(30);
    expect(actions).toEqual(["rewind"]);
  });

  test("finished tool calls collapse into a per-kind summary that expands", async () => {
    const stream = eventStream();
    setup = await testRender(
      <App
        events={stream.events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async () => ({ ok: true })}
      />,
      { width: 100, height: 30 }
    );

    stream.push({ type: "transcript", item: { id: "g1", kind: "assistant", title: "助手", body: "", state: "running", role: "group" } });
    await setup.waitForFrame((frame) => frame.includes("思考中"));
    stream.push({ type: "transcript", item: { id: "t1", kind: "tool", title: "文件已阅读", body: "calculator.html", state: "success", parentId: "g1" } });
    await setup.waitForFrame((frame) => frame.includes("已阅读 1 个文件"));
    stream.push({ type: "transcript", item: { id: "t2", kind: "thought", title: "思考", body: "740ms", state: "success", parentId: "g1" } });
    stream.push({ type: "transcript", item: { id: "t3", kind: "tool", title: "文件已阅读", body: "readme.md", state: "success", parentId: "g1" } });
    stream.push({ type: "transcript", item: { id: "t5", kind: "tool", title: "记忆已搜索", body: "notes", state: "success", parentId: "g1" } });
    stream.push({ type: "transcript", item: { id: "t4", kind: "tool", title: "命令已执行", body: "git status", state: "failed", parentId: "g1" } });
    // waitForFrame bails out early when the scheduler looks idle, which can
    // happen while act still holds queued updates; poll with forced renders.
    let collapsed = "";
    for (let i = 0; i < 40; i++) {
      await sleep(25);
      await setup.renderOnce();
      collapsed = setup.captureCharFrame();
      if (collapsed.includes("已阅读 2 个文件") && collapsed.includes("执行了工具 1 次") && collapsed.includes("git status")) break;
    }
    expect(collapsed).toContain("▸ 已阅读 2 个文件，执行了工具 1 次");
    // Collapsed: hidden rows stay out of the frame, failures stay visible.
    expect(collapsed).toContain("命令已执行");
    expect(collapsed).not.toContain("calculator.html");

    // Clicking the summary row expands the hidden rows.
    const summaryLine = collapsed.split("\n").find((line) => line.includes("已阅读 2 个文件"))!;
    const mx = Math.max(1, summaryLine.indexOf("已阅读"));
    const my = collapsed.split("\n").indexOf(summaryLine);
    await setup.mockMouse.click(mx, my);
    let opened = "";
    for (let i = 0; i < 40; i++) {
      await sleep(25);
      await setup.renderOnce();
      opened = setup.captureCharFrame();
      if (opened.includes("calculator.html") && opened.includes("▾ 已阅读 2 个文件")) break;
    }
    expect(opened).toContain("calculator.html");
  });

  test("Enter is ignored for a panel option while its action is in flight", async () => {
    const stream = eventStream();
    const panelActions: string[] = [];
    const releaseHolder: { release?: () => void } = {};
    setup = await testRender(
      <App
        events={stream.events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async (_sessionId: string, name: ActionName) => {
          if (name === "panel_action") {
            panelActions.push(name);
            await new Promise<void>((resolve) => { releaseHolder.release = resolve; });
          }
          return { ok: true };
        }}
      />,
      { width: 100, height: 30 }
    );

    stream.push({
      type: "panel",
      panel: { kind: "observe", title: "运行观察", options: [{ id: "go", label: "切换到项目概览" }] },
    });
    await setup.waitForFrame((frame) => frame.includes("切换到项目概览"));

    setup.mockInput.pressEnter();
    await setup.waitForFrame((frame) => frame.includes("处理中"));
    setup.mockInput.pressEnter();
    await sleep(50);

    expect(panelActions).toEqual(["panel_action"]);
    releaseHolder.release?.();
  });

  test("mention completion fires once after rapid typing", async () => {
    const stream = eventStream();
    const actions: ActionName[] = [];
    setup = await testRender(
      <App
        events={stream.events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async (_sessionId: string, name: ActionName) => {
          actions.push(name);
          return { ok: true, candidates: [] };
        }}
      />,
      { width: 100, height: 30 }
    );

    stream.push({
      type: "snapshot",
      snapshot: { ...initialState.snapshot, sessionId: "session-1", status: "ready" },
    });
    await setup.waitForFrame((frame) => frame.includes("从这里开始"));

    await setup.mockInput.typeText("@readme", 8);
    await sleep(300);

    expect(actions.filter((name) => name === "complete_mention").length).toBe(1);
  });

  test("approval resolves through keyboard and two-stage mouse click", async () => {
    const stream = eventStream();
    const resolutions: Record<string, unknown>[] = [];
    setup = await testRender(
      <App
        events={stream.events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async () => ({ ok: true })}
        onResolveInteraction={async (_sessionId, _id, result) => { resolutions.push(result); }}
      />,
      { width: 100, height: 30 }
    );
    stream.push({
      type: "interaction",
      id: "approval-1",
      kind: "approval",
      payload: { toolName: "run_bash", args: { command: "pytest" }, risk: "高风险", reason: "运行测试", persistAvailable: true },
    });
    await setup.waitForFrame((frame) => frame.includes("[2] 本项目后续允许"));

    setup.mockInput.pressEnter();
    await sleep(50);
    expect(resolutions).toEqual([{ decision: "approve" }]);
  });

  test("approval option can be selected and confirmed with the mouse", async () => {
    const stream = eventStream();
    const resolutions: Record<string, unknown>[] = [];
    setup = await testRender(
      <App
        events={stream.events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async () => ({ ok: true })}
        onResolveInteraction={async (_sessionId, _id, result) => { resolutions.push(result); }}
      />,
      { width: 100, height: 30 }
    );
    stream.push({
      type: "interaction",
      id: "approval-2",
      kind: "approval",
      payload: { toolName: "run_bash", args: { command: "pytest" }, risk: "高风险", reason: "运行测试", persistAvailable: false },
    });
    const frame = await setup.waitForFrame((frame) => frame.includes("[2] 拒绝"));
    const row = frame.split("\n").find((line) => line.includes("[2] 拒绝"))!;
    const x = Math.max(1, row.indexOf("[2]"));
    const y = frame.split("\n").indexOf(row);

    await setup.mockMouse.click(x + 1, y);
    await setup.waitForFrame((next) => next.includes("› [2] 拒绝"));
    expect(resolutions).toEqual([]);
    await setup.mockMouse.click(x + 1, y);
    await sleep(50);
    expect(resolutions).toEqual([{ decision: "deny" }]);
  });

  test("question other branch is escapable via arrow keys and submits custom text", async () => {
    const stream = eventStream();
    const resolutions: Record<string, unknown>[] = [];
    setup = await testRender(
      <App
        events={stream.events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async () => ({ ok: true })}
        onResolveInteraction={async (_sessionId, _id, result) => { resolutions.push(result); }}
      />,
      { width: 100, height: 30 }
    );
    stream.push({
      type: "interaction",
      id: "question-1",
      kind: "question",
      payload: {
        question: "接下来怎么做？",
        options: [
          { label: "直接修复", value: "fix", description: "", is_other: false },
          { label: "其他", value: "", description: "", is_other: true },
        ],
      },
    });
    await setup.waitForFrame((frame) => frame.includes("[2] 其他"));

    setup.mockInput.pressArrow("down");
    await setup.waitForFrame((frame) => frame.includes("其他说明"));
    setup.mockInput.pressArrow("up");
    await setup.waitForFrame((frame) => frame.includes("› [1] 直接修复") && !frame.includes("其他说明"));

    setup.mockInput.pressArrow("down");
    await setup.waitForFrame((frame) => frame.includes("其他说明"));
    await setup.mockInput.typeText("用方案B");
    setup.mockInput.pressEnter();
    await sleep(50);
    expect(resolutions).toEqual([{ selectedIndex: 1, customText: "用方案B" }]);
  });

  test("failed startup keeps the draft but blocks submission", async () => {
    const stream = eventStream();
    let submitCalls = 0;
    setup = await testRender(
      <App
        events={stream.events}
        onSubmit={async () => { submitCalls += 1; return { accepted: true } as never; }}
        onAction={async () => ({ ok: true })}
      />,
      { width: 100, height: 30 }
    );
    stream.push({ type: "progress", status: "failed", detail: "缺少运行依赖" });
    await setup.waitForFrame((frame) => frame.includes("会话启动失败"));
    await setup.mockInput.typeText("修复这个 bug");
    await sleep(200);
    setup.mockInput.pressEnter();
    await sleep(300);
    expect(setup.captureCharFrame()).toContain("会话未就绪");
    expect(submitCalls).toBe(0);
  });

  test("mouse click on a header button does not leave focus on it", async () => {
    setup = await testRender(
      <App
        events={eventStream().events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async () => ({ ok: true })}
      />,
      { width: 100, height: 30 }
    );
    await sleep(300);
    const headerLines = setup.captureCharFrame().split("\n");
    const labelY = headerLines.findIndex((line) => line.includes("＋ 新建"));
    const labelX = headerLines[labelY]?.indexOf("新建") ?? -1;
    expect(labelX).toBeGreaterThan(-1);

    await setup.mockMouse.click(labelX + 1, labelY);
    await sleep(150);

    const focused = findFocused((setup.renderer as unknown as { root: AnyNode }).root);
    const focusedKind = focused?.constructor?.name ?? "none";
    // Focus returns to the composer textarea instead of lingering on the
    // button; otherwise the hover highlight would stay after mouse-out.
    expect(focusedKind === "TextareaRenderable" || focusedKind === "none").toBe(true);
  });

  test.each([50, 70, 100])("right-side controls stay right-aligned at W=%d", async (terminalWidth) => {
    setup = await testRender(
      <App
        events={eventStream().events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async () => ({ ok: true })}
      />,
      { width: terminalWidth, height: 30 }
    );
    // Default snapshot already renders both toolbar groups.
    await sleep(300);
    const events = (() => {
      const root = (setup.renderer as unknown as { root: AnyNode }).root;
      return [root, ...(function collect(node: AnyNode): AnyNode[] {
        return (node._childrenInLayoutOrder ?? []).flatMap((child) => [child as AnyNode, ...collect(child as AnyNode)]);
      })(root)];
    })();
    const groups = events.filter((node) => (node._childrenInLayoutOrder?.length ?? 0) >= 3 && node.width >= 10);
    const rightGroup = groups.at(-1)!;
    // Right edge sits within 4 columns of the terminal edge (composer
    // padding + border); the group itself hugs its parent row's right edge.
    expect(terminalWidth - (rightGroup.x + rightGroup.width)).toBeLessThanOrEqual(4);
  });

  test.each([50, 100])("context gauge popover shows full details at W=%d", async (terminalWidth) => {
    setup = await testRender(
      <App
        events={eventStream().events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async () => ({ ok: true })}
      />,
      { width: terminalWidth, height: 30 }
    );
    await sleep(300);
    const all = (function collect(node: AnyNode): AnyNode[] {
      return [node, ...(node._childrenInLayoutOrder ?? []).flatMap((child) => collect(child as AnyNode))];
    })((setup.renderer as unknown as { root: AnyNode }).root);
    const groups = all.filter((node) => (node._childrenInLayoutOrder?.length ?? 0) >= 3 && node.width >= 10);
    const gauge = groups.at(-1)!._childrenInLayoutOrder?.[0]!;
    gauge.focus();
    await sleep(200);
    const frame = setup.captureCharFrame();
    expect(frame).toContain("已使用");
  });
});
