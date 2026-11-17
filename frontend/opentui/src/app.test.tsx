import { act } from "react";
import { afterEach, describe, expect, test } from "bun:test";
import { testRender } from "@opentui/react/test-utils";
import type { TestRendererSetup } from "@opentui/core/testing";
import type { Renderable } from "@opentui/core";
import { App } from "./app.tsx";
import type { ActionName, UiEvent, SessionEvent } from "./protocol.ts";
import { initialSessionState as initialState } from "./state.ts";

async function update(action: () => unknown) {
  await act(async () => { await action(); });
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
      return update(() => {
        queued.push({ ...event, sessionId: "test-session", sequence: ++sequence });
        const current = waiters;
        waiters = [];
        current.forEach((resolve) => resolve());
      });
    },
  };
}

describe("OpenTUI app interactions", () => {
  let setup: TestRendererSetup;

  function renderApp(...args: Parameters<typeof testRender>) {
    return update(async () => { setup = await testRender(...args); });
  }

  afterEach(() => update(() => setup?.renderer.destroy()));

  test.each([50, 100])("rewind appears only on hover and keeps layout stable at W=%d", async (width) => {
    const stream = eventStream();
    const actions: Array<{ name: ActionName; params?: Record<string, unknown> }> = [];
    await renderApp(<App events={stream.events} onAction={async (_sessionId, name, params) => { actions.push({ name, params }); return { ok: true }; }} />, { width, height: 30 });
    await stream.push({ type: "transcript", item: { id: "user-one", kind: "user", title: "你", body: "调整认证逻辑" } });
    await stream.push({ type: "transcript", item: { id: "group-one", kind: "assistant", title: "助手", body: "", role: "group", state: "success", turn: 1 } });
    await stream.push({ type: "transcript", item: { id: "answer-one", kind: "assistant", title: "助手", body: "认证逻辑已调整", parentId: "group-one", state: "success" } });
    await stream.push({ type: "recovery_available", turn: 1, pointId: "point-one" });
    const initial = await setup.waitForFrame((frame) => frame.includes("认证逻辑已调整"));
    expect(initial).not.toContain("回撤到此处");
    expect(initial.split("\n")[0]).not.toContain("回撤");
    const answerRow = initial.split("\n").findIndex((row) => row.includes("认证逻辑已调整"));
    const userRow = initial.split("\n").findIndex((row) => row.includes("调整认证逻辑"));
    await update(() => setup.mockMouse.moveTo(5, userRow));
    const hovered = await setup.waitForFrame((frame) => frame.includes("回撤到此处"));
    expect(hovered.split("\n").findIndex((row) => row.includes("认证逻辑已调整"))).toBe(answerRow);
    const buttonRow = hovered.split("\n").findIndex((row) => row.includes("回撤到此处"));
    const buttonColumn = hovered.split("\n")[buttonRow].indexOf("回撤到此处");
    await update(() => setup.mockMouse.moveTo(buttonColumn + 1, buttonRow));
    await setup.waitForFrame((frame) => frame.includes("回撤到此处"));
    await update(() => setup.mockMouse.click(buttonColumn + 1, buttonRow));
    expect(actions).toEqual([{ name: "rewind", params: { pointId: "point-one" } }]);
    await update(() => setup.mockMouse.moveTo(0, 0));
    await setup.waitForFrame((frame) => !frame.includes("回撤到此处"));
  });

  test("keyboard focus exposes rewind and running turns disable it", async () => {
    const stream = eventStream();
    const actions: ActionName[] = [];
    await renderApp(<App events={stream.events} onAction={async (_sessionId, name) => { actions.push(name); return { ok: true }; }} />, { width: 100, height: 30 });
    await stream.push({ type: "transcript", item: { id: "group-keyboard", kind: "assistant", title: "助手", body: "", role: "group", state: "success", turn: 1, recoveryPointId: "point-keyboard" } });
    await update(() => setup.renderOnce());
    const node = setup.renderer.root.findDescendantById("recovery-turn-group-keyboard")!;
    await update(() => node.focus());
    await setup.waitForFrame((frame) => frame.includes("回撤到此处"));
    await update(() => setup.mockInput.pressEnter());
    expect(actions).toEqual(["rewind"]);
    await stream.push({ type: "turn_state", state: "running" });
    await setup.waitForFrame((frame) => !frame.includes("回撤到此处"));
    await update(() => setup.mockInput.pressEnter());
    expect(actions).toEqual(["rewind"]);
  });

  test("finished tool calls collapse into a per-kind summary that expands", async () => {
    const stream = eventStream();
    await renderApp(
      <App
        events={stream.events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async () => ({ ok: true })}
      />,
      { width: 100, height: 30 }
    );

    await stream.push({ type: "transcript", item: { id: "g1", kind: "assistant", title: "助手", body: "", state: "running", role: "group" } });
    await setup.waitForFrame((frame) => frame.includes("思考中"));
    await stream.push({ type: "transcript", item: { id: "t1", kind: "tool", title: "文件已阅读", body: "calculator.html", state: "success", parentId: "g1" } });
    await setup.waitForFrame((frame) => frame.includes("已阅读 1 个文件"));
    await stream.push({ type: "transcript", item: { id: "t2", kind: "thought", title: "思考", body: "740ms", state: "success", parentId: "g1" } });
    await stream.push({ type: "transcript", item: { id: "t3", kind: "tool", title: "文件已阅读", body: "readme.md", state: "success", parentId: "g1" } });
    await stream.push({ type: "transcript", item: { id: "t5", kind: "tool", title: "记忆已搜索", body: "notes", state: "success", parentId: "g1" } });
    await stream.push({ type: "transcript", item: { id: "t4", kind: "tool", title: "命令已执行", body: "git status", state: "failed", parentId: "g1" } });
    // waitForFrame bails out early when the scheduler looks idle, which can
    // happen while act still holds queued updates; poll with forced renders.
    const collapsed = await setup.waitForFrame((frame) => frame.includes("已阅读 2 个文件") && frame.includes("执行了工具 1 次") && frame.includes("git status"));
    expect(collapsed).toContain("▸ 已阅读 2 个文件，执行了工具 1 次");
    // Collapsed: hidden rows stay out of the frame, failures stay visible.
    expect(collapsed).toContain("命令已执行");
    expect(collapsed).not.toContain("calculator.html");

    // Clicking the summary row expands the hidden rows.
    const summaryLine = collapsed.split("\n").find((line) => line.includes("已阅读 2 个文件"))!;
    const mx = Math.max(1, summaryLine.indexOf("已阅读"));
    const my = collapsed.split("\n").indexOf(summaryLine);
    await update(() => setup.mockMouse.click(mx, my));
    const opened = await setup.waitForFrame((frame) => frame.includes("calculator.html") && frame.includes("▾ 已阅读 2 个文件"));
    expect(opened).toContain("calculator.html");
  });

  test("subagent rows expand independently and survive live updates", async () => {
    const stream = eventStream();
    await renderApp(
      <App
        events={stream.events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async () => ({ ok: true })}
      />,
      { width: 110, height: 34 }
    );

    await stream.push({ type: "transcript", item: { id: "g-agent", kind: "assistant", title: "助手", body: "", state: "running", role: "group" } });
    await stream.push({ type: "transcript", item: { id: "agent-alpha", kind: "agent", title: "alpha", body: "运行中", state: "running", role: "group", parentId: "g-agent" } });
    await stream.push({ type: "transcript", item: { id: "agent-beta", kind: "agent", title: "beta", body: "运行中", state: "running", role: "group", parentId: "g-agent" } });
    await stream.push({ type: "transcript", item: { id: "alpha-tool", kind: "tool", title: "命令已执行", body: "alpha-output", state: "success", parentId: "agent-alpha" } });
    await stream.push({ type: "transcript", item: { id: "beta-tool", kind: "tool", title: "命令已执行", body: "beta-output", state: "success", parentId: "agent-beta" } });

    const collapsed = await setup.waitForFrame((frame) => frame.includes("alpha") && frame.includes("beta"));
    expect(collapsed).toContain("alpha");
    expect(collapsed).toContain("beta");
    // Both subagents start folded: neither timeline leaks into the transcript.
    expect(collapsed).not.toContain("alpha-output");
    expect(collapsed).not.toContain("beta-output");

    const headerLine = collapsed.split("\n").find((line) => line.includes("alpha"))!;
    const headerRow = collapsed.split("\n").indexOf(headerLine);
    const headerColumn = Math.max(1, headerLine.indexOf("alpha"));
    await update(() => setup.mockMouse.click(headerColumn, headerRow));
    const opened = await setup.waitForFrame((frame) => frame.includes("alpha-output"));
    expect(opened).toContain("alpha-output");
    // Expanding one agent does not expand its sibling.
    expect(opened).not.toContain("beta-output");

    // A live header update replaces the summary without collapsing the open
    // timeline or resetting its per-agent expansion state.
    await stream.push({ type: "transcript", item: { id: "agent-alpha", kind: "agent", title: "alpha", body: "已完成", state: "success", role: "group", parentId: "g-agent" } });
    const updated = await setup.waitForFrame((frame) => frame.includes("已完成"));
    expect(updated).toContain("已完成");
    expect(updated).toContain("alpha-output");
    expect(updated).not.toContain("beta-output");

    // Keyboard: Enter expands a focused agent, Space collapses it, and each
    // agent keeps its own expansion state.
    const betaNode = setup.renderer.root.findDescendantById("inline-agent-agent-beta")!;
    await update(() => betaNode.focus());
    await update(() => setup.mockInput.pressEnter());
    const betaOpened = await setup.waitForFrame((frame) => frame.includes("beta-output"));
    expect(betaOpened).toContain("beta-output");
    expect(betaOpened).toContain("alpha-output");

    await update(() => setup.mockInput.pressKey(" "));
    const betaClosed = await setup.waitForFrame((frame) => !frame.includes("beta-output"));
    expect(betaClosed).not.toContain("beta-output");
    expect(betaClosed).toContain("alpha-output");
  });

  test("Enter is ignored for a panel option while its action is in flight", async () => {
    const stream = eventStream();
    const panelActions: string[] = [];
    const releaseHolder: { release?: () => void } = {};
    await renderApp(
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

    // Commit the incoming event and keyboard effects before sending input.
    await stream.push({
      type: "panel",
      panel: { kind: "observe", title: "运行观察", options: [{ id: "go", label: "切换到项目概览" }] },
    });
    await setup.waitForFrame((frame) => frame.includes("切换到项目概览"));

    await update(() => setup.mockInput.pressEnter());
    await setup.waitForFrame((frame) => frame.includes("处理中"));
    await update(() => setup.mockInput.pressEnter());

    expect(panelActions).toEqual(["panel_action"]);
    await update(() => releaseHolder.release?.());
  });

  test("mention completion fires once after rapid typing", async () => {
    const stream = eventStream();
    const actions: ActionName[] = [];
    await renderApp(
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

    await stream.push({
      type: "snapshot",
      snapshot: { ...initialState.snapshot, sessionId: "test-session", status: "ready" },
    });
    await setup.waitForFrame((frame) => frame.includes("从这里开始"));

    await update(() => setup.mockInput.typeText("@readme", 8));
    await update(() => Bun.sleep(300));

    expect(actions.filter((name) => name === "complete_mention").length).toBe(1);
  });

  test("repeated Enter sends a pending draft only once", async () => {
    const stream = eventStream();
    const submissions: string[] = [];
    const pending = Promise.withResolvers<{ accepted: boolean }>();
    await renderApp(
      <App events={stream.events} onSubmit={async (_id, submission) => {
        submissions.push(submission.text);
        return pending.promise;
      }} />,
      { width: 100, height: 30 },
    );
    await setup.waitForFrame((frame) => frame.includes("从这里开始"));
    await update(() => setup.mockInput.typeText("一次提交"));
    await update(() => setup.mockInput.pressEnter());
    await update(() => setup.mockInput.pressEnter());
    expect(submissions).toEqual(["一次提交"]);
    await update(() => pending.resolve({ accepted: true }));
  });

  test("approval resolves through the keyboard", async () => {
    const stream = eventStream();
    const resolutions: Record<string, unknown>[] = [];
    await renderApp(
      <App
        events={stream.events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async () => ({ ok: true })}
        onResolveInteraction={async (_sessionId, _id, result) => { resolutions.push(result); }}
      />,
      { width: 100, height: 30 }
    );
    await stream.push({
      type: "interaction",
      id: "approval-1",
      kind: "approval",
      payload: { toolName: "bash", args: { command: "pytest" }, risk: "高风险", reason: "运行测试", persistAvailable: true },
    });
    const frame = await setup.waitForFrame((frame) => frame.includes("[2] 本项目后续允许"));
    expect(frame).toContain("执行命令");
    expect(frame).toContain("pytest");

    await update(() => setup.mockInput.pressEnter());
    expect(resolutions).toEqual([{ decision: "approve" }]);
  });

  test("approval option can be selected and confirmed with the mouse", async () => {
    const stream = eventStream();
    const resolutions: Record<string, unknown>[] = [];
    await renderApp(
      <App
        events={stream.events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async () => ({ ok: true })}
        onResolveInteraction={async (_sessionId, _id, result) => { resolutions.push(result); }}
      />,
      { width: 100, height: 30 }
    );
    await stream.push({
      type: "interaction",
      id: "approval-2",
      kind: "approval",
      payload: { toolName: "bash", args: { command: "pytest" }, risk: "高风险", reason: "运行测试", persistAvailable: false },
    });
    const frame = await setup.waitForFrame((frame) => frame.includes("[2] 拒绝"));
    const row = frame.split("\n").find((line) => line.includes("[2] 拒绝"))!;
    const x = Math.max(1, row.indexOf("[2]"));
    const y = frame.split("\n").indexOf(row);

    await update(() => setup.mockMouse.click(x + 1, y));
    await setup.waitForFrame((next) => next.includes("› [2] 拒绝"));
    expect(resolutions).toEqual([]);
    await update(() => setup.mockMouse.click(x + 1, y));
    expect(resolutions).toEqual([{ decision: "deny" }]);
  });

  test("question other branch is escapable via arrow keys and submits custom text", async () => {
    const stream = eventStream();
    const resolutions: Record<string, unknown>[] = [];
    await renderApp(
      <App
        events={stream.events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async () => ({ ok: true })}
        onResolveInteraction={async (_sessionId, _id, result) => { resolutions.push(result); }}
      />,
      { width: 100, height: 30 }
    );
    await stream.push({
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

    await update(() => setup.mockInput.pressArrow("down"));
    await setup.waitForFrame((frame) => frame.includes("其他说明"));
    await update(() => setup.mockInput.pressArrow("up"));
    await setup.waitForFrame((frame) => frame.includes("› [1] 直接修复") && !frame.includes("其他说明"));

    await update(() => setup.mockInput.pressArrow("down"));
    await setup.waitForFrame((frame) => frame.includes("其他说明"));
    await update(() => setup.mockInput.typeText("用方案B"));
    await update(() => setup.mockInput.pressEnter());
    expect(resolutions).toEqual([{ selectedIndex: 1, customText: "用方案B" }]);
  });

  test("failed startup keeps the draft but blocks submission", async () => {
    const stream = eventStream();
    let submitCalls = 0;
    await renderApp(
      <App
        events={stream.events}
        onSubmit={async () => { submitCalls += 1; return { accepted: true } as never; }}
        onAction={async () => ({ ok: true })}
      />,
      { width: 100, height: 30 }
    );
    await stream.push({ type: "progress", status: "failed", detail: "缺少运行依赖" });
    await setup.waitForFrame((frame) => frame.includes("会话启动失败"));
    await update(() => setup.mockInput.typeText("修复这个 bug"));
    await update(() => setup.mockInput.pressEnter());
    await setup.waitForFrame((frame) => frame.includes("会话未就绪"));
    expect(submitCalls).toBe(0);
  });

  test("mouse click on a header button does not leave focus on it", async () => {
    await renderApp(
      <App
        events={eventStream().events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async () => ({ ok: true })}
      />,
      { width: 100, height: 30 }
    );
    const header = await setup.waitForFrame((frame) => frame.includes("＋ 新建"));
    const headerLines = header.split("\n");
    const labelY = headerLines.findIndex((line) => line.includes("＋ 新建"));
    const labelX = headerLines[labelY]?.indexOf("新建") ?? -1;
    expect(labelX).toBeGreaterThan(-1);

    await update(() => setup.mockMouse.click(labelX + 1, labelY));

    const focused = findFocused((setup.renderer as unknown as { root: AnyNode }).root);
    const focusedKind = focused?.constructor?.name ?? "none";
    // Focus returns to the composer textarea instead of lingering on the
    // button; otherwise the hover highlight would stay after mouse-out.
    expect(focusedKind === "TextareaRenderable" || focusedKind === "none").toBe(true);
  });

  test.each([50, 70, 100])("right-side controls stay right-aligned at W=%d", async (terminalWidth) => {
    await renderApp(
      <App
        events={eventStream().events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async () => ({ ok: true })}
      />,
      { width: terminalWidth, height: 30 }
    );
    await update(() => setup.renderOnce());
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
    await renderApp(
      <App
        events={eventStream().events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async () => ({ ok: true })}
      />,
      { width: terminalWidth, height: 30 }
    );
    await update(() => setup.renderOnce());
    const all = (function collect(node: AnyNode): AnyNode[] {
      return [node, ...(node._childrenInLayoutOrder ?? []).flatMap((child) => collect(child as AnyNode))];
    })((setup.renderer as unknown as { root: AnyNode }).root);
    const groups = all.filter((node) => (node._childrenInLayoutOrder?.length ?? 0) >= 3 && node.width >= 10);
    const gauge = groups.at(-1)!._childrenInLayoutOrder?.[0]!;
    await update(() => gauge.focus());
    await setup.waitForFrame((frame) => frame.includes("已使用"));
  });
});
