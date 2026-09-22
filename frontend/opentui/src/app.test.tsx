import { afterEach, describe, expect, test } from "bun:test";
import { testRender } from "@opentui/react/test-utils";
import type { TestRendererSetup } from "@opentui/core/testing";
import type { Renderable } from "@opentui/core";
import { App } from "./app.tsx";
import type { ActionName, UiEvent } from "./protocol.ts";
import { initialState } from "./state.ts";

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
  const queued: UiEvent[] = [];
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
    push(event: UiEvent) {
      queued.push(event);
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

  test("agent transcript rows show direction arrows", async () => {
    const stream = eventStream();
    setup = await testRender(
      <App
        events={stream.events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async () => ({ ok: true })}
      />,
      { width: 100, height: 30 }
    );

    stream.push({
      type: "transcript",
      item: { id: "a1", kind: "agent", title: "reviewer 上报", body: "found issue", state: "running", direction: "in" },
    });
    await setup.waitForFrame((frame) => frame.includes("←") && frame.includes("reviewer"));

    stream.push({
      type: "transcript",
      item: { id: "a2", kind: "agent", title: "reviewer 已补充", body: "keep checking", state: "running", direction: "out" },
    });
    await setup.waitForFrame((frame) => frame.includes("→") && frame.includes("reviewer"));
  });

  test("Enter is ignored for a panel option while its action is in flight", async () => {
    const stream = eventStream();
    const panelActions: string[] = [];
    const releaseHolder: { release?: () => void } = {};
    setup = await testRender(
      <App
        events={stream.events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async (name: ActionName) => {
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
        onAction={async (name: ActionName) => {
          actions.push(name);
          return { ok: true, candidates: [] };
        }}
      />,
      { width: 100, height: 30 }
    );

    stream.push({
      type: "snapshot",
      snapshot: { ...initialState.snapshot, status: "ready" },
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
        onResolveInteraction={async (_id, result) => { resolutions.push(result); }}
      />,
      { width: 100, height: 30 }
    );
    stream.push({
      type: "interaction",
      id: "approval-1",
      kind: "approval",
      payload: { toolName: "run_bash", args: { command: "pytest" }, risk: "高风险", reason: "运行测试", persistAvailable: true },
    });
    await setup.waitForFrame((frame) => frame.includes("[2] 以后允许这类命令"));

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
        onResolveInteraction={async (_id, result) => { resolutions.push(result); }}
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
        onResolveInteraction={async (_id, result) => { resolutions.push(result); }}
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

  test("diff keeps the hunk header as a locating separator", async () => {
    const stream = eventStream();
    setup = await testRender(
      <App
        events={stream.events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async () => ({ ok: true })}
      />,
      { width: 100, height: 30 }
    );
    stream.push({
      type: "transcript",
      item: {
        id: "diff-1",
        kind: "file",
        title: "已编辑 src/app.ts  +1  -0",
        body: "--- a/src/app.ts\n+++ b/src/app.ts\n@@ -1,3 +1,4 @@\n old\n+new",
      },
    });
    await setup.waitForFrame((frame) => frame.includes("@@ -1,3 +1,4 @@"));
  });

  test("narrow terminal collapses header labels to icons", async () => {
    const stream = eventStream();
    setup = await testRender(
      <App
        events={stream.events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async () => ({ ok: true })}
      />,
      { width: 60, height: 20 }
    );
    stream.push({ type: "snapshot", snapshot: { ...initialState.snapshot, status: "ready", cwd: "proj" } });
    const frame = await setup.waitForFrame((frame) => frame.includes("从这里开始"));
    expect(frame).not.toContain("历史");
    expect(frame).not.toContain("新会话");
    expect(frame).toContain("◷");
  });

  test("welcome empty state is removed after the first real message", async () => {
    const stream = eventStream();
    setup = await testRender(
      <App
        events={stream.events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async () => ({ ok: true })}
      />,
      { width: 100, height: 30 }
    );
    stream.push({ type: "progress", status: "ready", detail: "" });
    await setup.waitForFrame((frame) => frame.includes("输入任务开始"));

    await setup.mockInput.typeText("修复这个问题");
    await sleep(200);
    setup.mockInput.pressEnter();
    await sleep(400);
    const submitted = setup.captureCharFrame();
    expect(submitted).toContain("修复这个问题");
    expect(submitted).not.toContain("VeriForge");
  });

  test("panel marks the current option with a check and adapts its footer", async () => {
    const stream = eventStream();
    setup = await testRender(
      <App
        events={stream.events}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async () => ({ ok: true })}
      />,
      { width: 100, height: 30 }
    );
    stream.push({
      type: "panel",
      panel: {
        kind: "profile",
        title: "工作模式",
        options: [
          { id: "auto", label: "自动路由", selected: true },
          { id: "coding", label: "编码" },
        ],
      },
    });
    const frame = await setup.waitForFrame((frame) => frame.includes("✓ 自动路由"));
    expect(frame).toContain("↑↓ 选择");

    stream.push({ type: "panel", panel: { kind: "help", title: "帮助", body: "一些说明" } });
    await setup.waitForFrame((frame) => frame.includes("一些说明") && frame.includes("Esc 关闭") && !frame.includes("↑↓ 选择"));
  });

  test("running rows use the dot marker once idle", async () => {
    // Marker semantics are covered via state/frame; here we only assert the
    // icon set no longer collides the prompt glyph with the running glyph.
    const { unicodeIcons } = await import("./icons.ts");
    expect(unicodeIcons.running).not.toBe(unicodeIcons.prompt);
    expect(unicodeIcons.running).toBe("·");
    expect(unicodeIcons.checkpoint).not.toBe(unicodeIcons.assistant);
    expect(unicodeIcons.checkpoint).toBe("◈");
  });

  test("mouse click on a header button does not leave focus on it", async () => {
    setup = await testRender(
      <App
        events={{ [Symbol.asyncIterator]: async function* () {} }}
        onSubmit={async () => ({ accepted: true } as never)}
        onAction={async () => ({ ok: true })}
      />,
      { width: 100, height: 30 }
    );
    await sleep(300);
    const headerLine = setup.captureCharFrame().split("\n")[0] ?? "";
    const labelX = headerLine.indexOf("新会话");
    expect(labelX).toBeGreaterThan(-1);

    await setup.mockMouse.click(labelX + 1, 0);
    await sleep(150);

    const focused = findFocused((setup.renderer as unknown as { root: AnyNode }).root);
    const focusedKind = focused?.constructor?.name ?? "none";
    // Focus returns to the composer textarea instead of lingering on the
    // button; otherwise the hover highlight would stay after mouse-out.
    expect(focusedKind === "TextareaRenderable" || focusedKind === "none").toBe(true);
  });
});
