import { afterEach, describe, expect, test } from "bun:test";
import { testRender } from "@opentui/react/test-utils";
import type { TestRendererSetup } from "@opentui/core/testing";
import { App } from "./app.tsx";
import type { ActionName, UiEvent } from "./protocol.ts";
import { initialState } from "./state.ts";

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
});
