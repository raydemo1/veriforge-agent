import { describe, expect, test } from "bun:test";
import { nerdIcons, resolveIcons, unicodeIcons } from "./icons.ts";
import { initialState, reduceEvent } from "./state.ts";
import { resolveTheme } from "./theme.ts";

describe("OpenTUI state", () => {
  test("session reset replaces transcript and clears transient interaction state", () => {
    const interacting = reduceEvent(initialState, { type: "interaction", id: "approval-1", kind: "approval", payload: { toolName: "write_file", args: {}, risk: "edit", reason: "confirm", persistAvailable: false } });
    const reset = reduceEvent(interacting, { type: "session_reset", snapshot: { ...initialState.snapshot, sessionId: "new-session" }, items: [{ id: "old", kind: "assistant", title: "助手", body: "restored" }] });
    expect(reset.interaction).toBeNull();
    expect(reset.items.map((item) => item.body)).toEqual(["restored"]);
    expect(reset.snapshot.sessionId).toBe("new-session");
  });
  test("auto theme follows detected terminal mode", () => {
    expect(resolveTheme("auto", "light").mode).toBe("light");
    expect(resolveTheme("dark", "light").mode).toBe("dark");
  });
  test("auto icons are safe unicode and nerd icons are opt-in", () => {
    expect(resolveIcons("auto")).toBe(unicodeIcons);
    expect(resolveIcons("nerd")).toBe(nerdIcons);
  });
  test("progress failed stops the loading state and shows the failure in the welcome row", () => {
    const failed = reduceEvent(initialState, { type: "progress", status: "failed", detail: "缺少运行依赖" });
    expect(failed.snapshot.status).toBe("failed");
    const welcome = failed.items.find((item) => item.id === "welcome");
    expect(welcome?.state).toBe("failed");
    expect(welcome?.title).toBe("会话启动失败");
    expect(welcome?.body).toBe("缺少运行依赖");
  });
  test("startup stages are localized and ready shows the input hint", () => {
    const connecting = reduceEvent(initialState, { type: "progress", status: "connecting tools", detail: "connecting tools" });
    expect(connecting.items[0].body).toBe("正在连接工具…");
    expect(connecting.items[0].state).toBe("running");

    const connectingExternal = reduceEvent(initialState, { type: "progress", status: "connecting external tools", detail: "" });
    expect(connectingExternal.items[0].body).toBe("正在连接外部工具…");
    expect(connectingExternal.items[0].state).toBe("running");

    const ready = reduceEvent(initialState, { type: "progress", status: "ready", detail: "Python 会话已就绪。" });
    expect(ready.items[0].state).toBe("success");
    expect(ready.items[0].body).toBe("今天想造点什么？");

    // external tools ready completes the whole startup: spinner must stop.
    const externalReady = reduceEvent(initialState, { type: "progress", status: "external tools ready", detail: "external tools ready" });
    expect(externalReady.items[0].state).toBe("success");
    expect(externalReady.items[0].body).toBe("今天想造点什么？");
  });
  test("first real transcript item replaces the welcome empty state", () => {
    const withContent = reduceEvent(initialState, { type: "transcript", item: { id: "user-1", kind: "user", title: "你", body: "开始干活" } });
    expect(withContent.items.some((item) => item.id === "welcome")).toBe(false);
    expect(withContent.items.map((item) => item.id)).toEqual(["user-1"]);
  });
  test("notices keep the welcome empty state until real content arrives", () => {
    const noticed = reduceEvent(initialState, { type: "notice", level: "info", text: "工具已同步" });
    expect(noticed.items.some((item) => item.id === "welcome")).toBe(true);
  });
});
