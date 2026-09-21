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

    const ready = reduceEvent(initialState, { type: "progress", status: "ready", detail: "Python 会话已就绪。" });
    expect(ready.items[0].state).toBe("success");
    expect(ready.items[0].body).toBe("输入任务开始，/ 查看命令，@ 添加上下文");
  });
});
