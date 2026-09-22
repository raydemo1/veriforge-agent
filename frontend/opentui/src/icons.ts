import type { IconPreference } from "./protocol.ts";

export type IconName = "prompt" | "assistant" | "session" | "newSession" | "search" | "tool" | "file" | "success" | "failure" | "pending" | "running" | "stop" | "warning" | "approval" | "question" | "profile" | "checkpoint" | "observe" | "close";
export type IconSet = Record<IconName, string>;

export const unicodeIcons: IconSet = {
  prompt: "›", assistant: "◆", session: "◷", newSession: "+", search: "⌕", tool: "◇", file: "▧",
  success: "✓", failure: "×", pending: "○", running: "·", stop: "■", warning: "!", approval: "!", question: "?",
  profile: "◎", checkpoint: "◈", observe: "◉", close: "×",
};
export const nerdIcons: IconSet = {
  prompt: "󰜴", assistant: "󰚩", session: "󰋚", newSession: "󰐕", search: "󰍉", tool: "󰒓", file: "󰈔",
  success: "󰄬", failure: "󰅖", pending: "󰔟", running: "󰁕", stop: "󰓛", warning: "󰀪", approval: "󰌶", question: "󰘥",
  profile: "󰀄", checkpoint: "󰆓", observe: "󰍹", close: "󰅖",
};
export function resolveIcons(preference: IconPreference): IconSet {
  return preference === "nerd" ? nerdIcons : unicodeIcons;
}
