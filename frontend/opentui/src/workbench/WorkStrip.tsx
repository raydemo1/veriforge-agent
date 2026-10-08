import type { WorkState } from "../protocol.ts";
import type { Theme } from "../theme.ts";
import type { WorkbenchTab } from "./strip.ts";
import { visibleSegments, workSegments } from "./strip.ts";

const TONE_COLOR = (theme: Theme, tone: string): string => {
  if (tone === "success") return theme.success;
  if (tone === "warning") return theme.warning;
  if (tone === "error") return theme.error;
  if (tone === "accent") return theme.accent;
  return theme.subtle;
};

export function WorkStrip({ work, width, theme, onOpen }: {
  work: WorkState;
  width: number;
  theme: Theme;
  onOpen: (tab: WorkbenchTab) => void;
}) {
  const segments = visibleSegments(workSegments(work), width);
  if (!segments.length) return null;
  const narrow = width < 70;

  return (
    <box
      style={{
        flexDirection: "row",
        alignItems: "center",
        height: 1,
        flexShrink: 0,
        gap: 1,
        paddingLeft: narrow ? 1 : 2,
        paddingRight: narrow ? 1 : 2,
      }}
    >
      {segments.map((segment, index) => {
        const color = TONE_COLOR(theme, segment.tone);
        const label = narrow || !segment.label ? "" : `${segment.label} `;
        return (
          <box
            key={segment.key}
            focusable
            onMouseDown={() => onOpen(segment.key)}
            style={{ flexDirection: "row", alignItems: "center" }}
          >
            {index > 0 ? <text fg={theme.subtle}>{"· "}</text> : null}
            <text fg={color}>{label}</text>
            <text fg={narrow ? color : theme.muted}>{segment.text}</text>
          </box>
        );
      })}
    </box>
  );
}
