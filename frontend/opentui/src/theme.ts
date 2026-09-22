import type { ThemePreference } from "./protocol.ts";

export type ThemeMode = "dark" | "light";
export type Theme = {
  mode: ThemeMode; background: string; surface: string; surfaceRaised: string; surfaceSelected: string;
  text: string; muted: string; subtle: string; accent: string; accentSoft: string; focus: string;
  success: string; warning: string; error: string; border: string; diffAdd: string; diffDelete: string;
  diffAddBackground: string; diffDeleteBackground: string;
};

export const themes: Record<ThemeMode, Theme> = {
  dark: {
    mode: "dark",
    background: "#0C1013",
    surface: "#13181C",
    surfaceRaised: "#192126",
    surfaceSelected: "#222C32",

    text: "#E8EDF0",
    muted: "#A9B3B9",
    subtle: "#727F87",

    accent: "#69BDE3",
    accentSoft: "#183643",
    focus: "#91D3F0",

    success: "#7BC68D",
    warning: "#D8B75F",
    error: "#E17D7D",

    border: "#2A353B",

    diffAdd: "#78C58B",
    diffDelete: "#DF7979",
    diffAddBackground: "#152B1C",
    diffDeleteBackground: "#321C1F",
  },
  light: {
    mode: "light",
    background: "#F7F9FA",
    surface: "#EEF2F4",
    surfaceRaised: "#E5EBEE",
    surfaceSelected: "#E0E7EA",

    text: "#182126",
    muted: "#50616A",
    subtle: "#74828A",

    accent: "#087A9F",
    accentSoft: "#D9EDF5",
    focus: "#056C8E",

    success: "#2D7A46",
    warning: "#896500",
    error: "#B44646",

    border: "#CDD6DB",

    diffAdd: "#307B49",
    diffDelete: "#B34848",
    diffAddBackground: "#E3F1E7",
    diffDeleteBackground: "#F5E4E4",
  },
};

export function resolveTheme(preference: ThemePreference, detected: ThemeMode | null): Theme {
  if (preference === "dark" || preference === "light") return themes[preference];
  return themes[detected ?? "dark"];
}
