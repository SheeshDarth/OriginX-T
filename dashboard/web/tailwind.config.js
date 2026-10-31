// Colours come only from src/styles/tokens.css; the default palette is switched off.
const v = (name) => `rgb(var(--${name}) / <alpha-value>)`;

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    colors: {
      transparent: "transparent",
      bg: v("bg"),
      surface: v("surface"),
      border: v("border"),
      fg: v("fg"),
      muted: v("muted"),
      accent: v("accent"),
      ok: v("ok"),
      warn: v("warn"),
      bad: v("bad"),
    },
    fontFamily: {
      sans: ["Inter", "system-ui", "sans-serif"],
      mono: ["JetBrains Mono", "ui-monospace", "monospace"],
    },
  },
};
