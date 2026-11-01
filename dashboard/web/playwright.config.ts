import { defineConfig } from "@playwright/test";

// The real API over the committed reports, and the dev server. Run: pnpm e2e
export default defineConfig({
  testDir: "e2e",
  use: { baseURL: "http://localhost:5173" },
  projects: [{ name: "light" }, { name: "dark", use: { colorScheme: "dark" } }], // contrast in both themes
  webServer: [
    {
      command: "python -m dashboard.api",
      cwd: "../..",
      url: "http://127.0.0.1:8000/v1/gates",
      reuseExistingServer: true,
    },
    {
      command: "pnpm dev",
      url: "http://localhost:5173",
      reuseExistingServer: true,
    },
  ],
});
