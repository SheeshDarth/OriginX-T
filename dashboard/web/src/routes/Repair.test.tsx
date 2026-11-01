import { afterAll, beforeAll, expect, test } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { Repair, recoveryText, x1Margin } from "./Repair";

// three rows of tests/fixtures/repair.jsonl (#106)
const row = (cell: string, strategy: string, recovery: number | null) => ({
  dataset: "wikitext2" as const,
  cell,
  strategy,
  k_replaced: 500,
  holdout_ppl: 57.3,
  benchmark_ppl: 62,
  distinct_2: 0.64,
  recovery,
  seed: 0,
});
const ROWS = [
  row("synthetic_r0.5_s0", "harm_guided", 0.7266),
  row("synthetic_r0.5_s0", "random", 0.4804),
  row("human_control_r0.5_s0", "random", null),
];
const server = setupServer(
  http.get("*/v1/repair/wikitext2", () => HttpResponse.json(ROWS)),
);
beforeAll(() => server.listen());
afterAll(() => server.close());

test("x1 margin and recovery text", () => {
  const syn = ROWS.filter((r) => r.cell === "synthetic_r0.5_s0");
  expect(x1Margin(syn)).toBe(25); // 0.7266 - 0.4804
  expect(x1Margin(ROWS.filter((r) => r.cell.startsWith("human")))).toBeNull();
  expect(recoveryText(ROWS.at(-1))).toBe("no damage");
  expect(recoveryText(undefined)).toBe("—");
});

test("renders a row per cell", async () => {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <Repair />
    </QueryClientProvider>,
  );
  expect(await screen.findByText("synthetic_r0.5_s0")).toBeTruthy();
  expect(screen.getByText("73%")).toBeTruthy();
  expect(screen.getByText("+25 pts ✓")).toBeTruthy();
  expect(screen.getByText("no damage")).toBeTruthy();
});
