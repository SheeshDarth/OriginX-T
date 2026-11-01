import { afterAll, beforeAll, expect, test } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { Overview } from "./Overview";

// shaped like the real GET /v1/gates on main
const GATES = [
  {
    id: "gate_0",
    name: "GATE 0: collapse spike",
    dataset: null,
    status: "passed",
    exploratory: false,
    rule: "r0",
    details: { per_seed: { "0": true, "1": true, "2": false } },
    source: "reports/collapse_spike/metrics.json",
  },
  {
    id: "gate_3",
    name: "GATE 3: early warning",
    dataset: "wikitext2",
    status: "failed",
    exploratory: true,
    rule: "r3",
    details: {
      trajectories: 15,
      never_collapse: 3,
      signals: {
        anisotropy_last: { auroc: 0.22, permutation_p: 0.97, warns: false },
        part_ratio_last: { auroc: 0.25, permutation_p: 0.95, warns: false },
      },
    },
    source: null,
  },
];

const server = setupServer(
  http.get("*/v1/gates", () => HttpResponse.json(GATES)),
);
beforeAll(() => server.listen());
afterAll(() => server.close());

test("the gate board shows status as text, the exploratory flag and the evidence numbers", async () => {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <Overview />
    </QueryClientProvider>,
  );
  expect(await screen.findByText("GATE 0: collapse spike")).toBeTruthy();
  expect(screen.getByText("passed")).toBeTruthy();
  expect(screen.getByText("failed")).toBeTruthy();
  expect(screen.getByText("exploratory")).toBeTruthy();
  expect(screen.getByText(/2\/3 seeds/)).toBeTruthy();
  expect(
    screen.getByText(
      /best AUROC 0\.25 \(part_ratio_last, permutation p 0\.95\)/,
    ),
  ).toBeTruthy();
});
