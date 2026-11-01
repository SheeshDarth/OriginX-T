import { afterAll, beforeAll, expect, test } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  createMemoryHistory,
  createRouter,
  RouterProvider,
} from "@tanstack/react-router";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { routeTree } from "../router";
import { curvesPlot } from "./Curves";
import { conformalSentence, verdict } from "../components/RiskPanel";

const row = (ratio: number, holdout_ppl: number) => ({
  cell: `synthetic_r${ratio}_s0`,
  type: "synthetic",
  ratio,
  seed: 0,
  n_train: 1000,
  n_contaminated: ratio * 1000,
  holdout_ppl,
  benchmark_ppl: 60,
  distinct_1: 0.17,
  distinct_2: 0.66,
  seconds: 75,
  config: "abc",
  shared_baseline: false,
});
const RESULTS = [row(0, 50), row(1, 60)];
const notProduced = () =>
  HttpResponse.json({ title: "Not found", status: 404 }, { status: 404 });

const server = setupServer(
  http.get("*/v1/grids/wikitext2/results", () => HttpResponse.json(RESULTS)),
  http.get("*/v1/grids/wikitext2/whitebox", () =>
    HttpResponse.json([
      {
        cell: "synthetic_r1_s0",
        type: "synthetic",
        ratio: 1,
        seed: 0,
        kind: "cell",
        signals: { anisotropy_last: 0.917 },
      },
    ]),
  ),
  http.get("*/v1/risk", () =>
    HttpResponse.json([
      {
        schema: "risk/1",
        dataset: "wikitext2",
        cell: "synthetic_r1_s0",
        p_degrade: 0.93,
        label: 1,
        conformal_set: [1],
        alpha: 0.1,
        model: "logreg@1",
        split: "test",
      },
    ]),
  ),
  http.get("*/v1/features/wikitext2", notProduced),
  http.get("*/v1/bench/samples", notProduced),
);
beforeAll(() => server.listen());
afterAll(() => server.close());

test("cell detail shows the outcome against its own ratio-0 run, its signals, and empty states for missing reports", async () => {
  const router = createRouter({
    routeTree,
    history: createMemoryHistory({
      initialEntries: ["/cells/wikitext2/synthetic_r1_s0"],
    }),
  });
  render(
    <QueryClientProvider client={new QueryClient()}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
  expect(await screen.findByText("60.0 +20.0% vs ratio 0")).toBeTruthy();
  expect(await screen.findByText("0.917")).toBeTruthy();
  expect(await screen.findByText("93%")).toBeTruthy();
  expect(screen.getByText("predicted to degrade")).toBeTruthy();
  expect(
    screen.getByText("At 90% coverage, the possible outcomes are {degrade}."),
  ).toBeTruthy();
  expect(
    await screen.findByText(
      "Features have not been computed for this grid yet.",
    ),
  ).toBeTruthy();
  expect(
    await screen.findByText(
      "No bench samples have been published for this cell yet.",
    ),
  ).toBeTruthy();
});

test("the curves plot labels its axes and draws one dot per selected run", () => {
  const plot = curvesPlot(
    [],
    RESULTS as never,
    "holdout",
    0,
    ["synthetic"],
    () => {},
  );
  expect(plot.textContent).toContain("Hidden-holdout perplexity");
  expect(plot.textContent).toContain("Contamination ratio");
  expect(plot.querySelectorAll("circle").length).toBe(2);
  const none = curvesPlot(
    [],
    RESULTS as never,
    "holdout",
    0,
    ["recursive"],
    () => {},
  );
  expect(none.querySelectorAll("circle").length).toBe(0);
});

test("conformal sets read as plain sentences and drive the verdict", () => {
  expect(conformalSentence([0, 1], 0.1)).toBe(
    "At 90% coverage, the possible outcomes are {no degrade, degrade}.",
  );
  expect(conformalSentence([], 0.2)).toMatch(/^At 80% coverage, no outcome/);
  expect(verdict([1]).text).toBe("predicted to degrade");
  expect(verdict([0]).text).toBe("predicted safe");
  expect(verdict([0, 1]).text).toBe("uncertain");
  expect(verdict([]).text).toBe("uncertain");
});
