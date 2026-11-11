import { afterAll, beforeAll, expect, test } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { Overview } from "./Overview";
import { baselineLine, est } from "../components/EvaluationCard";

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
    id: "gate_1",
    name: "GATE 1: risk model",
    dataset: null,
    status: "not_evaluable",
    exploratory: false,
    rule: "r1",
    details: { datasets: ["wikitext2"] },
    source: "reports/risk/evaluation.json",
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

const EVALUATION = {
  model: "logreg@1",
  label: "D4 default",
  features: ["f_lex_entropy", "f_model_loss_mean"],
  leave_one_type_out: {
    n: 60,
    n_degraded: 43,
    auroc: { value: 0.808, ci95: [0.688, 0.921] },
    ece: 0.239,
    reliability: [],
    conformal: {
      alpha: 0.1,
      target: 0.9,
      coverage: { value: 0.583, ci95: [0.467, 0.7] },
      mean_set_size: 0.8,
      n_cal_per_fold: [20],
    },
  },
  baselines: {
    ref_lm_loss: {
      column: "f_model_loss_mean",
      n: 60,
      auroc_model: 0.808,
      auroc_baseline: 0.837,
      difference: { value: -0.029, ci95: [-0.074, 0.0] },
      model_beats_baseline: false,
    },
    single_feature_auroc: { f_lex_entropy: 0.977 },
  },
  gate_1: { status: "not_evaluable", rule: "r", datasets: ["wikitext2"] },
};

const server = setupServer(
  http.get("*/v1/gates", () => HttpResponse.json(GATES)),
  http.get("*/v1/risk/evaluation", () => HttpResponse.json(EVALUATION)),
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
    screen.getByText(/features on 1 of 2 datasets needed \(wikitext2\)/),
  ).toBeTruthy();
  expect(
    screen.getByText(
      /best AUROC 0\.25 \(part_ratio_last, permutation p 0\.95\)/,
    ),
  ).toBeTruthy();
});

test("the evaluation card states AUROC, coverage against its target, n and the baseline verdict", async () => {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <Overview />
    </QueryClientProvider>,
  );
  expect(await screen.findByText("0.81 [0.69, 0.92]")).toBeTruthy();
  expect(screen.getByText("58% [47%, 70%]")).toBeTruthy();
  expect(screen.getByText(/target 90%/)).toBeTruthy();
  expect(screen.getByText(/60 datasets, 43/)).toBeTruthy();
  expect(
    screen.getByText(
      "Baseline ref-LM loss (AUROC 0.84): not beaten, difference -0.03 [-0.07, +0.00]",
    ),
  ).toBeTruthy();
});

test("with two datasets the card adds the within-dataset baselines and the cross-dataset numbers", async () => {
  const within = (auroc_baseline: number, value: number, beats: boolean) => ({
    column: "f_model_loss_mean",
    n: 30,
    auroc_model: 0.8,
    auroc_baseline,
    difference: { value, ci95: [value - 0.03, value + 0.03] },
    model_beats_baseline: beats,
  });
  server.use(
    http.get("*/v1/risk/evaluation", () =>
      HttpResponse.json({
        ...EVALUATION,
        baselines: {
          ...EVALUATION.baselines,
          ref_lm_loss: {
            ...EVALUATION.baselines.ref_lm_loss,
            per_dataset: {
              tinystories: within(0.83, -0.03, false),
              wikitext2: within(0.84, 0.05, true),
            },
          },
        },
        leave_one_dataset_out: {
          ...EVALUATION.leave_one_type_out,
          n: 180,
          auroc: { value: 0.73, ci95: [0.65, 0.8] },
          ece: 0.5,
        },
      }),
    ),
  );
  render(
    <QueryClientProvider client={new QueryClient()}>
      <Overview />
    </QueryClientProvider>,
  );
  expect(
    await screen.findByText(
      /^Baseline ref-LM loss, both datasets pooled \(AUROC 0\.84\): not beaten/,
    ),
  ).toBeTruthy();
  expect(
    screen.getByText(
      "Baseline ref-LM loss, within tinystories (AUROC 0.83): not beaten, difference -0.03 [-0.06, +0.00]",
    ),
  ).toBeTruthy();
  expect(
    screen.getByText(/^Baseline ref-LM loss, within wikitext2 .*: beaten/),
  ).toBeTruthy();
  expect(screen.getByText(/tested on the other \(180 datasets\)/)).toBeTruthy();
  expect(screen.getByText("0.73 [0.65, 0.80]")).toBeTruthy();
  server.resetHandlers();
});

test("a missing baseline or estimate reads as such", () => {
  expect(baselineLine(null)).toBe("Baseline ref-LM loss: not computed yet");
  expect(
    baselineLine({
      column: "f_model_loss_mean",
      n: 20,
      note: "one class only",
    }),
  ).toBe("Baseline ref-LM loss: one class only");
  expect([
    est(null),
    est({ value: 0.5 }),
    est({ value: 0.5, ci95: [0.4, 0.6] }),
  ]).toEqual(["—", "0.50", "0.50 [0.40, 0.60]"]);
});
