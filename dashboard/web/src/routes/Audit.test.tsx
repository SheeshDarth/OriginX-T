import { afterEach, expect, test } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { Audit, flaggedCsv, parseReport, type AuditReport } from "./Audit";
import { pct } from "../components/RiskPanel";

afterEach(cleanup);

// jsdom's File has no .text(); browsers do, so hand the input a file-like object
const file = (name: string, body: string) => ({ name, text: async () => body });

const REPORT: AuditReport = {
  schema: "audit/1",
  n_samples: 3,
  model: "logreg@1",
  p_degrade: 0.88,
  conformal_set: [1],
  alpha: 0.1,
  missing_features: ["f_model_ref_loss_mean"],
  contributions: [
    { feature: "f_lex_entropy", value: 2.1, contribution: 1.8 },
    { feature: "f_leak_max_sim", value: 0.3, contribution: -0.4 },
  ],
  n_flagged: 1,
  flagged: [
    {
      index: 1,
      sample_id: "s1",
      text: 'the quick "brown" fox',
      benchmark_similarity: 0.97,
      repeat_3gram: 0,
      reasons: ["near-copy of a benchmark item (similarity 0.97)"],
    },
  ],
};

test("an opened report shows the score, the set in words, the drivers, the gaps and the flags", async () => {
  render(<Audit />);
  const input = screen.getByLabelText(/Audit report/);
  fireEvent.change(input, {
    target: {
      files: [file("audit.json", JSON.stringify(REPORT))],
    },
  });
  expect(await screen.findByText("88%")).toBeTruthy();
  expect(screen.getByText("predicted to degrade")).toBeTruthy();
  expect(
    screen.getByText("At 90% coverage, the possible outcomes are {degrade}."),
  ).toBeTruthy();
  expect(screen.getByText("+1.80")).toBeTruthy();
  expect(screen.getByText("-0.40")).toBeTruthy();
  expect(screen.getByText(/f_model_ref_loss_mean/)).toBeTruthy();
  expect(screen.getByText("Flagged samples: 1 of 3")).toBeTruthy();
  expect(
    screen.getByText("near-copy of a benchmark item (similarity 0.97)"),
  ).toBeTruthy();
});

test("a file that is not an audit report says so", async () => {
  render(<Audit />);
  fireEvent.change(screen.getByLabelText(/Audit report/), {
    target: { files: [file("risk.json", '{"schema":"risk/1"}')] },
  });
  expect((await screen.findByRole("alert")).textContent).toBe(
    "risk.json: not an audit/1 report (schema: risk/1)",
  );
  expect(() => parseReport("{}")).toThrow(/schema: none/);
});

test("the CSV export quotes text and escapes quotes", () => {
  expect(flaggedCsv(REPORT).split("\n")[1]).toBe(
    '"1","s1","0.97","0","near-copy of a benchmark item (similarity 0.97)","the quick ""brown"" fox"',
  );
});

test("probabilities never read as certain", () => {
  expect([pct(0.9999), pct(0.88), pct(0.001)]).toEqual([">99%", "88%", "<1%"]);
});
