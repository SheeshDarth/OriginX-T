import { afterAll, afterEach, beforeAll, expect, test } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import {
  Audit,
  flaggedCsv,
  parseReport,
  safeName,
  sizeWarning,
  type AuditReport,
} from "./Audit";
import { pct } from "../components/RiskPanel";

afterEach(cleanup);

const renderAudit = () =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <Audit />
    </QueryClientProvider>,
  );

// jsdom's File has no .text(); browsers do, so hand the input a file-like object
const file = (name: string, body: string) => ({ name, text: async () => body });

const REPORT: AuditReport = {
  schema: "audit/1",
  n_samples: 3,
  model: "logreg@1",
  p_degrade: 0.88,
  conformal_set: [1],
  alpha: 0.1,
  features: { f_lex_entropy: 2.1 },
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
  renderAudit();
  const input = screen.getByLabelText(/open a saved report/);
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
  expect(screen.getByText(/This dataset has 3 samples/)).toBeTruthy(); // far from the 1000 it learned on
  expect(
    screen.getByText("near-copy of a benchmark item (similarity 0.97)"),
  ).toBeTruthy();
});

test("a file that is not an audit report says so", async () => {
  renderAudit();
  fireEvent.change(screen.getByLabelText(/open a saved report/), {
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

// the upload flow against the API's contract: 202 with a queued job, then polling until done
let polls = 0;
let uploaded: { body: string; query: string } | null = null;
const server = setupServer(
  http.post("*/v1/audits", async ({ request }) => {
    const url = new URL(request.url);
    uploaded = { body: await request.text(), query: url.search };
    if (url.searchParams.get("name") === "bad.jsonl")
      return HttpResponse.json(
        { title: "Invalid request", status: 422, detail: "line 2: not JSON" },
        { status: 422 },
      );
    return HttpResponse.json(
      {
        id: "a".repeat(32),
        status: "queued",
        name: "data.jsonl",
        n_samples: 3,
        created_at: "now",
      },
      { status: 202 },
    );
  }),
  http.get("*/v1/audits/:id", () => {
    polls += 1;
    const base = {
      id: "a".repeat(32),
      name: "data.jsonl",
      n_samples: 3,
      created_at: "now",
    };
    return HttpResponse.json(
      polls < 2
        ? { ...base, status: "running" }
        : { ...base, status: "done", report: REPORT, scores: [] },
    );
  }),
);
beforeAll(() => server.listen());
afterAll(() => server.close());

test("uploading a dataset starts an audit job, shows progress, then the report", async () => {
  polls = 0;
  renderAudit();
  fireEvent.change(screen.getByLabelText(/Dataset to audit/), {
    target: { files: [file("my data!.jsonl", '{"text": "a"}\n')] },
  });
  expect(await screen.findByRole("status")).toBeTruthy(); // "Auditing 3 samples ... (running)"
  expect(await screen.findByText("88%", {}, { timeout: 3000 })).toBeTruthy();
  expect(uploaded?.body).toBe('{"text": "a"}\n');
  expect(uploaded?.query).toContain("benchmark=wikitext2");
  expect(uploaded?.query).toContain("name=my%20data_.jsonl");
  expect(polls).toBeGreaterThanOrEqual(2);
});

test("an upload the API rejects shows its reason", async () => {
  renderAudit();
  fireEvent.click(screen.getByLabelText(/Check for copies/)); // leak check off
  fireEvent.change(screen.getByLabelText(/Dataset to audit/), {
    target: { files: [file("bad.jsonl", "x")] },
  });
  expect((await screen.findByRole("alert")).textContent).toBe(
    "bad.jsonl: line 2: not JSON",
  );
  expect(uploaded?.query).not.toContain("benchmark");
});

test("upload names are made safe for the API", () => {
  expect([
    safeName("my data!.jsonl"),
    safeName("ü/../x.jsonl"),
    safeName(""),
  ]).toEqual(["my data_.jsonl", "__.._x.jsonl", "upload"]);
});

test("only datasets far from the training size get a size warning", () => {
  expect([sizeWarning(1000), sizeWarning(500), sizeWarning(2000)]).toEqual([
    null,
    null,
    null,
  ]);
  expect(sizeWarning(20)).toMatch(/^This dataset has 20 samples/);
  expect(sizeWarning(5000)).toMatch(/unreliable/);
});
