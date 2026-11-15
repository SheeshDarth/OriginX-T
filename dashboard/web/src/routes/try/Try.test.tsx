import { afterAll, beforeAll, expect, test } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { Try } from "./Try";

const SAMPLE =
  '{"id":"a","text":"The quick brown fox"}\n{"id":"b","text":"jumps over the lazy dog"}\n';
const REPORT = {
  schema: "audit/1",
  n_samples: 2,
  model: "logreg@1",
  p_degrade: 0.97,
  conformal_set: [1],
  alpha: 0.1,
  features: {},
  missing_features: [],
  contributions: [
    { feature: "f_model_loss_std", value: 1.1, contribution: 4.5 },
  ],
  n_flagged: 0,
  flagged: [],
};
const JOB = {
  id: "j1",
  status: "done",
  name: "half_recursive.jsonl",
  n_samples: 2,
  benchmark: "wikitext2",
  created_at: "2026-11-14T00:00:00+00:00",
  stage: null,
  stages: [
    { stage: "features/lex", at: 0.4 },
    { stage: "done", at: 1.2 },
  ],
  report: REPORT,
  scores: [],
};
let posted: { url: string; body: string } | null = null;
const server = setupServer(
  http.get("*/samples/half_recursive.jsonl", () => HttpResponse.text(SAMPLE)),
  http.post("*/v1/audits", async ({ request }) => {
    posted = { url: request.url, body: await request.text() };
    return HttpResponse.json(
      { ...JOB, status: "queued", report: null },
      { status: 202 },
    );
  }),
  http.get("*/v1/audits/j1", () => HttpResponse.json(JOB)),
  http.get(
    "*/v1/risk/evaluation",
    () => new HttpResponse(null, { status: 404 }),
  ),
);
beforeAll(() => server.listen());
afterAll(() => server.close());

test("sample, preview, run, every stage shown, then the verdict", async () => {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <Try dwellMs={5} />
    </QueryClientProvider>,
  );
  (await screen.findByText("Half recursively generated")).click();
  expect(await screen.findByText("2 records · 1 KB")).toBeTruthy();
  expect(screen.getByText("The quick brown fox")).toBeTruthy();
  (await screen.findByRole("button", { name: "Run the audit" })).click();
  expect(
    await screen.findByText(
      "Likely to degrade a model fine-tuned on it",
      {},
      { timeout: 5000 },
    ),
  ).toBeTruthy();
  expect(posted?.url).toMatch(/name=half_recursive\.jsonl&benchmark=wikitext2/);
  expect(posted?.body).toBe(SAMPLE);
  expect(screen.getByText("97%")).toBeTruthy();
  expect(screen.getByText("Audit result")).toBeTruthy();
});

test("a file with a broken line cannot be run", async () => {
  server.use(
    http.get("*/samples/half_recursive.jsonl", () =>
      HttpResponse.text('{"text":"ok"}\nnope'),
    ),
  );
  render(
    <QueryClientProvider client={new QueryClient()}>
      <Try dwellMs={5} />
    </QueryClientProvider>,
  );
  (await screen.findByText("Half recursively generated")).click();
  expect(await screen.findByText("Line 2 is not valid JSON.")).toBeTruthy();
  expect(
    (screen.getByRole("button", { name: "Run the audit" }) as HTMLButtonElement)
      .disabled,
  ).toBe(true);
});
