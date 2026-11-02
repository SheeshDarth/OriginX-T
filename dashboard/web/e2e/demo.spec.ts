import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

// The demo's flows in a real browser against the real API (HANDBOOK §8.6, A12). Bench and
// repair flows join when their data exists.

async function noA11yViolations(page: Page) {
  const { violations } = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa"])
    .analyze();
  expect(
    violations.flatMap((v) =>
      v.nodes.map(
        (n) => `${v.id}: ${n.target} ${n.failureSummary?.split("\n")[1] ?? ""}`,
      ),
    ),
  ).toEqual([]);
}

test("overview: the gate board shows the committed verdicts", async ({
  page,
}) => {
  await page.goto("/");
  const gate2 = page.getByRole("row", { name: /GATE 2: dose-response/ });
  await expect(gate2.getByText("passed")).toBeVisible();
  await expect(
    page
      .getByRole("row", { name: /GATE 3: early warning/ })
      .getByText("exploratory"),
  ).toBeVisible();
  await noA11yViolations(page);
});

test("curves: filter a type, open a cell, see its outcome and risk", async ({
  page,
}) => {
  await page.goto("/curves");
  const dots = page.locator("figure svg circle");
  await expect(dots.first()).toBeVisible();
  const before = await dots.count();
  await page.getByLabel("human_control").uncheck();
  await expect.poll(() => dots.count()).toBeLessThan(before);
  await page.getByLabel("Metric").selectOption("benchmark");
  await expect(
    page.getByText(/^Benchmark-item perplexity by contamination ratio/),
  ).toBeVisible();
  await noA11yViolations(page);

  await page.goto("/cells/wikitext2/synthetic_r1_s0");
  await expect(page.getByText(/\+19\.3% vs ratio 0/)).toBeVisible();
  await expect(page.getByText("predicted to degrade")).toBeVisible();
  await noA11yViolations(page);
});

test("audit: upload a dataset, wait for the job, read the report", async ({
  page,
}) => {
  // a real audit needs the GPU and minutes; the job API is stubbed with its own contract
  const report = {
    schema: "audit/1",
    n_samples: 1000,
    model: "logreg@1",
    p_degrade: 0.91,
    conformal_set: [1],
    alpha: 0.1,
    features: {},
    missing_features: [],
    contributions: [
      { feature: "f_lex_entropy", value: 9.1, contribution: 1.5 },
    ],
    n_flagged: 1,
    flagged: [
      {
        index: 3,
        sample_id: "d-3",
        text: "leaked",
        benchmark_similarity: 1,
        repeat_3gram: 0,
        reasons: ["near-copy of a benchmark item (similarity 1.00)"],
      },
    ],
  };
  const job = {
    id: "f".repeat(32),
    name: "d.jsonl",
    n_samples: 1000,
    created_at: "now",
  };
  let polls = 0;
  await page.route("**/v1/audits", (r) =>
    r.fulfill({ status: 202, json: { ...job, status: "queued" } }),
  );
  await page.route("**/v1/audits/*", (r) =>
    r.fulfill({
      json:
        ++polls < 2
          ? { ...job, status: "running" }
          : { ...job, status: "done", report, scores: [] },
    }),
  );
  await page.goto("/audit");
  await page.getByLabel("Dataset to audit").setInputFiles({
    name: "d.jsonl",
    mimeType: "application/x-ndjson",
    buffer: Buffer.from('{"text": "x"}\n'),
  });
  await expect(page.getByRole("status")).toContainText("Auditing 1000 samples");
  await expect(page.getByText("predicted to degrade")).toBeVisible();
  await expect(page.getByText("Flagged samples: 1 of 1000")).toBeVisible();
  await noA11yViolations(page);
});
