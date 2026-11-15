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
  await page.goto("/overview");
  const gate2 = page.getByRole("row", {
    name: /GATE 2: dose-response wikitext2/,
  });
  await expect(gate2.getByText("passed")).toBeVisible();
  await expect(
    page
      .getByRole("row", { name: /GATE 3: early warning wikitext2/ })
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

test("monitor: pick a logged run, replay it, read its signals", async ({
  page,
}) => {
  // the committed X3 trainlogs (#98, #110) through the real run list and SSE stream
  await page.goto("/monitor");
  await expect(page.getByText(/^Healthy \(\d+\)$/)).toBeVisible();
  await noA11yViolations(page);

  await page.getByLabel("Filter runs").fill("recursive_r1_s0");
  await page.locator('a[href="/monitor/wikitext2_recursive_r1_s0"]').click();
  await expect(page).toHaveURL(/\/monitor\/wikitext2_recursive_r1_s0$/);
  // a finished log replays all 14 rows, then the stream ends
  await expect(page.getByText(/14 rows, last step 125/)).toBeVisible();
  await expect(
    page.getByText(/Finished: no new rows for a minute/),
  ).toBeVisible();
  await expect(page.locator("figure")).toHaveCount(7); // loss + six signals
  await expect(
    page.getByRole("row", { name: /Anisotropy, last layer/ }),
  ).toBeVisible();
  await noA11yViolations(page);
});

test("canary: the X2 verdict and one dot per matched cell", async ({
  page,
}) => {
  await page.goto("/canary");
  await expect(page.getByText(/ρ = 0\.78 \[0\.51, 0\.91\]/)).toBeVisible();
  await expect(page.getByText(/^passed \(bar/)).toBeVisible();
  await expect
    .poll(() => page.locator('figure svg [aria-label="dot"] path').count())
    .toBe(24);
  await noA11yViolations(page);
});

test("demo: hero, try it now, sample, stages, verdict with explanation", async ({
  page,
}) => {
  // the job API is stubbed (a real audit needs the models loaded); the page, samples and
  // the other endpoints are real
  const report = {
    schema: "audit/1",
    n_samples: 1000,
    model: "logreg@1",
    p_degrade: 0.97,
    conformal_set: [1],
    alpha: 0.1,
    features: {},
    missing_features: [],
    contributions: [
      { feature: "f_model_loss_std", value: 1.1, contribution: 4.5 },
      { feature: "f_lex_ttr", value: 0.5, contribution: -0.9 },
    ],
    n_flagged: 1,
    flagged: [
      {
        index: 3,
        sample_id: "d-3",
        text: "leaked text",
        benchmark_similarity: 0.98,
        repeat_3gram: 0.1,
        reasons: ["near-copy of a benchmark item (similarity 0.98)"],
      },
    ],
  };
  const scores = Array.from({ length: 60 }, (_, i) => ({
    sample_id: `s${i}`,
    method: "risk_reflm",
    score: -4 + (i % 20) / 10,
    higher_is: "worse",
  }));
  const job = {
    id: "e".repeat(32),
    name: "half_recursive.jsonl",
    n_samples: 1000,
    created_at: "now",
    stage: null,
    stages: [
      { stage: "features/lex", at: 0.3 },
      { stage: "features/leak", at: 0.5 },
      { stage: "features/model", at: 2.1 },
      { stage: "features/sem", at: 4.0 },
      { stage: "model", at: 5.2 },
      { stage: "flag", at: 5.3 },
      { stage: "scores/risk_reflm", at: 5.4 },
      { stage: "done", at: 8.9 },
    ],
  };
  // axe reads the colours mid-fade otherwise; reduced motion also proves the page works without animation
  await page.emulateMedia({ reducedMotion: "reduce" });
  let polls = 0;
  await page.route("**/v1/audits", (r) =>
    r.fulfill({ status: 202, json: { ...job, status: "queued", stages: [] } }),
  );
  await page.route("**/v1/audits/*", (r) =>
    r.fulfill({
      json:
        ++polls < 2
          ? { ...job, status: "running", stage: "features/lex", stages: [] }
          : { ...job, status: "done", report, scores },
    }),
  );

  await page.goto("/");
  await expect(
    page.getByRole("heading", { level: 1, name: /Know whether a dataset/ }),
  ).toBeVisible();
  await expect(page.getByText(/passed · \d+ failed/)).toBeVisible(); // the failures are shown too
  await noA11yViolations(page);

  await page.getByRole("link", { name: "Try it now" }).click();
  await expect(page).toHaveURL(/\/try$/);
  await page
    .getByRole("button", { name: /Half recursively generated/ })
    .click();
  await expect(page.getByText("1,000 records")).toBeVisible();
  await noA11yViolations(page);
  await page.getByRole("button", { name: "Run the audit" }).click();

  await expect(
    page.getByRole("heading", { name: "Analysing your dataset" }),
  ).toBeVisible();
  await expect(page.getByText("Reference language model")).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: "Likely to degrade a model fine-tuned on it",
    }),
  ).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText("97%", { exact: true })).toBeVisible();
  await expect(page.getByText("Flagged samples: 1 of 1,000")).toBeVisible();
  await expect(page.getByText(/not a guarantee/)).toBeVisible();
  await noA11yViolations(page);
});
