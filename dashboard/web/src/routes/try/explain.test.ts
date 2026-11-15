import { expect, test } from "vitest";
import {
  STAGES,
  activeStage,
  explain,
  previewJsonl,
  scanTexts,
  stageDurations,
} from "./explain";

const report = (set: number[], p: number) => ({
  schema: "audit/1" as const,
  n_samples: 1000,
  model: "logreg@1",
  p_degrade: p,
  conformal_set: set,
  alpha: 0.1,
  features: {},
  missing_features: [],
  contributions: [
    { feature: "f_model_loss_std", value: 1.1, contribution: 4.5 },
    { feature: "f_lex_repeat_3gram", value: 0.2, contribution: 1.7 },
    { feature: "f_lex_ttr", value: 0.5, contribution: -0.9 },
  ],
  n_flagged: 3,
  flagged: [],
});

test("previewJsonl counts records and names the first problem", () => {
  const ok = previewJsonl('{"text":"a b"}\n\n{"response":"c d"}\n');
  expect(ok).toEqual({ records: 2, samples: ["a b", "c d"], problem: null });
  expect(previewJsonl('{"text":"a"}\nnot json').problem).toBe(
    "Line 2 is not valid JSON.",
  );
  expect(previewJsonl("[1]").problem).toBe("Line 1 is not a JSON object.");
  expect(previewJsonl('{"id":1}').problem).toMatch(/Line 1 has no text/);
  expect(previewJsonl("").problem).toBe("The file has no records.");
});

test("scanTexts spreads through the file", () => {
  const file = Array.from({ length: 100 }, (_, i) =>
    JSON.stringify({ text: `t${i}` }),
  ).join("\n");
  const t = scanTexts(file, 10);
  expect(t).toHaveLength(10);
  expect(t[0]).toBe("t0");
  expect(t[9]).toBe("t90");
});

test("activeStage follows the job and the stage keys", () => {
  expect(activeStage("queued", null)).toBe(-1);
  expect(activeStage("running", null)).toBe(0);
  expect(activeStage("running", "features/model")).toBe(3);
  expect(activeStage("running", "model")).toBe(5); // the risk model, not the reference LM
  expect(activeStage("running", "scores/risk_knn")).toBe(7);
  expect(activeStage("done", null)).toBe(STAGES.length);
});

test("stageDurations come from the server's timestamps", () => {
  const log = [
    { stage: "features/lex", at: 3 },
    { stage: "features/leak", at: 3.2 },
    { stage: "features/model", at: 33 },
    { stage: "features/sem", at: 40 },
    { stage: "model", at: 50 },
    { stage: "flag", at: 50.1 },
    { stage: "scores/risk_reflm", at: 51 },
    { stage: "scores/risk_knn", at: 60 },
    { stage: "done", at: 70 },
  ];
  const d = stageDurations(log);
  expect(d[0]).toBe(3); // read and validate
  expect(d[2]).toBeCloseTo(29.8); // leak check
  expect(d[7]).toBe(19); // both score passes
  expect(stageDurations(log.slice(0, 3))[3]).toBeNull(); // still running
});

test("explain says what was found, in words", () => {
  const bad = explain(report([1], 0.999));
  expect(bad.tone).toBe("bad");
  expect(bad.headline).toMatch(/Likely to degrade/);
  expect(bad.paragraphs[0]).toMatch(/>99%.*90% coverage.*\{degrade\}/);
  expect(bad.paragraphs[1]).toContain("Reference-LM loss, spread");
  expect(bad.paragraphs[2]).toMatch(/3 of 1000 samples are flagged/);
  const ok = explain(report([0], 0.02));
  expect(ok.tone).toBe("ok");
  expect(ok.paragraphs[1]).toContain("Vocabulary diversity");
  const unsure = explain(report([0, 1], 0.5));
  expect(unsure.tone).toBe("warn");
  expect(unsure.headline).toMatch(/cannot call/);
});
