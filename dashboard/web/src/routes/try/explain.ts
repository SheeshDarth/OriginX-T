import type { components } from "../../api/schema";
import { conformalSentence, pct, verdict } from "../../components/RiskPanel";

type AuditReport = components["schemas"]["AuditReport"];

/** One step of an audit as the user sees it. `keys` are the API's `stage` strings it covers. */
export type Stage = {
  id: string;
  title: string;
  detail: string;
  keys: string[];
};

/** In the order the worker runs them (src/risk/audit.py, dashboard/api/app.py run_audit). */
export const STAGES: Stage[] = [
  {
    id: "read",
    title: "Read and validate",
    detail: "Parsing every record and loading the models.",
    keys: [],
  },
  {
    id: "lex",
    title: "Lexical statistics",
    detail: "Vocabulary diversity, repeated phrases and text length.",
    keys: ["features/lex"],
  },
  {
    id: "leak",
    title: "Benchmark leak check",
    detail: "How close each text is to a benchmark item, to catch near-copies.",
    keys: ["features/leak"],
  },
  {
    id: "model",
    title: "Reference language model",
    detail:
      "How predictable each text is to a fixed GPT-2. Generated text is unusually predictable.",
    keys: ["features/model"],
  },
  {
    id: "sem",
    title: "Semantic structure",
    detail:
      "Sentence embeddings: how tightly the texts cluster and whether rare topics survive.",
    keys: ["features/sem"],
  },
  {
    id: "risk",
    title: "Risk model",
    detail:
      "Combines the features into a probability of degradation, with a conformal prediction set.",
    keys: ["model"],
  },
  {
    id: "flag",
    title: "Flag samples",
    detail: "Benchmark near-copies and texts that repeat themselves.",
    keys: ["flag"],
  },
  {
    id: "scores",
    title: "Per-sample scores",
    detail:
      "A risk score for every text, from the reference model and from embedding density.",
    keys: ["scores/risk_reflm", "scores/risk_knn"],
  },
];

/** Index of the stage that is running now: -1 while queued, 0 before the first stage reports. */
export function activeStage(
  status: string,
  stage: string | null | undefined,
): number {
  if (status === "queued") return -1;
  if (status === "done") return STAGES.length;
  if (!stage) return 0;
  const i = STAGES.findIndex((s) => s.keys.includes(stage));
  return i < 0 ? 0 : i;
}

const LABELS: Record<string, string> = {
  f_lex_ttr: "Vocabulary diversity (type-token ratio)",
  f_lex_distinct_1: "Distinct words",
  f_lex_distinct_2: "Distinct word pairs",
  f_lex_distinct_3: "Distinct word triples",
  f_lex_entropy: "Word entropy",
  f_lex_len_mean: "Mean text length",
  f_lex_len_var: "Variation in text length",
  f_lex_burstiness: "Burstiness of repeated words",
  f_lex_repeat_3gram: "Repeated 3-grams",
  f_leak_max_sim: "Closest benchmark match",
  f_leak_mean_sim: "Average benchmark similarity",
  f_leak_frac_above_0_9: "Share of near-copies of benchmark items",
  f_model_loss_mean: "Reference-LM loss, mean",
  f_model_loss_p90: "Reference-LM loss, 90th percentile",
  f_model_loss_std: "Reference-LM loss, spread",
  f_model_rank_instability: "Token-rank instability",
  f_model_rank_log_mean: "Token rank, log-mean",
  f_model_rank_top10: "Share of top-10 token predictions",
  f_sem_cluster_entropy: "Topic spread (cluster entropy)",
  f_sem_knn_dist: "Embedding density (neighbour distance)",
  f_sem_mauve: "Match to human text (MAUVE)",
  f_sem_tail_retention: "Rare-topic retention",
};

/** A feature column in words; unknown columns keep their own name. */
export const featureLabel = (f: string): string => LABELS[f] ?? f;

export type Explanation = {
  headline: string;
  tone: "bad" | "ok" | "warn";
  paragraphs: string[];
};

/** Plain-language reading of a report (HANDBOOK §8.6: specific, never "insights detected"). */
export function explain(r: AuditReport): Explanation {
  const v = verdict(r.conformal_set);
  const cov = Math.round((1 - r.alpha) * 100);
  const up = r.contributions.filter((c) => c.contribution > 0);
  const down = r.contributions.filter((c) => c.contribution < 0);
  const name = (cs: typeof up) =>
    cs
      .slice(0, 2)
      .map((c) => featureLabel(c.feature))
      .join("; ");
  const flagged =
    r.n_flagged === 0
      ? "No sample is a benchmark near-copy or repeats itself."
      : `${r.n_flagged} of ${r.n_samples} samples are flagged as benchmark near-copies or as repeating themselves; they are listed below.`;
  if (v.text === "predicted to degrade")
    return {
      headline: "Likely to degrade a model fine-tuned on it",
      tone: "bad",
      paragraphs: [
        `The risk model puts the chance of degradation at ${pct(r.p_degrade)}. ${conformalSentence(r.conformal_set, r.alpha)} Degradation means held-out perplexity rising beyond normal run-to-run noise.`,
        up.length
          ? `The strongest push towards degradation came from: ${name(up)}.`
          : "No single feature dominates the score.",
        flagged,
      ],
    };
  if (v.text === "predicted safe")
    return {
      headline: "Looks safe to fine-tune on",
      tone: "ok",
      paragraphs: [
        `The risk model puts the chance of degradation at ${pct(r.p_degrade)}. ${conformalSentence(r.conformal_set, r.alpha)}`,
        down.length
          ? `The features that most favour safety: ${name(down)}.`
          : "No single feature dominates the score.",
        flagged,
      ],
    };
  return {
    headline: "The audit cannot call this one",
    tone: "warn",
    paragraphs: [
      `The risk model puts the chance of degradation at ${pct(r.p_degrade)}, too close to call at ${cov}% coverage. ${conformalSentence(r.conformal_set, r.alpha)} An honest "uncertain" is better than a guess.`,
      flagged,
    ],
  };
}

/** What the model was validated on, so a call is never read as a guarantee. */
export function caveat(
  auroc: number | null | undefined,
  coverage: number | null | undefined,
  target: number | null | undefined,
): string {
  const parts = [
    "The risk model was validated on controlled DistilGPT-2 experiments, one training recipe",
  ];
  if (auroc != null)
    parts.push(
      `its AUROC on contamination types it had not seen was ${auroc.toFixed(2)}`,
    );
  if (coverage != null && target != null)
    parts.push(
      `its prediction sets covered the true outcome ${(coverage * 100).toFixed(0)}% of the time against a ${(target * 100).toFixed(0)}% target`,
    );
  return `${parts.join("; ")}. Treat the call as a screen, not a guarantee.`;
}

/** A dataset file as the browser reads it before upload: record count and the first problem. */
export type Preview = {
  records: number;
  samples: string[];
  problem: string | null;
};

// the fields the server maps to prompt/response (src/ingestion/loaders.py)
const TEXT_KEYS = [
  "response",
  "output",
  "target",
  "completion",
  "answer",
  "text",
  "content",
  "code",
  "prompt",
  "input",
  "instruction",
  "question",
  "source_text",
  "context",
];

export function previewJsonl(text: string): Preview {
  const lines = text.split(/\r?\n/).filter((l) => l.trim());
  const samples: string[] = [];
  for (let i = 0; i < lines.length; i++) {
    let rec: unknown;
    try {
      rec = JSON.parse(lines[i]);
    } catch {
      return {
        records: lines.length,
        samples,
        problem: `Line ${i + 1} is not valid JSON.`,
      };
    }
    if (rec === null || typeof rec !== "object" || Array.isArray(rec))
      return {
        records: lines.length,
        samples,
        problem: `Line ${i + 1} is not a JSON object.`,
      };
    const o = rec as Record<string, unknown>;
    const body = TEXT_KEYS.map((k) => o[k]).find(
      (v) => typeof v === "string" && v.trim(),
    );
    if (body === undefined)
      return {
        records: lines.length,
        samples,
        problem: `Line ${i + 1} has no text: expected a field such as text, response or prompt.`,
      };
    if (samples.length < 3) samples.push(String(body));
  }
  return {
    records: lines.length,
    samples,
    problem: lines.length === 0 ? "The file has no records." : null,
  };
}

/** Up to `n` record texts spread through the file, for the scan panel while the audit runs. */
export function scanTexts(text: string, n = 30): string[] {
  const lines = text.split(/\r?\n/).filter((l) => l.trim());
  const step = Math.max(1, Math.floor(lines.length / n));
  const out: string[] = [];
  for (let i = 0; i < lines.length && out.length < n; i += step) {
    try {
      const o = JSON.parse(lines[i]) as Record<string, unknown>;
      const body = TEXT_KEYS.map((k) => o[k]).find(
        (v) => typeof v === "string" && v.trim(),
      );
      if (typeof body === "string") out.push(body);
    } catch {
      /* the preview already reported it */
    }
  }
  return out;
}

/** How long each UI stage really took (seconds), from the server's stage log; null while unknown. */
export function stageDurations(
  log: { stage: string; at: number }[],
): (number | null)[] {
  const start = STAGES.map((s, i) =>
    i === 0 ? 0 : (log.find((e) => s.keys.includes(e.stage))?.at ?? null),
  );
  const end = log.find((e) => e.stage === "done")?.at ?? null;
  return STAGES.map((_, i) => {
    const from = start[i];
    if (from == null) return null;
    let to: number | null = null;
    for (let k = i + 1; k < STAGES.length && to == null; k++) to = start[k];
    to ??= end;
    return to == null ? null : Math.max(0, to - from);
  });
}
