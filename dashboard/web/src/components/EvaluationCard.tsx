import { useQuery } from "@tanstack/react-query";
import { api, unwrap, type RiskEvaluation } from "../api/client";
import { QueryView } from "./QueryView";

type Estimate = { value?: number | null; ci95?: number[] | null };

/** "0.84 [0.71, 0.94]"; f formats each number. Missing parts read as "—". */
export function est(
  e: Estimate | null | undefined,
  f: (v: number) => string = (v) => v.toFixed(2),
): string {
  if (e?.value == null) return "—";
  return e.ci95?.length === 2
    ? `${f(e.value)} [${f(e.ci95[0])}, ${f(e.ci95[1])}]`
    : f(e.value);
}

const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
const signed = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;

/** H2: did the model beat the reference-LM baseline, with the paired CI of the AUROC difference. */
export function baselineLine(
  b: RiskEvaluation["baselines"]["ref_lm_loss"],
): string {
  if (!b) return "Baseline ref-LM loss: not computed yet";
  if (b.auroc_baseline == null)
    return `Baseline ref-LM loss: ${b.note ?? "not comparable"}`;
  return `Baseline ref-LM loss (AUROC ${b.auroc_baseline.toFixed(2)}): ${b.model_beats_baseline ? "beaten" : "not beaten"}, difference ${est(b.difference, signed)}`;
}

function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wide text-muted">{label}</div>
      <div className="font-mono text-lg">{value}</div>
      <div className="text-xs text-muted">{hint}</div>
    </div>
  );
}

/** The risk model's honest numbers: every cell scored by a model that never trained on its contamination type. */
export function EvaluationCard() {
  const ev = useQuery({
    queryKey: ["risk-evaluation"],
    queryFn: () => unwrap(api.GET("/v1/risk/evaluation")),
  });
  return (
    <section
      aria-labelledby="eval-title"
      className="mt-8 rounded border border-border bg-surface p-4"
    >
      <h2 id="eval-title" className="text-sm font-semibold">
        Risk model, held out by contamination type
      </h2>
      <QueryView
        query={ev}
        label="the risk evaluation"
        empty="The risk model has not been evaluated yet."
      >
        {(e) => {
          const t = e.leave_one_type_out;
          return (
            <>
              <p className="mb-3 text-xs text-muted">
                Each grid dataset is scored before training by a model that
                never saw its contamination type ({t.n} datasets, {t.n_degraded}{" "}
                measured as degraded; model{" "}
                <span className="font-mono">{e.model}</span>,{" "}
                {e.features.length} features). 95% bootstrap intervals in
                brackets.
              </p>
              {!t.auroc ? (
                <p className="text-sm text-muted">
                  Only one outcome among the held-out cells: AUROC is undefined.
                </p>
              ) : (
                <div className="grid gap-4 sm:grid-cols-3">
                  {t.auroc && (
                    <Stat
                      label="AUROC"
                      value={est(t.auroc)}
                      hint="ranking of degraded above safe; 0.5 = chance"
                    />
                  )}
                  {t.conformal && (
                    <Stat
                      label="Conformal coverage"
                      value={est(t.conformal.coverage, pct)}
                      hint={`target ${(t.conformal.target * 100).toFixed(0)}%; held-out types break the guarantee's exchangeability`}
                    />
                  )}
                  {t.ece != null && (
                    <Stat
                      label="Calibration error (ECE)"
                      value={t.ece.toFixed(2)}
                      hint="0 = probabilities match outcomes"
                    />
                  )}
                </div>
              )}
              <p className="mt-3 text-sm">
                {baselineLine(e.baselines.ref_lm_loss)}
              </p>
            </>
          );
        }}
      </QueryView>
    </section>
  );
}
