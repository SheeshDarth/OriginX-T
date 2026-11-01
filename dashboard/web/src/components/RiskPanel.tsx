import { useQuery } from "@tanstack/react-query";
import { api, unwrap } from "../api/client";
import { QueryView } from "./QueryView";

const OUTCOME = ["no degrade", "degrade"];

/** The conformal set in plain words (HANDBOOK §8.6). */
export function conformalSentence(set: number[], alpha: number): string {
  const cov = `At ${Math.round((1 - alpha) * 100)}% coverage`;
  if (set.length === 0)
    return `${cov}, no outcome is plausible enough: the model abstains.`;
  return `${cov}, the possible outcomes are {${set.map((y) => OUTCOME[y]).join(", ")}}.`;
}

/** Verdict from the conformal set, not from an arbitrary cut on p: a single label is a confident call. */
export function verdict(set: number[]): { text: string; className: string } {
  if (set.length === 1 && set[0] === 1)
    return {
      text: "predicted to degrade",
      className: "text-bad border-bad/40",
    };
  if (set.length === 1 && set[0] === 0)
    return { text: "predicted safe", className: "text-ok border-ok/40" };
  return { text: "uncertain", className: "text-warn border-warn/40" };
}

/** p(degrade) for one cell's training set, predicted before training, against what the grid then measured. */
export function RiskPanel({
  dataset,
  cell,
}: {
  dataset: string;
  cell: string;
}) {
  const risk = useQuery({
    queryKey: ["risk"],
    queryFn: () => unwrap(api.GET("/v1/risk")),
  });
  return (
    <div className="mb-6 rounded border border-border bg-surface p-4">
      <h2 className="mb-2 text-sm font-semibold">Risk prediction</h2>
      <QueryView
        query={risk}
        label="risk predictions"
        empty="The risk model has not been run yet."
      >
        {(rows) => {
          const r = rows.find((x) => x.dataset === dataset && x.cell === cell);
          if (!r)
            return (
              <p className="text-sm text-muted">
                No held-out prediction for this cell. Ratio-0 sets are never
                test cells: every type shares them.
              </p>
            );
          const v = verdict(r.conformal_set);
          return (
            <div className="flex flex-wrap items-baseline gap-x-6 gap-y-2 text-sm">
              <span>
                <span className="font-mono text-2xl">
                  {(r.p_degrade * 100).toFixed(0)}%
                </span>{" "}
                <span className="text-muted">p(degrade)</span>
              </span>
              <span
                className={`rounded border px-1.5 py-0.5 font-mono text-xs ${v.className}`}
              >
                {v.text}
              </span>
              <span>{conformalSentence(r.conformal_set, r.alpha)}</span>
              <span className="text-muted">
                Measured after training:{" "}
                <span className="text-fg">{OUTCOME[r.label]}</span> · model{" "}
                <span className="font-mono">{r.model}</span>, trained without
                this cell&apos;s contamination type
              </span>
            </div>
          );
        }}
      </QueryView>
    </div>
  );
}
