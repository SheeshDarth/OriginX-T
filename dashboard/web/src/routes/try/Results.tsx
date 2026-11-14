import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import * as Plot from "@observablehq/plot";
import { api, unwrap } from "../../api/client";
import type { components } from "../../api/schema";
import { PlotFigure, fit } from "../../components/charts";
import { pct, verdict } from "../../components/RiskPanel";
import { flaggedCsv } from "../Audit";
import { caveat, explain, featureLabel } from "./explain";

type AuditReport = components["schemas"]["AuditReport"];
type SampleScore = components["schemas"]["SampleScore"];

const TONE = {
  bad: "text-bad border-bad/40",
  ok: "text-ok border-ok/40",
  warn: "text-warn border-warn/40",
} as const;

/** A colour token as a plain rgb() string, for chart marks that cannot read CSS variables. */
function token(name: string): string {
  const v = getComputedStyle(document.documentElement)
    .getPropertyValue(`--${name}`)
    .trim()
    .split(/\s+/)
    .join(",");
  return v ? `rgb(${v})` : "currentColor";
}

function save(name: string, body: string, type: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([body], { type }));
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

function Gauge({ p, tone }: { p: number; tone: keyof typeof TONE }) {
  const fill = { bad: "bg-bad", ok: "bg-ok", warn: "bg-warn" }[tone];
  return (
    <div className="w-full max-w-md">
      <div
        role="img"
        aria-label={`p(degrade) ${pct(p)}`}
        className="relative h-2 rounded bg-border"
      >
        <div
          className={`anim-fade-up absolute inset-y-0 left-0 rounded ${fill}`}
          style={{ width: `${Math.max(1, p * 100)}%` }}
        />
      </div>
      <div className="mt-1 flex justify-between font-mono text-xs text-muted">
        <span>0%</span>
        <span>p(degrade)</span>
        <span>100%</span>
      </div>
    </div>
  );
}

function Drivers({ r }: { r: AuditReport }) {
  const [all, setAll] = useState(false);
  const rows = all ? r.contributions : r.contributions.slice(0, 8);
  const maxAbs = Math.max(
    ...r.contributions.map((c) => Math.abs(c.contribution)),
    1e-9,
  );
  return (
    <section aria-labelledby="drivers-title">
      <h2 id="drivers-title" className="mb-1 text-sm font-semibold">
        What drives the score
      </h2>
      <p className="mb-3 text-xs text-muted">
        Each feature&apos;s push on the log-odds of degradation (weight ×
        standardised value). Bars to the right push towards degrade, to the left
        towards safe.
      </p>
      <table className="w-full text-sm">
        <caption className="sr-only">
          Feature contributions to the score
        </caption>
        <tbody>
          {rows.map((c) => (
            <tr key={c.feature} className="border-b border-border">
              <th
                scope="row"
                className="w-72 py-1.5 pr-4 text-left text-xs font-normal"
                title={c.feature}
              >
                {featureLabel(c.feature)}
              </th>
              <td className="w-20 py-1.5 pr-4 text-right font-mono text-xs text-muted">
                {c.value == null ? "—" : c.value.toFixed(3)}
              </td>
              <td className="py-1.5">
                <div className="relative h-3">
                  <div className="absolute left-1/2 top-0 h-3 w-px bg-border" />
                  <div
                    aria-hidden="true"
                    className={`absolute top-0 h-3 ${c.contribution > 0 ? "bg-bad/70" : "bg-ok/70"}`}
                    style={{
                      left:
                        c.contribution > 0
                          ? "50%"
                          : `${50 - (50 * -c.contribution) / maxAbs}%`,
                      width: `${(50 * Math.abs(c.contribution)) / maxAbs}%`,
                    }}
                  />
                </div>
              </td>
              <td className="w-16 py-1.5 pl-4 text-right font-mono text-xs">
                {c.contribution > 0 ? "+" : ""}
                {c.contribution.toFixed(2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {r.contributions.length > 8 && (
        <button
          type="button"
          onClick={() => setAll(!all)}
          className="mt-2 text-xs text-accent underline"
        >
          {all ? "Show the top 8" : `Show all ${r.contributions.length}`}
        </button>
      )}
      {r.missing_features.length > 0 && (
        <p className="mt-2 text-xs text-warn">
          Not computed for this dataset, entered as the training average:{" "}
          {r.missing_features.map(featureLabel).join(", ")}.
        </p>
      )}
    </section>
  );
}

const METHODS = {
  risk_reflm: {
    label: "Reference language model",
    x: "Predictability to the reference model (higher = more machine-like)",
    caption:
      "Each bar counts texts at that predictability. Generated text sits to the right of human text, so a mixed dataset shows a second hump on the right.",
  },
  risk_knn: {
    label: "Embedding density",
    x: "Closeness to the other texts in the set (higher = more redundant)",
    caption:
      "Near-duplicates and templated text sit to the right, because their nearest neighbour is almost identical.",
  },
} as const;

function Distribution({ scores }: { scores: SampleScore[] }) {
  const [method, setMethod] = useState<keyof typeof METHODS>("risk_reflm");
  const rows = useMemo(
    () =>
      scores.filter(
        (s) => s.method === method && s.score != null,
      ) as (SampleScore & { score: number })[],
    [scores, method],
  );
  if (!scores.some((s) => s.score != null)) return null;
  const m = METHODS[method];
  return (
    <section aria-labelledby="dist-title">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="dist-title" className="text-sm font-semibold">
          How your samples score
        </h2>
        <label className="text-xs text-muted">
          Score{" "}
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value as keyof typeof METHODS)}
            className="ml-1 rounded border border-border bg-surface px-1 py-0.5 text-fg"
          >
            {Object.entries(METHODS).map(([k, v]) => (
              <option key={k} value={k}>
                {v.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <PlotFigure
        make={() =>
          fit(
            Plot.plot({
              width: 760,
              height: 240,
              marginLeft: 48,
              marginBottom: 48,
              style: { background: "transparent", color: "currentColor" },
              x: { label: m.x, labelAnchor: "center" },
              y: { label: "Samples", grid: true },
              marks: [
                Plot.rectY(rows, {
                  ...Plot.binX({ y: "count" }, { x: "score", thresholds: 40 }),
                  fill: token("accent"),
                }),
                Plot.ruleY([0]),
              ],
            }),
            760,
            240,
          )
        }
        deps={[rows, method]}
        caption={m.caption}
      />
    </section>
  );
}

function Flagged({ r }: { r: AuditReport }) {
  const [q, setQ] = useState("");
  const shown = r.flagged.filter((f) =>
    `${f.sample_id} ${f.reasons.join(" ")} ${f.text}`
      .toLowerCase()
      .includes(q.trim().toLowerCase()),
  );
  return (
    <section aria-labelledby="flag-title">
      <div className="mb-2 flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <h2 id="flag-title" className="text-sm font-semibold">
          Flagged samples: {r.n_flagged.toLocaleString()} of{" "}
          {r.n_samples.toLocaleString()}
        </h2>
        {r.flagged.length > 0 && (
          <>
            <button
              type="button"
              onClick={() => save("flagged.csv", flaggedCsv(r), "text/csv")}
              className="text-xs text-accent underline"
            >
              Export CSV
            </button>
            <button
              type="button"
              onClick={() =>
                save(
                  "audit.json",
                  JSON.stringify(r, null, 2),
                  "application/json",
                )
              }
              className="text-xs text-accent underline"
            >
              Export JSON
            </button>
          </>
        )}
      </div>
      {r.flagged.length === 0 ? (
        <p className="text-sm text-muted">
          No sample is a benchmark near-copy or repeats itself.
        </p>
      ) : (
        <>
          <p className="mb-2 text-xs text-muted">
            A sample is flagged when it is a near-copy of a benchmark item or
            repeats its own phrases. Remove these first: they are the cheapest
            fix.
          </p>
          <label className="mb-2 block text-xs text-muted">
            Filter{" "}
            <input
              type="search"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="id, reason or text"
              className="ml-1 w-56 rounded border border-border bg-surface px-2 py-1 text-fg"
            />
          </label>
          <div className="max-h-96 overflow-auto rounded border border-border">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-surface text-left text-xs uppercase tracking-wide text-muted">
                <tr className="border-b border-border">
                  <th className="px-3 py-2 font-medium">Sample</th>
                  <th className="px-3 py-2 font-medium">Why</th>
                  <th className="px-3 py-2 font-medium">Benchmark sim.</th>
                  <th className="px-3 py-2 font-medium">Repeats</th>
                  <th className="px-3 py-2 font-medium">Text</th>
                </tr>
              </thead>
              <tbody>
                {shown.slice(0, 200).map((f) => (
                  <tr
                    key={f.index}
                    className="border-b border-border align-top"
                  >
                    <td className="px-3 py-1.5 font-mono text-xs">
                      {f.sample_id}
                    </td>
                    <td className="px-3 py-1.5 text-xs">
                      {f.reasons.join("; ")}
                    </td>
                    <td className="px-3 py-1.5 font-mono text-xs">
                      {f.benchmark_similarity == null
                        ? "—"
                        : f.benchmark_similarity.toFixed(2)}
                    </td>
                    <td className="px-3 py-1.5 font-mono text-xs">
                      {f.repeat_3gram == null
                        ? "—"
                        : `${(f.repeat_3gram * 100).toFixed(0)}%`}
                    </td>
                    <td className="max-w-md px-3 py-1.5 text-xs text-muted">
                      <span className="line-clamp-2">{f.text}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {shown.length > 200 && (
            <p className="mt-1 text-xs text-muted">
              Showing 200 of {shown.length}. Export for the full list.
            </p>
          )}
        </>
      )}
    </section>
  );
}

/** Step 3: the verdict in words, the numbers behind it, and the samples to blame. */
export function Results({
  report,
  scores,
  name,
  onAgain,
}: {
  report: AuditReport;
  scores: SampleScore[];
  name: string;
  onAgain: () => void;
}) {
  const ev = useQuery({
    queryKey: ["risk-evaluation"],
    queryFn: () => unwrap(api.GET("/v1/risk/evaluation")),
  });
  const x = explain(report);
  const v = verdict(report.conformal_set);
  const lo = ev.data?.leave_one_type_out;
  const note = caveat(
    lo?.auroc?.value,
    lo?.conformal?.coverage?.value,
    lo?.conformal?.target,
  );
  return (
    <div className="space-y-10">
      <section
        aria-labelledby="verdict-title"
        className="anim-fade-up rounded border border-border bg-surface p-6"
      >
        <div className="flex flex-wrap items-start justify-between gap-6">
          <div className="min-w-0 max-w-xl">
            <p className="font-mono text-xs text-muted">
              {name} · {report.n_samples.toLocaleString()} samples
            </p>
            <h2
              id="verdict-title"
              tabIndex={-1}
              className="mt-1 text-2xl font-semibold tracking-tight"
            >
              {x.headline}
            </h2>
            <div className="mt-4 space-y-2 text-sm leading-relaxed">
              {x.paragraphs.map((p, i) => (
                <p key={i}>{p}</p>
              ))}
            </div>
          </div>
          <div className="space-y-3">
            <span
              className={`inline-block rounded border px-2 py-1 font-mono text-xs ${TONE[x.tone]}`}
            >
              {v.text}
            </span>
            <div className="font-mono text-4xl tabular-nums">
              {pct(report.p_degrade)}
            </div>
            <Gauge p={report.p_degrade} tone={x.tone} />
          </div>
        </div>
        <p className="mt-5 border-t border-border pt-3 text-xs text-muted">
          {note}
        </p>
      </section>

      <Drivers r={report} />
      <Distribution scores={scores} />
      <Flagged r={report} />

      <div className="flex gap-4 border-t border-border pt-4 text-sm">
        <button
          type="button"
          onClick={onAgain}
          className="rounded border border-border px-3 py-1.5 transition-colors hover:border-accent/60"
        >
          Audit another dataset
        </button>
      </div>
    </div>
  );
}
