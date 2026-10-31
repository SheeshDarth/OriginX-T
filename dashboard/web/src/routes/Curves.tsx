import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import * as Plot from "@observablehq/plot";
import {
  api,
  unwrap,
  type Curve,
  type Dataset,
  type GridResult,
} from "../api/client";
import { PlotFigure, TYPE_COLORS, TYPES } from "../components/charts";
import { QueryView } from "../components/QueryView";

type Metric = "holdout" | "benchmark";
const METRIC_LABEL: Record<Metric, string> = {
  holdout: "Hidden-holdout perplexity",
  benchmark: "Benchmark-item perplexity",
};

/** Mean line + min-max band over seeds (or one seed's line), with every cell as a clickable dot. */
export function curvesPlot(
  curves: Curve[],
  results: GridResult[],
  metric: Metric,
  seed: number | "all",
  types: string[],
  onPick: (cell: GridResult) => void,
) {
  const y = `${metric}_ppl` as const;
  const cs = curves.filter((c) => types.includes(c.type));
  const rs = results.filter(
    (r) => types.includes(r.type) && (seed === "all" || r.seed === seed),
  );
  const marks =
    seed === "all"
      ? [
          Plot.areaY(cs, {
            x: "ratio",
            y1: `${y}_min`,
            y2: `${y}_max`,
            fill: "type",
            fillOpacity: 0.12,
          }),
          Plot.lineY(cs, {
            x: "ratio",
            y: `${y}_mean`,
            stroke: "type",
            strokeWidth: 2,
          }),
        ]
      : [
          Plot.lineY(rs, {
            x: "ratio",
            y,
            stroke: "type",
            strokeWidth: 2,
            sort: "ratio",
          }),
        ];
  const plot = Plot.plot({
    width: 900,
    height: 420,
    marginLeft: 56,
    style: { background: "transparent", color: "currentColor" },
    x: {
      label: "Contamination ratio (share of training set)",
      ticks: [0, 0.25, 0.5, 0.75, 1],
    },
    y: { label: `${METRIC_LABEL[metric]} (lower is better)`, grid: true },
    color: {
      domain: TYPES,
      range: TYPES.map((t) => TYPE_COLORS[t]),
    },
    marks: [
      ...marks,
      Plot.dot(rs, { x: "ratio", y, stroke: "type", r: 2.5 }),
      Plot.dot(
        rs,
        Plot.pointer({ x: "ratio", y, stroke: "type", r: 6, strokeWidth: 2 }),
      ),
      Plot.tip(
        rs,
        Plot.pointer({
          x: "ratio",
          y,
          title: (r: GridResult) =>
            `${r.cell}\n${METRIC_LABEL[metric]}: ${r[y].toFixed(1)}\nClick for details`,
        }),
      ),
    ],
  });
  plot.addEventListener("click", () => {
    const v = (plot as unknown as { value: GridResult | null }).value;
    if (v) onPick(v);
  });
  return plot;
}

export function Curves() {
  const [dataset, setDataset] = useState<Dataset>("wikitext2");
  const [metric, setMetric] = useState<Metric>("holdout");
  const [seed, setSeed] = useState<number | "all">("all");
  const [types, setTypes] = useState<string[]>(TYPES);
  const navigate = useNavigate();
  const path = { params: { path: { dataset } } };
  const curves = useQuery({
    queryKey: ["curves", dataset],
    queryFn: () => unwrap(api.GET("/v1/grids/{dataset}/curves", path)),
  });
  const results = useQuery({
    queryKey: ["results", dataset],
    queryFn: () => unwrap(api.GET("/v1/grids/{dataset}/results", path)),
  });
  const seeds = [...new Set((results.data ?? []).map((r) => r.seed))].sort();
  const nSeeds = seed === "all" ? seeds.length : 1;

  return (
    <section aria-labelledby="curves-title">
      <h1 id="curves-title" className="text-lg font-semibold">
        Collapse curves
      </h1>
      <p className="mb-4 text-sm text-muted">
        How perplexity changes as more of the training set is replaced by
        contamination. Click a point to open that cell.
      </p>
      <div className="mb-4 flex flex-wrap items-center gap-4 text-sm">
        <label>
          Dataset{" "}
          <select
            value={dataset}
            onChange={(e) => setDataset(e.target.value as Dataset)}
            className="ml-1 rounded border border-border bg-surface px-1 py-0.5"
          >
            <option value="wikitext2">wikitext2</option>
            <option value="tinystories">tinystories</option>
          </select>
        </label>
        <label>
          Metric{" "}
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value as Metric)}
            className="ml-1 rounded border border-border bg-surface px-1 py-0.5"
          >
            <option value="holdout">Hidden holdout</option>
            <option value="benchmark">Benchmark items</option>
          </select>
        </label>
        <label>
          Seed{" "}
          <select
            value={String(seed)}
            onChange={(e) =>
              setSeed(e.target.value === "all" ? "all" : Number(e.target.value))
            }
            className="ml-1 rounded border border-border bg-surface px-1 py-0.5"
          >
            <option value="all">All (mean, min–max band)</option>
            {seeds.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <fieldset className="flex flex-wrap gap-3">
          <legend className="sr-only">Contamination types</legend>
          {TYPES.map((t) => (
            <label key={t} className="flex items-center gap-1">
              <input
                type="checkbox"
                checked={types.includes(t)}
                onChange={(e) =>
                  setTypes(
                    e.target.checked
                      ? [...types, t]
                      : types.filter((x) => x !== t),
                  )
                }
              />
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ background: TYPE_COLORS[t] }}
              />
              <span className="font-mono text-xs">{t}</span>
            </label>
          ))}
        </fieldset>
      </div>
      <QueryView
        query={curves}
        label="curves"
        empty={`The ${dataset} grid has not been run yet.`}
      >
        {(cs) => (
          <QueryView
            query={results}
            label="grid results"
            empty={`The ${dataset} grid has not been run yet.`}
          >
            {(rs) => (
              <PlotFigure
                deps={[cs, rs, metric, seed, types]}
                make={() =>
                  curvesPlot(cs, rs, metric, seed, types, (r) =>
                    navigate({
                      to: "/cells/$dataset/$cell",
                      params: { dataset, cell: r.cell },
                    }),
                  )
                }
                caption={`${METRIC_LABEL[metric]} by contamination ratio, ${nSeeds} seed${nSeeds === 1 ? "" : "s"}, DistilGPT-2 + LoRA on ${dataset}. ${
                  seed === "all"
                    ? "Line = mean, band = min–max over seeds, dots = single runs. "
                    : ""
                }Look at how fast each type rises above the ratio-0 baseline.`}
              />
            )}
          </QueryView>
        )}
      </QueryView>
    </section>
  );
}
