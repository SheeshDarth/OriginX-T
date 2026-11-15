import { useQuery } from "@tanstack/react-query";
import * as Plot from "@observablehq/plot";
import { api, unwrap, type Canary as CanaryData } from "../api/client";
import { PlotFigure, TYPE_COLORS, TYPES, fit } from "../components/charts";
import { QueryView } from "../components/QueryView";

type Pair = CanaryData["pairs"][number];

const pct = (d: number) => `${(d * 100).toFixed(1)}%`;
export const rhoText = (c: CanaryData) =>
  `ρ = ${c.rho.toFixed(2)} [${c.ci[0].toFixed(2)}, ${c.ci[1].toFixed(2)}], n = ${c.n_pairs} cells`;

/** Proxy damage (x) against target damage (y), one dot per matched cell. */
export function canaryPlot(pairs: Pair[]) {
  const plot = Plot.plot({
    width: 640,
    height: 420,
    marginLeft: 56,
    style: { background: "transparent", color: "currentColor" },
    x: {
      label: "Proxy damage: DistilGPT-2 (holdout ppl vs its ratio 0)",
      tickFormat: pct,
      grid: true,
    },
    y: {
      label: "Target damage: Pythia-410M (holdout ppl vs its ratio 0)",
      tickFormat: pct,
      grid: true,
    },
    color: { domain: TYPES, range: TYPES.map((t) => TYPE_COLORS[t]) },
    symbol: { domain: [0.5, 1], range: ["circle", "triangle"] },
    marks: [
      Plot.dot(pairs, {
        x: "proxy_damage",
        y: "target_damage",
        stroke: "type",
        symbol: "ratio",
        r: 5,
      }),
      Plot.tip(
        pairs,
        Plot.pointer({
          x: "proxy_damage",
          y: "target_damage",
          title: (p: Pair) =>
            `${p.type}, ratio ${p.ratio}, seed ${p.seed}\nproxy ${pct(p.proxy_damage)}, target ${pct(p.target_damage)}`,
        }),
      ),
    ],
  });
  return fit(plot, 640, 420);
}

export function Canary() {
  const query = useQuery({
    queryKey: ["canary"],
    queryFn: () => unwrap(api.GET("/v1/x2/canary")),
  });
  return (
    <section>
      <h1 className="mb-1 text-xl font-semibold">Canary (X2)</h1>
      <p className="mb-4 text-sm text-muted">
        Does the small model&apos;s damage rank the larger model&apos;s? Each
        dot is one cell (type, ratio, seed) trained on the same data by both
        models. Damage = holdout perplexity ÷ the same seed&apos;s ratio-0
        perplexity − 1.
      </p>
      <QueryView
        query={query}
        label="the X2 verdict"
        empty="X2 has not run yet: no target-model results."
      >
        {(c) => (
          <div className="grid gap-6 lg:grid-cols-[auto_1fr]">
            <PlotFigure
              make={() => canaryPlot(c.pairs)}
              deps={[c]}
              caption="Dots up and to the right: both models are hurt. Circles are ratio 0.5, triangles ratio 1."
            />
            <div className="space-y-3 text-sm">
              <dl className="space-y-3">
                <div>
                  <dt className="text-muted">Gate X2</dt>
                  <dd className="font-medium">
                    {c.status} (bar: ρ ≥ {c.rho_min}, CI above 0)
                  </dd>
                </div>
                <div>
                  <dt className="text-muted">Rank correlation</dt>
                  <dd className="font-mono tabular-nums">{rhoText(c)}</dd>
                </div>
              </dl>
              <p className="max-w-prose text-muted">
                The pairing rule is ours: the handbook did not fix it. Dose
                drives much of the rank, so ρ partly says &quot;more
                contamination, more damage&quot; in both models. 95% bootstrap
                interval over cells.
              </p>
            </div>
          </div>
        )}
      </QueryView>
    </section>
  );
}
