import { useQuery } from "@tanstack/react-query";
import { api, unwrap, type RepairResult } from "../api/client";
import { QueryView } from "../components/QueryView";

// §9.3 order: ours first, baselines after, oracle last as the ceiling
const STRATEGIES = [
  "harm_guided",
  "repr_sim",
  "risk_guided",
  "knn_guided",
  "random",
  "perplexity_filter",
  "oracle",
];

export type SeedSummary = {
  group: string; // cell id without its seed, e.g. "synthetic_r0.5"
  strategy: string;
  n: number;
  mean: number;
  min: number;
  max: number;
  distinct2: number;
};

/** Each (type, ratio) and strategy across its seeds: mean recovery, its range and mean distinct-2. */
export function acrossSeeds(rows: RepairResult[]): SeedSummary[] {
  const groups = new Map<string, RepairResult[]>();
  for (const r of rows) {
    if (r.recovery == null) continue;
    const key = `${r.cell.replace(/_s\d+$/, "")}|${r.strategy}`;
    groups.set(key, [...(groups.get(key) ?? []), r]);
  }
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
  return [...groups.entries()].map(([key, rs]) => {
    const [group, strategy] = key.split("|");
    const rec = rs.map((r) => r.recovery as number);
    return {
      group,
      strategy,
      n: rs.length,
      mean: mean(rec),
      min: Math.min(...rec),
      max: Math.max(...rec),
      distinct2: mean(rs.map((r) => r.distinct_2)),
    };
  });
}

const pct = (x: number) => Math.round(x * 100);
export const summaryText = (s?: SeedSummary) =>
  !s
    ? "—"
    : s.n < 2
      ? `${pct(s.mean)}%`
      : `${pct(s.mean)}% [${pct(s.min)}, ${pct(s.max)}]`;

export const recoveryText = (r?: RepairResult) =>
  !r
    ? "—"
    : r.recovery == null
      ? "no damage"
      : `${Math.round(r.recovery * 100)}%`;

/** X1 (§11): harm-guided recovers >= 10 points more than random on this cell. */
export function x1Margin(rows: RepairResult[]): number | null {
  const by = (s: string) => rows.find((r) => r.strategy === s)?.recovery;
  const h = by("harm_guided");
  const r = by("random");
  return h == null || r == null ? null : Math.round((h - r) * 100);
}

export function Repair() {
  const dataset = "wikitext2";
  const query = useQuery({
    queryKey: ["repair", dataset],
    queryFn: () =>
      unwrap(
        api.GET("/v1/repair/{dataset}", { params: { path: { dataset } } }),
      ),
  });
  return (
    <section>
      <h1 className="mb-1 text-xl font-semibold">Repair</h1>
      <p className="mb-4 text-sm text-muted">
        Replace the k most harmful training rows, retrain, and see how much of
        the lost holdout perplexity comes back. Recovery = (contaminated −
        repaired) / (contaminated − clean).
      </p>
      <QueryView
        query={query}
        label="repair results"
        empty="No repair results yet: X1 has not run on this dataset."
      >
        {(rows) => {
          const cells = [...new Set(rows.map((r) => r.cell))];
          const summary = acrossSeeds(rows);
          const groups = [...new Set(summary.map((s) => s.group))];
          return (
            <div className="overflow-x-auto">
              <h2 className="mb-1 font-semibold">Across seeds</h2>
              <p className="mb-2 text-sm text-muted">
                Mean recovery over seeds, with the lowest and highest seed in
                brackets. Hover for distinct-2 of the repaired set (higher =
                more diverse).
              </p>
              <table className="mb-6 w-full text-sm">
                <caption className="sr-only">
                  Mean recovery by strategy across seeds
                </caption>
                <thead>
                  <tr className="border-b border-border text-left">
                    <th scope="col" className="py-1 pr-4">
                      Type, ratio
                    </th>
                    {STRATEGIES.map((s) => (
                      <th key={s} scope="col" className="py-1 pr-4 font-mono">
                        {s}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {groups.map((g) => (
                    <tr key={g} className="border-b border-border">
                      <th
                        scope="row"
                        className="py-1 pr-4 font-mono font-normal"
                      >
                        {g}
                      </th>
                      {STRATEGIES.map((st) => {
                        const s = summary.find(
                          (x) => x.group === g && x.strategy === st,
                        );
                        return (
                          <td
                            key={st}
                            className="py-1 pr-4 tabular-nums"
                            title={
                              s
                                ? `${s.n} seed(s), distinct-2 ${s.distinct2.toFixed(3)}`
                                : undefined
                            }
                          >
                            {summaryText(s)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
              <h2 className="mb-1 font-semibold">Per cell</h2>
              <table className="w-full text-sm">
                <caption className="sr-only">
                  Recovery by strategy and cell
                </caption>
                <thead>
                  <tr className="border-b border-border text-left">
                    <th scope="col" className="py-1 pr-4">
                      Cell
                    </th>
                    {STRATEGIES.map((s) => (
                      <th key={s} scope="col" className="py-1 pr-4 font-mono">
                        {s}
                      </th>
                    ))}
                    <th scope="col" className="py-1">
                      Harm − random
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {cells.map((cell) => {
                    const rs = rows.filter((r) => r.cell === cell);
                    const m = x1Margin(rs);
                    return (
                      <tr key={cell} className="border-b border-border">
                        <th
                          scope="row"
                          className="py-1 pr-4 font-mono font-normal"
                        >
                          {cell}
                        </th>
                        {STRATEGIES.map((s) => {
                          const r = rs.find((x) => x.strategy === s);
                          return (
                            <td
                              key={s}
                              className="py-1 pr-4 tabular-nums"
                              title={
                                r
                                  ? `k=${r.k_replaced}, holdout ppl ${r.holdout_ppl}, distinct-2 ${r.distinct_2}`
                                  : undefined
                              }
                            >
                              {recoveryText(r)}
                            </td>
                          );
                        })}
                        <td className="py-1 tabular-nums">
                          {m == null
                            ? "—"
                            : `${m > 0 ? "+" : ""}${m} pts${m >= 10 ? " ✓" : ""}`}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          );
        }}
      </QueryView>
    </section>
  );
}
