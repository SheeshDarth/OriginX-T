import { useQuery } from "@tanstack/react-query";
import { api, unwrap, type RepairResult } from "../api/client";
import { QueryView } from "../components/QueryView";

// §9.3 order: ours first, baselines after, oracle last as the ceiling
const STRATEGIES = [
  "harm_guided",
  "repr_sim",
  "risk_guided",
  "random",
  "perplexity_filter",
  "oracle",
];

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
          return (
            <div className="overflow-x-auto">
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
