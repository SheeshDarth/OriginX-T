import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "@tanstack/react-router";
import { api, unwrap, type Dataset, type GridResult } from "../api/client";
import { TYPE_COLORS } from "../components/charts";
import { QueryView } from "../components/QueryView";

const fmt = (v: number | null | undefined, d = 3) =>
  v == null ? "—" : v.toFixed(d);

/** "+19.4% vs ratio 0" against the same type and seed at ratio 0 (the cell's own clean baseline). */
export function vsBaseline(
  row: GridResult,
  rows: GridResult[],
  key: "holdout_ppl" | "benchmark_ppl",
) {
  const base = rows.find(
    (r) => r.type === row.type && r.seed === row.seed && r.ratio === 0,
  );
  if (!base || row.ratio === 0) return "";
  const pct = (100 * (row[key] - base[key])) / base[key];
  return `${pct >= 0 ? "+" : ""}${pct.toFixed(1)}% vs ratio 0`;
}

function KeyValues({ rows }: { rows: [string, string][] }) {
  return (
    <table className="w-full text-sm">
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k} className="border-b border-border">
            <th
              scope="row"
              className="py-1 pr-4 text-left font-mono text-xs font-normal text-muted"
            >
              {k}
            </th>
            <td className="py-1 text-right font-mono">{v}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function CellDetail() {
  const { dataset, cell } = useParams({ from: "/cells/$dataset/$cell" }) as {
    dataset: Dataset;
    cell: string;
  };
  const path = { params: { path: { dataset } } };
  const results = useQuery({
    queryKey: ["results", dataset],
    queryFn: () => unwrap(api.GET("/v1/grids/{dataset}/results", path)),
  });
  const whitebox = useQuery({
    queryKey: ["whitebox", dataset],
    queryFn: () => unwrap(api.GET("/v1/grids/{dataset}/whitebox", path)),
  });
  const features = useQuery({
    queryKey: ["features", dataset],
    queryFn: () => unwrap(api.GET("/v1/features/{dataset}", path)),
  });
  const row = results.data?.find((r) => r.cell === cell);
  const samples = useQuery({
    queryKey: ["bench", dataset, cell],
    enabled: !!row,
    queryFn: () =>
      unwrap(
        api.GET("/v1/bench/samples", {
          params: {
            query: {
              dataset,
              type: row!.type as never,
              ratio: row!.ratio,
              seed: row!.seed,
              limit: 20,
            },
          },
        }),
      ),
  });

  return (
    <section aria-labelledby="cell-title">
      <Link to="/curves" className="text-xs text-accent underline">
        ← Collapse curves
      </Link>
      <h1 id="cell-title" className="mt-1 font-mono text-lg font-semibold">
        {row && (
          <span
            className="mr-2 inline-block h-2.5 w-2.5 rounded-full"
            style={{ background: TYPE_COLORS[row.type] }}
          />
        )}
        {cell}
      </h1>
      <p className="mb-4 text-sm text-muted">
        One fine-tune of the {dataset} grid: its outcome, white-box signals,
        dataset features and training samples.
      </p>

      <div className="grid gap-6 md:grid-cols-3">
        <div>
          <h2 className="mb-2 text-sm font-semibold">Outcome</h2>
          <QueryView
            query={results}
            label="grid results"
            empty="This grid has not been run yet."
          >
            {(rows) => {
              const r = rows.find((x) => x.cell === cell);
              if (!r)
                return (
                  <p className="text-sm text-muted">
                    No cell {cell} in this grid.
                  </p>
                );
              return (
                <KeyValues
                  rows={[
                    ["type", r.type],
                    ["ratio", fmt(r.ratio, 2)],
                    ["seed", String(r.seed)],
                    [
                      "train samples",
                      `${r.n_train} (${r.n_contaminated} contaminated)`,
                    ],
                    [
                      "holdout ppl",
                      `${fmt(r.holdout_ppl, 1)} ${vsBaseline(r, rows, "holdout_ppl")}`,
                    ],
                    [
                      "benchmark ppl",
                      `${fmt(r.benchmark_ppl, 1)} ${vsBaseline(r, rows, "benchmark_ppl")}`,
                    ],
                    [
                      "distinct-1 / -2",
                      `${fmt(r.distinct_1)} / ${fmt(r.distinct_2)}`,
                    ],
                    ["train time", `${fmt(r.seconds, 0)} s`],
                  ]}
                />
              );
            }}
          </QueryView>
        </div>
        <div>
          <h2 className="mb-2 text-sm font-semibold">White-box signals</h2>
          <QueryView
            query={whitebox}
            label="white-box signals"
            empty="White-box signals have not been measured for this grid."
          >
            {(rows) => {
              const w = rows.find((x) => x.cell === cell);
              if (!w)
                return (
                  <p className="text-sm text-muted">
                    No checkpoint measured for this cell.
                  </p>
                );
              return (
                <KeyValues
                  rows={Object.entries(w.signals).map(([k, v]) => [k, fmt(v)])}
                />
              );
            }}
          </QueryView>
        </div>
        <div>
          <h2 className="mb-2 text-sm font-semibold">Dataset features</h2>
          <QueryView
            query={features}
            label="features"
            empty="Features have not been computed for this grid yet."
          >
            {(rows) => {
              const f = rows.find((x) => x.cell === cell);
              if (!f)
                return (
                  <p className="text-sm text-muted">
                    No feature row for this cell.
                  </p>
                );
              return (
                <KeyValues
                  rows={Object.entries(f.features).map(([k, v]) => [k, fmt(v)])}
                />
              );
            }}
          </QueryView>
        </div>
      </div>

      <h2 className="mb-2 mt-8 text-sm font-semibold">Training samples</h2>
      <QueryView
        query={samples}
        label="samples"
        empty="No bench samples have been published for this cell yet."
      >
        {(rows) => (
          <table className="w-full text-sm">
            <thead className="text-left text-xs uppercase tracking-wide text-muted">
              <tr className="border-b border-border">
                <th className="py-2 pr-4 font-medium">Source</th>
                <th className="py-2 pr-4 font-medium">Generation</th>
                <th className="py-2 pr-4 font-medium">Text</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((s) => (
                <tr
                  key={s.sample_id}
                  className="border-b border-border align-top"
                >
                  <td className="py-1 pr-4 font-mono text-xs">{s.source}</td>
                  <td className="py-1 pr-4 font-mono text-xs">
                    {s.generation}
                  </td>
                  <td className="py-1 pr-4">{s.response}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </QueryView>
    </section>
  );
}
