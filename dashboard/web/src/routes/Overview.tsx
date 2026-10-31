import { useQuery } from "@tanstack/react-query";
import { api, unwrap, type Gate } from "../api/client";
import { QueryView } from "../components/QueryView";
import { StatusBadge } from "../components/StatusBadge";

const REPO = "https://github.com/SheeshDarth/OriginX-T/blob/main/";

type Signal = { auroc?: number | null; permutation_p?: number | null };

/** The numbers behind a status, in one line. */
export function evidence(g: Gate): string {
  const d = g.details as Record<string, unknown> | null | undefined;
  if (!d) return "";
  if (g.id === "gate_0") {
    const seeds = Object.values(d.per_seed as Record<string, boolean>);
    return `${seeds.filter(Boolean).length}/${seeds.length} seeds raise holdout ppl`;
  }
  if (g.id === "gate_2") {
    return Object.entries(d.types as Record<string, string>)
      .map(([t, s]) => `${t} ${s}`)
      .join(", ");
  }
  if (g.id === "gate_3") {
    const head = `${d.trajectories} trajectories, ${d.never_collapse} never collapse`;
    const best = Object.entries(d.signals as Record<string, Signal>).sort(
      ([, a], [, b]) => (b.auroc ?? 0) - (a.auroc ?? 0),
    )[0];
    if (!best) return head;
    const [name, s] = best;
    return `${head}; best AUROC ${s.auroc?.toFixed(2)} (${name}, permutation p ${s.permutation_p?.toFixed(2)})`;
  }
  return "";
}

export function Overview() {
  const gates = useQuery({
    queryKey: ["gates"],
    queryFn: () => unwrap(api.GET("/v1/gates")),
  });

  return (
    <section aria-labelledby="overview-title">
      <h1 id="overview-title" className="text-lg font-semibold">
        Gates
      </h1>
      <p className="mb-4 text-sm text-muted">
        Each gate is a go/no-go rule fixed before its results existed.
      </p>
      <QueryView query={gates} label="gates" empty="No gates reported yet.">
        {(data) => (
          <table className="w-full border-collapse text-sm">
            <thead className="text-left text-xs uppercase tracking-wide text-muted">
              <tr className="border-b border-border">
                <th className="py-2 pr-4 font-medium">Gate</th>
                <th className="py-2 pr-4 font-medium">Dataset</th>
                <th className="py-2 pr-4 font-medium">Status</th>
                <th className="py-2 pr-4 font-medium">Rule</th>
                <th className="py-2 pr-4 font-medium">Evidence</th>
              </tr>
            </thead>
            <tbody>
              {data.map((g) => (
                <tr
                  key={`${g.id}-${g.dataset ?? ""}`}
                  id={g.id}
                  className="border-b border-border align-top"
                >
                  <td className="py-2 pr-4 font-medium">{g.name}</td>
                  <td className="py-2 pr-4 font-mono text-xs">
                    {g.dataset ?? "—"}
                  </td>
                  <td className="whitespace-nowrap py-2 pr-4">
                    <StatusBadge status={g.status} />
                    {g.exploratory && (
                      <span className="ml-2 text-xs text-muted">
                        exploratory
                      </span>
                    )}
                  </td>
                  <td className="py-2 pr-4 text-muted">{g.rule}</td>
                  <td className="py-2 pr-4">
                    {evidence(g)}
                    {g.source && (
                      <a
                        href={REPO + g.source}
                        className="ml-2 font-mono text-xs text-accent underline"
                      >
                        source
                      </a>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </QueryView>
    </section>
  );
}
