import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, unwrap } from "../api/client";
import type { components } from "../api/schema";
import { conformalSentence, pct, verdict } from "../components/RiskPanel";

/** The audit/1 report (src/risk/audit.py), as GET /v1/audits/{id} returns it once done. */
export type AuditReport = components["schemas"]["AuditReport"];
type AuditJob = components["schemas"]["AuditJob"];

export function parseReport(text: string): AuditReport {
  const r = JSON.parse(text);
  if (r?.schema !== "audit/1")
    throw new Error(`not an audit/1 report (schema: ${r?.schema ?? "none"})`);
  return r as AuditReport;
}

const csvCell = (v: unknown) => `"${String(v ?? "").replace(/"/g, '""')}"`;

export function flaggedCsv(r: AuditReport): string {
  const head = [
    "index",
    "sample_id",
    "benchmark_similarity",
    "repeat_3gram",
    "reasons",
    "text",
  ];
  const rows = r.flagged.map((f) =>
    [
      f.index,
      f.sample_id,
      f.benchmark_similarity,
      f.repeat_3gram,
      f.reasons.join("; "),
      f.text,
    ]
      .map(csvCell)
      .join(","),
  );
  return [head.join(","), ...rows].join("\n") + "\n";
}

function download(name: string, body: string, type: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([body], { type }));
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

const fmt = (v: number | null, d = 3) => (v == null ? "—" : v.toFixed(d));

/** The risk model learned from 1000-sample grid datasets, and lexical features such as
 *  type-token ratio depend on dataset size, so a score far from that size is out of distribution. */
export function sizeWarning(n: number): string | null {
  if (n >= 500 && n <= 2000) return null;
  return `This dataset has ${n} samples; the risk model learned from 1000-sample datasets and some features (type-token ratio, distinct n-grams) change with size, so treat the score as unreliable. The flagged samples below do not depend on size.`;
}

function Report({ r }: { r: AuditReport }) {
  const v = verdict(r.conformal_set);
  const maxAbs = Math.max(
    ...r.contributions.map((c) => Math.abs(c.contribution)),
    1e-9,
  );
  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-baseline gap-x-6 gap-y-2 rounded border border-border bg-surface p-4 text-sm">
        <span>
          <span className="font-mono text-3xl">{pct(r.p_degrade)}</span>{" "}
          <span className="text-muted">p(degrade)</span>
        </span>
        <span
          className={`rounded border px-1.5 py-0.5 font-mono text-xs ${v.className}`}
        >
          {v.text}
        </span>
        <span>{conformalSentence(r.conformal_set, r.alpha)}</span>
        <span className="text-muted">
          {r.n_samples} samples · model{" "}
          <span className="font-mono">{r.model}</span>
        </span>
        {sizeWarning(r.n_samples) && (
          <p className="w-full text-xs text-warn">{sizeWarning(r.n_samples)}</p>
        )}
      </div>

      <section>
        <h2 className="mb-1 text-sm font-semibold">What drives the score</h2>
        <p className="mb-2 text-xs text-muted">
          Each feature&apos;s push on the log-odds (weight × standardised
          value). Right = towards degrade.
        </p>
        <table className="w-full text-sm">
          <tbody>
            {r.contributions.map((c) => (
              <tr key={c.feature} className="border-b border-border">
                <th
                  scope="row"
                  className="w-64 py-1 pr-4 text-left font-mono text-xs font-normal"
                >
                  {c.feature}
                </th>
                <td className="w-24 py-1 pr-4 text-right font-mono text-xs">
                  {fmt(c.value)}
                </td>
                <td className="py-1">
                  <div className="relative h-3">
                    <div className="absolute left-1/2 top-0 h-3 w-px bg-border" />
                    <div
                      aria-hidden
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
                <td className="w-20 py-1 pl-4 text-right font-mono text-xs">
                  {c.contribution > 0 ? "+" : ""}
                  {c.contribution.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {r.missing_features.length > 0 && (
          <p className="mt-2 text-xs text-warn">
            Not computed for this dataset, entered as the training average:{" "}
            {r.missing_features.join(", ")}.
          </p>
        )}
      </section>

      <section>
        <div className="mb-2 flex items-baseline gap-4">
          <h2 className="text-sm font-semibold">
            Flagged samples: {r.n_flagged} of {r.n_samples}
          </h2>
          <button
            onClick={() => download("flagged.csv", flaggedCsv(r), "text/csv")}
            className="text-xs text-accent underline"
          >
            Export CSV
          </button>
          <button
            onClick={() =>
              download(
                "audit.json",
                JSON.stringify(r, null, 2),
                "application/json",
              )
            }
            className="text-xs text-accent underline"
          >
            Export JSON
          </button>
        </div>
        {r.flagged.length === 0 ? (
          <p className="text-sm text-muted">
            No sample is a benchmark near-copy or repeats itself.
          </p>
        ) : (
          <table className="w-full text-sm">
            <thead className="text-left text-xs uppercase tracking-wide text-muted">
              <tr className="border-b border-border">
                <th className="py-2 pr-4 font-medium">Sample</th>
                <th className="py-2 pr-4 font-medium">Benchmark sim.</th>
                <th className="py-2 pr-4 font-medium">Repeated 3-grams</th>
                <th className="py-2 pr-4 font-medium">Why</th>
                <th className="py-2 pr-4 font-medium">Text</th>
              </tr>
            </thead>
            <tbody>
              {r.flagged.map((f) => (
                <tr key={f.index} className="border-b border-border align-top">
                  <td className="py-1 pr-4 font-mono text-xs">{f.sample_id}</td>
                  <td className="py-1 pr-4 font-mono text-xs">
                    {fmt(f.benchmark_similarity, 2)}
                  </td>
                  <td className="py-1 pr-4 font-mono text-xs">
                    {f.repeat_3gram == null
                      ? "—"
                      : `${(f.repeat_3gram * 100).toFixed(0)}%`}
                  </td>
                  <td className="py-1 pr-4">{f.reasons.join("; ")}</td>
                  <td className="py-1 pr-4 text-muted">{f.text}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}

/** The upload's name as the API accepts it (letters, digits, space, dot, dash, underscore). */
export const safeName = (name: string) =>
  name.replace(/[^\w .-]/g, "_").slice(0, 120) || "upload";

const finished = (job: AuditJob | null | undefined) =>
  job?.status === "done" || job?.status === "failed";

export function Audit() {
  const [opened, setOpened] = useState<AuditReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [leakCheck, setLeakCheck] = useState(true);

  const job = useQuery({
    queryKey: ["audit", jobId],
    enabled: jobId !== null,
    queryFn: () =>
      unwrap(
        api.GET("/v1/audits/{audit_id}", {
          params: { path: { audit_id: jobId! } },
        }),
      ),
    refetchInterval: (q) => (finished(q.state.data) ? false : 1000),
  });

  async function upload(file: File | undefined) {
    if (!file) return;
    setOpened(null);
    setError(null);
    setJobId(null);
    const {
      data,
      error: problem,
      response,
    } = await api.POST("/v1/audits", {
      params: {
        query: {
          name: safeName(file.name),
          benchmark: leakCheck ? "wikitext2" : undefined,
        },
      },
      body: await file.text(),
      bodySerializer: (body) => body as string,
      headers: { "Content-Type": "application/x-ndjson" },
    });
    if (data) setJobId(data.id);
    else
      setError(
        `${file.name}: ${(problem as { detail?: string })?.detail ?? `HTTP ${response.status}`}`,
      );
  }

  async function open(file: File | undefined) {
    if (!file) return;
    setJobId(null);
    try {
      setOpened(parseReport(await file.text()));
      setError(null);
    } catch (e) {
      setOpened(null);
      setError(`${file.name}: ${(e as Error).message}`);
    }
  }

  const j = job.data;
  const report = j?.report ?? opened;
  return (
    <section aria-labelledby="audit-title">
      <h1 id="audit-title" className="text-lg font-semibold">
        Audit a dataset
      </h1>
      <p className="mb-4 text-sm text-muted">
        Will fine-tuning on this dataset damage the model? Upload it as JSONL
        (one record per line; prompt/response or text, up to 20 MB).
      </p>
      <div className="mb-6 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
        <label>
          Dataset to audit{" "}
          <input
            type="file"
            accept=".jsonl,.json,application/x-ndjson"
            onChange={(e) => upload(e.target.files?.[0])}
            className="ml-2 text-sm"
          />
        </label>
        <label className="flex items-center gap-1">
          <input
            type="checkbox"
            checked={leakCheck}
            onChange={(e) => setLeakCheck(e.target.checked)}
          />
          Check for copies of the WikiText benchmark items
        </label>
        <label className="text-muted">
          or open a saved report{" "}
          <input
            type="file"
            accept=".json,application/json"
            onChange={(e) => open(e.target.files?.[0])}
            className="ml-2 text-sm"
          />
        </label>
      </div>
      {error && (
        <p role="alert" className="mb-4 text-sm text-bad">
          {error}
        </p>
      )}
      {job.isError && (
        <p role="alert" className="mb-4 text-sm text-bad">
          Lost track of the audit ({job.error.message}). The server keeps jobs
          only until it restarts.
        </p>
      )}
      {j && !finished(j) && (
        <p role="status" className="mb-4 text-sm text-muted">
          Auditing {j.n_samples} samples of {j.name} ({j.status}). About 3
          minutes per 1000 samples on a GPU.
        </p>
      )}
      {j?.status === "failed" && (
        <p role="alert" className="mb-4 text-sm text-bad">
          The audit failed: {j.error}
        </p>
      )}
      {report ? (
        <Report r={report} />
      ) : (
        !error &&
        !j && <p className="text-sm text-muted">No dataset audited yet.</p>
      )}
    </section>
  );
}
