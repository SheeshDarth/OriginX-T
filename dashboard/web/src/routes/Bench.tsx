import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, unwrap, type Dataset } from "../api/client";
import type { paths } from "../api/schema";
import { TYPES } from "../components/charts";
import { QueryView } from "../components/QueryView";

type BenchType = NonNullable<
  paths["/v1/bench/samples"]["get"]["parameters"]["query"]
>["type"];
const BENCH_TYPES = [...TYPES, "duplicate"] as BenchType[];
const RATIOS = [0, 0.25, 0.5, 0.75, 1];
type Part = { op: "same" | "del" | "add"; text: string };

/** Character-level diff (LCS). ponytail: O(n·m) table, fine for bench rows (≤150 words). */
export function charDiff(a: string, b: string): Part[] {
  const n = a.length;
  const m = b.length;
  const L = new Uint16Array((n + 1) * (m + 1));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      L[i * (m + 1) + j] =
        a[i] === b[j]
          ? L[(i + 1) * (m + 1) + j + 1] + 1
          : Math.max(L[(i + 1) * (m + 1) + j], L[i * (m + 1) + j + 1]);
  const out: Part[] = [];
  const push = (op: Part["op"], ch: string) =>
    out.at(-1)?.op === op
      ? (out[out.length - 1].text += ch)
      : out.push({ op, text: ch });
  let i = 0;
  let j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && a[i] === b[j]) {
      push("same", a[i++]);
      j++;
    } else if (
      j < m &&
      (i === n || L[i * (m + 1) + j + 1] >= L[(i + 1) * (m + 1) + j])
    )
      push("add", b[j++]);
    else push("del", a[i++]);
  }
  return out;
}

function Diff({ original, copy }: { original: string; copy: string }) {
  return (
    <p className="whitespace-pre-wrap font-mono text-xs">
      {charDiff(original, copy).map((p, k) =>
        p.op === "same" ? (
          <span key={k}>{p.text}</span>
        ) : p.op === "del" ? (
          <del
            key={k}
            className="bg-bad/30 font-semibold"
            title="only in the original"
          >
            {p.text}
          </del>
        ) : (
          <ins
            key={k}
            className="bg-accent/30 font-semibold no-underline"
            title="only in the copy"
          >
            {p.text}
          </ins>
        ),
      )}
    </p>
  );
}

export function Bench() {
  const [dataset, setDataset] = useState<Dataset>("wikitext2");
  const [type, setType] = useState<BenchType>("benchmark_near");
  const [ratio, setRatio] = useState(0.5);
  const [seed, setSeed] = useState(0);
  const query = useQuery({
    queryKey: ["bench", dataset, type, ratio, seed],
    queryFn: () =>
      unwrap(
        api.GET("/v1/bench/samples", {
          params: {
            query: { dataset, type, ratio, seed, limit: 100 },
          },
        }),
      ),
  });
  const select = "ml-1 rounded border border-border bg-surface px-1 py-0.5";
  return (
    <section>
      <h1 className="mb-1 text-xl font-semibold">Bench browser</h1>
      <p className="mb-4 text-sm text-muted">
        ORIGIN-T-Bench samples by type and ratio. A copy shows a character diff
        against the text it was made from: struck out is only in the original,
        highlighted is only in the copy.
      </p>
      <div className="mb-4 flex flex-wrap gap-4 text-sm">
        <label>
          Dataset
          <select
            value={dataset}
            onChange={(e) => setDataset(e.target.value as Dataset)}
            className={select}
          >
            <option value="wikitext2">wikitext2</option>
            <option value="tinystories">tinystories</option>
          </select>
        </label>
        <label>
          Type
          <select
            value={type}
            onChange={(e) => setType(e.target.value as BenchType)}
            className={select}
          >
            {BENCH_TYPES.map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>
        </label>
        <label>
          Ratio
          <select
            value={ratio}
            onChange={(e) => setRatio(Number(e.target.value))}
            className={select}
          >
            {RATIOS.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </label>
        <label>
          Seed
          <select
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            className={select}
          >
            {[0, 1, 2].map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
      </div>
      <QueryView
        query={query}
        label="bench samples"
        empty="This bench file is not built yet: python -m src.bench.build"
      >
        {(rows) => (
          <ul className="space-y-3">
            {rows.map((s) => (
              <li
                key={s.sample_id}
                className="rounded border border-border bg-surface p-3"
              >
                <p className="mb-1 font-mono text-xs text-muted">
                  {s.sample_id} · {s.source}
                  {s.source === "benchmark_near" &&
                    ` · similarity ${s.benchmark_near_score.toFixed(2)}`}
                </p>
                {s.original ? (
                  <Diff original={s.original} copy={s.response} />
                ) : (
                  <p className="text-sm">{s.response}</p>
                )}
              </li>
            ))}
          </ul>
        )}
      </QueryView>
    </section>
  );
}
