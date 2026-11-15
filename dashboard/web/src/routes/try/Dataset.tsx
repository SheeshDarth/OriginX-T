import { useRef, useState } from "react";
import { previewJsonl, type Preview } from "./explain";

export type Sample = {
  file: string;
  title: string;
  about: string;
  expect: string;
};

/** public/samples/*: 1,000 WikiText-2 records each, built from the grid's pools (see README there). */
export const SAMPLES: Sample[] = [
  {
    file: "human_wikitext.jsonl",
    title: "Clean human text",
    about: "1,000 WikiText-2 paragraphs, all written by people.",
    expect: "Expected: safe",
  },
  {
    file: "half_recursive.jsonl",
    title: "Half recursively generated",
    about:
      "500 human paragraphs plus 500 produced by a model trained on its own output.",
    expect: "Expected: degrade",
  },
  {
    file: "leaked_benchmark.jsonl",
    title: "Half leaked benchmark text",
    about: "500 human paragraphs plus 500 near-copies of benchmark items.",
    expect: "Expected: flagged samples",
  },
];

export const MAX_MB = 20;
const kb = (n: number) =>
  n > 1e6
    ? `${(n / 1e6).toFixed(1)} MB`
    : `${Math.max(1, Math.round(n / 1e3))} KB`;

/** Step 1: choose a file, see what the audit will read, then start. */
export function Dataset({
  onStart,
  error,
  busy,
}: {
  onStart: (file: File, text: string, leakCheck: boolean) => void;
  error: string | null;
  busy: boolean;
}) {
  const [picked, setPicked] = useState<{
    file: File;
    text: string;
    preview: Preview;
  } | null>(null);
  const [over, setOver] = useState(false);
  const [leak, setLeak] = useState(true);
  const [local, setLocal] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  async function take(file: File | undefined) {
    if (!file) return;
    setLocal(null);
    if (file.size > MAX_MB * 2 ** 20) {
      setPicked(null);
      setLocal(`${file.name} is over ${MAX_MB} MB.`);
      return;
    }
    const text = await file.text();
    setPicked({ file, text, preview: previewJsonl(text) });
  }

  async function sample(s: Sample) {
    setLocal(null);
    try {
      const res = await fetch(`/samples/${s.file}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      const file = new File([text], s.file, { type: "application/x-ndjson" });
      setPicked({ file, text, preview: previewJsonl(text) });
    } catch (e) {
      setLocal(`Could not load the sample (${(e as Error).message}).`);
    }
  }

  const p = picked?.preview;
  const n = p?.records ?? 0;
  const ready = picked && p && !p.problem;
  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <div className="min-w-0">
        <label
          htmlFor="dataset-file"
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            void take(e.dataTransfer.files[0]);
          }}
          className={`flex cursor-pointer flex-col items-center justify-center rounded border border-dashed px-6 py-12 text-center transition-colors ${over ? "border-accent bg-accent/5" : "border-border bg-surface hover:border-accent/60"}`}
        >
          <svg
            viewBox="0 0 24 24"
            className="mb-3 h-7 w-7 text-muted"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            aria-hidden="true"
          >
            <path
              d="M12 16V4m0 0l-4 4m4-4l4 4M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <span className="text-sm font-medium">
            Drop a dataset here, or{" "}
            <span className="text-accent underline">browse</span>
          </span>
          <span className="mt-1 text-xs text-muted">
            JSONL, one record per line, with a text, response or prompt field.
            Up to {MAX_MB} MB.
          </span>
          <input
            id="dataset-file"
            ref={input}
            type="file"
            accept=".jsonl,.ndjson,.json,application/x-ndjson"
            onChange={(e) => void take(e.target.files?.[0])}
            className="sr-only"
          />
        </label>

        {(local || error) && (
          <p role="alert" className="mt-3 text-sm text-bad">
            {local ?? error}
          </p>
        )}

        {picked && p && (
          <div className="anim-fade-up mt-6 rounded border border-border bg-surface p-4">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <span className="font-mono text-sm">{picked.file.name}</span>
              <span className="font-mono text-xs text-muted">
                {n.toLocaleString()} records · {kb(picked.file.size)}
              </span>
            </div>
            {p.problem ? (
              <p role="alert" className="mt-2 text-sm text-bad">
                {p.problem}
              </p>
            ) : (
              <>
                <p className="mb-1 mt-3 text-xs uppercase tracking-wide text-muted">
                  First records
                </p>
                <ul className="space-y-1 text-xs text-muted">
                  {p.samples.map((s, i) => (
                    <li key={i} className="truncate font-mono">
                      {s}
                    </li>
                  ))}
                </ul>
                {(n < 500 || n > 2000) && (
                  <p className="mt-3 text-xs text-warn">
                    The risk model learned from 1,000-sample datasets. With {n}{" "}
                    records, treat the risk score as unreliable; the flagged
                    samples do not depend on size.
                  </p>
                )}
              </>
            )}
            <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={leak}
                  onChange={(e) => setLeak(e.target.checked)}
                />
                Check for copies of WikiText benchmark items
              </label>
              <button
                type="button"
                disabled={!ready || busy}
                onClick={() =>
                  picked && onStart(picked.file, picked.text, leak)
                }
                className="rounded bg-accent px-4 py-2 text-sm font-medium text-bg transition-opacity disabled:opacity-40"
              >
                {busy ? "Starting…" : "Run the audit"}
              </button>
            </div>
          </div>
        )}
      </div>

      <aside aria-labelledby="samples-title">
        <h2 id="samples-title" className="mb-2 text-sm font-semibold">
          No dataset at hand? Try a sample
        </h2>
        <ul className="space-y-2">
          {SAMPLES.map((s) => (
            <li key={s.file}>
              <button
                type="button"
                onClick={() => void sample(s)}
                className="w-full rounded border border-border bg-surface p-3 text-left transition-colors hover:border-accent/60"
              >
                <span className="block text-sm font-medium">{s.title}</span>
                <span className="block text-xs text-muted">{s.about}</span>
                <span className="mt-1 block font-mono text-xs text-muted">
                  {s.expect}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </aside>
    </div>
  );
}
