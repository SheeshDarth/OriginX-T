import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, unwrap } from "../../api/client";
import { safeName } from "../Audit";
import { Dataset } from "./Dataset";
import { Pipeline } from "./Pipeline";
import { Results } from "./Results";
import { scanTexts } from "./explain";

const STEPS = ["Dataset", "Analysis", "Results"];

function Stepper({ at }: { at: number }) {
  return (
    <ol aria-label="Progress" className="mb-8 flex items-center gap-3 text-sm">
      {STEPS.map((s, i) => {
        const state = i < at ? "done" : i === at ? "active" : "todo";
        return (
          <li
            key={s}
            aria-current={state === "active" ? "step" : undefined}
            className="flex items-center gap-3"
          >
            <span
              className={`flex h-6 w-6 items-center justify-center rounded-full border font-mono text-xs transition-colors ${
                state === "active"
                  ? "border-accent bg-accent text-bg"
                  : state === "done"
                    ? "border-accent text-accent"
                    : "border-border text-muted"
              }`}
            >
              {state === "done" ? "✓" : i + 1}
            </span>
            <span className={state === "todo" ? "text-muted" : "font-medium"}>
              {s}
            </span>
            {i < STEPS.length - 1 && (
              <span
                aria-hidden="true"
                className={`h-px w-10 sm:w-16 ${i < at ? "bg-accent" : "bg-border"}`}
              />
            )}
          </li>
        );
      })}
    </ol>
  );
}

export function Try({ dwellMs }: { dwellMs?: number } = {}) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [texts, setTexts] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [settled, setSettled] = useState(false);
  const heading = useRef<HTMLHeadingElement>(null);

  const job = useQuery({
    queryKey: ["audit", jobId],
    enabled: jobId !== null,
    queryFn: () =>
      unwrap(
        api.GET("/v1/audits/{audit_id}", {
          params: { path: { audit_id: jobId! } },
        }),
      ),
    refetchInterval: (q) =>
      q.state.data?.status === "done" || q.state.data?.status === "failed"
        ? false
        : 1000,
  });
  const j = job.data;
  const done = j?.status === "done" && j.report != null && settled;
  const at = done ? 2 : jobId && j?.status !== "failed" ? 1 : 0;

  useEffect(() => {
    heading.current?.focus();
  }, [at]);

  async function start(file: File, text: string, leak: boolean) {
    setBusy(true);
    setError(null);
    try {
      const {
        data,
        error: problem,
        response,
      } = await api.POST("/v1/audits", {
        params: {
          query: {
            name: safeName(file.name),
            benchmark: leak ? "wikitext2" : undefined,
          },
        },
        body: text,
        bodySerializer: (body) => body as string,
        headers: { "Content-Type": "application/x-ndjson" },
      });
      if (data) {
        setTexts(scanTexts(text));
        setSettled(false);
        setJobId(data.id);
      } else
        setError(
          `${file.name}: ${(problem as { detail?: string })?.detail ?? `HTTP ${response.status}`}`,
        );
    } catch {
      setError(
        "Could not reach the API. Start it with: python -m dashboard.api",
      );
    } finally {
      setBusy(false);
    }
  }

  function again() {
    setJobId(null);
    setTexts([]);
    setError(null);
  }

  return (
    <section aria-labelledby="try-title">
      <h1
        id="try-title"
        ref={heading}
        tabIndex={-1}
        className="mb-1 text-xl font-semibold outline-none"
      >
        {at === 0 && "Audit a dataset"}
        {at === 1 && "Analysing your dataset"}
        {at === 2 && "Audit result"}
      </h1>
      <p className="mb-6 max-w-2xl text-sm text-muted">
        {at === 0 &&
          "Will fine-tuning on this dataset damage the model? Upload it, or start from a sample."}
        {at === 1 &&
          "The audit measures the dataset, scores it with a risk model, and flags the samples to blame."}
        {at === 2 &&
          "The verdict, the evidence behind it, and the samples that need attention."}
      </p>
      <Stepper at={at} />
      {at === 0 && (
        <>
          {j?.status === "failed" && (
            <p role="alert" className="mb-4 text-sm text-bad">
              The audit failed: {j.error}
            </p>
          )}
          {job.isError && (
            <p role="alert" className="mb-4 text-sm text-bad">
              Lost track of the audit ({job.error.message}). The server keeps
              jobs only until it restarts.
            </p>
          )}
          <Dataset onStart={start} error={error} busy={busy} />
        </>
      )}
      {at === 1 && j && (
        <>
          <Pipeline
            status={j.status}
            stage={j.stage}
            stages={j.stages ?? []}
            nSamples={j.n_samples}
            name={j.name}
            texts={texts}
            onSettled={() => setSettled(true)}
            dwellMs={dwellMs}
          />
          <button
            type="button"
            onClick={again}
            className="mt-6 text-xs text-muted underline"
          >
            Stop watching and start over
          </button>
        </>
      )}
      {at === 2 && j?.report && (
        <Results
          report={j.report}
          scores={j.scores ?? []}
          name={j.name ?? "your dataset"}
          onAgain={again}
        />
      )}
    </section>
  );
}
