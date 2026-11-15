import { useEffect, useRef, useState } from "react";
import { STAGES, activeStage, stageDurations } from "./explain";

const clock = (s: number) =>
  `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

function Mark({ state }: { state: "done" | "active" | "todo" }) {
  if (state === "done")
    return (
      <svg
        viewBox="0 0 20 20"
        className="h-5 w-5 shrink-0 text-ok"
        aria-hidden="true"
      >
        <circle cx="10" cy="10" r="9" fill="none" stroke="currentColor" />
        <path
          d="M6 10.5l2.8 2.8L14 7.6"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  if (state === "active")
    return (
      <svg
        viewBox="0 0 20 20"
        className="anim-spin h-5 w-5 shrink-0 text-accent"
        aria-hidden="true"
      >
        <circle
          cx="10"
          cy="10"
          r="8"
          fill="none"
          stroke="currentColor"
          strokeOpacity="0.2"
          strokeWidth="2"
        />
        <path
          d="M10 2a8 8 0 0 1 8 8"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
        />
      </svg>
    );
  return (
    <svg
      viewBox="0 0 20 20"
      className="h-5 w-5 shrink-0 text-border"
      aria-hidden="true"
    >
      <circle cx="10" cy="10" r="9" fill="none" stroke="currentColor" />
    </svg>
  );
}

/** The records the audit is working through, one at a time, with a scan line over them. */
function Scanner({ texts }: { texts: string[] }) {
  const [i, setI] = useState(0);
  useEffect(() => {
    if (texts.length < 2) return;
    const t = setInterval(() => setI((n) => (n + 1) % texts.length), 2200);
    return () => clearInterval(t);
  }, [texts]);
  if (texts.length === 0) return null;
  return (
    <figure className="relative overflow-hidden rounded border border-border bg-surface p-4">
      <figcaption className="mb-2 font-mono text-xs text-muted">
        record {i + 1} of {texts.length} shown from your file
      </figcaption>
      <p
        key={i}
        className="anim-fade-up min-h-[7.5rem] break-words font-mono text-xs leading-relaxed text-fg"
      >
        {texts[i].length > 420 ? `${texts[i].slice(0, 420)}…` : texts[i]}
      </p>
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-x-0 top-0 h-full overflow-hidden"
      >
        <div className="anim-scan h-1/2 w-full bg-accent/10" />
      </div>
    </figure>
  );
}

/** Shortest time a stage stays on screen, so a fast audit still shows every real stage. */
export const MIN_DWELL_MS = 650;

/** The audit's real stages, driven by the job's stage log. Each is shown for at least MIN_DWELL_MS,
 *  with the time the server actually spent on it; `onSettled` fires once every stage has been shown. */
export function Pipeline({
  status,
  stage,
  stages,
  nSamples,
  name,
  texts,
  onSettled,
  dwellMs = MIN_DWELL_MS,
}: {
  status: string;
  stage: string | null | undefined;
  stages: { stage: string; at: number }[];
  nSamples: number;
  name: string | null | undefined;
  texts: string[];
  onSettled: () => void;
  dwellMs?: number;
}) {
  const target = activeStage(status, stage);
  const [shown, setShown] = useState(0);
  const started = useRef(Date.now());
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (shown >= Math.max(target, 0)) return;
    const t = setTimeout(() => setShown((n) => n + 1), dwellMs);
    return () => clearTimeout(t);
  }, [shown, target, dwellMs]);
  useEffect(() => {
    if (shown >= STAGES.length) onSettled();
  }, [shown, onSettled]);
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(t);
  }, []);
  const elapsed = Math.round((now - started.current) / 1000);
  const took = stageDurations(stages);
  const active = target < 0 ? -1 : shown;
  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-8 lg:grid-cols-[minmax(0,1fr)_22rem]">
      <div className="min-w-0">
        <p role="status" aria-live="polite" className="sr-only">
          {active < 0
            ? "Waiting in the queue"
            : `Step ${Math.min(active + 1, STAGES.length)} of ${STAGES.length}: ${STAGES[Math.min(active, STAGES.length - 1)].title}`}
        </p>
        <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2 text-sm">
          <span>
            Auditing <span className="font-mono">{name ?? "your dataset"}</span>{" "}
            <span className="text-muted">
              ({nSamples.toLocaleString()} samples)
            </span>
          </span>
          <span className="font-mono text-muted">{clock(elapsed)} elapsed</span>
        </div>
        <ol className="space-y-px overflow-hidden rounded border border-border bg-border">
          {STAGES.map((s, i) => {
            const state =
              i < active ? "done" : i === active ? "active" : "todo";
            const t = took[i];
            return (
              <li
                key={s.id}
                aria-current={state === "active" ? "step" : undefined}
                className={`flex gap-3 bg-surface px-4 py-3 transition-opacity ${state === "todo" ? "opacity-60" : ""}`}
              >
                <Mark state={state} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="text-sm font-medium">{s.title}</span>
                    <span className="font-mono text-xs text-muted">
                      {state === "done" && t != null
                        ? t < 0.1
                          ? "<0.1s"
                          : `${t.toFixed(1)}s`
                        : ""}
                      {state === "active" ? "running" : ""}
                    </span>
                  </div>
                  <p className="text-xs text-muted">{s.detail}</p>
                  {state === "active" && (
                    <div
                      aria-hidden="true"
                      className="relative mt-2 h-0.5 overflow-hidden rounded bg-border"
                    >
                      <div className="anim-sweep absolute h-full w-1/4 bg-accent" />
                    </div>
                  )}
                </div>
              </li>
            );
          })}
        </ol>
        <p className="mt-3 text-xs text-muted">
          These are the audit&apos;s real stages and the time each took. Each is
          held on screen for a moment so you can follow; the first run on a
          server also loads the models, which takes about a minute.
        </p>
      </div>
      <Scanner texts={texts} />
    </div>
  );
}
