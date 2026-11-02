import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "@tanstack/react-router";
import * as Plot from "@observablehq/plot";
import {
  api,
  unwrap,
  type AlarmMonitor,
  type RunSummary,
  type TrainlogRow,
} from "../../api/client";
import { useTrainlogStream, type StreamStatus } from "../../api/stream";
import { PlotFigure, TYPE_COLORS } from "../../components/charts";
import { QueryView } from "../../components/QueryView";

/** The six GATE 3 signals, in trainlog/1 order, with what a reader should take from each. */
export const SIGNALS: { key: string; label: string }[] = [
  { key: "anisotropy_last", label: "Anisotropy, last layer" },
  { key: "eff_rank_last", label: "Effective rank, last layer" },
  { key: "part_ratio_last", label: "Participation ratio, last layer" },
  { key: "dead_fraction_mean", label: "Dead MLP neurons (share)" },
  { key: "stable_rank_attn", label: "Stable rank, attention weights" },
  { key: "delta_rel_norm", label: "Distance from the base weights" },
];
const LABEL = Object.fromEntries(SIGNALS.map((s) => [s.key, s.label]));
// non-text marks: >= 3:1 against both the light and the dark background
const ALARM = "#dc2626";
const BAND = "#8a8a8a";

const STATUS_TEXT: Record<StreamStatus, string> = {
  connecting: "Connecting…",
  live: "Receiving the log",
  reconnecting: "Connection dropped, reconnecting…",
  ended: "Finished: no new rows for a minute",
  error: "Connection lost",
};

/** Runs split into healthy (the control arm and clean ratio-0 runs), contaminated and other. */
export function groupRuns(runs: RunSummary[]) {
  const healthy = (r: RunSummary) =>
    r.type === "human_control" || r.ratio === 0;
  return {
    healthy: runs.filter((r) => r.dataset && healthy(r)),
    contaminated: runs.filter((r) => r.dataset && !healthy(r)),
    other: runs.filter((r) => !r.dataset),
  };
}

/** The first row with the alarm set: when it fired and on which signal. */
export function firstAlarm(rows: TrainlogRow[]) {
  const r = rows.find((x) => x.alarm);
  return r ? { step: r.step, signal: r.alarm_signal ?? null } : null;
}

/** The monitor's healthy band for one signal, placed on this run's own step-0 value:
 *  `base + center ± threshold × scale` per step. Empty without a finite threshold. */
export function band(
  rows: TrainlogRow[],
  monitor: AlarmMonitor | null | undefined,
  signal: string,
) {
  const base = rows[0]?.signals[signal];
  const t = monitor?.threshold;
  if (base == null || t == null || !monitor?.signals.includes(signal))
    return [];
  const center = monitor.center[signal] ?? {};
  const scale = monitor.scale[signal] ?? {};
  return Object.keys(center)
    .map(Number)
    .sort((a, b) => a - b)
    .map((step) => ({
      step,
      lo: base + center[step] - t * scale[step],
      hi: base + center[step] + t * scale[step],
    }));
}

function signalPlot(
  rows: TrainlogRow[],
  signal: string,
  color: string,
  monitor: AlarmMonitor | null | undefined,
  alarmStep: number | null,
) {
  const data = rows
    .map((r) => ({ step: r.step, v: r.signals[signal] }))
    .filter((d) => d.v != null);
  const b = band(rows, monitor, signal);
  return Plot.plot({
    width: 300,
    height: 170,
    marginLeft: 56,
    style: { background: "transparent", color: "currentColor" },
    x: { label: "Step" },
    y: { label: LABEL[signal] ?? signal, grid: true, nice: true },
    marks: [
      b.length
        ? Plot.areaY(b, {
            x: "step",
            y1: "lo",
            y2: "hi",
            fill: BAND,
            fillOpacity: 0.2,
          })
        : null,
      Plot.lineY(data, { x: "step", y: "v", stroke: color, strokeWidth: 2 }),
      Plot.dot(data, { x: "step", y: "v", fill: color, r: 2 }),
      alarmStep != null
        ? Plot.ruleX([alarmStep], {
            stroke: ALARM,
            strokeWidth: 2,
            strokeDasharray: "4 3",
          })
        : null,
    ],
  });
}

function lossPlot(
  rows: TrainlogRow[],
  color: string,
  alarmStep: number | null,
) {
  const data = rows
    .filter((r) => r.holdout_loss != null)
    .map((r) => ({ step: r.step, ppl: Math.exp(r.holdout_loss as number) }));
  return Plot.plot({
    width: 900,
    height: 260,
    marginLeft: 56,
    style: { background: "transparent", color: "currentColor" },
    x: { label: "Optimizer step" },
    y: {
      label: "Hidden-holdout perplexity (lower is better)",
      grid: true,
      nice: true,
    },
    marks: [
      Plot.lineY(data, { x: "step", y: "ppl", stroke: color, strokeWidth: 2 }),
      Plot.dot(data, { x: "step", y: "ppl", fill: color, r: 2.5 }),
      alarmStep != null
        ? Plot.ruleX([alarmStep], {
            stroke: ALARM,
            strokeWidth: 2,
            strokeDasharray: "4 3",
          })
        : null,
      Plot.tip(
        data,
        Plot.pointerX({
          x: "step",
          y: "ppl",
          title: (d: { step: number; ppl: number }) =>
            `step ${d.step}\nperplexity ${d.ppl.toFixed(2)}`,
        }),
      ),
    ],
  });
}

const fmt = (v: number | null | undefined, d = 4) =>
  v == null ? "—" : v.toFixed(d);

function RunView({
  runId,
  run,
}: {
  runId: string;
  run: RunSummary | undefined;
}) {
  const { rows, status, events } = useTrainlogStream(runId);
  const monitor = useQuery({
    queryKey: ["x3-monitor"],
    queryFn: () => unwrap(api.GET("/v1/x3/monitor")),
  });
  const color = TYPE_COLORS[run?.type ?? ""] ?? BAND;
  const alarm = firstAlarm(rows);
  const last = rows.at(-1);
  const first = rows[0];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <h2 className="font-mono text-sm font-semibold">{runId}</h2>
        <span className="text-xs text-muted">
          {run?.type ?? "unknown type"}
          {run?.ratio != null && ` · ratio ${run.ratio}`}
          {run?.seed != null && ` · seed ${run.seed}`}
        </span>
      </div>
      <p aria-live="polite" className="text-sm">
        <span className="text-muted">{STATUS_TEXT[status]}</span>
        {" · "}
        {rows.length} rows{last && `, last step ${last.step}`}
        {alarm ? (
          <strong className="ml-2 text-bad">
            Alarm at step {alarm.step}
            {alarm.signal &&
              `: ${LABEL[alarm.signal] ?? alarm.signal} left the healthy band`}
          </strong>
        ) : (
          rows.length > 0 && <span className="ml-2 text-ok">No alarm</span>
        )}
      </p>

      {monitor.data === null && (
        <p className="text-xs text-muted">
          No calibrated X3 monitor yet, so there are no healthy bands to draw.
          They appear once the alarm is calibrated (
          <code className="font-mono">configs/x3.yaml</code>, after CP0).
        </p>
      )}
      {monitor.data?.threshold_infinite && (
        <p className="text-xs text-warn">
          The monitor has too few calibration runs for its false-alarm target,
          so it never alarms.
        </p>
      )}

      {rows.length === 0 ? (
        <div
          aria-label="Waiting for rows"
          className="h-40 animate-pulse rounded bg-border/50"
        />
      ) : (
        <>
          <PlotFigure
            deps={[rows, color, alarm?.step]}
            make={() => lossPlot(rows, color, alarm?.step ?? null)}
            caption={`Hidden-holdout perplexity of ${runId} over training steps.${
              alarm ? ` Dashed line: the alarm at step ${alarm.step}.` : ""
            } Contaminated runs still improve, just less than healthy runs: damage is a gap, not a rise.`}
          />
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {SIGNALS.map((s) => (
              <PlotFigure
                key={s.key}
                deps={[rows, color, monitor.data, alarm?.step]}
                make={() =>
                  signalPlot(
                    rows,
                    s.key,
                    color,
                    monitor.data,
                    alarm?.step ?? null,
                  )
                }
                caption={`${s.label} by step${
                  monitor.data?.threshold != null
                    ? "; grey band = healthy runs"
                    : ""
                }.`}
              />
            ))}
          </div>
          <table className="w-full border-collapse text-sm">
            <caption className="mb-1 text-left text-xs text-muted">
              Signals at step 0 and at the latest step
            </caption>
            <thead className="text-left text-xs uppercase tracking-wide text-muted">
              <tr className="border-b border-border">
                <th className="py-2 pr-4 font-medium">Signal</th>
                <th className="py-2 pr-4 text-right font-medium">Step 0</th>
                <th className="py-2 pr-4 text-right font-medium">
                  Step {last?.step}
                </th>
                <th className="py-2 pr-4 text-right font-medium">Change</th>
              </tr>
            </thead>
            <tbody>
              {SIGNALS.map((s) => {
                const a = first?.signals[s.key];
                const b = last?.signals[s.key];
                return (
                  <tr key={s.key} className="border-b border-border">
                    <th scope="row" className="py-1 pr-4 text-left font-normal">
                      {s.label}
                      {alarm?.signal === s.key && (
                        <span className="ml-2 text-xs text-bad">alarm</span>
                      )}
                    </th>
                    <td className="py-1 pr-4 text-right font-mono text-xs">
                      {fmt(a)}
                    </td>
                    <td className="py-1 pr-4 text-right font-mono text-xs">
                      {fmt(b)}
                    </td>
                    <td className="py-1 pr-4 text-right font-mono text-xs">
                      {a == null || b == null
                        ? "—"
                        : `${b - a > 0 ? "+" : ""}${(b - a).toFixed(4)}`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      )}

      <section aria-labelledby="events-title">
        <h3 id="events-title" className="mb-1 text-sm font-semibold">
          Events
        </h3>
        <ol className="space-y-1 text-sm">
          {first && (
            <li>Step {first.step}: the base model, before any training.</li>
          )}
          {alarm && (
            <li className="text-bad">
              Step {alarm.step}: alarm,{" "}
              {LABEL[alarm.signal ?? ""] ?? alarm.signal ?? "a signal"} left the
              healthy band. It stays raised for the rest of the run.
            </li>
          )}
          {events.map((e) => (
            <li key={`${e.kind}-${e.at}`} className="text-muted">
              {e.kind === "reset" &&
                "A new run restarted this log; the rows above are the new run's."}
              {e.kind === "ended" &&
                `Finished at step ${last?.step ?? "?"}: the log has been quiet for a minute.`}
              {e.kind === "lost" && "The connection to the API was lost."}
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}

function RunList({
  runs,
  selected,
}: {
  runs: RunSummary[];
  selected: string | null;
}) {
  const [filter, setFilter] = useState("");
  const shown = runs.filter((r) => r.run_id.includes(filter.trim()));
  const groups = groupRuns(shown);
  const list = (title: string, rs: RunSummary[]) =>
    rs.length > 0 && (
      <div>
        <h3 className="mb-1 text-xs uppercase tracking-wide text-muted">
          {title} ({rs.length})
        </h3>
        <ul className="space-y-0.5">
          {rs.map((r) => (
            <li key={r.run_id}>
              <Link
                to="/monitor/$runId"
                params={{ runId: r.run_id }}
                aria-current={r.run_id === selected ? "page" : undefined}
                className="flex items-center justify-between gap-2 rounded px-1 py-0.5 font-mono text-xs hover:bg-border/50 aria-[current=page]:bg-border"
              >
                <span className="flex items-center gap-1">
                  <span
                    aria-hidden
                    className="inline-block h-2 w-2 shrink-0 rounded-full"
                    style={{ background: TYPE_COLORS[r.type ?? ""] ?? BAND }}
                  />
                  {r.cell ?? r.run_id}
                </span>
                {r.alarm_step != null && (
                  <span className="text-bad">alarm @{r.alarm_step}</span>
                )}
              </Link>
            </li>
          ))}
        </ul>
      </div>
    );
  return (
    <nav aria-label="Logged runs" className="space-y-4">
      <label className="block text-sm">
        Filter runs
        <input
          type="search"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="e.g. recursive_r1"
          className="mt-1 w-full rounded border border-border bg-surface px-2 py-1 font-mono text-xs"
        />
      </label>
      <div className="max-h-[70vh] space-y-4 overflow-y-auto pr-1">
        {list("Contaminated", groups.contaminated)}
        {list("Healthy", groups.healthy)}
        {list("Other", groups.other)}
        {shown.length === 0 && (
          <p className="text-xs text-muted">No run matches.</p>
        )}
      </div>
    </nav>
  );
}

export function Monitor() {
  const { runId } = useParams({ strict: false }) as { runId?: string };
  const runs = useQuery({
    queryKey: ["runs"],
    queryFn: () => unwrap(api.GET("/v1/runs")),
    refetchInterval: 15_000, // new runs appear; the ETag makes an unchanged list a 304
  });

  return (
    <section aria-labelledby="monitor-title">
      <h1 id="monitor-title" className="text-lg font-semibold">
        Live monitor
      </h1>
      <p className="mb-4 text-sm text-muted">
        White-box signals of a fine-tuning run, measured every few steps on the
        hidden holdout (X3). A run being trained now streams in as it goes; a
        finished run replays its log.
      </p>
      <QueryView query={runs} label="runs" empty="No run has been logged yet.">
        {(rs) =>
          rs.length === 0 ? (
            <p className="text-sm text-muted">
              No run has been logged yet. Log one with{" "}
              <code className="font-mono">python -m src.evaluation.live</code>.
            </p>
          ) : (
            <div className="grid gap-6 md:grid-cols-[17rem_1fr]">
              <RunList runs={rs} selected={runId ?? null} />
              {runId ? (
                <RunView
                  key={runId}
                  runId={runId}
                  run={rs.find((r) => r.run_id === runId)}
                />
              ) : (
                <p className="text-sm text-muted">
                  Pick a run. Healthy runs show how the signals move when
                  nothing is wrong; compare a contaminated run against them.
                </p>
              )}
            </div>
          )
        }
      </QueryView>
    </section>
  );
}
