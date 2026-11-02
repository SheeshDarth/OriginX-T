import { afterAll, afterEach, beforeAll, expect, test, vi } from "vitest";
import { act, cleanup, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  createMemoryHistory,
  createRouter,
  RouterProvider,
} from "@tanstack/react-router";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { routeTree } from "../../router";
import type { AlarmMonitor, RunSummary, TrainlogRow } from "../../api/client";
import { band, firstAlarm, groupRuns } from "./Monitor";

/** jsdom has no EventSource: this one lets a test push SSE events by hand. */
class FakeEventSource {
  static CLOSED = 2;
  static last: FakeEventSource;
  readyState = 1;
  listeners: Record<string, ((e: { data?: string }) => void)[]> = {};
  constructor(public url: string) {
    FakeEventSource.last = this;
  }
  addEventListener(type: string, fn: (e: { data?: string }) => void) {
    (this.listeners[type] ??= []).push(fn);
  }
  close() {
    this.readyState = FakeEventSource.CLOSED;
  }
  emit(type: string, data?: unknown) {
    act(() => {
      for (const fn of this.listeners[type] ?? [])
        fn({ data: data === undefined ? undefined : JSON.stringify(data) });
    });
  }
}

const SIGNALS = {
  anisotropy_last: 0.935,
  eff_rank_last: 372,
  part_ratio_last: 2.06,
  dead_fraction_mean: 0.0054,
  stable_rank_attn: 89.9,
  delta_rel_norm: 0,
};
const row = (step: number, anisotropy: number, alarm = false): TrainlogRow => ({
  run_id: "wikitext2_synthetic_r1_s0",
  step,
  holdout_loss: 4.3 - step / 1000,
  signals: { ...SIGNALS, anisotropy_last: anisotropy },
  alarm,
  alarm_signal: alarm ? "anisotropy_last" : null,
});
const summary = (
  cell: string,
  type: string,
  ratio: number,
  alarm_step: number | null = null,
): RunSummary => ({
  run_id: `wikitext2_${cell}`,
  dataset: "wikitext2",
  cell,
  type,
  ratio,
  seed: 0,
  n_rows: 14,
  last_step: 125,
  alarm_step,
  alarm_signal: alarm_step == null ? null : "anisotropy_last",
  updated_at: "2026-11-01T04:37:58+00:00",
});
const RUNS = [
  summary("synthetic_r1_s0", "synthetic", 1, 20),
  summary("human_control_r1_s0", "human_control", 1),
  summary("synthetic_r0_s7", "synthetic", 0),
];
const MONITOR: AlarmMonitor = {
  signals: ["anisotropy_last"],
  one_sided: false,
  center: { anisotropy_last: { "10": 0.001, "20": 0.002 } },
  scale: { anisotropy_last: { "10": 0.0005, "20": 0.001 } },
  alpha: 0.05,
  n_calibration: 20,
  threshold: 3,
  threshold_infinite: false,
};

let monitor: AlarmMonitor | null = null;
const server = setupServer(
  http.get("*/v1/runs", () => HttpResponse.json(RUNS)),
  http.get("*/v1/x3/monitor", () =>
    monitor
      ? HttpResponse.json(monitor)
      : HttpResponse.json({ title: "Not found", status: 404 }, { status: 404 }),
  ),
);
beforeAll(() => {
  server.listen();
  vi.stubGlobal("EventSource", FakeEventSource);
});
afterEach(() => {
  cleanup(); // no vitest globals, so Testing Library does not unmount between tests by itself
  monitor = null;
});
afterAll(() => {
  server.close();
  vi.unstubAllGlobals();
});

function renderAt(path: string) {
  const router = createRouter({
    routeTree,
    history: createMemoryHistory({ initialEntries: [path] }),
  });
  render(
    <QueryClientProvider client={new QueryClient()}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
}

test("helpers: groups, first alarm, the band on the run's own baseline", () => {
  const g = groupRuns([...RUNS, { ...RUNS[0], run_id: "tiny", dataset: null }]);
  expect(g.healthy.map((r) => r.cell)).toEqual([
    "human_control_r1_s0",
    "synthetic_r0_s7",
  ]);
  expect(g.contaminated.map((r) => r.cell)).toEqual(["synthetic_r1_s0"]);
  expect(g.other.map((r) => r.run_id)).toEqual(["tiny"]);

  const rows = [
    row(0, 0.935),
    row(10, 0.93),
    row(20, 0.92, true),
    row(30, 0.91, true),
  ];
  expect(firstAlarm(rows)).toEqual({ step: 20, signal: "anisotropy_last" });
  expect(firstAlarm(rows.slice(0, 2))).toBeNull();

  const b = band(rows, MONITOR, "anisotropy_last");
  expect(b.map((x) => x.step)).toEqual([10, 20]);
  expect(b[1].lo).toBeCloseTo(0.935 + 0.002 - 3 * 0.001);
  expect(b[1].hi).toBeCloseTo(0.935 + 0.002 + 3 * 0.001);
  expect(band(rows, MONITOR, "eff_rank_last")).toEqual([]); // not monitored
  expect(
    band(rows, { ...MONITOR, threshold: null }, "anisotropy_last"),
  ).toEqual([]);
  expect(band([], MONITOR, "anisotropy_last")).toEqual([]);
});

test("the run list groups healthy and contaminated runs and marks alarms", async () => {
  renderAt("/monitor");
  expect(await screen.findByText("Healthy (2)")).toBeTruthy();
  expect(screen.getByText("Contaminated (1)")).toBeTruthy();
  expect(screen.getByText("alarm @20")).toBeTruthy();
  expect(screen.getByText(/Pick a run/)).toBeTruthy();
});

test("a run streams in, raises its alarm, survives a reset and ends", async () => {
  renderAt("/monitor/wikitext2_synthetic_r1_s0");
  await screen.findByText("wikitext2_synthetic_r1_s0");
  const es = FakeEventSource.last;
  expect(es.url).toMatch(/\/v1\/runs\/wikitext2_synthetic_r1_s0\/stream$/);

  es.emit("open");
  es.emit("row", row(0, 0.935));
  es.emit("row", row(10, 0.93));
  expect(await screen.findByText("No alarm")).toBeTruthy();
  es.emit("row", row(10, 0.93)); // a duplicate step is ignored
  es.emit("row", row(20, 0.92, true));
  expect(screen.getByText(/3 rows, last step 20/)).toBeTruthy();
  expect(
    screen.getByText(
      /Alarm at step 20: Anisotropy, last layer left the healthy band/,
    ),
  ).toBeTruthy();
  expect(await screen.findByText(/No calibrated X3 monitor yet/)).toBeTruthy();

  es.emit("reset");
  expect(screen.getByText(/0 rows/)).toBeTruthy();
  expect(screen.getByText(/A new run restarted this log/)).toBeTruthy();
  es.emit("row", row(0, 0.935));
  es.emit("end");
  expect(screen.getByText(/Finished: no new rows for a minute/)).toBeTruthy();
  expect(es.readyState).toBe(FakeEventSource.CLOSED);
});

test("with a calibrated monitor the signal charts carry the healthy band", async () => {
  monitor = MONITOR;
  renderAt("/monitor/wikitext2_synthetic_r1_s0");
  await screen.findByText("wikitext2_synthetic_r1_s0");
  FakeEventSource.last.emit("row", row(0, 0.935));
  FakeEventSource.last.emit("row", row(10, 0.93));
  expect(
    await screen.findByText(
      /Anisotropy, last layer by step; grey band = healthy runs/,
    ),
  ).toBeTruthy();
  expect(screen.queryByText(/No calibrated X3 monitor yet/)).toBeNull();
});
