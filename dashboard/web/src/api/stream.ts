import { useEffect, useState } from "react";
import { API_URL, type TrainlogRow } from "./client";

export type StreamStatus =
  "connecting" | "live" | "reconnecting" | "ended" | "error";

/** Something that happened to the stream itself (not a row), for the event log. */
export type StreamEvent = { at: number; kind: "reset" | "ended" | "lost" };

/** A run's trainlog from GET /v1/runs/{id}/stream (SSE): the logged rows first, then live
 *  ones. `reset` (a new run restarted the log) clears the rows; `end` (the log went quiet)
 *  closes the stream. On a dropped connection the browser reconnects by itself and sends
 *  Last-Event-ID, so no row arrives twice. */
export function useTrainlogStream(runId: string | null) {
  const [rows, setRows] = useState<TrainlogRow[]>([]);
  const [status, setStatus] = useState<StreamStatus>("connecting");
  const [events, setEvents] = useState<StreamEvent[]>([]);

  useEffect(() => {
    setRows([]);
    setEvents([]);
    if (!runId) return;
    setStatus("connecting");
    const log = (kind: StreamEvent["kind"]) =>
      setEvents((e) => [...e, { at: Date.now(), kind }]);
    const es = new EventSource(
      `${API_URL}/v1/runs/${encodeURIComponent(runId)}/stream`,
    );
    es.addEventListener("open", () => setStatus("live"));
    es.addEventListener("row", (e) => {
      const row = JSON.parse((e as MessageEvent<string>).data) as TrainlogRow;
      setRows((rs) =>
        rs.some((r) => r.step === row.step) ? rs : [...rs, row],
      );
    });
    es.addEventListener("reset", () => {
      setRows([]);
      log("reset");
    });
    es.addEventListener("end", () => {
      es.close();
      setStatus("ended");
      log("ended");
    });
    es.addEventListener("error", () => {
      if (es.readyState === EventSource.CLOSED) {
        setStatus("error");
        log("lost");
      } else setStatus("reconnecting");
    });
    return () => es.close();
  }, [runId]);

  return { rows, status, events };
}
