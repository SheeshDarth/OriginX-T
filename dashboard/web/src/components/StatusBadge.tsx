import type { Gate } from "../api/client";

const STYLE: Record<Gate["status"], string> = {
  passed: "text-ok border-ok/40",
  failed: "text-bad border-bad/40",
  not_evaluable: "text-warn border-warn/40",
  pending: "text-muted border-border",
};

/** Colour plus a text label, never colour alone. */
export function StatusBadge({ status }: { status: Gate["status"] }) {
  return (
    <span
      className={`inline-block rounded border px-1.5 py-0.5 font-mono text-xs ${STYLE[status]}`}
    >
      {status.replace("_", " ")}
    </span>
  );
}
