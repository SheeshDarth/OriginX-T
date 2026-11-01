import type { ReactNode } from "react";
import type { UseQueryResult } from "@tanstack/react-query";

/** Loading skeleton, error, "not produced yet" empty state, or the data. */
export function QueryView<T>({
  query,
  label,
  empty,
  children,
}: {
  query: UseQueryResult<T | null>;
  label: string;
  empty: string;
  children: (data: T) => ReactNode;
}) {
  if (query.isPending)
    return (
      <div
        aria-label={`Loading ${label}`}
        className="h-24 animate-pulse rounded bg-border/50"
      />
    );
  if (query.isError)
    return (
      <p role="alert" className="text-sm text-bad">
        Could not load {label} ({query.error.message}). Is the API running?{" "}
        <code className="font-mono">python -m dashboard.api</code>
      </p>
    );
  if (query.data == null) return <p className="text-sm text-muted">{empty}</p>;
  return <>{children(query.data)}</>;
}
