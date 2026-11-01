import createClient from "openapi-fetch";
import type { components, paths } from "./schema";

type S = components["schemas"];
export type Gate = S["Gate"];
export type Curve = S["Curve"];
export type GridResult = S["GridResult"];
export type WhiteboxCell = S["WhiteboxCell"];
export type FeatureRow = S["FeatureRow"];
export type Dataset = "wikitext2" | "tinystories";

export const api = createClient<paths>({
  baseUrl: import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000",
  fetch: (req) => globalThis.fetch(req), // looked up per call, so test mocks (MSW) apply
});

/** The response body; null on 404, which the API uses for "not produced yet" (HANDBOOK §7.7).
 *  Anything else throws, so TanStack Query shows the error state. */
export async function unwrap<T>(
  p: Promise<{ data?: T; error?: unknown; response: Response }>,
): Promise<T | null> {
  const { data, error, response } = await p;
  if (response.status === 404) return null;
  if (error !== undefined || data === undefined)
    throw Object.assign(new Error(`HTTP ${response.status}`), {
      status: response.status,
    });
  return data;
}
