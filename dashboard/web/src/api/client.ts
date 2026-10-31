import createClient from "openapi-fetch";
import type { components, paths } from "./schema";

export type Gate = components["schemas"]["Gate"];

export const api = createClient<paths>({
  baseUrl: import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000",
  fetch: (req) => globalThis.fetch(req), // looked up per call, so test mocks (MSW) apply
});

/** The response body, or throw so TanStack Query shows the error state. A 404 means "not produced yet". */
export async function unwrap<T>(
  p: Promise<{ data?: T; error?: unknown; response: Response }>,
): Promise<T> {
  const { data, error, response } = await p;
  if (error !== undefined || data === undefined)
    throw Object.assign(new Error(`HTTP ${response.status}`), {
      status: response.status,
    });
  return data;
}
