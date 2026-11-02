import { afterAll, beforeAll, expect, test } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { Bench, charDiff } from "./Bench";

const sample = (
  sample_id: string,
  response: string,
  original: string | null,
) => ({
  sample_id,
  prompt: "",
  response,
  source: original ? "benchmark_near" : "human",
  generation: 0,
  contamination_ratio: 0.5,
  benchmark_near_score: original ? 0.97 : 0,
  split: "train",
  original,
});
const server = setupServer(
  http.get("*/v1/bench/samples", () =>
    HttpResponse.json([
      sample("v1-near3", "The sky is blue", "the sky is blue."),
      sample("k1", "A cat sat.", null),
    ]),
  ),
);
beforeAll(() => server.listen());
afterAll(() => server.close());

test("char diff marks exactly the edits and rebuilds both texts", () => {
  const a = "the sky is blue.";
  const b = "The sky is blue";
  const d = charDiff(a, b);
  expect(
    d
      .filter((p) => p.op !== "add")
      .map((p) => p.text)
      .join(""),
  ).toBe(a);
  expect(
    d
      .filter((p) => p.op !== "del")
      .map((p) => p.text)
      .join(""),
  ).toBe(b);
  expect(
    d
      .filter((p) => p.op !== "same")
      .map((p) => p.text)
      .sort(),
  ).toEqual([".", "T", "t"]);
  expect(charDiff("abc", "abc")).toEqual([{ op: "same", text: "abc" }]);
});

test("a copy shows its diff, a human row shows plain text", async () => {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <Bench />
    </QueryClientProvider>,
  );
  expect(await screen.findByText(/similarity 0.97/)).toBeTruthy();
  expect(
    screen.getByTitle("only in the copy", { exact: false }).textContent,
  ).toBe("T");
  expect(screen.getByText("A cat sat.")).toBeTruthy();
});
