import { afterAll, beforeAll, expect, test } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { Canary, rhoText } from "./Canary";

const BODY = {
  status: "passed" as const,
  rho: 0.7765,
  ci: [0.5077, 0.9066],
  rho_min: 0.7,
  n_pairs: 2,
  pairs: [
    {
      type: "synthetic" as const,
      ratio: 0.5,
      seed: 0,
      proxy_damage: 0.05,
      target_damage: 0.08,
    },
    {
      type: "recursive" as const,
      ratio: 1,
      seed: 1,
      proxy_damage: 0.2,
      target_damage: 0.3,
    },
  ],
};
let found = true;
const server = setupServer(
  http.get("*/v1/x2/canary", () =>
    found ? HttpResponse.json(BODY) : new HttpResponse(null, { status: 404 }),
  ),
);
beforeAll(() => server.listen());
afterAll(() => server.close());

const view = () =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <Canary />
    </QueryClientProvider>,
  );

test("rho text carries the interval and n", () => {
  expect(rhoText(BODY)).toBe("ρ = 0.78 [0.51, 0.91], n = 2 cells");
});

test("shows the verdict and a dot per cell", async () => {
  view();
  expect(await screen.findByText(/ρ = 0\.78 \[0\.51, 0\.91\]/)).toBeTruthy();
  expect(screen.getByText(/passed \(bar: ρ ≥ 0\.7/)).toBeTruthy();
  expect(
    document.querySelectorAll("figure svg circle, figure svg path").length,
  ).toBeGreaterThanOrEqual(2);
});

test("says so when X2 has not run", async () => {
  found = false;
  view();
  expect(await screen.findByText(/X2 has not run yet/)).toBeTruthy();
});
