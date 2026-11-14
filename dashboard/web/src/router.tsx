import {
  createRootRoute,
  createRoute,
  createRouter,
  Link,
  Outlet,
} from "@tanstack/react-router";
import { CommandPalette, type Command } from "./components/CommandPalette";
import { TYPES } from "./components/charts";
import { Audit } from "./routes/Audit";
import { Bench } from "./routes/Bench";
import { Canary } from "./routes/Canary";
import { CellDetail } from "./routes/CellDetail";
import { Curves } from "./routes/Curves";
import { Home } from "./routes/Home";
import { Monitor } from "./routes/monitor/Monitor";
import { Overview } from "./routes/Overview";
import { Repair } from "./routes/Repair";
import { Try } from "./routes/try/Try";

const COMMANDS: Command[] = [
  { label: "Home", to: "/" },
  { label: "Try it: audit a dataset", to: "/try" },
  { label: "Evidence: gate board", to: "/overview" },
  { label: "Collapse curves", to: "/curves" },
  { label: "Audit a dataset", to: "/audit" },
  { label: "Live monitor (X3)", to: "/monitor" },
  { label: "Canary (X2)", to: "/canary" },
  ...["gate_0", "gate_1", "gate_2", "gate_3", "x1", "x2", "x3"].map((id) => ({
    label: `Gate ${id.replace("gate_", "").toUpperCase()}`,
    to: "/overview",
    hash: id,
  })),
  // every WikiText cell; the palette jumps straight to its detail page
  ...[0, 1, 2].flatMap((seed) =>
    [0, 0.25, 0.5, 0.75, 1].flatMap((ratio) =>
      TYPES.map((type) => {
        const cell = `${type}_r${ratio}_s${seed}`;
        return { label: `Cell ${cell}`, to: `/cells/wikitext2/${cell}` };
      }),
    ),
  ),
];

const root = createRootRoute({
  component: () => (
    <div className="min-h-screen">
      <header className="flex items-center gap-6 border-b border-border bg-surface px-4 py-2">
        <Link to="/" className="shrink-0 font-mono text-sm font-semibold">
          ORIGIN-T
        </Link>
        <nav
          aria-label="Main"
          className="flex min-w-0 gap-4 overflow-x-auto whitespace-nowrap text-sm [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
        >
          <Link to="/try" className="text-muted [&.active]:text-fg">
            Try it
          </Link>
          <Link to="/overview" className="text-muted [&.active]:text-fg">
            Evidence
          </Link>
          <Link to="/curves" className="text-muted [&.active]:text-fg">
            Collapse curves
          </Link>
          <Link to="/audit" className="text-muted [&.active]:text-fg">
            Audit
          </Link>
          <Link to="/repair" className="text-muted [&.active]:text-fg">
            Repair
          </Link>
          <Link to="/bench" className="text-muted [&.active]:text-fg">
            Bench
          </Link>
          <Link to="/monitor" className="text-muted [&.active]:text-fg">
            Live monitor
          </Link>
          <Link to="/canary" className="text-muted [&.active]:text-fg">
            Canary
          </Link>
        </nav>
        <kbd className="ml-auto hidden shrink-0 font-mono text-xs text-muted sm:block">
          Ctrl K
        </kbd>
      </header>
      <main className="mx-auto max-w-6xl p-4">
        <Outlet />
      </main>
      <CommandPalette commands={COMMANDS} />
    </div>
  ),
});

export const routeTree = root.addChildren([
  createRoute({ getParentRoute: () => root, path: "/", component: Home }),
  createRoute({ getParentRoute: () => root, path: "/try", component: Try }),
  createRoute({
    getParentRoute: () => root,
    path: "/overview",
    component: Overview,
  }),
  createRoute({ getParentRoute: () => root, path: "/audit", component: Audit }),
  createRoute({
    getParentRoute: () => root,
    path: "/repair",
    component: Repair,
  }),
  createRoute({ getParentRoute: () => root, path: "/bench", component: Bench }),
  createRoute({
    getParentRoute: () => root,
    path: "/curves",
    component: Curves,
  }),
  createRoute({
    getParentRoute: () => root,
    path: "/cells/$dataset/$cell",
    component: CellDetail,
  }),
  createRoute({
    getParentRoute: () => root,
    path: "/monitor",
    component: Monitor,
  }),
  createRoute({
    getParentRoute: () => root,
    path: "/monitor/$runId",
    component: Monitor,
  }),
  createRoute({
    getParentRoute: () => root,
    path: "/canary",
    component: Canary,
  }),
]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
