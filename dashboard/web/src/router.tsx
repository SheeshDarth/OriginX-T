import {
  createRootRoute,
  createRoute,
  createRouter,
  Link,
  Outlet,
} from "@tanstack/react-router";
import { CommandPalette, type Command } from "./components/CommandPalette";
import { TYPES } from "./components/charts";
import { CellDetail } from "./routes/CellDetail";
import { Curves } from "./routes/Curves";
import { Overview } from "./routes/Overview";

const COMMANDS: Command[] = [
  { label: "Overview", to: "/" },
  { label: "Collapse curves", to: "/curves" },
  ...["gate_0", "gate_1", "gate_2", "gate_3", "x1", "x2", "x3"].map((id) => ({
    label: `Gate ${id.replace("gate_", "").toUpperCase()}`,
    to: "/",
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
        <span className="font-mono text-sm font-semibold">ORIGIN-T</span>
        <nav aria-label="Main" className="flex gap-4 text-sm">
          <Link
            to="/"
            activeOptions={{ exact: true }}
            className="text-muted [&.active]:text-fg"
          >
            Overview
          </Link>
          <Link to="/curves" className="text-muted [&.active]:text-fg">
            Collapse curves
          </Link>
        </nav>
        <kbd className="ml-auto font-mono text-xs text-muted">Ctrl K</kbd>
      </header>
      <main className="mx-auto max-w-6xl p-4">
        <Outlet />
      </main>
      <CommandPalette commands={COMMANDS} />
    </div>
  ),
});

export const routeTree = root.addChildren([
  createRoute({ getParentRoute: () => root, path: "/", component: Overview }),
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
]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
