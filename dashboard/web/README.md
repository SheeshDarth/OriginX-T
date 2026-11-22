# ORIGIN-T Console (`dashboard/web/`)

The product UI (HANDBOOK §8.6). React 18 + TypeScript + Vite, TanStack Router/Query, Tailwind on
`src/styles/tokens.css` (the only colour source).

```bash
corepack enable            # or: npm i -g pnpm@9
pnpm i
python -m dashboard.api    # from the repo root, serves http://127.0.0.1:8000
pnpm dev                   # http://localhost:5173; set VITE_API_URL to point elsewhere
```

Screens: `/` landing page (hero chart from the committed grid), `/try` upload, live analysis and
explained result, `/overview` gate board, `/curves`, `/audit` (advanced, with saved reports), `/repair`,
`/bench`, `/monitor` (X3), `/canary` (X2). Command palette: Ctrl K.

`public/samples/` holds three 1,000-record datasets for the demo (see its README). The first audit
after the API starts loads the models (about a minute). A stage that runs over 90 s shows a note
about laptop power modes: the GPU slows down a lot on battery.

- `pnpm typecheck && pnpm lint && pnpm test`: what CI runs.
- `pnpm e2e`: the demo flows in a real browser (light and dark) against the real API, with axe WCAG AA checks. It starts both servers itself. First time: `pnpm exec playwright install chromium`.
- `pnpm gen:api`: regenerate `src/api/schema.d.ts` after `dashboard/api/openapi.json` changes.
  Never hand-write request or response types.
- Tests mock the API with MSW, so they need no backend.
