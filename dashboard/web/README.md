# ORIGIN-T Console (`dashboard/web/`)

The product UI (HANDBOOK §8.6). React 18 + TypeScript + Vite, TanStack Router/Query, Tailwind on
`src/styles/tokens.css` (the only colour source).

```bash
corepack enable            # or: npm i -g pnpm@9
pnpm i
python -m dashboard.api    # from the repo root, serves http://127.0.0.1:8000
pnpm dev                   # http://localhost:5173; set VITE_API_URL to point elsewhere
```

- `pnpm typecheck && pnpm lint && pnpm test`: what CI runs.
- `pnpm e2e`: the demo flows in a real browser (light and dark) against the real API, with axe WCAG AA checks. It starts both servers itself. First time: `pnpm exec playwright install chromium`.
- `pnpm gen:api`: regenerate `src/api/schema.d.ts` after `dashboard/api/openapi.json` changes.
  Never hand-write request or response types.
- Tests mock the API with MSW, so they need no backend.
