# Sample datasets for the demo (`/try`)

Three 1,000-record JSONL files, built from the grid's cached pools (`data/processed/grid_pools/d497f2268060/`, WikiText-2) so the demo has something to audit:

| File | Contents | Expected audit |
|---|---|---|
| `human_wikitext.jsonl` | 1,000 human paragraphs (`human_control`, seed 0) | safe |
| `half_recursive.jsonl` | 500 human (`human_control`, seed 1) + 500 recursively generated (`recursive`, seed 0) | degrade |
| `leaked_benchmark.jsonl` | 500 human + 500 `benchmark_near` copies of benchmark items (`benchmark_near`, seed 0) | flagged samples |

Records are `{"id", "text"}`, shuffled with seed 0. The human text is from WikiText-2 (CC BY-SA 3.0); the generated text comes from the DistilGPT-2 pools (`docs/paper/track_b.md` §1). The expected-audit column is what the committed risk model returned when the demo was built; it is a description, not a test.
