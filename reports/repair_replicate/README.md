# B9 independent-pool replicate (WikiText-2)

The full X1 repair grid (B9) run end to end on **regenerated pools** on Siddharth's RTX 4050
(6 GB), 2026-11-09/10. It is *not* the official B9: the grid's own pools and checkpoints are on
the grid machine (#64), and sampling is not reproducible across machines (§12.3). So these
contaminated sets, checkpoints and recoveries are this machine's own, with the same config
fingerprint (`d497f2268060`), code (`main` + #135) and rules.

| file | what |
|---|---|
| `grid_results.jsonl` | the replicate's own grid (4 types × ratios 0, 0.5, 1 × seeds 0-2): the contaminated and clean perplexities recovery is measured against |
| `wikitext2.jsonl` | repair/1 (§7.5): 24 cells × 7 strategies + 3 `+rare` = 240 rows |
| `wikitext2_summary.json` | pilot, Gate X1, D10 secondary, H5, benchmark recovery (`src.mitigation.repair`) |

Pipeline: `src.finetune.grid` → `src.mitigation.harm` → `risk_reflm`/`risk_knn` via
`src.features.scores.score_cell` → `src.mitigation.repair`, with `configs/repair.yaml` pointed at
the replicate's folders. Results are in `docs/paper/track_b.md` §5.

**Read ratio 1 with care.** At ratio 1 every row is contaminated, so k = N and every strategy
replaces the whole set: all ratio-1 recoveries are ≈ 1 by construction. Comparisons between
strategies are informative only at ratio 0.5. The summary's pooled D10 secondary includes the
ratio-1 cells; the ratio-0.5 numbers are in Track B §5.
