"""Gate X2 verdict: does the proxy's damage rank the target's damage? (HANDBOOK 10.5, 11).

Damage of a cell = holdout ppl / the same seed's ratio-0 ppl - 1, so two models with different
perplexity scales are comparable. Pairs are the target's cells (ratio > 0) matched to the proxy's
by (type, ratio, seed). Spearman rho with a bootstrap CI over pairs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path



def damage(path: str) -> dict[tuple, float]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    base = {r["seed"]: r["holdout_ppl"] for r in rows if r["ratio"] == 0}
    return {(r["type"], r["ratio"], r["seed"]): r["holdout_ppl"] / base[r["seed"]] - 1
            for r in rows if r["ratio"] > 0 and r["seed"] in base}


def verdict(proxy: dict, target: dict, rho_min=0.7, n_boot=10000, seed=0) -> dict:
    import numpy as np  # lazy: CI installs neither numpy nor scipy
    from scipy.stats import spearmanr

    keys = sorted(set(proxy) & set(target))
    x = np.array([proxy[k] for k in keys])
    y = np.array([target[k] for k in keys])
    rho = float(spearmanr(x, y)[0])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(keys), len(keys))
        r = spearmanr(x[i], y[i])[0]
        if not np.isnan(r):
            boots.append(r)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ok = rho >= rho_min and lo > 0
    pairs = [{"type": k[0], "ratio": k[1], "seed": k[2], "proxy_damage": float(a), "target_damage": float(b)}
             for k, a, b in zip(keys, x, y)]
    return {"n_pairs": len(keys), "rho": rho, "pairs": pairs, "ci": [float(lo), float(hi)], "rho_min": rho_min,
            "status": "passed" if ok else "failed"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proxy", default="reports/grid/results.jsonl")
    ap.add_argument("--target", default="reports/x2/results.jsonl")
    ap.add_argument("--out", default="reports/x2/gate_x2.json")
    a = ap.parse_args()
    v = verdict(damage(a.proxy), damage(a.target))
    Path(a.out).write_text(json.dumps(v, indent=1), encoding="utf-8")
    print(f"Gate X2 {v['status'].upper()}: rho {v['rho']:.2f} CI {v['ci'][0]:.2f}..{v['ci'][1]:.2f} on {v['n_pairs']} pairs -> {a.out}")


if __name__ == "__main__":
    main()
