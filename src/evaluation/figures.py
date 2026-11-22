"""Paper figures for Track C, drawn only from committed reports (no recomputation).

    python -m src.evaluation.figures            # writes the three PNGs to reports/figures/

- ``x3_alarm.png``: anisotropy trajectories against the healthy band, the alarm and breach step
  of every collapsing run, and the lead-time histogram (`reports/x3/`, `reports/trainlogs/`).
- ``x2_scatter.png``: proxy damage against target damage with Spearman rho (`reports/x2/gate_x2.json`).
- ``gate3_tinystories.png``: AUROC with its interval for each white-box signal on the confirmatory
  GATE 3 run (`reports/grid_tinystories/earlywarning.json`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TYPE_COLORS = {"synthetic": "#2a78d6", "recursive": "#eb6834", "paraphrased": "#1baf7a", "benchmark_near": "#eda100",
               "human_control": "#8a8a8a"}
ACCENT = "#1f6bbf"
SIGNAL = "anisotropy_last"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100})
    return plt


def _change(run_id: str, root: Path) -> tuple[list[int], list[float]]:
    """The signal's change since step 0, at every logged step after 0."""
    rows = _rows(root / f"reports/trainlogs/wikitext2_{run_id}.jsonl")
    base = rows[0][f"sig_{SIGNAL}"]
    kept = [r for r in rows if r["step"] > 0]
    return [r["step"] for r in kept], [r[f"sig_{SIGNAL}"] - base for r in kept]


def x3_alarm(root: Path = Path(".")):
    """Left: the signal against the healthy band. Right: how many steps the alarm led the breach."""
    import yaml

    plt = _plt()
    gate = _json(root / "reports/x3/gate_x3.json")
    mon = _json(root / "reports/x3/monitor.json")
    cfg = yaml.safe_load((root / "configs/x3.yaml").read_text(encoding="utf-8"))
    centre, scale, thr = mon["center"][SIGNAL], mon["scale"][SIGNAL], mon["threshold"]
    steps = sorted(int(s) for s in centre)
    lo = [centre[str(s)] - thr * scale[str(s)] for s in steps]
    hi = [centre[str(s)] + thr * scale[str(s)] for s in steps]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [2.2, 1]})
    ax.fill_between(steps, lo, hi, color="#d9d9d9", alpha=0.7, label=f"healthy band (alarm at {thr:.1f} spreads)")
    for i, run in enumerate(cfg["runs"]["healthy_test"]):
        x, y = _change(run, root)
        ax.plot(x, y, color="#8a8a8a", lw=0.8, alpha=0.7, label="healthy runs (held out)" if i == 0 else None)
    seen = set()
    for r in gate["runs"]:
        kind = r["run_id"].split("_")[1]
        x, y = _change(r["run_id"].removeprefix("wikitext2_"), root)
        ax.plot(x, y, color=TYPE_COLORS[kind], lw=1.4, alpha=0.9, label=kind if kind not in seen else None)
        seen.add(kind)
        ax.scatter([r["alarm_step"]], [y[x.index(r["alarm_step"])]], marker="o", s=22, color="black", zorder=5)
        ax.scatter([r["breach_step"]], [y[x.index(r["breach_step"])]], marker="x", s=26, color="black", zorder=5)
    ax.scatter([], [], marker="o", s=22, color="black", label="alarm step")
    ax.scatter([], [], marker="x", s=26, color="black", label="loss breach step")
    ax.set(xlabel="training step", ylabel="change in anisotropy since step 0",
           title="Contaminated runs drift out of the healthy band")
    ax.legend(fontsize=8, loc="lower left", frameon=False)

    leads = [r["lead_steps"] for r in gate["runs"]]
    top = max(leads)
    counts = {v: leads.count(v) for v in range(0, top + 1, 10)}
    bx.bar([str(v) for v in counts], counts.values(), color=[("#8a8a8a" if v == 0 else ACCENT) for v in counts])
    bx.set(xlabel="steps the alarm came before the breach", ylabel="collapsing runs",
           title=f"{gate['early_alarm_rate'] * 100:.0f}% alarmed early (bar 50%)")
    bx.grid(axis="x", visible=False)
    fig.text(0.99, 0.01, f"one signal shown ({SIGNAL}); the alarm uses all six. False alarms: "
             f"{gate['observed_false_alarm_rate'] * 100:.0f}% of {gate['n']['healthy_test']} held-out healthy runs.",
             ha="right", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def x2_scatter(root: Path = Path(".")):
    plt = _plt()
    g = _json(root / "reports/x2/gate_x2.json")
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for p in g["pairs"]:
        ax.scatter(p["proxy_damage"] * 100, p["target_damage"] * 100, color=TYPE_COLORS[p["type"]],
                   marker="o" if p["ratio"] == 0.5 else "^", s=48, alpha=0.9, edgecolor="white", linewidth=0.6)
    for t, c in TYPE_COLORS.items():
        if any(p["type"] == t for p in g["pairs"]):
            ax.scatter([], [], color=c, label=t.replace("_", " "))
    ax.scatter([], [], color="#444", marker="o", label="ratio 0.5")
    ax.scatter([], [], color="#444", marker="^", label="ratio 1")
    lo, hi = g["ci"]
    ax.set(xlabel="proxy damage: DistilGPT-2 (% rise in held-out perplexity)",
           ylabel="target damage: Pythia-410M (% rise)",
           title=f"Spearman ρ = {g['rho']:.2f}, 95% CI [{lo:.2f}, {hi:.2f}], {g['n_pairs']} matched runs")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.text(0.99, 0.01, "damage = holdout ppl / the same seed's ratio-0 ppl − 1", ha="right", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def gate3_tinystories(root: Path = Path(".")):
    plt = _plt()
    e = _json(root / "reports/grid_tinystories/earlywarning.json")
    names = list(e["signals"])
    vals = [e["signals"][n] for n in names]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    y = list(range(len(names)))[::-1]
    for yi, v in zip(y, vals):
        lo, hi = v["auroc_ci"]
        ax.plot([lo, hi], [yi, yi], color="#8a8a8a", lw=2)
        ax.scatter([v["auroc"]], [yi], color=ACCENT, zorder=3)
        ax.text(1.02, yi, f"p = {v['permutation_p']:.2f}", va="center", fontsize=8, color="#555")
    ax.axvline(0.5, color="#bbb", ls="--", lw=1)
    ax.set(yticks=y, yticklabels=[n.replace("_", " ") for n in names], xlim=(0, 1.15), xlabel="AUROC (0.5 = chance)",
           title=f"Confirmatory GATE 3 on TinyStories: {e['status']}\n"
                 f"{e['trajectories']} trajectories, {e['never_collapse']} never collapse")
    ax.grid(axis="y", visible=False)
    fig.text(0.99, 0.01, f"bar: permutation p < {e['alpha_adjusted']:.4f} (Bonferroni, 6 signals)",
             ha="right", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


FIGURES = {"x3_alarm": x3_alarm, "x2_scatter": x2_scatter, "gate3_tinystories": gate3_tinystories}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".")
    p.add_argument("--out", default="reports/figures")
    a = p.parse_args(argv)
    root, out = Path(a.root), Path(a.root) / a.out
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        fig = fn(root)
        fig.savefig(out / f"{name}.png", dpi=200)
        print(f"wrote {out / f'{name}.png'}")


if __name__ == "__main__":
    main()
