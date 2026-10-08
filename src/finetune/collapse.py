"""Sprint-0 collapse spike: train on your own output, generation after generation.

Gen-0 is fine-tuned on human text. Its output becomes Gen-1's training data,
Gen-1's output becomes Gen-2's, and so on. Every generation starts from the same
base model, so the only thing that changes is the data. That isolates the
effect ORIGIN-T exists to predict: training data drifting from human to
model-made.

    python -m src.finetune.collapse --config configs/collapse_spike.yaml

Writes ``{output_dir}/metrics.json`` and ``reports/figures/collapse_spike.png``,
and prints whether GATE 0 passes.

``run_generations`` takes the trainer, generator and evaluator as functions, so
the loop is tested on CPU with stubs. ``main`` plugs in the real ones.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..evaluation.metrics import distinct_n, perplexity
from ..evaluation.tracking import track_run
from ..generation.model_based import Generate, hf_generator, make_synthetic
from ..ingestion.loaders import load_dataset
from ..ingestion.schema import Sample
from .lora import train_lora


def training_text(s: Sample) -> str:
    """The string a sample is fine-tuned on (shared by the spike and the grid)."""
    return f"{s.prompt}\n{s.response}".strip()


def run_generations(
    human: Sequence[Sample],
    *,
    generations: int,
    train: Callable[[list[str], int], Any],
    generator_for: Callable[[Any], Generate],
    evaluate: Callable[[Any], dict[str, float]],
    on_row: Callable[[dict[str, float]], None] = lambda row: None,
) -> list[dict[str, float]]:
    """Train Gen-0..``generations`` and return one metrics row per generation.

    ``train(texts, g)`` returns a model handle, ``generator_for(model)`` turns it
    into a ``generate`` callable, and ``evaluate(model)`` scores it (e.g. holdout
    perplexity). Data diversity is measured here on each generation's training set.
    ``on_row`` sees each row as soon as it exists, so a long run can be logged live.
    """
    data = list(human)
    rows: list[dict[str, float]] = []
    for g in range(generations + 1):
        texts = [training_text(s) for s in data]
        model = train(texts, g)
        row = {
            "generation": g,
            **evaluate(model),
            "distinct_1": distinct_n(texts, 1),
            "distinct_2": distinct_n(texts, 2),
        }
        rows.append(row)
        on_row(row)
        if g < generations:
            data = make_synthetic(data, generator_for(model))
    return rows


def gate_passes(rows: Sequence[dict[str, float]], min_increase: float) -> bool:
    """GATE 0: last-generation holdout perplexity is ``min_increase`` above Gen-0's."""
    first, last = rows[0]["holdout_ppl"], rows[-1]["holdout_ppl"]
    return last >= first * (1 + min_increase)


def _pick(samples: list[Sample], n: int, lo: int, hi: int, rng: random.Random) -> list[Sample]:
    fits = [s for s in samples if lo <= len(s.response.split()) <= hi]
    return rng.sample(fits, min(n, len(fits)))


def load_holdout(d: dict[str, Any]) -> list[str]:
    """The fixed hidden-holdout texts every run is scored against.

    Drawn with a constant seed, so the spike and the grid (and every seed in
    them) see the same texts and differ only in what they trained on.
    """
    picked = _pick(
        load_dataset(d["holdout"]), d["n_holdout"], d["min_words"], d["max_words"],
        random.Random("holdout"),
    )
    return [s.response for s in picked]


def _plot(runs: dict[int, list[dict[str, float]]], path: Path) -> None:
    """Mean across seeds as a line, the min-max range across seeds as a band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(runs.values())
    gens = [r["generation"] for r in rows[0]]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.5))
    panels = ((left, ("holdout_ppl",)), (right, ("distinct_1", "distinct_2")))
    for ax, keys in panels:
        for key in keys:
            per_gen = [[run[g][key] for run in rows] for g in range(len(gens))]
            ax.plot(gens, [sum(v) / len(v) for v in per_gen], marker="o", label=key)
            ax.fill_between(gens, [min(v) for v in per_gen], [max(v) for v in per_gen], alpha=0.2)
        ax.set_xticks(gens)
    left.set(title="Hidden-holdout perplexity", xlabel="generation", ylabel="perplexity")
    right.set(title="Training-data diversity", xlabel="generation", ylabel="unique share")
    right.legend()
    fig.suptitle(f"Collapse spike, {len(rows)} seed(s): mean line, min-max band", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Sprint-0 collapse spike.")
    parser.add_argument("--config", default="configs/collapse_spike.yaml")
    args = parser.parse_args(argv)

    import yaml
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()  # per-sample generation warnings drown the progress lines
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    d, t = cfg["data"], cfg["train"]
    kept = load_dataset(d["kept"])
    # One fixed holdout for every seed, so seeds differ only in training data
    # and training randomness, never in what they are scored against.
    holdout = load_holdout(d)

    def run_seed(seed: int) -> list[dict[str, float]]:
        print(f"seed {seed}", flush=True)
        rng = random.Random(seed)
        human = _pick(kept, d["n_train"], d["min_words"], d["max_words"], rng)
        models_dir = Path(cfg["models_dir"]) / f"seed_{seed}"

        def train(texts: list[str], g: int) -> Path:
            print(f"  gen {g}: training on {len(texts)} texts", flush=True)
            return train_lora(
                texts,
                models_dir / f"gen_{g}",
                base_model=cfg["base_model"],
                seed=seed * 100 + g,
                device=cfg["device"],
                **t,
            )

        def generator_for(model_dir: Path) -> Generate:
            return hf_generator(
                str(model_dir),
                max_new_tokens=cfg["generate"]["max_new_tokens"],
                batch_size=cfg["generate"]["batch_size"],
                seed=seed,
                device=cfg["device"],
            )

        def evaluate(model_dir: Path) -> dict[str, float]:
            ppl = perplexity(
                model_dir, holdout, max_len=t["max_len"], batch_size=t["batch_size"],
                device=cfg["device"],
            )
            print(f"    holdout perplexity {ppl:.2f}", flush=True)
            return {"holdout_ppl": ppl}

        params = {
            "seed": seed, "base_model": cfg["base_model"], "generations": cfg["generations"],
            "n_train": d["n_train"], "n_holdout": d["n_holdout"], **t,
        }
        with track_run(cfg.get("tracking"), f"spike-seed{seed}", params, {"kind": "spike"}) as run:
            return run_generations(
                human,
                generations=cfg["generations"],
                train=train,
                generator_for=generator_for,
                evaluate=evaluate,
                on_row=lambda row: run.log_metrics(
                    {k: v for k, v in row.items() if k != "generation"}, step=int(row["generation"])
                ),
            )

    runs = {seed: run_seed(seed) for seed in cfg["seeds"]}
    per_seed = {seed: gate_passes(rows, cfg["gate_min_ppl_increase"]) for seed, rows in runs.items()}
    passed = all(per_seed.values())

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(
            {"config": cfg, "runs": runs, "gate_0_per_seed": per_seed, "gate_0_passed": passed},
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot(runs, Path("reports/figures/collapse_spike.png"))
    print(f"GATE 0 {'PASSED' if passed else 'NOT PASSED'} on every seed: {per_seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
