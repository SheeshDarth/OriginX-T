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
from ..generation.model_based import Generate, hf_generator, make_synthetic
from ..ingestion.loaders import load_dataset
from ..ingestion.schema import Sample
from .lora import train_lora


def run_generations(
    human: Sequence[Sample],
    *,
    generations: int,
    train: Callable[[list[str], int], Any],
    generator_for: Callable[[Any], Generate],
    evaluate: Callable[[Any], dict[str, float]],
) -> list[dict[str, float]]:
    """Train Gen-0..``generations`` and return one metrics row per generation.

    ``train(texts, g)`` returns a model handle, ``generator_for(model)`` turns it
    into a ``generate`` callable, and ``evaluate(model)`` scores it (e.g. holdout
    perplexity). Data diversity is measured here on each generation's training set.
    """
    data = list(human)
    rows: list[dict[str, float]] = []
    for g in range(generations + 1):
        texts = [f"{s.prompt}\n{s.response}".strip() for s in data]
        model = train(texts, g)
        rows.append(
            {
                "generation": g,
                **evaluate(model),
                "distinct_1": distinct_n(texts, 1),
                "distinct_2": distinct_n(texts, 2),
            }
        )
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


def _plot(rows: Sequence[dict[str, float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gens = [r["generation"] for r in rows]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.5))
    left.plot(gens, [r["holdout_ppl"] for r in rows], marker="o")
    left.set(title="Hidden-holdout perplexity", xlabel="generation", ylabel="perplexity")
    for key in ("distinct_1", "distinct_2"):
        right.plot(gens, [r[key] for r in rows], marker="o", label=key)
    right.set(title="Training-data diversity", xlabel="generation", ylabel="unique share")
    right.legend()
    for ax in (left, right):
        ax.set_xticks(gens)
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
    cfg =yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    d, t = cfg["data"], cfg["train"]
    rng = random.Random(cfg["seed"])
    human = _pick(load_dataset(d["kept"]), d["n_train"], d["min_words"], d["max_words"], rng)
    holdout = [
        s.response
        for s in _pick(
            load_dataset(d["holdout"]), d["n_holdout"], d["min_words"], d["max_words"], rng
        )
    ]
    models_dir = Path(cfg["models_dir"])

    def train(texts: list[str], g: int) -> Path:
        print(f"gen {g}: training on {len(texts)} texts", flush=True)
        return train_lora(
            texts,
            models_dir / f"gen_{g}",
            base_model=cfg["base_model"],
            seed=cfg["seed"] + g,
            device=cfg["device"],
            **t,
        )

    def generator_for(model_dir: Path) -> Generate:
        return hf_generator(
            str(model_dir),
            max_new_tokens=cfg["generate"]["max_new_tokens"],
            seed=cfg["seed"],
            device=cfg["device"],
        )

    def evaluate(model_dir: Path) -> dict[str, float]:
        ppl = perplexity(
            model_dir, holdout, max_len=t["max_len"], batch_size=t["batch_size"],
            device=cfg["device"],
        )
        print(f"  holdout perplexity {ppl:.2f}", flush=True)
        return {"holdout_ppl": ppl}

    rows = run_generations(
        human,
        generations=cfg["generations"],
        train=train,
        generator_for=generator_for,
        evaluate=evaluate,
    )
    passed = gate_passes(rows, cfg["gate_min_ppl_increase"])

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps({"config": cfg, "generations": rows, "gate_0_passed": passed}, indent=2),
        encoding="utf-8",
    )
    _plot(rows, Path("reports/figures/collapse_spike.png"))
    print(f"GATE 0 {'PASSED' if passed else 'NOT PASSED'}: see {out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
