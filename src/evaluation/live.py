"""Live training signals (X3, HANDBOOK §10.4): trainlog/1 rows while a model trains.

    python -m src.evaluation.live --config configs/grid.yaml --cells synthetic_r1_s0 human_control_r1_s0
    python -m src.evaluation.live --config configs/grid.yaml --preset x3   # the X3 calibration subset

``TrainLogger`` watches a run, every ``every`` optimizer steps (and at step 0 and the last
step): it puts the model in eval mode, measures it on a fixed probe set (the grid's hidden
holdout texts), restores train mode, and appends one §7.6 row to
``reports/trainlogs/<run_id>.jsonl``. Rows are appended and flushed one by one, so
``GET /v1/runs/{run_id}/stream`` can tail the file while the run is going (append, not the
usual write-then-rename: a reader only ever sees whole lines).

Signals are the six GATE 3 ones, named as in ``whitebox.jsonl``'s headline:
- ``eff_rank_last``, ``part_ratio_last``, ``anisotropy_last``, ``dead_fraction_mean``:
  ``whitebox.model_signals`` on the probe set, without its weight SVDs.
- ``stable_rank_attn``: mean over blocks of the stable rank of the *effective* attention
  input matrix (frozen weight + LoRA delta; a LoRA layer's ``.weight`` is the frozen one).
- ``delta_rel_norm``: ``||W - W0|| / ||W0||`` over every block matrix, W0 the step-0 weights.
- ``holdout_loss``: token-weighted mean cross-entropy on the probe set (nats).

Measuring never changes the weights (no in-place LoRA merge) and draws no random numbers,
so a logged run trains exactly like an unlogged one. ``alarm`` is always ``false`` here:
the alarm itself is C14 (#37).

Plug-ins: ``train_lora(..., on_step=logger)`` for our loop, ``trainer_callback(logger)`` for a
Hugging Face ``Trainer``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional

from . import whitebox

SIGNALS = ("eff_rank_last", "part_ratio_last", "anisotropy_last", "dead_fraction_mean",
           "stable_rank_attn", "delta_rel_norm")

# (model, tokenizer, logger) -> {"holdout_loss": float, <each of SIGNALS>: float}
Measure = Callable[[Any, Any, "TrainLogger"], dict[str, Optional[float]]]


def _effective_weights(model: Any) -> dict[str, Any]:
    """Every block matrix as the model currently computes with it, keyed like ``block_weights``."""
    out = {}
    for i, block in enumerate(whitebox._blocks(model)):
        for kind, (part, name) in whitebox.WEIGHT_KINDS.items():
            layer = getattr(getattr(block, part), name)
            if hasattr(layer, "base_layer"):  # a PEFT LoRA layer: frozen weight + its delta
                w = layer.base_layer.weight
                for adapter in getattr(layer, "active_adapters", []):
                    if adapter in getattr(layer, "lora_A", {}):
                        w = w + layer.get_delta_weight(adapter)
            else:
                w = layer.weight
            out[f"{kind}.{i}"] = w.detach().float()
    return out


def measure(model: Any, tok: Any, logger: "TrainLogger") -> dict[str, Optional[float]]:
    """The holdout loss and the six signals, on the logger's probe texts."""
    import torch

    device = next(model.parameters()).device
    batches = []
    side, tok.padding_side = tok.padding_side, "right"  # whitebox assumes right padding
    try:
        for i in range(0, len(logger.probe), logger.batch_size):
            enc = tok(list(logger.probe[i : i + logger.batch_size]), return_tensors="pt", padding=True,
                      truncation=True, max_length=logger.max_len).to(device)
            batches.append((enc.input_ids, enc.attention_mask))
    finally:
        tok.padding_side = side

    nll, count = 0.0, 0
    with torch.no_grad():
        for ids, mask in batches:
            labels = ids.masked_fill(mask == 0, -100)
            n = int((labels[:, 1:] != -100).sum())
            nll += model(input_ids=ids, attention_mask=mask, labels=labels).loss.item() * n
            count += n
    head = whitebox.headline(whitebox.model_signals(model, batches, weight_signals=False))

    weights = _effective_weights(model)
    if logger._w0 is None:
        logger._w0 = {k: w.clone() for k, w in weights.items()}
    attn = [whitebox.stable_rank(torch.linalg.svdvals(w).tolist())
            for k, w in weights.items() if k.startswith("attn_qkv.")]
    moved = sum(float((w - logger._w0[k]).pow(2).sum()) for k, w in weights.items())
    base = sum(float(w0.pow(2).sum()) for w0 in logger._w0.values())
    return {
        "holdout_loss": nll / count if count else None,
        **{k: head.get(k) for k in SIGNALS[:4]},
        "stable_rank_attn": sum(attn) / len(attn),
        "delta_rel_norm": math.sqrt(moved / base),
    }


class TrainLogger:
    """Writes a trainlog/1 row every ``every`` steps; pass it as ``train_lora(on_step=...)``."""

    def __init__(self, path: str | Path, run_id: str, probe: Sequence[str], *, every: int = 10,
                 batch_size: int = 16, max_len: int = 256, measure: Measure = measure,
                 on_row: Callable[[dict[str, Any]], None] = lambda row: None) -> None:
        if every < 1:
            raise ValueError("every must be >= 1")
        if not probe:
            raise ValueError("the probe set is empty")
        self.path, self.run_id, self.probe = Path(path), run_id, list(probe)
        self.every, self.batch_size, self.max_len = every, batch_size, max_len
        self._measure, self._on_row = measure, on_row
        self._w0: Optional[dict[str, Any]] = None
        self.rows: list[dict[str, Any]] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")  # a new run starts a new log

    def __call__(self, step: int, model: Any, tok: Any, last: bool = False) -> None:
        if step % self.every and not last:
            return
        if self.rows and self.rows[-1]["step"] == step:  # the last step may also be a multiple
            return
        was_training = getattr(model, "training", False)
        model.eval()
        try:
            values = self._measure(model, tok, self)
        finally:
            if was_training:
                model.train()
        row = {"schema": "trainlog/1", "run_id": self.run_id, "step": step,
               "holdout_loss": values.get("holdout_loss"),
               **{f"sig_{k}": values.get(k) for k in SIGNALS}, "alarm": False, "alarm_signal": None}
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
        self.rows.append(row)
        self._on_row(row)


def trainer_callback(logger: TrainLogger) -> Any:
    """A ``transformers.TrainerCallback`` that drives ``logger`` from a Hugging Face Trainer."""
    from transformers import TrainerCallback

    class _Callback(TrainerCallback):
        def _tok(self, kwargs: dict[str, Any]) -> Any:
            return kwargs.get("processing_class") or kwargs.get("tokenizer")

        def on_train_begin(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
            logger(0, model, self._tok(kwargs))

        def on_step_end(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
            logger(state.global_step, model, self._tok(kwargs), state.global_step >= state.max_steps)

    return _Callback()


# Healthy and collapsing runs for X3 (#37): every type at full dose, plus the clean baseline.
PRESETS = {"x3": [f"{t}_r1_s{s}" for s in (0, 1, 2)
                  for t in ("human_control", "synthetic", "recursive", "paraphrased", "benchmark_near")]
                 + [f"synthetic_r0_s{s}" for s in (0, 1, 2)]}


def main(argv: Optional[list[str]] = None) -> None:
    import argparse
    import shutil

    import yaml

    from ..finetune.collapse import load_holdout, training_text
    from ..finetune.grid import Cell, make_cells, training_set
    from ..finetune.lora import train_lora
    from ..ingestion.loaders import load_dataset

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/grid.yaml")
    p.add_argument("--dataset", default="wikitext2")
    p.add_argument("--cells", nargs="*", default=[], help="grid cell ids, e.g. synthetic_r1_s0")
    p.add_argument("--preset", choices=sorted(PRESETS))
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--out-dir", default="reports/trainlogs")
    p.add_argument("--models-dir", default="data/models/live", help="scratch; deleted after each run")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    wanted = list(dict.fromkeys(args.cells + PRESETS.get(args.preset or "", [])))
    cells = {c.id: c for c in make_cells(cfg["types"], cfg["ratios"], cfg["seeds"])}
    unknown = [c for c in wanted if c not in cells]
    if unknown or not wanted:
        raise SystemExit(f"unknown or no cells: {unknown or wanted}; choose from this grid's cells")
    kept, probe, t = load_dataset(cfg["data"]["kept"]), load_holdout(cfg["data"]), cfg["train"]
    for cell_id in wanted:
        cell: Cell = cells[cell_id]
        run_id = f"{args.dataset}_{cell_id}"
        logger = TrainLogger(Path(args.out_dir) / f"{run_id}.jsonl", run_id, probe, every=args.every,
                             batch_size=t["batch_size"], max_len=t["max_len"],
                             on_row=lambda r: print(f"  {r['run_id']} step {r['step']:>4}  "
                                                    f"loss {r['holdout_loss']:.4f}", flush=True))
        texts = [training_text(s) for s in training_set(cfg, cell, kept=kept)]
        out = Path(args.models_dir) / run_id
        # the grid's own seed for this cell, so the logged run is the grid's run
        train_lora(texts, out, base_model=cfg["base_model"], seed=cell.seed * 100, device=cfg["device"],
                   on_step=logger, **t)
        shutil.rmtree(out, ignore_errors=True)
        last = logger.rows[-1]
        print(f"{run_id}: {len(logger.rows)} rows, final holdout ppl {math.exp(last['holdout_loss']):.2f}",
              flush=True)


if __name__ == "__main__":
    main()
