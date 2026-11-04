"""Per-sample harm scores for X1 (HANDBOOK §9.3, B7): ``harm_lora_grad`` and ``harm_repr_sim``.

    python -m src.mitigation.harm --config configs/grid.yaml
    python -m src.mitigation.harm --config configs/grid.yaml --cells synthetic_r0.5_s0 recursive_r0.5_s0

Each training sample of a cell is scored at **that cell's trained checkpoint**
(``models_dir/<cell>``, kept by the grid; copy it over, §12.3) by how much it pulls the
model away from held-out human text. Both scores are negated cosines, so higher is worse
and both lie in [-1, 1]:

- ``harm_lora_grad`` (TracIn-style, one checkpoint): minus the cosine between the
  sample's loss gradient and the gradient of the mean held-out loss. A gradient step on a
  sample changes the held-out loss by ``-lr * <g_sample, g_target>``, so a sample whose
  gradient points against the target's *raises* held-out loss: harmful. Cosine rather
  than the raw dot product, because short texts have larger gradients and would otherwise
  dominate (LESS, Xia et al. 2024). Gradients are taken with respect to a fresh LoRA
  adapter on the merged checkpoint (rank ``train.lora_r``, as the grid trains). Its B
  matrices start at zero, so only their gradients are non-zero: each is the full weight
  gradient projected through the adapter's random A, a fixed random projection shared by
  every sample. That keeps the vectors small (about 110k numbers for DistilGPT-2) and the
  scoring cheap.
- ``harm_repr_sim``: the representation-similarity baseline that beat influence
  functions at flagging harmful fine-tuning data. Minus the cosine between the sample's
  mean-pooled last hidden state and the centroid of the held-out texts', after removing
  the mean of all embeddings (a causal LM's hidden states share a large common direction
  that would otherwise put every cosine near 1).

**The target is not the evaluation set.** The grid scores every model on 200 texts drawn
from the hidden holdout (``collapse.load_holdout``). Selecting data by its effect on
those same texts, then reporting recovery on them, would tune the repair to its own test
set. The target is a disjoint draw of ``n_holdout`` texts from the rest of the hidden
holdout, never trained on. ``true_source`` is copied into each row for evaluation only;
no score reads it.

Rows go to ``reports/scores/<method>/<dataset>.jsonl`` (``--scores-dir``) (scores/1, §7.3), replacing the
scored cells' earlier rows. AUROC against ``true_source != "human"`` for these methods,
and for Track C's ``risk_*`` scores on the same cells, goes to
``reports/scores/<dataset>_harm_eval.json``. torch, transformers and peft are imported
only inside the functions that need them.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

METHODS = ("harm_lora_grad", "harm_repr_sim")
# The X1 pilot (B8): one seed, ratio 0.5, the two collapse types that degrade most.
PILOT_CELLS = ("synthetic_r0.5_s0", "recursive_r0.5_s0")


def neg_cosine(vectors: Sequence[Sequence[float]], target: Sequence[float]) -> list[Optional[float]]:
    """``-cos(v, target)`` for each vector; None for a zero vector (no direction to compare)."""
    t_norm = math.sqrt(sum(x * x for x in target))
    out: list[Optional[float]] = []
    for v in vectors:
        v_norm = math.sqrt(sum(x * x for x in v))
        if not v_norm or not t_norm:
            out.append(None)
            continue
        out.append(round(-sum(a * b for a, b in zip(v, target)) / (v_norm * t_norm), 6))
    return out


def target_texts(d: Mapping[str, Any]) -> list[str]:
    """Held-out texts for the harm target: from the hidden holdout, disjoint from the evaluation texts."""
    from ..finetune.collapse import _pick, load_holdout
    from ..ingestion.loaders import load_dataset

    evaluated = set(load_holdout(d))
    rest = [s for s in load_dataset(d["holdout"]) if s.response not in evaluated]
    return [s.response for s in _pick(rest, d["n_holdout"], d["min_words"], d["max_words"],
                                      random.Random("harm-target"))]


def _load(model_dir: str | Path, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    return model, tok


def lora_grad_harm(
    model_dir: str | Path, texts: Sequence[str], targets: Sequence[str], *,
    lora_r: int = 8, max_len: int = 256, seed: int = 0, device: str = "cpu",
) -> list[Optional[float]]:
    """``harm_lora_grad`` for each of ``texts`` at the checkpoint in ``model_dir`` (see the module doc)."""
    import torch
    from peft import LoraConfig, get_peft_model

    model, tok = _load(model_dir, device)
    torch.manual_seed(seed)  # the adapter's A: the random projection, the same for every sample
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, task_type="CAUSAL_LM"))
    model.eval()  # no dropout: a sample's gradient must not depend on the order it is scored in
    for n, p in model.named_parameters():
        p.requires_grad_("lora_B" in n)
    params = [p for p in model.parameters() if p.requires_grad]

    def grad(text: str) -> "torch.Tensor":
        batch = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        model.zero_grad(set_to_none=True)
        model(**batch, labels=batch.input_ids).loss.backward()
        return torch.cat([p.grad.reshape(-1) for p in params]).double()

    target = sum(grad(t) for t in targets) / len(targets)  # gradient of the mean per-text held-out loss
    t_norm = target.norm()
    out: list[Optional[float]] = []
    for text in texts:
        g = grad(text)
        g_norm = g.norm()
        out.append(None if not g_norm or not t_norm else round(-float(g @ target / (g_norm * t_norm)), 6))
    return out


def repr_sim_harm(
    model_dir: str | Path, texts: Sequence[str], targets: Sequence[str], *,
    max_len: int = 256, batch_size: int = 16, device: str = "cpu",
) -> list[Optional[float]]:
    """``harm_repr_sim`` for each of ``texts`` at the checkpoint in ``model_dir`` (see the module doc)."""
    import torch

    model, tok = _load(model_dir, device)
    model.eval()

    @torch.no_grad()
    def embed(batch_texts: Sequence[str]) -> "torch.Tensor":
        rows = []
        for i in range(0, len(batch_texts), batch_size):
            batch = tok(list(batch_texts[i:i + batch_size]), return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len).to(device)
            hidden = model(**batch, output_hidden_states=True).hidden_states[-1]
            mask = batch.attention_mask.unsqueeze(-1).to(hidden.dtype)
            rows.append(((hidden * mask).sum(1) / mask.sum(1)).double().cpu())
        return torch.cat(rows)

    train, held = embed(texts), embed(targets)
    mean = torch.cat([train, held]).mean(0)
    return neg_cosine((train - mean).tolist(), (held - mean).mean(0).tolist())


def score_cell(
    dataset: str, cell: str, samples: Sequence[Any], texts: Sequence[str], targets: Sequence[str],
    model_dir: str | Path, *, train: Mapping[str, Any], device: str,
    lora_grad=lora_grad_harm, repr_sim=repr_sim_harm,
) -> dict[str, list[dict[str, Any]]]:
    """``{method: scores/1 rows}`` for one cell; ``samples[i]`` is the sample behind ``texts[i]``."""
    values = {
        "harm_lora_grad": lora_grad(model_dir, texts, targets, lora_r=train["lora_r"],
                                    max_len=train["max_len"], device=device),
        "harm_repr_sim": repr_sim(model_dir, texts, targets, max_len=train["max_len"],
                                  batch_size=train["batch_size"], device=device),
    }
    return {
        method: [{"schema": "scores/1", "dataset": dataset, "cell": cell, "sample_id": s.sample_id,
                  "method": method, "score": v, "higher_is": "worse", "true_source": s.source}
                 for s, v in zip(samples, vals)]
        for method, vals in values.items()
    }


def merge_rows(old: Sequence[Mapping[str, Any]], new: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """``old`` without the cells in ``new``, then ``new``: re-scoring a cell replaces its rows."""
    cells = {r["cell"] for r in new}
    return [r for r in old if r["cell"] not in cells] + list(new)


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    import yaml

    from ..features.scores import _write, evaluate
    from ..features.suite import _device
    from ..finetune.collapse import training_text
    from ..finetune.grid import make_cells, training_set
    from ..ingestion.loaders import load_dataset

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/grid.yaml", help="the grid config the cells were trained under")
    p.add_argument("--dataset", default="wikitext2")
    p.add_argument("--cells", nargs="+", default=list(PILOT_CELLS), help="cell ids (default: the X1 pilot)")
    p.add_argument("--device", default="auto")
    p.add_argument("--scores-dir", default="reports/scores",
                   help="where scores/1 rows go (repair.yaml's scores_dir reads them)")
    args = p.parse_args(argv)

    grid = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    by_id = {c.id: c for c in make_cells(grid["types"], grid["ratios"], grid["seeds"])}
    unknown = [c for c in args.cells if c not in by_id]
    if unknown:
        raise SystemExit(f"{unknown} are not cells of {args.config}")
    missing = [c for c in args.cells if not (Path(grid["models_dir"]) / c).exists()]
    if missing:
        raise SystemExit(f"no checkpoint for {missing} under {grid['models_dir']}. Copy the grid's "
                         "checkpoints from the machine that ran it (HANDBOOK §12.3).")
    device = _device(args.device)
    kept = load_dataset(grid["data"]["kept"])
    targets = target_texts(grid["data"])
    new: dict[str, list[dict[str, Any]]] = {m: [] for m in METHODS}
    for i, cid in enumerate(args.cells, 1):
        samples = training_set(grid, by_id[cid], kept=kept)
        rows = score_cell(args.dataset, cid, samples, [training_text(s) for s in samples], targets,
                          Path(grid["models_dir"]) / cid, train=grid["train"], device=device)
        for method, r in rows.items():
            new[method] += r
        print(f"  [{i}/{len(args.cells)}] {cid}", flush=True)

    root = Path(args.scores_dir)
    for method, rows in new.items():
        path = root / method / f"{args.dataset}.jsonl"
        _write(path, merge_rows(_read(path), rows))
    scored = set(args.cells)
    report: dict[str, Any] = {
        m: evaluate([r for r in _read(root / m / f"{args.dataset}.jsonl") if r["cell"] in scored])
        for m in (*METHODS, "risk_reflm", "risk_knn") if (root / m / f"{args.dataset}.jsonl").exists()
    }
    report["cells"] = sorted(scored)
    (root / f"{args.dataset}_harm_eval.json").write_text(json.dumps(report, indent=2) + "\n",
                                                         encoding="utf-8", newline="\n")
    for m, r in report.items():
        if m != "cells" and r["pooled"]:
            print(f"{m}: AUROC {r['pooled']['auroc']:.3f} [{r['pooled']['ci'][0]:.3f}, {r['pooled']['ci'][1]:.3f}]")


if __name__ == "__main__":
    main()
