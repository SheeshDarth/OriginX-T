"""White-box representational signals, measured on the checkpoints the grid already made.

TRD section 6: effective rank / participation ratio, weight stable rank, embedding
anisotropy and neuron-death rate, "computed from checkpoints already produced in the
fine-tuning grid". Nothing here trains. For every checkpoint it runs the fixed
hidden-holdout texts through the model once and reads its weights:

- **effective rank** (Roy & Vetterli, 2007) and **participation ratio** of each layer's
  token representations: how many directions the model actually uses. Collapse should
  shrink both. Taken on the centred covariance spectrum, singular values s = sqrt(lambda).
- **anisotropy** (Ethayarajh, 2019): the mean cosine similarity between two different
  tokens' representations, over *all* pairs, exactly (it needs only the sum of the unit
  vectors, so nothing is sampled). 0 is isotropic, 1 is every token pointing one way.
- **neuron-death rate**: the share of each MLP's hidden units whose activation never
  exceeds ``eps`` on any token.
- **weight stable rank** ``||W||_F^2 / ||W||_2^2`` and effective rank of the attention and
  MLP matrices, and, given the base model, how far training moved them
  (``delta_rel_norm = ||W - W0||_F / ||W0||_F``).

GPT-2's first position is an outlier (a huge-norm "attention sink" state). Left in, it
dominates every covariance and anisotropy figure, so by default it is excluded
(``exclude_first_token``); it applies to the activations as well as the representations.

    python -m src.evaluation.whitebox --config configs/whitebox.yaml
    python -m src.evaluation.whitebox --config configs/whitebox.yaml --summarize-only

Layout, so most of it is testable without an ML stack:

1. ``effective_rank`` / ``participation_ratio`` / ``stable_rank``: formulas on a list of
   singular values or eigenvalues, standard library only.
2. ``RepresentationStats`` / ``ActivationMax`` / ``weight_stats``: the streaming and
   linear-algebra pieces (numpy, imported inside).
3. ``model_signals`` / ``compute_checkpoint``: runs a GPT-2-style model (torch and
   transformers, imported inside).
4. ``run_whitebox`` and ``main``: which checkpoints to measure (from the grid's
   ``results.jsonl``), resume, staleness, summary. They take ``compute`` as a function,
   so they are tested with stubs.

Rows are appended to ``whitebox.jsonl`` one checkpoint at a time, so a cut-short run
resumes. A row records the grid row's ``holdout_ppl`` it was measured for: if a cell is
re-run (``--rerun-types``), its checkpoint changes, that number no longer matches, and the
cell is measured again. Shared ratio-0 baselines have one checkpoint for all types and are
copied, not re-measured. The recursion's intermediate checkpoints
(``pool_recursive_s<seed>_gen<g>``, kept with ``keep_models``) are measured too, as the
recursion stages of the trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

# --- 1. formulas ------------------------------------------------------------------------------------


def _non_negative(values: Sequence[float], what: str) -> list[float]:
    vals = [float(v) for v in values]
    if any(v < 0 or math.isnan(v) for v in vals):
        raise ValueError(f"{what} must be non-negative numbers, got {vals[:5]}...")
    if not any(v > 0 for v in vals):
        raise ValueError(f"{what} must not be empty or all zero")
    return vals


def effective_rank(singular_values: Sequence[float]) -> float:
    """exp of the Shannon entropy of ``s / sum(s)`` (Roy & Vetterli, 2007).

    k equal singular values give k, one non-zero value gives 1, and the result does not
    depend on the scale of the matrix.
    """
    vals = _non_negative(singular_values, "singular values")
    total = sum(vals)
    entropy = -sum(p * math.log(p) for p in (v / total for v in vals) if p > 0)
    return math.exp(entropy)


def participation_ratio(eigenvalues: Sequence[float]) -> float:
    """``(sum lambda)^2 / sum lambda^2``: the number of directions sharing the variance.

    k equal eigenvalues give k; one dominant eigenvalue gives about 1.
    """
    vals = _non_negative(eigenvalues, "eigenvalues")
    return sum(vals) ** 2 / sum(v * v for v in vals)


def stable_rank(singular_values: Sequence[float]) -> float:
    """``sum s^2 / max(s)^2``: a noise-robust rank (``||W||_F^2 / ||W||_2^2``).

    The identity matrix of size n gives n; any rank-1 matrix gives 1.
    """
    vals = _non_negative(singular_values, "singular values")
    return sum(v * v for v in vals) / max(vals) ** 2


# --- 2. streaming statistics and weights (numpy) ----------------------------------------------------


class RepresentationStats:
    """Spectrum and anisotropy of token representations, accumulated batch by batch.

    Only running sums are kept (``n``, ``sum x``, ``sum x x^T``, ``sum x/|x|``), so the
    token matrix is never stored. Everything is float64.
    """

    def __init__(self, dim: int) -> None:
        import numpy as np

        self.dim = dim
        self.n = 0
        self._sum = np.zeros(dim)
        self._outer = np.zeros((dim, dim))
        self._unit_sum = np.zeros(dim)
        self._n_unit = 0

    def update(self, x: Any) -> None:
        """Add a ``(tokens, dim)`` block of representations."""
        import numpy as np

        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"expected shape (n, {self.dim}), got {x.shape}")
        self.n += len(x)
        self._sum += x.sum(axis=0)
        self._outer += x.T @ x
        norms = np.linalg.norm(x, axis=1)
        nonzero = norms > 0
        self._unit_sum += (x[nonzero] / norms[nonzero, None]).sum(axis=0)
        self._n_unit += int(nonzero.sum())

    def eigenvalues(self) -> Any:
        """Eigenvalues of the (biased) covariance, largest first, negatives from rounding clipped."""
        import numpy as np

        if self.n < 2:
            raise ValueError(f"need at least 2 tokens, have {self.n}")
        mean = self._sum / self.n
        cov = self._outer / self.n - np.outer(mean, mean)
        return np.clip(np.linalg.eigvalsh((cov + cov.T) / 2)[::-1], 0.0, None)

    def result(self) -> dict[str, Any]:
        import numpy as np

        lam = self.eigenvalues()
        n = self._n_unit
        # sum over pairs i != j of u_i . u_j  =  |sum u|^2 - n
        anisotropy = (float(self._unit_sum @ self._unit_sum) - n) / (n * (n - 1)) if n >= 2 else 0.0
        # The limits of collapse: no variance (every token the same vector) or no non-zero
        # vectors at all. There are no directions in use, so rank 0 (the formulas are
        # undefined there) and anisotropy 0 where there is nothing to compare. Flagged, so a
        # degenerate layer is not read as an isotropic one.
        no_variance = not lam.any()
        return {
            "n_tokens": self.n,
            "effective_rank": 0.0 if no_variance else effective_rank(np.sqrt(lam)),
            "participation_ratio": 0.0 if no_variance else participation_ratio(lam),
            "anisotropy": anisotropy,
            "degenerate": bool(no_variance or n < 2),
        }


class ActivationMax:
    """The largest activation each unit ever reaches, for the neuron-death rate."""

    def __init__(self, units: int) -> None:
        import numpy as np

        self.max = np.full(units, -np.inf)
        self.n = 0

    def update(self, acts: Any) -> None:
        """Add a ``(tokens, units)`` block."""
        import numpy as np

        acts = np.asarray(acts)
        if acts.ndim != 2 or acts.shape[1] != len(self.max):
            raise ValueError(f"expected shape (n, {len(self.max)}), got {acts.shape}")
        if len(acts):
            self.max = np.maximum(self.max, acts.max(axis=0))
            self.n += len(acts)

    def dead_fraction(self, eps: float) -> float:
        """Share of units that never exceeded ``eps`` on any token."""
        if self.n == 0:
            raise ValueError("no tokens seen")
        return float((self.max <= eps).mean())


def weight_stats(w: Any) -> dict[str, float]:
    """Stable rank, effective rank and norms of one weight matrix.

    An all-zero matrix has rank 0 (the formulas are undefined there), not an error: a
    fully collapsed checkpoint must still be measurable.
    """
    import numpy as np

    s = np.linalg.svd(np.asarray(w, dtype=np.float64), compute_uv=False)
    zero = not s.any()
    return {
        "stable_rank": 0.0 if zero else stable_rank(s),
        "effective_rank": 0.0 if zero else effective_rank(s),
        "frobenius": float(np.sqrt((s**2).sum())),
        "spectral": float(s[0]),
    }


def delta_norms(w: Any, w0: Any) -> tuple[float, float, Optional[float]]:
    """``(||W - W0||_F, ||W0||_F, effective rank of W - W0 or None if it is zero)``."""
    import numpy as np

    d = np.asarray(w, dtype=np.float64) - np.asarray(w0, dtype=np.float64)
    fro = float(np.linalg.norm(d))
    rank = None if fro == 0.0 else effective_rank(np.linalg.svd(d, compute_uv=False))
    return fro, float(np.linalg.norm(np.asarray(w0, dtype=np.float64))), rank


# --- 3. a GPT-2-style model (torch, transformers) ---------------------------------------------------

# the four matrices of a GPT-2 block, by the name used in the signals
WEIGHT_KINDS = {
    "attn_qkv": ("attn", "c_attn"),
    "attn_out": ("attn", "c_proj"),
    "mlp_in": ("mlp", "c_fc"),
    "mlp_out": ("mlp", "c_proj"),
}


def _blocks(model: Any) -> Any:
    inner = getattr(model, "transformer", model)
    blocks = getattr(inner, "h", None)
    if blocks is None:
        raise ValueError("white-box signals support GPT-2-style models (transformer.h blocks)")
    return blocks


def block_weights(model: Any) -> dict[str, Any]:
    """Every block's four matrices as numpy arrays, keyed ``"<kind>.<block>"``."""
    out = {}
    for i, block in enumerate(_blocks(model)):
        for kind, (part, name) in WEIGHT_KINDS.items():
            out[f"{kind}.{i}"] = getattr(getattr(block, part), name).weight.detach().cpu().double().numpy()
    return out


def model_signals(
    model: Any,
    batches: Sequence[tuple[Any, Any]],
    *,
    eps: float = 0.01,
    exclude_first_token: bool = True,
    base_weights: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """All signals for one model, from ``batches`` of ``(input_ids, attention_mask)`` tensors.

    The model must already be in eval mode on the device the tensors are on. Padding is
    never counted: tokens are selected by the attention mask, so the result does not
    depend on how the texts were batched.
    """
    import numpy as np
    import torch

    blocks = list(_blocks(model))
    layer_stats: list[RepresentationStats] = []
    act_max: list[ActivationMax] = []
    state: dict[str, Any] = {"valid": None}
    hooks = []

    def make_hook(i: int, mlp: Any):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            acts = mlp.act(output)  # the unit activations, whatever form the activation takes
            keep = state["valid"]
            while len(act_max) <= i:
                act_max.append(ActivationMax(acts.shape[-1]))
            act_max[i].update(acts[keep].detach().float().cpu().numpy())

        return hook

    for i, block in enumerate(blocks):
        hooks.append(block.mlp.c_fc.register_forward_hook(make_hook(i, block.mlp)))
    try:
        with torch.no_grad():
            for input_ids, attention_mask in batches:
                valid = attention_mask.bool().clone()
                if exclude_first_token:
                    valid[:, 0] = False
                state["valid"] = valid
                out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                for layer, hidden in enumerate(out.hidden_states):
                    while len(layer_stats) <= layer:
                        layer_stats.append(RepresentationStats(hidden.shape[-1]))
                    layer_stats[layer].update(hidden[valid].detach().double().cpu().numpy())
    finally:
        for h in hooks:
            h.remove()

    layers = [{"layer": i, **s.result()} for i, s in enumerate(layer_stats)]
    mlp = [{"block": i, "dead_fraction": a.dead_fraction(eps)} for i, a in enumerate(act_max)]

    weights = block_weights(model)
    per_kind: dict[str, dict[str, float]] = {}
    for kind in WEIGHT_KINDS:
        stats = [weight_stats(w) for key, w in weights.items() if key.startswith(kind + ".")]
        per_kind[kind] = {
            "stable_rank": float(np.mean([s["stable_rank"] for s in stats])),
            "effective_rank": float(np.mean([s["effective_rank"] for s in stats])),
        }

    signals: dict[str, Any] = {
        "n_tokens": layers[-1]["n_tokens"], "layers": layers, "mlp": mlp, "weights": per_kind,
    }
    if base_weights is not None:
        moved = norm0 = 0.0
        ranks = []
        for key, w in weights.items():
            fro, fro0, rank = delta_norms(w, base_weights[key])
            moved += fro**2
            norm0 += fro0**2
            if rank is not None:
                ranks.append(rank)
        signals["delta"] = {
            "rel_norm": math.sqrt(moved) / math.sqrt(norm0),
            "effective_rank": float(np.mean(ranks)) if ranks else None,
        }
    return signals


def headline(signals: Mapping[str, Any]) -> dict[str, Optional[float]]:
    """The scalar signals the risk model and the plots use, from a full ``model_signals`` result."""
    layers = signals["layers"]
    last = layers[-1]
    out: dict[str, Optional[float]] = {
        "eff_rank_last": last["effective_rank"],
        "eff_rank_mean": sum(l["effective_rank"] for l in layers) / len(layers),
        "part_ratio_last": last["participation_ratio"],
        "anisotropy_last": last["anisotropy"],
        "anisotropy_mean": sum(l["anisotropy"] for l in layers) / len(layers),
        "dead_fraction_mean": sum(m["dead_fraction"] for m in signals["mlp"]) / len(signals["mlp"]),
        "stable_rank_attn": signals["weights"]["attn_qkv"]["stable_rank"],
        "stable_rank_mlp": signals["weights"]["mlp_in"]["stable_rank"],
    }
    if "delta" in signals:
        out["delta_rel_norm"] = signals["delta"]["rel_norm"]
        out["delta_eff_rank"] = signals["delta"]["effective_rank"]
    return out


@lru_cache(maxsize=2)
def _base_weights(base_model: str) -> dict[str, Any]:
    """The base model's block weights, loaded once for all the checkpoints measured."""
    from transformers import AutoModelForCausalLM

    return block_weights(AutoModelForCausalLM.from_pretrained(base_model).eval())


def compute_checkpoint(
    checkpoint: str | Path,
    texts: Sequence[str],
    *,
    max_len: int,
    batch_size: int,
    device: str,
    eps: float,
    exclude_first_token: bool,
    base_model: Optional[str],
) -> dict[str, Any]:
    """Load a checkpoint and measure it on ``texts`` (tokenised as the grid scores them)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(checkpoint)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device).eval()
    batches = []
    for i in range(0, len(texts), batch_size):
        enc = tok(list(texts[i : i + batch_size]), return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(device)
        batches.append((enc.input_ids, enc.attention_mask))
    base = _base_weights(base_model) if base_model else None
    return model_signals(model, batches, eps=eps, exclude_first_token=exclude_first_token,
                         base_weights=base)


# --- 4. which checkpoints, resume, summary ----------------------------------------------------------

_RECURSION_DIR = re.compile(r"^pool_recursive_s(\d+)_gen(\d+)$")


def settings_fingerprint(grid_fingerprint: str, settings: Mapping[str, Any]) -> str:
    """Short hash of everything that changes what a row means (grid settings plus these)."""
    blob = json.dumps({"grid": grid_fingerprint, **settings}, sort_keys=True, default=str)
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=6).hexdigest()


def plan(
    grid_rows: Sequence[Mapping[str, Any]], models_dir: str | Path, seeds: Sequence[int]
) -> tuple[list[dict[str, Any]], list[str]]:
    """What to measure: ``(items, missing cells)``.

    Each grid cell with its own checkpoint is an item. A shared ratio-0 baseline has none:
    it points at the trained baseline of its seed (``source``), and is copied from it.
    Recursion intermediates found on disk for the configured seeds are items too
    (``kind="recursion_stage"``). ``missing`` lists cells with no checkpoint to measure.
    """
    models_dir = Path(models_dir)
    items: list[dict[str, Any]] = []
    missing: list[str] = []
    own = {r["cell"]: models_dir / r["cell"] for r in grid_rows if (models_dir / r["cell"]).is_dir()}
    trained_baseline = {
        r["seed"]: r["cell"] for r in grid_rows
        if r["ratio"] == 0 and not r.get("shared_baseline") and r["cell"] in own
    }
    for r in grid_rows:
        base = {"cell": r["cell"], "type": r["type"], "ratio": r["ratio"], "seed": r["seed"],
                "grid_holdout_ppl": r["holdout_ppl"]}
        if r["cell"] in own:
            items.append({**base, "kind": "cell", "checkpoint": own[r["cell"]]})
        elif r.get("shared_baseline") and r["seed"] in trained_baseline:
            items.append({**base, "kind": "shared_baseline", "source": trained_baseline[r["seed"]]})
        else:
            missing.append(r["cell"])
    if models_dir.is_dir():
        for path in sorted(models_dir.iterdir()):
            m = _RECURSION_DIR.match(path.name)
            if m and path.is_dir() and int(m.group(1)) in set(seeds):
                items.append({"cell": path.name, "type": "recursive", "ratio": None, "seed": int(m.group(1)),
                              "generation": int(m.group(2)), "kind": "recursion_stage", "checkpoint": path})
    return items, missing


def load_rows(path: str | Path, fingerprint: str) -> dict[str, dict[str, Any]]:
    """Measured rows by cell id (the last row for a cell wins). Refuses other settings."""
    path = Path(path)
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("settings") != fingerprint:
            raise ValueError(
                f"{path} holds rows made under different settings ({row.get('settings')}, now "
                f"{fingerprint}). Point output_dir somewhere new, or delete the file."
            )
        rows[row["cell"]] = row
    return rows


def is_current(item: Mapping[str, Any], row: Optional[Mapping[str, Any]]) -> bool:
    """Is ``row`` a measurement of the checkpoint ``item`` describes?

    A grid cell that was re-run has a new checkpoint and a different ``holdout_ppl``, so its
    old row is out of date. Recursion stages have no grid row, so presence is enough.
    """
    if row is None:
        return False
    return row.get("grid_holdout_ppl") == item.get("grid_holdout_ppl")


def run_whitebox(
    items: Sequence[Mapping[str, Any]],
    *,
    compute: Callable[[Path], dict[str, Any]],
    done: Mapping[str, Mapping[str, Any]],
    on_result: Callable[[dict[str, Any]], None],
    settings: str,
) -> list[dict[str, Any]]:
    """Measure every item that has no current row; return one row per item.

    ``compute(checkpoint)`` returns a full ``model_signals`` dict. Baselines are copied from
    their source, measured first. A row is written as soon as it exists.
    """
    rows: dict[str, dict[str, Any]] = {}
    ordered = sorted(items, key=lambda it: it["kind"] == "shared_baseline")  # sources first
    for item in ordered:
        cell = item["cell"]
        existing = done.get(cell)
        if item["kind"] == "shared_baseline":
            source = rows.get(item["source"])
            if source is None:
                raise ValueError(f"baseline source {item['source']} was not measured")
            row = {**source, "cell": cell, "type": item["type"], "kind": "shared_baseline",
                   "shared_from": item["source"], "grid_holdout_ppl": item["grid_holdout_ppl"]}
            row.pop("headline", None)
            if existing is not None and all(
                existing.get(k) == row.get(k) for k in ("type", "shared_from", "grid_holdout_ppl", "signals")
            ):
                rows[cell] = dict(existing)  # a copy of the same measurement: nothing to write
                continue
        elif is_current(item, existing):
            rows[cell] = dict(existing)
            continue
        else:
            signals = compute(Path(item["checkpoint"]))
            row = {
                **{k: item[k] for k in ("cell", "type", "ratio", "seed", "kind") if k in item},
                **({"generation": item["generation"]} if "generation" in item else {}),
                **({"grid_holdout_ppl": item["grid_holdout_ppl"]} if "grid_holdout_ppl" in item else {}),
                "checkpoint_name": Path(item["checkpoint"]).name,
                "signals": signals,
            }
        row["headline"] = headline(row["signals"])
        row["settings"] = settings
        rows[cell] = row
        on_result(row)
    return [rows[it["cell"]] for it in items]


HEADLINE_KEYS = (
    "eff_rank_last", "eff_rank_mean", "part_ratio_last", "anisotropy_last", "anisotropy_mean",
    "dead_fraction_mean", "stable_rank_attn", "stable_rank_mlp", "delta_rel_norm", "delta_eff_rank",
)


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean, min and max across seeds of each headline signal, per (type, ratio); recursion stages by generation."""
    def stats(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n_seeds": len({r["seed"] for r in group})}
        for key in HEADLINE_KEYS:
            vals = [r["headline"][key] for r in group if r["headline"].get(key) is not None]
            if vals:
                out[key] = {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}
        return out

    by_cell: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    by_gen: dict[int, list[Mapping[str, Any]]] = {}
    for r in rows:
        if r["kind"] == "recursion_stage":
            by_gen.setdefault(r["generation"], []).append(r)
        else:
            by_cell.setdefault((r["type"], r["ratio"]), []).append(r)
    return {
        "curves": [{"type": t, "ratio": ratio, **stats(g)} for (t, ratio), g in sorted(by_cell.items())],
        "recursion_stages": [{"generation": gen, **stats(g)} for gen, g in sorted(by_gen.items())],
    }


def format_table(summary: Mapping[str, Any], key: str) -> str:
    """Mean of one headline signal by ratio (rows) and type (columns), for the run log."""
    curves = summary["curves"]
    kinds = sorted({c["type"] for c in curves})
    ratios = sorted({c["ratio"] for c in curves})
    cell = {(c["type"], c["ratio"]): c[key]["mean"] for c in curves if key in c}
    lines = ["ratio  " + "".join(f"{t:>16}" for t in kinds)]
    for r in ratios:
        vals = "".join(f"{cell[(t, r)]:>16.4f}" if (t, r) in cell else f"{'-':>16}" for t in kinds)
        lines.append(f"{r:<7g}" + vals)
    return "\n".join(lines)


_COLORS = {"synthetic": "#2a78d6", "recursive": "#eb6834", "paraphrased": "#1baf7a", "benchmark_near": "#eda100"}
_MARKERS = {"synthetic": "o", "recursive": "s", "paraphrased": "^", "benchmark_near": "D"}
_PANELS = (
    ("eff_rank_last", "Effective rank, last layer", "lower = fewer directions in use"),
    ("anisotropy_last", "Anisotropy, last layer", "higher = representations point one way"),
    ("dead_fraction_mean", "Dead MLP neurons", "share that never activate"),
    ("delta_rel_norm", "Distance from the base weights", "||W - W0|| / ||W0||"),
)


def _plot(summary: Mapping[str, Any], path: Path) -> None:
    """Mean across seeds as a line, the min-max range as a band, one panel per signal."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = summary["curves"]
    panels = [p for p in _PANELS if any(p[0] in c for c in curves)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.8), squeeze=False)
    for ax, (key, title, note) in zip(axes[0], panels):
        for kind in _COLORS:
            pts = sorted((c for c in curves if c["type"] == kind and key in c), key=lambda c: c["ratio"])
            if not pts:
                continue
            x = [c["ratio"] for c in pts]
            ax.plot(x, [c[key]["mean"] for c in pts], color=_COLORS[kind], marker=_MARKERS[kind],
                    markersize=5, linewidth=2, label=kind)
            ax.fill_between(x, [c[key]["min"] for c in pts], [c[key]["max"] for c in pts],
                            color=_COLORS[kind], alpha=0.15, linewidth=0)
        ax.set_title(f"{title}\n({note})", fontsize=9)
        ax.set_xlabel("contamination ratio")
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


_SETTING_KEYS = ("grid_config", "eps", "exclude_first_token", "delta_from_base", "output_dir", "figure")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measure white-box signals on the grid's checkpoints.")
    parser.add_argument("--config", default="configs/whitebox.yaml")
    parser.add_argument("--summarize-only", action="store_true",
                        help="measure nothing; rebuild the summary, tables and figure from whitebox.jsonl")
    args = parser.parse_args(argv)

    import yaml

    from ..finetune.collapse import load_holdout
    from ..finetune.grid import config_fingerprint, load_results

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    missing_keys = [k for k in _SETTING_KEYS if k not in cfg]
    if missing_keys:
        raise ValueError(f"{args.config} is missing {missing_keys}")
    grid = yaml.safe_load(Path(cfg["grid_config"]).read_text(encoding="utf-8"))
    grid_fp = config_fingerprint(grid)
    grid_rows = list(load_results(Path(grid["output_dir"]) / "results.jsonl", grid_fp).values())
    if not grid_rows:
        raise SystemExit(f"no grid results in {grid['output_dir']}/results.jsonl")
    settings = settings_fingerprint(grid_fp, {
        "eps": cfg["eps"], "exclude_first_token": cfg["exclude_first_token"],
        "delta_from_base": cfg["delta_from_base"], "max_len": grid["train"]["max_len"],
    })
    out = Path(cfg["output_dir"])
    rows_path = out / "whitebox.jsonl"
    done = load_rows(rows_path, settings)

    items, missing = plan(grid_rows, grid["models_dir"], grid["seeds"])
    if missing:
        print(f"WARNING: {len(missing)} grid cell(s) have no checkpoint in {grid['models_dir']} "
              f"(was the grid run with keep_models?): {missing[:6]}{'...' if len(missing) > 6 else ''}")

    if not args.summarize_only:
        texts = load_holdout(grid["data"])
        out.mkdir(parents=True, exist_ok=True)

        def compute(checkpoint: Path) -> dict[str, Any]:
            print(f"  measuring {checkpoint.name}", flush=True)
            return compute_checkpoint(
                checkpoint, texts, max_len=grid["train"]["max_len"], batch_size=grid["train"]["batch_size"],
                device=grid["device"], eps=cfg["eps"], exclude_first_token=cfg["exclude_first_token"],
                base_model=grid["base_model"] if cfg["delta_from_base"] else None,
            )

        def on_result(row: dict[str, Any]) -> None:
            with rows_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")

        todo = [it for it in items if not is_current(it, done.get(it["cell"]))]
        print(f"{len(items)} checkpoints, {len(items) - len(todo)} already measured, {len(todo)} to do", flush=True)
        run_whitebox(items, compute=compute, done=done, on_result=on_result, settings=settings)
        done = load_rows(rows_path, settings)

    wanted = {it["cell"] for it in items}
    rows = [r for k, r in done.items() if k in wanted]
    if not rows:
        raise SystemExit(f"no white-box rows in {rows_path}")
    summary = summarize(rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "whitebox_summary.json").write_text(
        json.dumps({"settings": settings, "grid_fingerprint": grid_fp, "missing_cells": missing, **summary}, indent=2),
        encoding="utf-8",
    )
    _plot(summary, Path(cfg["figure"]))
    for key in ("eff_rank_last", "anisotropy_last", "dead_fraction_mean"):
        print(f"\n{key} (mean over seeds):\n" + format_table(summary, key))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
