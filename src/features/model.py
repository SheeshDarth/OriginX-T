"""Reference-LM features of a dataset (HANDBOOK §10.7): the ``model`` feature family.

How surprising a dataset's texts are to a **fixed** reference model, ``gpt2``. It is never
one of the grid's checkpoints, so a feature describes the data, not a model trained on it.
Generated text tends to be easy for a language model: low loss, and tokens the model ranks
near the top of its prediction (the GLTR observation). Columns (the suite writes them as
``f_model_<name>``):

- ``loss_mean``, ``loss_p90``, ``loss_std``: over texts, of each text's mean next-token loss
  (nats). p90 is the 90th percentile, linear interpolation.
- ``rank_log_mean``: mean over texts of the mean ``ln(rank)`` of each true next token
  (rank 1 = the model's top guess, so 0 means every token was the top guess).
- ``rank_top10``: mean over texts of the share of tokens ranked in the model's top 10.
- ``rank_instability``: mean over texts of the standard deviation of ``ln(rank)`` within the
  text. Human text mixes predictable and surprising tokens; generated text is more uniform.

Every text has equal weight. A text with fewer than two tokens has no next token to score
and is left out; with no scorable text at all every value is ``None``.

``token_stats`` runs the model (torch and transformers are imported inside it) and is the
only part that needs them; ``aggregate`` is plain arithmetic, so tests stub the first and
check the second by hand. ``per_text`` gives the per-sample values (§7.3, C10).

``ctx`` is the suite's context dict; this family reads ``device`` (default ``"cpu"``),
``batch_size`` (8: a batch's logits over the 50k vocabulary are ~0.4 GB) and ``max_len``
(256, the grid's training length) from it.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any, Optional

VERSION = 1  # bump when a column's definition changes (the suite recomputes)
REFERENCE_MODEL = "openai-community/gpt2"  # "gpt2"; the full id resolves from the local cache
COLUMNS = ("loss_mean", "loss_p90", "loss_std", "rank_log_mean", "rank_top10", "rank_instability")

# One text's scored tokens: next-token losses (nats) and the ranks of the true tokens (1 = top).
TokenStats = tuple[list[float], list[int]]


@lru_cache(maxsize=2)
def _load(device: str) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(REFERENCE_MODEL)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(REFERENCE_MODEL).to(device).eval()
    return tok, model


# ponytail: unbounded, keyed by text. A grid's ~18k distinct texts fit easily and the shared
# human halves are scored once; key on a hash if corpora grow past memory.
_CACHE: dict[tuple[str, int], TokenStats] = {}


def token_stats(texts: Sequence[str], *, device: str = "cpu", batch_size: int = 8,
                max_len: int = 256) -> list[TokenStats]:
    """Next-token losses and true-token ranks of each text under the reference model."""
    todo = list(dict.fromkeys(t for t in texts if (t, max_len) not in _CACHE))
    if todo:
        import torch

        tok, model = _load(device)
        with torch.no_grad():
            for i in range(0, len(todo), batch_size):
                chunk = todo[i : i + batch_size]
                batch = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                            max_length=max_len).to(device)
                logits = model(**batch).logits[:, :-1].float()
                target = batch.input_ids[:, 1:]
                mask = batch.attention_mask[:, 1:].bool()
                losses = torch.nn.functional.cross_entropy(
                    logits.transpose(1, 2), target, reduction="none")
                true_logit = logits.gather(-1, target.unsqueeze(-1))
                ranks = (logits > true_logit).sum(-1) + 1  # ties count in the token's favour
                for j, text in enumerate(chunk):
                    m = mask[j]
                    _CACHE[(text, max_len)] = (losses[j][m].tolist(), ranks[j][m].tolist())
    return [_CACHE[(t, max_len)] for t in texts]


def _text_values(losses: Sequence[float], ranks: Sequence[int]) -> Optional[dict[str, float]]:
    if not losses:
        return None
    logs = [math.log(r) for r in ranks]
    return {
        "loss": statistics.fmean(losses),
        "rank_log": statistics.fmean(logs),
        "top10": sum(r <= 10 for r in ranks) / len(ranks),
        "instability": statistics.pstdev(logs),
    }


def _p90(values: list[float]) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=10, method="inclusive")[8]


def aggregate(stats: Sequence[TokenStats]) -> dict[str, Optional[float]]:
    """The family's columns from per-text token stats."""
    per = [v for v in (_text_values(*s) for s in stats) if v is not None]
    if not per:
        return dict.fromkeys(COLUMNS)
    loss = [p["loss"] for p in per]
    return {
        "loss_mean": statistics.fmean(loss),
        "loss_p90": _p90(loss),
        "loss_std": statistics.pstdev(loss),
        "rank_log_mean": statistics.fmean(p["rank_log"] for p in per),
        "rank_top10": statistics.fmean(p["top10"] for p in per),
        "rank_instability": statistics.fmean(p["instability"] for p in per),
    }


def _settings(ctx: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    ctx = ctx or {}
    return {"device": ctx.get("device", "cpu"), "batch_size": ctx.get("batch_size", 8),
            "max_len": ctx.get("max_len", 256)}


def model(texts: list[str], ctx: Optional[Mapping[str, Any]] = None) -> dict[str, Optional[float]]:
    """The ``model`` family: reference-LM loss and token-rank features of ``texts``."""
    return aggregate(token_stats(texts, **_settings(ctx)))


def per_text(texts: list[str], ctx: Optional[Mapping[str, Any]] = None) -> list[Optional[dict[str, float]]]:
    """Each text's own values (``loss``, ``rank_log``, ``top10``, ``instability``); ``None`` if unscorable."""
    return [_text_values(*s) for s in token_stats(texts, **_settings(ctx))]
