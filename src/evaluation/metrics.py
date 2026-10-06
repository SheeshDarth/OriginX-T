"""Collapse metrics: model perplexity on held-out text, and data diversity.

``perplexity`` measures the model: a collapsing model assigns lower probability
to real human text, so its hidden-holdout perplexity rises. ``distinct_n``
measures the data: recursive generations repeat themselves, so the share of
unique n-grams falls.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path


def distinct_n(texts: Sequence[str], n: int) -> float:
    """Unique n-grams / total n-grams over all texts (whitespace tokens). 0.0 if none."""
    grams = [
        tuple(words[i : i + n])
        for words in (t.split() for t in texts)
        for i in range(len(words) - n + 1)
    ]
    return len(set(grams)) / len(grams) if grams else 0.0


def perplexity(
    model_dir: str | Path,
    texts: Sequence[str],
    *,
    max_len: int = 256,
    batch_size: int = 8,
    device: str = "cpu",
) -> float:
    """Token-weighted perplexity of the model at ``model_dir`` on ``texts``."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device).eval()

    nll, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = tok(
                list(texts[i : i + batch_size]),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            ).to(device)
            labels = batch.input_ids.masked_fill(batch.attention_mask == 0, -100)
            # loss is the mean over predicted tokens; weight it back to a sum
            predicted = int((labels[:, 1:] != -100).sum())
            nll += model(**batch, labels=labels).loss.item() * predicted
            count += predicted
    return math.exp(nll / count)
