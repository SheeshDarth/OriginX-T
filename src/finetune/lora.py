"""Fine-tune a small causal LM with LoRA and save a merged checkpoint.

A plain PyTorch loop rather than ``transformers.Trainer``: the collapse grid
trains many tiny models, and a dozen lines of loop are easier to read, seed and
debug than Trainer's configuration surface.

The LoRA weights are merged into the base model before saving, so the result
loads like any Hugging Face model (``hf_generator(path)``, ``perplexity(path)``).
torch/transformers/peft are imported inside the function so the package imports
without them.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional

# on_step(step, model, tokenizer, last): step 0 is the untrained model, then after every
# optimizer step; ``last`` is True once, after the final step. It must leave the model as
# it found it (src.evaluation.live.TrainLogger does).
OnStep = Callable[[int, Any, Any, bool], None]


def train_lora(
    texts: Sequence[str],
    out_dir: str | Path,
    *,
    base_model: str = "distilgpt2",
    epochs: int = 1,
    lr: float = 2e-4,
    batch_size: int = 8,
    max_len: int = 256,
    lora_r: int = 8,
    seed: int = 0,
    device: str = "cpu",
    on_step: Optional[OnStep] = None,
) -> Path:
    """Train LoRA adapters on ``texts`` from ``base_model``. Returns ``out_dir``.

    ``on_step`` (optional) watches training, e.g. to log signals every N steps (X3).
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model)
    # ponytail: full-precision LoRA, not the TRD's 4-bit QLoRA. DistilGPT-2
    # fits in 4GB without it; add bitsandbytes quantization for bigger models.
    model = get_peft_model(
        model, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, task_type="CAUSAL_LM")
    )
    model.to(device).train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    rng = random.Random(seed)
    step = 0
    total = epochs * -(-len(texts) // batch_size)
    if on_step is not None:
        on_step(0, model, tok, total == 0)
    for _ in range(epochs):
        order = list(texts)
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            batch = tok(
                order[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            ).to(device)
            labels = batch.input_ids.masked_fill(batch.attention_mask == 0, -100)
            model(**batch, labels=labels).loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            if on_step is not None:
                on_step(step, model, tok, step == total)

    out_dir = Path(out_dir)
    model.merge_and_unload().save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir
