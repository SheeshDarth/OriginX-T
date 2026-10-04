"""Generate synthetic, recursive, and paraphrased contamination with a model.

All three need a language model, but the model is passed in as a plain
``generate(text) -> text`` callable. That keeps the bookkeeping (ids,
provenance, generation depth) deterministic and testable on CPU with a stub,
and lets the same code run against DistilGPT-2 or Qwen2.5 (TRD section 4)
by swapping the callable. ``hf_generator`` builds the real one.

- **synthetic**: the model writes the response. Prompt/response data gets the
  prompt; raw corpora (empty prompt) get the first half of the human text as a
  prefix to continue, so the length and topic stay close to the original.
- **recursive**: the same thing, run on synthetic output. Depth comes from
  ``generation``, so calling ``make_synthetic`` on Gen-1 data gives Gen-2.
- **paraphrased**: the model rewrites the human response, so the meaning stays
  the same and the wording changes.
"""

from __future__ import annotations

from typing import Callable, Sequence

from ..ingestion.schema import Sample

Generate = Callable[[str], str]

PARAPHRASE_TEMPLATE = (
    "Rewrite the following text in different words, keeping the meaning.\n\n"
    "Text: {text}\n\nRewritten:"
)


def _prefix(text: str) -> str:
    words = text.split()
    return " ".join(words[: max(1, len(words) // 2)])


def make_synthetic(samples: Sequence[Sample], generate: Generate) -> list[Sample]:
    """Replace each response with model output and mark it one generation deeper.

    Human input (generation 0) becomes ``synthetic`` Gen-1, and Gen-1 input
    becomes ``recursive`` Gen-2. Run it again on the output for deeper
    recursion. Raw-corpus samples keep their human prefix, because that is what
    the model continued.
    """
    out: list[Sample] = []
    for s in samples:
        depth = s.generation + 1
        if s.prompt:
            response = generate(s.prompt).strip()
        else:
            prefix = _prefix(s.response)
            response = f"{prefix} {generate(prefix).strip()}".strip()
        out.append(
            Sample(
                sample_id=f"{s.sample_id}-g{depth}",
                prompt=s.prompt,
                response=response,
                label=s.label,
                source="synthetic" if depth == 1 else "recursive",
                generation=depth,
                split=s.split,
            ).validate()
        )
    return out


def make_paraphrased(
    samples: Sequence[Sample],
    generate: Generate,
    *,
    template: str = PARAPHRASE_TEMPLATE,
) -> list[Sample]:
    """Have the model rewrite each response in new words. Prompts are left as they are.

    Generation is 1 because the text is model-written, even though the content
    comes from a human.
    """
    return [
        Sample(
            sample_id=f"{s.sample_id}-para",
            prompt=s.prompt,
            response=generate(template.format(text=s.response)).strip(),
            label=s.label,
            source="paraphrased",
            generation=1,
            split=s.split,
        ).validate()
        for s in samples
    ]


def hf_generator(
    model_name: str = "distilgpt2",
    *,
    max_new_tokens: int = 128,
    seed: int = 0,
    device: int | str = -1,
) -> Generate:
    """Build a ``generate`` callable backed by a Hugging Face text-generation pipeline.

    Returns only the new text, without the input. Sampling is seeded once, so a
    run is reproducible on a given machine and library version.
    """
    from transformers import pipeline, set_seed  # heavy import, only when used

    set_seed(seed)
    pipe = pipeline("text-generation", model=model_name, device=device)

    def generate(text: str) -> str:
        # ponytail: one call per sample, no batching. Batch via pipe(list) if
        # throughput on the full corpus matters.
        return pipe(
            text,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            return_full_text=False,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )[0]["generated_text"]

    return generate
