"""Generate synthetic, recursive, and paraphrased contamination with a model.

All three need a language model, but the model is passed in as a plain
``generate(texts) -> texts`` callable that takes a whole list of inputs, so a
real model can batch them on the GPU. That keeps the bookkeeping (ids,
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

from collections.abc import Callable, Sequence

from ..ingestion.schema import Sample

# Takes N input texts, returns N generated texts in the same order.
Generate = Callable[[list[str]], list[str]]

PARAPHRASE_TEMPLATE = (
    "Rewrite the following text in different words, keeping the meaning.\n\n"
    "Text: {text}\n\nRewritten:"
)


def _prefix(text: str) -> str:
    words = text.split()
    return " ".join(words[: max(1, len(words) // 2)])


def _call(generate: Generate, inputs: list[str]) -> list[str]:
    outputs = generate(inputs)
    if len(outputs) != len(inputs):
        raise ValueError(f"generate returned {len(outputs)} texts for {len(inputs)} inputs")
    return [o.strip() for o in outputs]


def make_synthetic(samples: Sequence[Sample], generate: Generate) -> list[Sample]:
    """Replace each response with model output and mark it one generation deeper.

    Human input (generation 0) becomes ``synthetic`` Gen-1, and Gen-1 input
    becomes ``recursive`` Gen-2. Run it again on the output for deeper
    recursion. Raw-corpus samples keep their human prefix, because that is what
    the model continued.
    """
    # Raw-corpus samples are continued from a prefix, which is kept in the output.
    kept = ["" if s.prompt else _prefix(s.response) for s in samples]
    inputs = [s.prompt or k for s, k in zip(samples, kept)]
    outputs = _call(generate, inputs)

    out: list[Sample] = []
    for s, k, gen in zip(samples, kept, outputs):
        depth = s.generation + 1
        out.append(
            Sample(
                sample_id=f"{s.sample_id}-g{depth}",
                prompt=s.prompt,
                response=f"{k} {gen}".strip(),
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
    outputs = _call(generate, [template.format(text=s.response) for s in samples])
    return [
        Sample(
            sample_id=f"{s.sample_id}-para",
            prompt=s.prompt,
            response=gen,
            label=s.label,
            source="paraphrased",
            generation=1,
            split=s.split,
        ).validate()
        for s, gen in zip(samples, outputs)
    ]


def hf_generator(
    model_name: str = "distilgpt2",
    *,
    max_new_tokens: int = 128,
    batch_size: int = 16,
    seed: int = 0,
    device: int | str = -1,
) -> Generate:
    """Build a batched ``generate`` callable backed by a Hugging Face pipeline.

    Returns only the new text, without the input. Sampling is seeded once, so a
    run is reproducible on a given machine, library version and batch size.
    """
    from transformers import pipeline, set_seed  # heavy import, only when used

    set_seed(seed)
    pipe = pipeline("text-generation", model=model_name, device=device)
    # GPT-style models have no pad token, and batched generation must pad on
    # the left so every prompt ends right where generation starts.
    pipe.tokenizer.pad_token = pipe.tokenizer.pad_token or pipe.tokenizer.eos_token
    pipe.tokenizer.padding_side = "left"

    def generate(texts: list[str]) -> list[str]:
        results = pipe(
            texts,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            return_full_text=False,
            pad_token_id=pipe.tokenizer.pad_token_id,
        )
        return [r[0]["generated_text"] for r in results]

    return generate
