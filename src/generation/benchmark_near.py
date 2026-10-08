"""Generate benchmark-near contamination — leaked eval items in disguise.

Benchmark leakage rarely looks like an exact copy. An item gets reformatted,
recased, or rewrapped somewhere in a data pipeline and lands in training as a
near-duplicate that exact-match and n-gram overlap checks miss, while the model
still memorises the answer. That is the failure ORIGIN-T has to detect, so the
generator has to produce it.

This is the one contamination type that needs no model: the perturbations are
surface-level and meaning-preserving, so they run on CPU and stay deterministic.
Genuine paraphrase (same meaning, different words) needs a generation model and
belongs with the synthetic generators.

``benchmark_near_score`` is *measured* per sample with ``difflib``, not assumed
— an exact copy scores 1.0, a reformatted one lower, and the detector is then
scored against a real number rather than a label we invented.

Why the text is never recased as a whole. An earlier version rewrote about half
of the perturbed copies in full ALL CAPS or all lowercase. That is still the
same content, and ``similarity()`` rightly scores it 1.0, but GPT-2 tokenizes
``THE RAILWAY`` and ``the railway`` as completely different tokens from ``The
railway``: ALL CAPS changes up to 80-87% of a passage's pre-tokens. So the
"near-duplicate" was not near to the model. In the full Sprint-6 grid (PR #12),
training on these copies *raised* benchmark-item perplexity by 4.7% at ratio 1
instead of lowering it, while 100% verbatim copies lowered it by 4.9% (1 epoch)
and 13.4% (3 epochs): the recasing hid the leak it was meant to simulate, and it
also caused outsized holdout damage (+8% against +1.2% for verbatim copies).
Every perturbation here is therefore a surface edit that leaves most tokens as
they were: a flipped first letter or sentence start, a double space after a
sentence, swapped punctuation.
"""

from __future__ import annotations

import difflib
import random
import re
from typing import Optional, Sequence

from ..ingestion.schema import Sample

_PUNCT_SWAPS = {".": " .", ",": " ,", "!": ".", "?": " ?", ";": ",", "'": "", '"': ""}


_LETTER = r"[^\W\d_]"
_FIRST_LETTER = re.compile(_LETTER)
# the first letter after a sentence end, allowing a closing quote or bracket in between
_AFTER_SENTENCE_END = re.compile(rf"[.!?][\"')\]]*\s+(?={_LETTER})")


def _recase(text: str, rng: random.Random) -> str:
    """Flip the case of the first letter, or of every sentence start. Never the whole text.

    A pipeline that lower-cases or capitalises sentence starts changes one token
    per sentence. Recasing everything would change nearly all of them (see the
    module docstring), so if an edit would leave a mixed-case text entirely upper
    or lower case, the text is returned unchanged instead.
    """
    first = _FIRST_LETTER.search(text)
    if first is None:
        return text
    starts = [first.start()] + [m.end() for m in _AFTER_SENTENCE_END.finditer(text)]
    chars = list(text)
    for i in starts[:1] if rng.random() < 0.5 else starts:
        chars[i] = chars[i].swapcase()
    out = "".join(chars)
    mixed = text not in (text.lower(), text.upper())
    return text if mixed and out in (out.lower(), out.upper()) else out


def _respace(text: str, rng: random.Random) -> str:
    """Collapse whitespace, or pad it the way old typewriter-style text does.

    Padding is a double space after each sentence end (or, if there is none, at one
    place), not after every word: doubling every space makes up to half of the
    tokens a lone-space filler and changes the context of every word.
    """
    collapsed = re.sub(r"\s+", " ", text).strip()
    if rng.random() < 0.5:
        return collapsed
    padded = re.sub(r"(?<=[.!?]) (?=\S)", "  ", collapsed)
    if padded == collapsed:
        gaps = [m.start() for m in re.finditer(" ", collapsed)]
        if gaps:
            i = rng.choice(gaps)
            padded = collapsed[:i] + "  " + collapsed[i + 1 :]
    return padded


def _repunct(text: str, rng: random.Random) -> str:
    for old, new in _PUNCT_SWAPS.items():
        if old in text and rng.random() < 0.5:
            text = text.replace(old, new)
    return text


_PERTURBATIONS = (_recase, _respace, _repunct)


def perturb(text: str, rng: random.Random) -> str:
    """Apply one or two surface edits. Meaning is preserved; the string is not.

    The edits are mild by construction: the result keeps at least about 80% of the
    original's whitespace-separated words and most of its GPT-2 tokens.
    """
    for op in rng.sample(_PERTURBATIONS, rng.randint(1, 2)):
        text = op(text, rng)
    return text.strip() or text


def similarity(a: str, b: str) -> float:
    """Content similarity in [0, 1], ignoring case and whitespace (1.0 == same content).

    Deliberately blind to formatting: an ALL-CAPS or re-wrapped copy of a
    benchmark item is a *complete* leak of the content, so a detector should find
    it, and it must score 1.0. Comparing raw characters would score it near zero
    and hand the detector wrong ground truth.

    That is a statement about content, not about what a model will memorise: a
    fully recased copy scores 1.0 here yet GPT-2 sees different tokens, which is
    why :func:`perturb` no longer recases whole texts (module docstring, PR #12).
    """
    fold = lambda s: re.sub(r"\s+", " ", s).strip().lower()  # noqa: E731
    return difflib.SequenceMatcher(None, fold(a), fold(b)).ratio()


def make_near_duplicates(
    benchmark: Sequence[Sample],
    *,
    count: Optional[int] = None,
    exact_fraction: float = 0.3,
    seed: int = 0,
) -> list[Sample]:
    """Turn benchmark items into leaked-looking training samples.

    ``exact_fraction`` of them are verbatim copies (the easy case a detector
    must not miss); the rest are surface-perturbed (the hard case). Each result
    is marked ``source="benchmark_near"`` and carries its measured similarity to
    the item it came from.

    Ids are suffixed rather than reused so a near-duplicate is never mistaken
    for the original benchmark row.
    """
    if not 0.0 <= exact_fraction <= 1.0:
        raise ValueError(f"exact_fraction must be in [0, 1], got {exact_fraction}")
    if not benchmark:
        raise ValueError("benchmark must not be empty")

    count = len(benchmark) if count is None else count
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")

    rng = random.Random(seed)
    n_exact = round(count * exact_fraction)

    out: list[Sample] = []
    for i in range(count):
        # ponytail: cycle the pool when count exceeds it, rather than sampling
        # with replacement -- keeps coverage even, matters at small benchmark sizes.
        item = benchmark[i % len(benchmark)]
        verbatim = i < n_exact
        response = item.response if verbatim else perturb(item.response, rng)
        prompt = item.prompt if verbatim else (perturb(item.prompt, rng) if item.prompt else "")

        out.append(
            Sample(
                sample_id=f"{item.sample_id}-near{i}",
                prompt=prompt,
                response=response,
                label=item.label,
                source="benchmark_near",
                generation=0,  # copied, not model-generated
                benchmark_near_score=similarity(item.response, response),
                split=item.split,
            ).validate()
        )
    return out
