"""Carve the trust-critical hidden holdout out of a normalized corpus.

The hidden holdout is the set ORIGIN-T measures degradation against, so it must
never enter any training mixture. Assignment is by a **stable hash of the
sample's text** (the leak check's fingerprint), not a shuffle: the same text
always lands on the same side of the split, even if the corpus grows or rows
are reordered later. That means a holdout built today stays valid when new data
arrives — a reshuffle would silently leak previously-held-out samples into
training.

Hashing the text rather than the id matters because real corpora repeat lines
(WikiText-2 has ``= = History = =`` 129 times). Split by id, those copies land
on both sides and the leak check fails on a clean corpus; split by text, every
copy goes to the same side.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

from .check_leakage import text_fingerprint
from .loaders import load_dataset, write_jsonl
from .schema import Sample

HOLDOUT_SPLIT = "hidden_holdout"


def _bucket(key: str, salt: str) -> float:
    """Map a key to a stable float in [0, 1).

    Uses blake2b rather than :func:`hash` because Python's built-in hash is
    randomized per process (PYTHONHASHSEED) and would give a different split on
    every run.
    """
    digest = hashlib.blake2b(f"{salt}:{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def is_holdout(key: str, *, fraction: float, salt: str = "origin-t") -> bool:
    """True if this key (a text fingerprint) belongs in the hidden holdout."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    return _bucket(key, salt) < fraction


def carve_holdout(
    samples: Iterable[Sample], *, fraction: float = 0.1, salt: str = "origin-t"
) -> tuple[list[Sample], list[Sample]]:
    """Split samples into ``(kept, holdout)``.

    Holdout samples get ``split="hidden_holdout"``; kept samples are returned
    unchanged. Deterministic for a given ``fraction`` and ``salt``.
    """
    kept: list[Sample] = []
    holdout: list[Sample] = []
    for sample in samples:
        if is_holdout(text_fingerprint(sample), fraction=fraction, salt=salt):
            holdout.append(replace(sample, split=HOLDOUT_SPLIT).validate())
        else:
            kept.append(sample)
    return kept, holdout


def main(argv: Optional[list[str]] = None) -> int:
    """Carve each input corpus into `*_kept.jsonl` and `*_hidden_holdout.jsonl`.

        python -m src.ingestion.holdout --input data/processed/wikitext2_train.jsonl
    """
    parser = argparse.ArgumentParser(description="Carve the hidden holdout out of a corpus.")
    parser.add_argument("--input", required=True, nargs="+", help="normalized JSONL file(s)")
    parser.add_argument("--fraction", type=float, default=0.1, help="holdout share (default 0.1)")
    parser.add_argument("--salt", default="origin-t", help="split salt; changing it re-splits")
    args = parser.parse_args(argv)

    for path in args.input:
        source = Path(path)
        kept, holdout = carve_holdout(
            load_dataset(source), fraction=args.fraction, salt=args.salt
        )
        # Write beside the input rather than over it — re-running is then non-destructive.
        write_jsonl(kept, source.with_name(f"{source.stem}_kept.jsonl"))
        write_jsonl(holdout, source.with_name(f"{source.stem}_hidden_holdout.jsonl"))
        print(f"{source.name}: {len(kept)} kept, {len(holdout)} held out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
