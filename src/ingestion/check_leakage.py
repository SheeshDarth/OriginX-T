"""Assert the hidden holdout never appears in any training file.

Holdout leakage silently inflates every downstream result — a model that has
seen the holdout looks healthy even while collapsing — so this runs as a gate
before any fine-tune reads data.

Two independent checks, because ids alone are not enough: contamination
generators copy and rewrite text, so the same passage can reappear under a
fresh id.

  1. **id overlap**    — a holdout ``sample_id`` present in a training file.
  2. **text overlap**  — identical normalized text (case/whitespace-folded)
                          under any id.

    python -m src.ingestion.check_leakage \
        --holdout data/processed/wikitext2_hidden_holdout.jsonl \
        --train data/processed/wikitext2_train.jsonl

Exit code is 1 when leakage is found, so CI fails loudly.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .loaders import load_dataset
from .schema import Sample

_WHITESPACE = re.compile(r"\s+")


def text_fingerprint(sample: Sample) -> str:
    """Stable hash of a sample's text, folded so trivial edits still match."""
    combined = f"{sample.prompt}\n{sample.response}".strip().lower()
    normalized = _WHITESPACE.sub(" ", combined)
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()


@dataclass
class LeakReport:
    """Result of a leakage check. Truthy when leaked, so `if report:` reads well."""

    id_overlap: list[str] = field(default_factory=list)
    text_overlap: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.id_overlap and not self.text_overlap

    def __bool__(self) -> bool:  # truthy == leaked
        return not self.clean

    def summary(self) -> str:
        if self.clean:
            return "OK: no holdout leakage detected."
        lines = ["LEAKAGE DETECTED"]
        if self.id_overlap:
            lines.append(f"  {len(self.id_overlap)} sample_id(s) shared with training data")
            lines += [f"    - {i}" for i in self.id_overlap[:5]]
        if self.text_overlap:
            lines.append(f"  {len(self.text_overlap)} holdout text(s) reappear in training data")
            lines += [f"    - {i}" for i in self.text_overlap[:5]]
        return "\n".join(lines)


def find_leaks(holdout: Iterable[Sample], train: Iterable[Sample]) -> LeakReport:
    """Compare a holdout set against pooled training samples."""
    train_samples = list(train)
    train_ids = {s.sample_id for s in train_samples}
    train_texts = {text_fingerprint(s) for s in train_samples}

    report = LeakReport()
    for sample in holdout:
        if sample.sample_id in train_ids:
            report.id_overlap.append(sample.sample_id)
        if text_fingerprint(sample) in train_texts:
            report.text_overlap.append(sample.sample_id)
    return report


def check_files(holdout_path: str | Path, train_paths: Sequence[str | Path]) -> LeakReport:
    """Load a holdout file and training files from disk, then compare."""
    holdout = load_dataset(holdout_path)
    train: list[Sample] = []
    for path in train_paths:
        train.extend(load_dataset(path))
    return find_leaks(holdout, train)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail if the hidden holdout leaks into training data."
    )
    parser.add_argument("--holdout", required=True, help="holdout JSONL file")
    parser.add_argument("--train", required=True, nargs="+", help="training/validation JSONL files")
    args = parser.parse_args(argv)

    report = check_files(args.holdout, args.train)
    print(report.summary())
    return 1 if report else 0


if __name__ == "__main__":
    raise SystemExit(main())
