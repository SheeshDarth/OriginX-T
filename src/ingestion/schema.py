"""Canonical sample schema for ORIGIN-T (TRD section 7).

This is the single shared contract between every module: ingestion produces
``Sample`` records, and generation, features, fine-tuning, and evaluation all
consume them. Change this schema only via a PR reviewed by the whole team.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Optional

# Provenance of a sample. `human` is the clean baseline; the rest are the
# contamination types the generator (src/generation) produces.
SOURCE_TYPES: tuple[str, ...] = (
    "human",
    "synthetic",
    "recursive",
    "paraphrased",
    "benchmark_near",
    "duplicate",
)

# Dataset partitions. `hidden_holdout` is the trust-sensitive set that must
# never appear in any training mixture.
SPLITS: tuple[str, ...] = ("train", "validation", "test", "hidden_holdout")


@dataclass
class Sample:
    """One dataset record.

    ``sample_id`` is the only required field; loaders always assign it. A
    record must carry text in ``prompt`` and/or ``response`` — raw corpora
    (e.g. WikiText) put their text in ``response`` with an empty ``prompt``,
    while prompt/response datasets fill both.
    """

    sample_id: str
    prompt: str = ""
    response: str = ""
    label: Optional[str] = None
    source: str = "human"          # one of SOURCE_TYPES
    generation: int = 0            # 0 = human, 1 = first-gen synthetic, 2+ = recursive depth
    contamination_ratio: float = 0.0   # dataset-level mixture ratio, 0..1
    benchmark_near_score: float = 0.0  # similarity to a known benchmark item, 0..1
    split: str = "train"           # one of SPLITS

    def validate(self) -> "Sample":
        """Raise ``ValueError`` if any field is out of contract. Returns self."""
        if not self.sample_id:
            raise ValueError("sample_id must be a non-empty string")
        if self.source not in SOURCE_TYPES:
            raise ValueError(
                f"source {self.source!r} not in {SOURCE_TYPES} (sample {self.sample_id})"
            )
        if self.split not in SPLITS:
            raise ValueError(
                f"split {self.split!r} not in {SPLITS} (sample {self.sample_id})"
            )
        if self.generation < 0:
            raise ValueError(f"generation must be >= 0 (sample {self.sample_id})")
        if not 0.0 <= self.contamination_ratio <= 1.0:
            raise ValueError(
                f"contamination_ratio must be in [0, 1] (sample {self.sample_id})"
            )
        if not 0.0 <= self.benchmark_near_score <= 1.0:
            raise ValueError(
                f"benchmark_near_score must be in [0, 1] (sample {self.sample_id})"
            )
        if not (self.prompt or self.response):
            raise ValueError(
                f"sample {self.sample_id} has empty prompt and response"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sample":
        """Build a Sample from a dict, ignoring keys outside the schema."""
        known = cls.field_names()
        return cls(**{k: v for k, v in data.items() if k in known})
