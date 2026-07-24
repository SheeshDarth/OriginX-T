"""Dataset loaders and normalizer for ORIGIN-T.

Reads JSONL / CSV / TXT into the canonical :class:`~src.ingestion.schema.Sample`
schema. Pure standard library — no pandas — so it runs anywhere, including a
cold Kaggle/Colab kernel. Field names are mapped from common aliases so the same
loader handles raw corpora (WikiText) and prompt/response datasets alike.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from .schema import Sample

# First matching alias wins. `text`/`content` map to `response` so raw corpora
# land as response-only records.
_PROMPT_ALIASES = ("prompt", "input", "instruction", "question", "source_text", "context")
_RESPONSE_ALIASES = ("response", "output", "target", "completion", "answer", "text", "content", "code")
_ID_ALIASES = ("sample_id", "id", "uid", "_id")


def _first(record: dict[str, Any], aliases: Iterable[str]) -> Optional[Any]:
    for key in aliases:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _to_str(value: Any) -> str:
    return "" if value is None else str(value)


def _to_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))  # tolerate "2" and "2.0"


def _to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def normalize_record(
    record: dict[str, Any],
    index: int,
    *,
    stem: str,
    source: Optional[str] = None,
    split: Optional[str] = None,
    generation: Optional[int] = None,
) -> Sample:
    """Map one raw dict to a validated :class:`Sample`.

    Explicit ``source`` / ``split`` / ``generation`` arguments win over values
    found in the record, which in turn win over the schema defaults. Missing
    ids get a deterministic ``{stem}-{index:07d}`` id so repeated loads are
    reproducible.
    """
    raw_id = _first(record, _ID_ALIASES)
    sample_id = _to_str(raw_id) if raw_id is not None else f"{stem}-{index:07d}"

    sample = Sample(
        sample_id=sample_id,
        prompt=_to_str(_first(record, _PROMPT_ALIASES)),
        response=_to_str(_first(record, _RESPONSE_ALIASES)),
        label=(None if record.get("label") in (None, "") else _to_str(record.get("label"))),
        source=source or _to_str(record.get("source")) or "human",
        generation=generation if generation is not None else _to_int(record.get("generation")),
        contamination_ratio=_to_float(record.get("contamination_ratio")),
        benchmark_near_score=_to_float(record.get("benchmark_near_score")),
        split=split or _to_str(record.get("split")) or "train",
    )
    return sample.validate()


def _read_raw(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else [data]
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    if suffix == ".txt":
        # One record per non-empty line; the text becomes `response`.
        return [
            {"response": line.strip()}
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    raise ValueError(f"unsupported file type: {path.suffix} ({path})")


def load_dataset(
    path: str | Path,
    *,
    source: Optional[str] = None,
    split: Optional[str] = None,
    generation: Optional[int] = None,
) -> list[Sample]:
    """Load and normalize a dataset file into a list of validated Samples."""
    path = Path(path)
    stem = path.stem
    return [
        normalize_record(
            raw, i, stem=stem, source=source, split=split, generation=generation
        )
        for i, raw in enumerate(_read_raw(path))
    ]


def write_jsonl(samples: Iterable[Sample], path: str | Path) -> Path:
    """Write Samples to a JSONL file (the canonical on-disk bench format)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
    return path
