"""Normalize base corpora (WikiText-2, TinyStories, ...) into the Sample schema.

Config-driven: a YAML file lists each Hugging Face dataset, its splits, the text
field, and an optional row cap. Every row becomes a canonical
:class:`~src.ingestion.schema.Sample` (``source=human``, ``generation=0``) written
to ``{output_dir}/{name}_{split}.jsonl`` through the shared loader.

    python -m src.ingestion.normalize_corpora --config configs/corpora.yaml

``datasets`` and ``pyyaml`` are imported lazily so this module — and its unit
tests — load without a network or those packages installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Optional

from .loaders import normalize_record, write_jsonl
from .schema import Sample


def _normalize_rows(
    rows: Iterable[Any],
    *,
    name: str,
    split: str,
    source: str = "human",
    text_field: str = "text",
    max_rows: Optional[int] = None,
) -> list[Sample]:
    """Turn raw dataset rows into validated Samples (pure — no I/O).

    Blank/whitespace-only rows are skipped, since raw corpora such as WikiText
    are full of empty lines. Text becomes ``response`` so these land as
    response-only records. Ids are deterministic (``{name}_{split}-{index}``),
    keyed on the *source row index* so a given corpus always yields the same
    ids on re-run.
    """
    samples: list[Sample] = []
    for index, row in enumerate(rows):
        raw = row.get(text_field) if isinstance(row, dict) else row
        text = ("" if raw is None else str(raw)).strip()
        if not text:
            continue
        samples.append(
            normalize_record(
                {"response": text},
                index,
                stem=f"{name}_{split}",
                source=source,
                split=split,
                generation=0,
            )
        )
        if max_rows is not None and len(samples) >= max_rows:
            break
    return samples


def load_config(path: str | Path) -> dict[str, Any]:
    """Read the corpora YAML config."""
    import yaml  # lazy: keeps this module importable without pyyaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def normalize_dataset(
    entry: dict[str, Any], output_dir: str | Path
) -> dict[str, tuple[Path, int]]:
    """Load one configured HF dataset and write a JSONL file per split."""
    from datasets import load_dataset as hf_load  # lazy: heavy optional dep

    results: dict[str, tuple[Path, int]] = {}
    for split in entry["splits"]:
        rows = hf_load(entry["hf_path"], entry.get("hf_config"), split=split)
        samples = _normalize_rows(
            rows,
            name=entry["name"],
            split=split,
            source=entry.get("source", "human"),
            text_field=entry.get("text_field", "text"),
            max_rows=entry.get("max_rows"),
        )
        out_path = Path(output_dir) / f"{entry['name']}_{split}.jsonl"
        write_jsonl(samples, out_path)
        results[split] = (out_path, len(samples))
    return results


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Normalize base corpora into the ORIGIN-T Sample schema."
    )
    parser.add_argument(
        "--config", default="configs/corpora.yaml", help="path to the corpora YAML config"
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    output_dir = config.get("output_dir", "data/processed")
    for entry in config["datasets"]:
        for split, (path, count) in normalize_dataset(entry, output_dir).items():
            print(f"{entry['name']}/{split}: {count} samples -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
