"""ORIGIN-T data ingestion: canonical schema + dataset loaders."""

from .schema import SOURCE_TYPES, SPLITS, Sample
from .loaders import load_dataset, normalize_record, write_jsonl
from .holdout import HOLDOUT_SPLIT, carve_holdout, is_holdout
from .check_leakage import LeakReport, check_files, find_leaks

__all__ = [
    "Sample",
    "SOURCE_TYPES",
    "SPLITS",
    "load_dataset",
    "normalize_record",
    "write_jsonl",
    "HOLDOUT_SPLIT",
    "carve_holdout",
    "is_holdout",
    "LeakReport",
    "check_files",
    "find_leaks",
]
