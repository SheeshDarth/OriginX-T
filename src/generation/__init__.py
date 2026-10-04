"""ORIGIN-T contamination generation: mixing and generators."""

from .mixer import mix, ratio_sweep
from .benchmark_near import make_near_duplicates, perturb, similarity
from .model_based import hf_generator, make_paraphrased, make_synthetic

__all__ = [
    "mix", "ratio_sweep", "make_near_duplicates", "perturb", "similarity",
    "make_synthetic", "make_paraphrased", "hf_generator",
]
