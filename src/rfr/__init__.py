"""Rank from References (RFR) package."""

from .core import CandidateResult, evaluate_candidates
from .io import Dataset, load_long_table

__all__ = ["CandidateResult", "evaluate_candidates", "Dataset", "load_long_table"]
