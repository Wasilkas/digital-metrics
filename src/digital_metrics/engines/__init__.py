"""Scoring engines: pluggable metrics pipelines selected by ``Evaluation``."""

from .backend import BackendEngine
from .base import EvaluationResult, ScoringEngine, ScoringInputs
from .native import NativeEngine, compute_metrics_from_matches

__all__ = [
    "BackendEngine",
    "EvaluationResult",
    "NativeEngine",
    "ScoringEngine",
    "ScoringInputs",
    "compute_metrics_from_matches",
]
