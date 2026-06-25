"""Scoring-engine interface and shared data types.

A :class:`ScoringEngine` turns prepared GT/prediction DataFrames (:class:`ScoringInputs`)
into an :class:`EvaluationResult`. :class:`~metrics.evaluation.Evaluation` owns
data loading, prediction generation and split selection, then delegates the
actual scoring to one engine — :class:`~metrics.engines.native.NativeEngine`
(custom pipeline) or :class:`~metrics.engines.backend.BackendEngine` (external
library) — and copies the result onto its public attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..types import DetectionMetrics, Metrics, PredictMatch


@dataclass(frozen=True)
class ScoringInputs:
    """Prepared inputs for a single scoring run.

    ``gt_df`` is the ground truth for the evaluation split (already selected and,
    when needed, predicted). ``preds_df`` is the preprocessed predictions (used by
    the native pipeline); ``raw_preds_df`` is the unpreprocessed predictions (used
    for mAP and by the external backends). ``split_df`` is the full ground truth
    (all splits), needed for calibration. ``split`` is the split name (for logs).
    """

    gt_df: pd.DataFrame
    preds_df: pd.DataFrame
    raw_preds_df: pd.DataFrame
    split_df: pd.DataFrame
    split: str
    find_best_confs: bool
    calibration_split: str | None


@dataclass
class EvaluationResult:
    """Everything a scoring run produces, copied onto ``Evaluation`` afterwards.

    ``detection_metrics`` is populated only by the external backends; ``matches`` /
    ``unfiltered_matches`` only by the native engine. ``cm`` is ``None`` when the
    engine produces no confusion matrix (the ``torchmetrics`` backend).
    """

    metrics: dict[str, Metrics]
    best_confidences: dict[str, float]
    cm: npt.NDArray[np.int64] | None
    class_labels: list[str]
    detection_metrics: dict[str, DetectionMetrics] = field(default_factory=dict)
    matches: dict[str, list[PredictMatch]] = field(default_factory=dict)
    unfiltered_matches: dict[str, list[PredictMatch]] = field(default_factory=dict)


class ScoringEngine(Protocol):
    """A pluggable metrics engine selected by ``Evaluation`` from ``backend``."""

    def resolve_calibration_split(self, calibration_split: str | None) -> str | None:
        """Return the calibration split this engine will actually honour.

        Called before prediction generation so the auto-predict path covers the
        right splits. An engine that cannot calibrate on a split returns ``None``
        (logging why); otherwise it returns the split unchanged.
        """
        ...

    def run(self, inputs: ScoringInputs) -> EvaluationResult:
        """Score ``inputs`` and return the full result bundle."""
        ...
