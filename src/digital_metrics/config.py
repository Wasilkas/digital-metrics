"""Grouped configuration objects for the :class:`~metrics.evaluation.Evaluation` constructor.

The flat keyword arguments on ``Evaluation`` remain fully supported; these
dataclasses are an optional, tidier alternative for callers who would rather pass
a few grouped configs than the dozen-plus flat kwargs. Each group, when passed,
supplies all of its fields and takes precedence over the corresponding flat
kwargs (which are then ignored for that group). Their defaults mirror the flat
defaults exactly, so ``Evaluation(preds, split)`` and
``Evaluation(preds, split, scoring=ScoringConfig())`` behave identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .matching import MatchingStrategy
from .scoring import APMethod, ConfidenceOptimization


@dataclass
class ScoringConfig:
    """Matching and metric-computation options.

    Mirrors the flat kwargs ``iou_threshold``, ``matching_strategy``,
    ``ap_method``, ``confidence_optimization`` and ``skip_cohen_kappa``.
    """

    iou_threshold: float = 0.5
    matching_strategy: MatchingStrategy = "iou_prior"
    ap_method: APMethod = "interp"
    confidence_optimization: ConfidenceOptimization = "per_class"
    skip_cohen_kappa: bool = True


@dataclass
class PreprocessConfig:
    """Ground-truth dedup and predictions preprocessing thresholds.

    Mirrors the flat kwargs ``preprocess`` (as ``dedup_gt``),
    ``preprocess_preds_conf_threshold`` (as ``conf_threshold``) and the two NMS
    thresholds. ``None`` disables a suppression type.
    """

    dedup_gt: bool = False
    conf_threshold: float | None = None
    nms_containment_threshold: float | None = None
    nms_iou_threshold: float | None = None


@dataclass
class InferenceConfig:
    """YOLO auto-prediction options for the ``weights_path`` flow.

    Mirrors the flat kwargs ``weights_path`` and ``predict_kwargs``.
    """

    weights_path: str | None = None
    predict_kwargs: dict[str, Any] | None = None
