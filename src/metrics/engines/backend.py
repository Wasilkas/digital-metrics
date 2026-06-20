"""Backend scoring engine: scores via an external library (ultralytics/torchmetrics)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger

from ..backends import (
    Backend,
    compute_detection_metrics,
    compute_torchmetrics_metrics,
    compute_ultralytics_confusion_matrix,
    compute_ultralytics_metrics,
    find_torchmetrics_confidence,
    find_ultralytics_confidence,
)
from ..calibration import ConfidenceCalibrator
from ..scoring import ConfidenceOptimization
from ..types import DetectionMetrics, Metrics
from ..validation import validate_dataframes
from .base import EvaluationResult, ScoringInputs


class BackendEngine:
    """Score a split through an external metrics library, adapted to native ``Metrics``.

    The raw backend output is kept as ``detection_metrics``; ``metrics`` holds the
    same numbers reconstructed onto :class:`Metrics` so the dashboards/CI plots keep
    working. Both backends support calibration on a held-out split. The
    ``"ultralytics"`` backend also fills the confusion matrix; ``"torchmetrics"``
    has none.
    """

    def __init__(
        self,
        *,
        backend: Backend,
        classes: list[str],
        confidence_optimization: ConfidenceOptimization,
        calibrator: ConfidenceCalibrator,
    ) -> None:
        self._backend = backend
        self._classes = classes
        self._confidence_optimization = confidence_optimization
        self._calibrator = calibrator

    def resolve_calibration_split(self, calibration_split: str | None) -> str | None:
        """Both backends honour a calibration split, so this is a no-op."""
        return calibration_split

    def run(self, inputs: ScoringInputs) -> EvaluationResult:
        gt_df = inputs.gt_df
        raw_preds_df = inputs.raw_preds_df
        # Backends score the raw predictions (YOLO val style); no conf/NMS preprocessing.
        validate_dataframes(raw_preds_df, gt_df, self._classes)
        split_image_names = gt_df["image_name"].unique().tolist()

        best_confidences = {c: 0.0 for c in self._classes}
        if inputs.calibration_split is None:
            logger.info(
                f"Computing metrics with the '{self._backend}' backend on split '{inputs.split}'..."
            )
            detection_metrics = compute_detection_metrics(
                gt_df,
                raw_preds_df,
                backend=self._backend,
                classes=self._classes,
                split_image_names=split_image_names,
            )
        else:
            detection_metrics, best_confidences = self._calibrate(inputs, split_image_names)

        metrics = self._adapt(detection_metrics, gt_df)

        cm: npt.NDArray[np.int64] | None
        if self._backend == "ultralytics":
            cm, class_labels = compute_ultralytics_confusion_matrix(
                gt_df,
                raw_preds_df,
                classes=self._classes,
                split_image_names=split_image_names,
            )
        else:
            cm, class_labels = None, []

        return EvaluationResult(
            metrics=metrics,
            best_confidences=best_confidences,
            cm=cm,
            class_labels=class_labels,
            detection_metrics=detection_metrics,
        )

    def _calibrate(
        self, inputs: ScoringInputs, split_image_names: list[str]
    ) -> tuple[dict[str, DetectionMetrics], dict[str, float]]:
        """Backend metrics with the operating point calibrated on a split.

        Finds the F1-optimal confidence on ``calibration_split`` (per the configured
        ``confidence_optimization`` mode), then reads the eval split's P/R/F1 at that
        confidence while AP stays over the full curve, and returns the chosen
        threshold(s) as ``best_confidences``. Works for both backends — each reads
        P/R/F1 off its own confidence curve.
        """
        assert inputs.calibration_split is not None
        gt_df = inputs.gt_df
        raw_preds_df = inputs.raw_preds_df
        cal_gt = self._calibrator.validate_calibration_gt(
            inputs.split_df, inputs.calibration_split, gt_df
        )
        cal_image_names = cal_gt["image_name"].unique().tolist()
        logger.info(
            f"Calibrating '{self._backend}' confidence on '{inputs.calibration_split}' "
            f"({len(cal_gt)} GT rows, mode={self._confidence_optimization})..."
        )
        find_confidence = (
            find_ultralytics_confidence
            if self._backend == "ultralytics"
            else find_torchmetrics_confidence
        )
        compute_metrics = (
            compute_ultralytics_metrics
            if self._backend == "ultralytics"
            else compute_torchmetrics_metrics
        )
        conf = find_confidence(
            cal_gt,
            raw_preds_df,
            classes=self._classes,
            split_image_names=cal_image_names,
            mode=self._confidence_optimization,
        )
        if isinstance(conf, dict):
            best_confidences = {c: conf.get(c, 0.0) for c in self._classes}
        else:
            best_confidences = {c: conf for c in self._classes}

        detection_metrics = compute_metrics(
            gt_df,
            raw_preds_df,
            classes=self._classes,
            split_image_names=split_image_names,
            conf_threshold=conf,
        )
        return detection_metrics, best_confidences

    def _adapt(
        self, detection_metrics: dict[str, DetectionMetrics], gt_df: pd.DataFrame
    ) -> dict[str, Metrics]:
        """Map external ``DetectionMetrics`` onto native ``Metrics`` for the dashboards.

        The backends report only precision/recall/f1 and AP at a self-selected
        operating point — no box-level TP/FP/FN. We reconstruct float counts from
        the per-class ground-truth size ``N`` (= TP + FN, known from ``gt_df``) so
        the reproduced precision/recall/f1 equal the backend's exactly::

            TP = recall * N        FN = N - TP        FP = TP * (1 - p) / p   (p > 0)

        Wilson CIs then follow from these counts — the recall CI is grounded in the
        true ``N``; the precision CI is approximate because FP is reconstructed, not
        counted. ``cohen_kappa`` is set to ``-1`` (not provided by external
        backends) and the confidence threshold to ``0.0`` (the operating point is
        internal to the backend). Classes with no GT in the split get NaN AP,
        matching the native convention.
        """
        gt_counts = gt_df["instance_label"].value_counts().to_dict()
        result: dict[str, Metrics] = {}
        for c in self._classes:
            n_gt = int(gt_counts.get(c, 0))
            dm = detection_metrics.get(c)
            if dm is None or n_gt == 0:
                result[c] = Metrics(
                    ap50=float("nan"),
                    ap75=float("nan"),
                    ap50_95=float("nan"),
                    cohen_kappa=-1,
                )
                continue
            tp = dm.recall * n_gt
            fn = n_gt - tp
            fp = tp * (1.0 - dm.precision) / dm.precision if dm.precision > 0 else 0.0
            result[c] = Metrics(
                tp=tp,
                fp=fp,
                fn=fn,
                ap50=dm.ap50,
                ap75=dm.ap75,
                ap50_95=dm.ap50_95,
                cohen_kappa=-1,
            )
        return result
