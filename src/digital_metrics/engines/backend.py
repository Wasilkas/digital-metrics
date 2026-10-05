"""Backend scoring engine: scores via an external library (ultralytics/torchmetrics)."""

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
    backend results adapted onto :class:`Metrics` for reporting, with observed
    counts and intervals only when explicit thresholds are used. Both backends
    support calibration on a held-out split. The
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
        validate_dataframes(raw_preds_df, gt_df)
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
        for name, metric in metrics.items():
            metric.confidence = best_confidences.get(name, 0.0)

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

        Selects realizable thresholds on the calibration split, then reports
        exact retained TP/FP/FN and P/R/F1 on the evaluation split. Scalar global
        calibration and independent COCO per-class calibration are exact on the
        observed-score grid. Mixed-class Ultralytics per-class calibration finds
        a deterministic coordinate-local realized macro-F1 optimum; it does not
        enumerate the joint threshold grid. AP uses the complete predictions.
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

        Explicit thresholds carry observed counts and Wilson intervals.
        Unthresholded summaries reconstruct compatibility counts from recall,
        precision and the GT size; those counts and missing summaries are marked
        unavailable with ``counts_observed=False`` and NaN Wilson intervals.
        ``cohen_kappa`` is unavailable (-1); ``run`` copies calibrated thresholds
        into each resulting metric. Classes without GT retain NaN AP.
        """
        gt_counts = gt_df["instance_label"].value_counts().to_dict()
        result: dict[str, Metrics] = {}
        for c in self._classes:
            n_gt = int(gt_counts.get(c, 0))
            dm = detection_metrics.get(c)
            if dm is None:
                result[c] = Metrics(
                    counts_observed=False,
                    ap50=float("nan"),
                    ap75=float("nan"),
                    ap50_95=float("nan"),
                    cohen_kappa=-1,
                )
                continue
            tp = dm.tp if dm.tp is not None else dm.recall * n_gt
            fn = dm.fn if dm.fn is not None else n_gt - tp
            fp = (
                dm.fp
                if dm.fp is not None
                else (tp * (1.0 - dm.precision) / dm.precision if dm.precision > 0 else 0.0)
            )
            result[c] = Metrics(
                counts_observed=dm.tp is not None and dm.fp is not None and dm.fn is not None,
                tp=tp,
                fp=fp,
                fn=fn,
                ap50=dm.ap50,
                ap75=dm.ap75,
                ap50_95=dm.ap50_95,
                cohen_kappa=-1,
            )
        return result
