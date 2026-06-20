"""Native scoring engine: custom matching, calibration, mAP, kappa, CM."""

from __future__ import annotations

import numpy as np
from loguru import logger
from tqdm import tqdm

from ..calibration import ConfidenceCalibrator
from ..matching import MatchingStrategy, match_boxes
from ..scoring import APMethod, compute_kappa, compute_map, get_confusion_matrix, slice_by_conf
from ..types import Metrics, PredictMatch
from ..validation import validate_dataframes
from .base import EvaluationResult, ScoringInputs

_BBOX_COLS = ["bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]


def compute_metrics_from_matches(
    matches: dict[str, list[PredictMatch]],
    classes: list[str],
    best_confidences: dict[str, float],
) -> dict[str, Metrics]:
    """Tally TP/FP/FN per class from match records into ``Metrics`` objects."""
    result: dict[str, Metrics] = {}
    for c in classes:
        m = Metrics()
        m.confidence = best_confidences.get(c, 0.0)
        for match in matches.get(c, []):
            pred_type = match.type.lower()
            setattr(m, pred_type, getattr(m, pred_type) + 1)
        result[c] = m
    return result


class NativeEngine:
    """This library's own evaluation pipeline (the default ``backend=None``)."""

    def __init__(
        self,
        *,
        classes: list[str],
        iou_threshold: float,
        matching_strategy: MatchingStrategy,
        ap_method: APMethod,
        skip_cohen_kappa: bool,
        calibrator: ConfidenceCalibrator,
    ) -> None:
        self._classes = classes
        self._iou_threshold = iou_threshold
        self._matching_strategy = matching_strategy
        self._ap_method = ap_method
        self._skip_cohen_kappa = skip_cohen_kappa
        self._calibrator = calibrator

    def resolve_calibration_split(self, calibration_split: str | None) -> str | None:
        """The native engine calibrates on any split, so this is a no-op."""
        return calibration_split

    def run(self, inputs: ScoringInputs) -> EvaluationResult:
        gt_df = inputs.gt_df
        preds_df = inputs.preds_df
        validate_dataframes(preds_df, gt_df, self._classes)

        split_image_names = gt_df["image_name"].unique().tolist()

        logger.info("Matching boxes...")
        matches = match_boxes(
            gt_df,
            preds_df,
            self._iou_threshold,
            strategy=self._matching_strategy,
            split_image_names=split_image_names,
        )
        logger.info("Matching complete.")

        best_confidences = {c: 0.0 for c in self._classes}
        if inputs.calibration_split is not None:
            best_confidences = self._calibrator.calibrate_native(
                inputs.split_df, inputs.calibration_split, gt_df, preds_df
            )
        elif inputs.find_best_confs:
            logger.info("Finding best confidence thresholds (in-sample)...")
            best_confidences = self._calibrator.find_from_matches(matches)
            logger.info("Best thresholds found.")

        logger.info("Filtering by best confidence thresholds...")
        unfiltered_matches = matches
        sliced = slice_by_conf(matches, self._classes, best_confidences)
        logger.info("Filtering complete.")

        logger.info("Computing metrics and confusion matrix...")
        metrics = compute_metrics_from_matches(sliced, self._classes, best_confidences)
        compute_map(
            gt_df,
            inputs.raw_preds_df,
            metrics,
            split_image_names,
            method=self._ap_method,
            strategy=self._matching_strategy,
        )
        self._compute_cohen_kappa(metrics, inputs)
        cm, class_labels = get_confusion_matrix(sliced, self._classes)
        logger.info("Metrics and confusion matrix computed.")

        return EvaluationResult(
            metrics=metrics,
            best_confidences=best_confidences,
            cm=cm,
            class_labels=class_labels,
            matches=sliced,
            unfiltered_matches=unfiltered_matches,
        )

    def _compute_cohen_kappa(self, metrics: dict[str, Metrics], inputs: ScoringInputs) -> None:
        # Sentinel -1 for every class (including any absent from this split), so
        # the column is uniform when kappa is skipped.
        if self._skip_cohen_kappa:
            for m in metrics.values():
                m.cohen_kappa = -1
            return

        gt_df = inputs.gt_df
        preds_df = inputs.preds_df
        missing = {"image_width", "image_height"} - set(gt_df.columns)
        if missing:
            raise ValueError(
                f"Cohen's kappa needs the optional column(s) {sorted(missing)} in the "
                "ground-truth DataFrame (image pixel dimensions for the masks). Add them "
                "or keep skip_cohen_kappa=True."
            )

        for c in tqdm(
            gt_df["instance_label"].unique(),
            desc="Computing Cohen's Kappa",
            total=gt_df["instance_label"].nunique(),
        ):
            kappas: list[float] = []
            class_gt = gt_df[gt_df["instance_label"] == c]
            preds_gt = preds_df[preds_df["instance_label"] == c]

            for image_name in class_gt["image_name"].unique():
                gt_boxes = class_gt[class_gt["image_name"] == image_name]
                pred_boxes = preds_gt[preds_gt["image_name"] == image_name]
                # Plain (n, 4) arrays: compute_kappa indexes each box positionally.
                gt_box_list = gt_boxes[_BBOX_COLS].to_numpy(np.float64)
                pred_box_list = pred_boxes[_BBOX_COLS].to_numpy(np.float64)
                # Use this image's own dimensions, not the first image's.
                kappa = compute_kappa(
                    gt_box_list,
                    pred_box_list,
                    (int(gt_boxes.iloc[0]["image_width"]), int(gt_boxes.iloc[0]["image_height"])),
                )
                kappas.append(kappa)

            metrics[c].cohen_kappa = float(np.mean(kappas)) if kappas else -1.0
            logger.debug(f"Kappa for {c}: {metrics[c].cohen_kappa}")
