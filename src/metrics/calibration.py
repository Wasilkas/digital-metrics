"""Confidence-threshold calibration for the native evaluation pipeline.

The :class:`ConfidenceCalibrator` owns the logic for choosing per-class (or
global, YOLO-style) confidence thresholds: in-sample optimisation from a set of
match records, and out-of-sample calibration on a held-out split. It is a
config-only collaborator — all DataFrames are passed in — so it stays pure and
independently testable. :class:`~metrics.evaluation.Evaluation` builds one at
construction and delegates to it.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from .matching import MatchingStrategy, match_boxes
from .scoring import (
    ConfidenceOptimization,
    find_best_confidences,
    find_best_global_confidence,
)
from .types import PredictMatch


class ConfidenceCalibrator:
    """Choose confidence thresholds for the native evaluation pipeline."""

    def __init__(
        self,
        *,
        classes: list[str],
        iou_threshold: float,
        matching_strategy: MatchingStrategy,
        confidence_optimization: ConfidenceOptimization,
    ) -> None:
        """Initialise the calibrator.

        Args:
            classes: Class vocabulary to optimise thresholds over.
            iou_threshold: IoU threshold for box matching during calibration.
            matching_strategy: Box-matching strategy ("iou_prior", "greedy",
                "hungarian").
            confidence_optimization: ``"per_class"`` (independent threshold per
                class) or ``"global"`` (single YOLO-style threshold shared by all
                classes).
        """
        self._classes = classes
        self._iou_threshold = iou_threshold
        self._matching_strategy = matching_strategy
        self._confidence_optimization = confidence_optimization

    def validate_calibration_gt(
        self,
        split_df: pd.DataFrame,
        calibration_split: str,
        eval_gt: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return the validated ground truth for *calibration_split*.

        Shared by the native and backend calibration paths. ``eval_gt`` (the
        evaluation split) must be passed so leakage can be detected.

        Args:
            split_df: Full ground-truth DataFrame (all splits).
            calibration_split: Split value to calibrate on (e.g. ``"val"``).
            eval_gt: Ground truth for the evaluation split, used for leak detection.

        Raises:
            ValueError: If split_df has no "split" column, the calibration split
                has no rows, or it shares an ``image_name`` with the evaluation
                split (which would leak calibration data into the evaluation).
        """
        if "split" not in split_df.columns:
            raise ValueError(
                f"calibration_split={calibration_split!r} requires split_df to have "
                "a 'split' column, but none was found."
            )
        cal_gt = split_df[split_df["split"] == calibration_split]
        if cal_gt.empty:
            available = split_df["split"].unique().tolist()
            raise ValueError(
                f"No ground-truth rows found for calibration split {calibration_split!r}. "
                f"Available splits: {available}"
            )

        overlap = set(cal_gt["image_name"]) & set(eval_gt["image_name"])
        if overlap:
            sample = sorted(overlap)[:5]
            raise ValueError(
                f"Calibration split {calibration_split!r} shares "
                f"{len(overlap)} image_name(s) with the evaluation split "
                f"(e.g. {sample}). Predictions are matched to ground truth via "
                "image_name, so overlapping images would leak calibration data "
                "into the evaluation. Fix the 'split' labels in split_df so each "
                "image_name belongs to exactly one split."
            )
        return cal_gt

    def calibrate_native(
        self,
        split_df: pd.DataFrame,
        calibration_split: str,
        eval_gt: pd.DataFrame,
        preds_df: pd.DataFrame,
    ) -> dict[str, float]:
        """Find confidence thresholds on *calibration_split* (out-of-sample).

        Validates the calibration split, matches predictions against its ground
        truth, then optimises thresholds from those matches.

        Args:
            split_df: Full ground-truth DataFrame (all splits).
            calibration_split: Split value to calibrate on (e.g. ``"val"``).
            eval_gt: Ground truth for the evaluation split (leak detection).
            preds_df: Predictions to match against the calibration ground truth.

        Returns:
            Dict mapping class name → best confidence threshold.

        Raises:
            ValueError: Propagated from :meth:`validate_calibration_gt`.
        """
        cal_gt = self.validate_calibration_gt(split_df, calibration_split, eval_gt)
        cal_image_names = cal_gt["image_name"].unique().tolist()
        logger.info(
            f"Calibrating confidence thresholds on '{calibration_split}' split "
            f"({len(cal_gt)} GT rows)..."
        )
        cal_matches = match_boxes(
            cal_gt,
            preds_df,
            self._iou_threshold,
            strategy=self._matching_strategy,
            split_image_names=cal_image_names,
        )
        thresholds = self.find_from_matches(cal_matches)
        logger.info("Threshold calibration complete.")
        return thresholds

    def find_from_matches(self, matches: dict[str, list[PredictMatch]]) -> dict[str, float]:
        """Choose confidence thresholds from match records (in-sample).

        ``"per_class"`` returns an independent threshold per class;
        ``"global"`` returns the same YOLO-style threshold for every class.
        Warns when the chosen threshold keeps every detection (no effect).
        """
        if self._confidence_optimization == "global":
            threshold = find_best_global_confidence(matches, self._classes)
            thresholds = {c: threshold for c in self._classes}
        else:
            thresholds = find_best_confidences(matches, self._classes)
        self._warn_if_thresholds_unoptimized(matches, thresholds)
        return thresholds

    def _warn_if_thresholds_unoptimized(
        self, matches: dict[str, list[PredictMatch]], thresholds: dict[str, float]
    ) -> None:
        """Warn when an optimised threshold keeps every detection.

        A threshold equal to the minimum prediction confidence does not discard
        anything, so confidence optimisation had no effect — typically because the
        predictions match the ground truth so well that the optimal cut is none
        (e.g. identical pred/GT boxes, where the F1-optimal threshold is the floor).
        """

        def min_confidence(records: list[PredictMatch]) -> float | None:
            confs = [m.confidence for m in records if m.type != "FN"]
            return min(confs) if confs else None

        if self._confidence_optimization == "global":
            all_confs = [m.confidence for recs in matches.values() for m in recs if m.type != "FN"]
            if not all_confs:
                return
            global_min = min(all_confs)
            threshold = next(iter(thresholds.values()), 0.0)
            if threshold <= global_min:
                logger.warning(
                    f"Global confidence threshold ({threshold:.6g}) equals the minimum "
                    f"prediction confidence ({global_min:.6g}); it keeps every detection, "
                    "so confidence optimisation had no effect (predictions may match GT "
                    "closely)."
                )
            return

        for c in self._classes:
            mc = min_confidence(matches.get(c, []))
            if mc is None:
                continue
            if thresholds.get(c, 0.0) <= mc:
                logger.warning(
                    f"Confidence threshold for class '{c}' ({thresholds[c]:.6g}) equals the "
                    f"minimum prediction confidence ({mc:.6g}); it keeps every detection, so "
                    "confidence optimisation had no effect for this class."
                )
