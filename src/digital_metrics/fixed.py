"""Fixed native evaluation without calibration, preprocessing or application policy."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from .engines import compute_metrics_from_matches
from .matching import MatchingStrategy, match_boxes
from .scoring import APMethod, get_confusion_matrix, slice_by_conf
from .scoring.ap import PRObservation, _compute_map_with_observations, _ordered_observations
from .types import Metrics, PredictMatch
from .validation import (
    BBOX_COLS,
    normalize_image_ids,
    validate_dataframes,
    validate_iou,
    validate_option,
)


@dataclass(frozen=True)
class FixedEvaluationResult:
    """Exact fixed-threshold evidence plus independent unfiltered AP observations."""

    metrics: dict[str, Metrics]
    matches: dict[str, list[PredictMatch]]
    unfiltered_matches: dict[str, list[PredictMatch]]
    cm: npt.NDArray[np.int64]
    class_labels: list[str]
    gt_status: pd.DataFrame
    pred_status: pd.DataFrame
    pr_curves: list[PRObservation]


def evaluate_fixed(
    gt_df: pd.DataFrame,
    prepared_predictions: pd.DataFrame,
    ap_predictions: pd.DataFrame,
    *,
    classes: list[str],
    thresholds: Mapping[str, float],
    split_image_names: list[str] | None = None,
    iou_threshold: float = 0.5,
    matching_strategy: MatchingStrategy = "greedy",
    ap_method: APMethod = "interp",
    skip_cohen_kappa: bool = True,
) -> FixedEvaluationResult:
    """Score frozen thresholds with distinct fixed and unfiltered AP populations.

    The caller supplies scoped truth, geometry-valid unfiltered AP predictions
    and optionally preprocessed fixed predictions. No calibration, label
    restoration, unknown-class policy or source lineage is applied. Missing
    thresholds retain the existing slice_by_conf default of zero. Inputs and
    their row IDs are preserved; output statuses join using those row IDs.
    """
    validate_iou(iou_threshold)
    validate_option("matching_strategy", matching_strategy, ("greedy", "iou_prior", "hungarian"))
    validate_option("ap_method", ap_method, ("interp", "continuous"))
    if not skip_cohen_kappa:
        raise ValueError("Fixed evaluation currently requires skip_cohen_kappa=True")
    for name, threshold in thresholds.items():
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"Threshold for {name!r} must be finite and in [0, 1]")
    validate_dataframes(prepared_predictions, gt_df)
    validate_dataframes(ap_predictions, gt_df)
    scope = pd.DataFrame(
        {
            "image_name": (
                gt_df["image_name"].unique().tolist()
                if split_image_names is None
                else split_image_names
            )
        }
    )
    truth, predictions, ap_preds, scope = normalize_image_ids(
        gt_df, prepared_predictions, ap_predictions, scope
    )
    image_names = scope["image_name"].tolist()
    truth = truth[truth["image_name"].isin(image_names)]
    unfiltered_matches = match_boxes(
        truth,
        predictions,
        iou_threshold,
        strategy=matching_strategy,
        split_image_names=image_names,
    )
    normalized = dict(thresholds)
    sliced = slice_by_conf(unfiltered_matches, classes, normalized)
    metrics = compute_metrics_from_matches(sliced, classes, normalized)
    observations = _compute_map_with_observations(
        truth.dropna(subset=BBOX_COLS),
        ap_preds,
        metrics,
        image_names,
        ap_method,
        matching_strategy,
    )
    cm, labels = get_confusion_matrix(sliced, classes)
    gt_status, pred_status = _statuses(truth, predictions, classes, sliced)
    return FixedEvaluationResult(
        metrics=metrics,
        matches=sliced,
        unfiltered_matches=unfiltered_matches,
        cm=cm,
        class_labels=labels,
        gt_status=gt_status,
        pred_status=pred_status,
        pr_curves=_ordered_observations(observations, classes, ap_method),
    )


def _statuses(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    classes: list[str],
    matches: dict[str, list[PredictMatch]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detected = {m.gt_index for records in matches.values() for m in records if m.type == "TP"}
    scored_gt = truth.dropna(subset=BBOX_COLS)
    scored_gt = scored_gt[scored_gt["instance_label"].isin(classes)]
    gt_status = pd.DataFrame(
        {
            "gt_index": scored_gt.index.to_numpy(),
            "image_name": scored_gt["image_name"].to_numpy(),
            "instance_label": scored_gt["instance_label"].to_numpy(),
            "detected": scored_gt.index.isin(list(detected)),
        }
    )
    # Prediction identity is its supplied index; keep caller row order. This
    # accepts the legacy library's arbitrary hashable IDs without coercion.
    surviving = {
        m.pred_index: m.type == "TP"
        for records in matches.values()
        for m in records
        if m.pred_index != -1
    }
    scored_preds = predictions.loc[predictions.index.isin(surviving)]
    pred_status = pd.DataFrame(
        {
            "pred_index": scored_preds.index.to_numpy(),
            "image_name": scored_preds["image_name"].to_numpy(),
            "instance_label": scored_preds["instance_label"].to_numpy(),
            "is_tp": [surviving[index] for index in scored_preds.index],
        }
    )
    return gt_status.astype({"detected": bool}), pred_status.astype({"is_tp": bool})
