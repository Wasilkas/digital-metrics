from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..grouping import image_row_indices
from ..types import PredictMatch
from .assignment import (
    MatchedPairs,
    assign_greedy,
    assign_hungarian,
    assign_iou_prior,
)
from .iou import compute_iou_matrix

MatchingStrategy = Literal["greedy", "hungarian", "iou_prior"]

_BBOX_COLS = ["bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]


def _resolve_matching_scope(
    gt_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    split_image_names: list[str] | None,
) -> tuple[pd.DataFrame, list[str] | npt.NDArray[np.str_]]:
    """Return (scoped preds, images to iterate) for the matching loop.

    When split_image_names is provided, preds are filtered to that list and
    the iteration set is the union of split_image_names and gt_df images
    (so no GT image is ever skipped).  When None, preds are unfiltered and
    only images present in gt_df are iterated.
    """
    if split_image_names is not None:
        scoped_preds = preds_df[preds_df["image_name"].isin(split_image_names)]
        all_images: list[str] | npt.NDArray[np.str_] = list(
            set(split_image_names) | set(gt_df["image_name"].unique())
        )
        return scoped_preds, all_images
    return preds_df, gt_df["image_name"].unique()


def match_boxes(
    gt_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    iou_threshold: float,
    strategy: MatchingStrategy = "greedy",
    split_image_names: list[str] | None = None,
) -> dict[str, list[PredictMatch]]:
    """Match ground-truth and prediction boxes per image.

    Args:
        gt_df: Ground-truth DataFrame with standard schema.
        preds_df: Predictions DataFrame with standard schema.
        iou_threshold: Minimum IoU required to consider a match valid.
        strategy: "greedy" (confidence-sorted, YOLO-style), "iou_prior"
            (IoU-sorted, label-aware) or "hungarian" (globally optimal
            assignment via linear_sum_assignment).
        split_image_names: Complete list of image names in the split,
            including empty images (no GT boxes).  Predictions on empty
            images are counted as FPs.  When None, only images present
            in gt_df are processed.

    Returns:
        Dict mapping class name → list of PredictMatch objects.
    """
    matches: dict[str, list[PredictMatch]] = {}
    preds_df, all_images = _resolve_matching_scope(gt_df, preds_df, split_image_names)

    # Empty images carry placeholder GT rows (None label, NaN bbox coords)
    # purely to keep the image in scope so predictions on it count as FPs.
    # Drop them once so they never form phantom GT boxes/FNs and so the IoU
    # matrix columns stay aligned with the gt rows used in _build_matches.
    gt_df = gt_df.dropna(subset=_BBOX_COLS)

    # Extract every column to numpy once over the whole frame, then slice each
    # image's rows by positional index inside the loop.  Per-image DataFrame
    # access (groupby chopping + df[cols]) is the dominant cost otherwise.
    gt_bbox = gt_df[_BBOX_COLS].to_numpy(np.float32)
    gt_label = gt_df["instance_label"].to_numpy(dtype=object)
    gt_index = gt_df.index.to_numpy()
    pred_bbox = preds_df[_BBOX_COLS].to_numpy(np.float32)
    pred_label = preds_df["instance_label"].to_numpy(dtype=object)
    pred_index = preds_df.index.to_numpy()
    pred_conf = preds_df["confidence"].to_numpy(np.float64)

    gt_rows = image_row_indices(gt_df)
    pred_rows = image_row_indices(preds_df)
    empty = np.empty(0, dtype=np.intp)

    for image_name in all_images:
        g = gt_rows.get(image_name, empty)
        p = pred_rows.get(image_name, empty)

        if strategy == "greedy" and p.size > 1:
            # Greedy consumes the highest-IoU GT in confidence order, so order
            # rows by confidence descending; stable keeps ties in row order.
            p = p[np.argsort(-pred_conf[p], kind="stable")]

        iou_matrix = compute_iou_matrix(pred_bbox[p], gt_bbox[g])

        g_label, p_label = gt_label[g], pred_label[p]
        pairs = _assign(strategy, iou_matrix, iou_threshold, p_label, g_label)

        # iou_prior records the closest cross-class GT for unmatched preds so
        # the confusion matrix captures label confusions; greedy and hungarian
        # always book an unmatched prediction as "background".
        _build_matches(
            matches,
            iou_matrix,
            pairs,
            iou_threshold,
            g_label,
            gt_index[g],
            p_label,
            pred_index[p],
            pred_conf[p],
            cross_class_fp=strategy == "iou_prior",
        )

    return matches


def _assign(
    strategy: MatchingStrategy,
    iou_matrix: npt.NDArray[np.float64],
    iou_threshold: float,
    pred_labels: npt.NDArray[np.object_],
    gt_labels: npt.NDArray[np.object_],
) -> MatchedPairs:
    """Dispatch to the geometric assignment kernel for the given strategy."""
    if strategy == "hungarian":
        return assign_hungarian(iou_matrix, iou_threshold)
    if strategy == "iou_prior":
        label_match = pred_labels[:, None] == gt_labels[None, :]
        return assign_iou_prior(iou_matrix, iou_threshold, valid_mask=label_match)
    return assign_greedy(iou_matrix, iou_threshold)


def _build_matches(
    matches: dict[str, list[PredictMatch]],
    iou_matrix: npt.NDArray[np.float64],
    pairs: MatchedPairs,
    iou_threshold: float,
    gt_labels: npt.NDArray[np.object_],
    gt_indices: npt.NDArray[np.int64],
    pred_labels: npt.NDArray[np.object_],
    pred_indices: npt.NDArray[np.int64],
    pred_confs: npt.NDArray[np.float64],
    *,
    cross_class_fp: bool,
) -> None:
    """Turn positional (pred, gt) pairs into PredictMatch records in place.

    The label/index/confidence arrays are this image's rows, already sliced from
    the whole-frame columns (and reordered to match ``iou_matrix`` row order).
    Matched predictions record the actual GT label (label mismatch is left for
    PredictMatch.type to classify as FP).  Unmatched predictions become FPs
    against "background"; when ``cross_class_fp`` is set, the closest GT label
    is recorded instead if it overlaps (IoU >= threshold) with a different
    class.  Any GT left unmatched becomes an FN.
    """
    n_preds = len(pred_labels)
    n_gts = len(gt_labels)
    pred_to_gt = dict(pairs)
    matched_gts = {gt_j for _, gt_j in pairs}

    for pred_pos in range(n_preds):
        pred_label = str(pred_labels[pred_pos])
        match_iou: float | None = None

        if pred_pos in pred_to_gt:
            gt_pos = pred_to_gt[pred_pos]
            gt_label = str(gt_labels[gt_pos])
            gt_index = int(gt_indices[gt_pos])
            match_iou = float(iou_matrix[pred_pos, gt_pos])
        else:
            gt_label = "background"
            gt_index = -1
            if cross_class_fp and n_gts > 0:
                best_gt_j = int(np.argmax(iou_matrix[pred_pos]))
                best_iou = float(iou_matrix[pred_pos, best_gt_j])
                closest_label = str(gt_labels[best_gt_j])
                if best_iou >= iou_threshold and closest_label != pred_label:
                    gt_label = closest_label
                    gt_index = int(gt_indices[best_gt_j])
                    match_iou = best_iou

        matches.setdefault(pred_label, []).append(
            PredictMatch(
                pred_label=pred_label,
                gt_label=gt_label,
                pred_index=int(pred_indices[pred_pos]),
                gt_index=gt_index,
                confidence=float(pred_confs[pred_pos]),
                iou=match_iou,
            )
        )

    for gt_pos in range(n_gts):
        if gt_pos in matched_gts:
            continue
        gt_label = str(gt_labels[gt_pos])
        matches.setdefault(gt_label, []).append(
            PredictMatch(
                pred_label="background",
                gt_label=gt_label,
                pred_index=-1,
                gt_index=int(gt_indices[gt_pos]),
                confidence=0.0,
            )
        )
