from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .iou import compute_iou_matrix
from .types import PredictMatch

MatchingStrategy = Literal["greedy", "hungarian"]


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
        strategy: "greedy" (confidence-sorted, YOLO-style) or "hungarian"
            (globally optimal assignment via linear_sum_assignment).
        split_image_names: Complete list of image names in the split,
            including empty images (no GT boxes).  Predictions on empty
            images are counted as FPs.  When None, only images present
            in gt_df are processed.

    Returns:
        Dict mapping class name → list of PredictMatch objects.
    """
    if strategy == "greedy":
        return _match_boxes_greedy(gt_df, preds_df, iou_threshold, split_image_names)
    return _match_boxes_hungarian(gt_df, preds_df, iou_threshold, split_image_names)


def _row_index(row: Any) -> int:
    """Return the integer index label of a pandas row."""
    return int(row.name)


def _match_boxes_greedy(
    gt_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    iou_threshold: float,
    split_image_names: list[str] | None = None,
) -> dict[str, list[PredictMatch]]:
    """Greedy (confidence-sorted) box matching — YOLO-compatible."""
    matches: dict[str, list[PredictMatch]] = {}
    preds_df, all_images = _resolve_matching_scope(gt_df, preds_df, split_image_names)

    for image_name in all_images:
        gt = gt_df[gt_df["image_name"] == image_name]
        preds = preds_df[preds_df["image_name"] == image_name]

        preds_sorted = preds.sort_values(by="confidence", ascending=False)

        gt_bboxes = gt[["bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]].dropna().values
        pred_bboxes = preds_sorted[["bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]].values
        iou_matrix = compute_iou_matrix(pred_bboxes, gt_bboxes)

        matched_gt = np.zeros(len(gt_bboxes), dtype=bool)

        for i, (_, pred_row) in enumerate(preds_sorted.iterrows()):
            ious = iou_matrix[i]
            pred_label: str = str(pred_row["instance_label"])

            if len(ious) > 0:
                max_iou_idx = int(np.argmax(ious))
                max_iou = float(ious[max_iou_idx])
            else:
                max_iou_idx = -1
                max_iou = 0.0

            if max_iou >= iou_threshold and max_iou_idx >= 0 and not matched_gt[max_iou_idx]:
                gt_label: str = str(gt.iloc[max_iou_idx]["instance_label"])
                gt_index: int = _row_index(gt.iloc[max_iou_idx])
                # Always consume the GT regardless of label match.
                # Label comparison only determines TP vs FP.
                matched_gt[max_iou_idx] = True
            else:
                gt_label = "background"
                gt_index = -1

            match = PredictMatch(
                pred_label=pred_label,
                gt_label=gt_label,
                pred_index=_row_index(pred_row),
                gt_index=gt_index,
                confidence=float(pred_row["confidence"]),
            )
            matches.setdefault(pred_label, []).append(match)

        # Unmatched GTs become FN entries.
        for i in np.where(~matched_gt)[0]:
            gt_row = gt.iloc[int(i)]
            gt_label = str(gt_row["instance_label"])
            fn_match = PredictMatch(
                pred_label="background",
                gt_label=gt_label,
                pred_index=-1,
                gt_index=_row_index(gt_row),
                confidence=0.0,
            )
            matches.setdefault(gt_label, []).append(fn_match)

    return matches


def _match_boxes_hungarian(
    gt_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    iou_threshold: float,
    split_image_names: list[str] | None = None,
) -> dict[str, list[PredictMatch]]:
    """Hungarian (globally optimal) box matching.

    Uses scipy.optimize.linear_sum_assignment on the negative IoU matrix.
    Assignment is geometry-first; confidence plays no role in pairing.
    """
    matches: dict[str, list[PredictMatch]] = {}
    preds_df, all_images = _resolve_matching_scope(gt_df, preds_df, split_image_names)

    for image_name in all_images:
        gt = gt_df[gt_df["image_name"] == image_name]
        preds = preds_df[preds_df["image_name"] == image_name]

        gt_bboxes = gt[["bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]].dropna().values
        pred_bboxes = preds[["bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]].values

        n_preds = len(pred_bboxes)
        n_gts = len(gt_bboxes)

        if n_preds == 0 and n_gts == 0:
            continue

        iou_matrix = compute_iou_matrix(pred_bboxes, gt_bboxes)  # (n_preds, n_gts)

        matched_preds: set[int] = set()
        matched_gts: set[int] = set()

        if n_preds > 0 and n_gts > 0:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            for pred_i, gt_j in zip(row_ind, col_ind, strict=True):
                pred_i, gt_j = int(pred_i), int(gt_j)
                if iou_matrix[pred_i, gt_j] < iou_threshold:
                    continue  # below threshold → both become FP/FN

                pred_row = preds.iloc[pred_i]
                pred_label = str(pred_row["instance_label"])
                gt_label = str(gt.iloc[gt_j]["instance_label"])

                match = PredictMatch(
                    pred_label=pred_label,
                    gt_label=gt_label,
                    pred_index=_row_index(pred_row),
                    gt_index=_row_index(gt.iloc[gt_j]),
                    confidence=float(pred_row["confidence"]),
                )
                matches.setdefault(pred_label, []).append(match)
                matched_preds.add(pred_i)
                matched_gts.add(gt_j)

        # Unmatched predictions → FP
        for pred_i in range(n_preds):
            if pred_i in matched_preds:
                continue
            pred_row = preds.iloc[pred_i]
            pred_label = str(pred_row["instance_label"])
            match = PredictMatch(
                pred_label=pred_label,
                gt_label="background",
                pred_index=_row_index(pred_row),
                gt_index=-1,
                confidence=float(pred_row["confidence"]),
            )
            matches.setdefault(pred_label, []).append(match)

        # Unmatched GTs → FN
        for gt_j in range(n_gts):
            if gt_j in matched_gts:
                continue
            gt_row = gt.iloc[gt_j]
            gt_label = str(gt_row["instance_label"])
            fn_match = PredictMatch(
                pred_label="background",
                gt_label=gt_label,
                pred_index=-1,
                gt_index=_row_index(gt_row),
                confidence=0.0,
            )
            matches.setdefault(gt_label, []).append(fn_match)

    return matches
