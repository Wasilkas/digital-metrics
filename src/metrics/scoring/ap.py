from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..matching import (
    MatchingStrategy,
    assign_greedy,
    assign_hungarian,
    assign_iou_prior,
    compute_iou_matrix,
)
from ..types import Metrics

_IOU_THRESHOLDS = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]

APMethod = Literal["interp", "continuous"]

# Per-class GT boxes keyed by image, used inside compute_map.
_GtByImg = dict[str, npt.NDArray[np.float32]]

# One image's matching inputs for a class: (global prediction indices, IoU matrix).
# The IoU matrix is ``None`` when the image has no GT of this class, so every
# prediction there is a false positive at any threshold.
_ImageMatch = tuple[npt.NDArray[np.intp], npt.NDArray[np.float64] | None]


def _resolve_split_images(
    gt_df: pd.DataFrame,
    split_image_names: list[str] | None,
) -> list[str] | npt.NDArray[np.str_]:
    if split_image_names is not None:
        return split_image_names
    return gt_df["image_name"].unique()


def compute_ap(
    recall: npt.NDArray[np.float64],
    precision: npt.NDArray[np.float64],
    method: APMethod = "continuous",
) -> float:
    """Compute average precision from a precision-recall curve.

    Two methods are available:

    ``"continuous"`` (default) — VOC 2010+ rectangle-area integration.
    Prepends (0, 0) and appends (1, 0) sentinels, computes the right-to-left
    precision envelope, then sums rectangle areas at recall change points.

    ``"interp"`` — 101-point COCO interpolation, byte-equivalent to
    ``ultralytics.utils.metrics.compute_ap`` with ``method="interp"``.
    Uses Ultralytics-compatible sentinels (``mpre[0] = 1.0``,
    ``mrec[-1] = recall[-1] + 1e-4``) and integrates with ``np.trapezoid``
    over 101 equally-spaced recall points.  Returns 0.0 when no predictions
    were made (empty recall array).
    """
    if method == "interp":
        if len(recall) == 0:
            return 0.0
        mrec = np.concatenate(([0.0], recall, [recall[-1] + 1e-4]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        x = np.linspace(0, 1, 101)
        return float(np.trapezoid(np.interp(x, mrec, mpre), x))

    # "continuous": VOC 2010+ rectangle-area integration
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _precompute_image_matches(
    pred_boxes: npt.NDArray[np.float32],
    pred_img_ids: npt.NDArray[np.str_],
    gt_by_img: _GtByImg,
) -> list[_ImageMatch]:
    """Group a class's predictions by image and compute each image's IoU once.

    The IoU matrix between an image's predictions and its GTs does not depend on
    the IoU threshold, so computing it here (once per class) and reusing it across
    all ten thresholds avoids ~10x redundant IoU work.  Predictions stay grouped
    in their confidence-descending input order via their global indices, so the
    cumulative P-R curve is traced correctly downstream.
    """
    img_to_pred_indices: dict[str, list[int]] = {}
    for i, img_id in enumerate(pred_img_ids):
        img_to_pred_indices.setdefault(str(img_id), []).append(i)

    matches: list[_ImageMatch] = []
    for img_id, pred_indices in img_to_pred_indices.items():
        idx = np.asarray(pred_indices, dtype=np.intp)
        gt_boxes = gt_by_img.get(img_id)
        if gt_boxes is None or gt_boxes.size == 0:
            matches.append((idx, None))
        else:
            matches.append((idx, compute_iou_matrix(pred_boxes[idx], gt_boxes)))
    return matches


def _assign_tp_fp(
    image_matches: list[_ImageMatch],
    n: int,
    thr: float,
    strategy: MatchingStrategy,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Per-class, per-image TP/FP assignment for the mAP precision-recall curve.

    Runs the configured ``strategy`` kernel at IoU threshold ``thr`` over the
    precomputed per-image IoU matrices (see :func:`_precompute_image_matches`).
    Labels need no checking — ``compute_map`` already works one class at a time.
    A prediction matched to a GT is a TP, everything else (including predictions
    on images with no GT of this class) is an FP.  Results stay in the
    confidence-sorted global index order so the cumulative P-R curve is correct.
    """
    tp = np.zeros(n, dtype=np.float32)
    fp = np.zeros(n, dtype=np.float32)

    for idx, iou_matrix in image_matches:
        if iou_matrix is None:
            fp[idx] = 1.0
            continue

        if strategy == "hungarian":
            pairs = assign_hungarian(iou_matrix, thr)
        elif strategy == "iou_prior":
            pairs = assign_iou_prior(iou_matrix, thr)
        else:
            pairs = assign_greedy(iou_matrix, thr)

        matched_local = {pred_i for pred_i, _ in pairs}
        for local_i, global_i in enumerate(idx):
            if local_i in matched_local:
                tp[global_i] = 1.0
            else:
                fp[global_i] = 1.0

    return tp, fp


def compute_map(
    gt_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    metrics: dict[str, Metrics],
    split_image_names: list[str] | None = None,
    method: APMethod = "continuous",
    strategy: MatchingStrategy = "greedy",
) -> None:
    """Compute mAP50/75/50-95 per class and write into metrics in place.

    Args:
        gt_df: Ground-truth DataFrame (current split).
        preds_df: Full predictions DataFrame (not confidence-filtered). May
            contain predictions for images outside the current split — these
            are dropped before scoring.
        metrics: Dict of Metrics objects; ap50/ap75/ap50_95 are set in place.
        split_image_names: Complete list of image names in the split, including
            empty images.  When provided, predictions on empty images are
            counted as FPs.  When None, falls back to images in gt_df.
        method: AP integration method — ``"continuous"`` (default, VOC 2010+)
            or ``"interp"`` (101-point COCO, Ultralytics-compatible).
        strategy: Box-matching strategy for the inner mAP loop —
            ``"greedy"`` (default, confidence-sorted, YOLO-style),
            ``"iou_prior"`` (IoU-sorted, Ultralytics non-scipy style), or
            ``"hungarian"`` (globally optimal per-image assignment via
            ``scipy.optimize.linear_sum_assignment``).
    """
    x1, y1, x2, y2 = "bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"

    split_images = _resolve_split_images(gt_df, split_image_names)
    preds_df = preds_df[preds_df["image_name"].isin(split_images)]

    classes_in_split = set(gt_df["instance_label"].unique().tolist())
    classes = sorted(classes_in_split)
    npos_by_class = gt_df.groupby("instance_label").size().to_dict()

    ap_per_class: dict[str, dict[float, float]] = {
        c: {t: float("nan") for t in _IOU_THRESHOLDS} for c in classes
    }

    for c in classes:
        gt_c = gt_df[gt_df["instance_label"] == c]
        pred_c = preds_df[preds_df["instance_label"] == c]

        # Extract bboxes once, then slice each image's rows by positional index.
        # Per-group DataFrame column selection (df_img[[...]].values) is the
        # dominant cost here when a class spans tens of thousands of images.
        gt_boxes_c = gt_c[[x1, y1, x2, y2]].to_numpy(np.float32)
        gt_by_img: _GtByImg = {
            str(img_id): gt_boxes_c[idx]
            for img_id, idx in gt_c.groupby("image_name").indices.items()
        }

        npos = npos_by_class.get(c, 0)
        if npos == 0:
            # No GT for this class → AP stays NaN at every threshold (pre-filled).
            continue

        pred_c = pred_c.sort_values(by="confidence", ascending=False)
        pred_boxes = pred_c[[x1, y1, x2, y2]].values.astype(np.float32)
        pred_img_ids = pred_c["image_name"].values

        # Compute each image's IoU matrix once, then reuse it across all thresholds.
        image_matches = _precompute_image_matches(pred_boxes, pred_img_ids, gt_by_img)
        n_preds = len(pred_boxes)

        for thr in _IOU_THRESHOLDS:
            tp, fp = _assign_tp_fp(image_matches, n_preds, thr, strategy)

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            rec = tp_cum / max(npos, 1)
            prec = np.divide(tp_cum, (tp_cum + fp_cum + 1e-12))
            ap_per_class[c][thr] = compute_ap(rec, prec, method)

    for c in classes:
        if c in metrics:
            metrics[c].ap50 = ap_per_class[c][0.5]
            metrics[c].ap75 = ap_per_class[c][0.75]
            metrics[c].ap50_95 = float(np.nanmean(list(ap_per_class[c].values())))

    for c, m in metrics.items():
        if c not in classes_in_split:
            m.ap50 = float("nan")
            m.ap75 = float("nan")
            m.ap50_95 = float("nan")
