from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from .iou import compute_iou_matrix
from .matching import MatchingStrategy
from .types import Metrics

_IOU_THRESHOLDS = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]

APMethod = Literal["interp", "continuous"]

# Type alias for the per-class GT structure used inside compute_map.
_GtByImg = dict[str, dict[str, npt.NDArray[np.float32]]]


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


def _tp_fp_greedy(
    pred_boxes: npt.NDArray[np.float32],
    pred_img_ids: npt.NDArray[np.str_],
    gt_by_img: _GtByImg,
    thr: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Confidence-sorted greedy TP/FP assignment (YOLO-style).

    Processes predictions in decreasing confidence order (caller's
    responsibility to pass them pre-sorted).  Each prediction claims the
    highest-IoU unmatched GT; if IoU >= thr the prediction is TP, otherwise FP.
    """
    n = len(pred_boxes)
    tp = np.zeros(n, dtype=np.float32)
    fp = np.zeros(n, dtype=np.float32)

    for v in gt_by_img.values():
        v["matched_flags"].fill(False)

    for i, (img_id, pbox) in enumerate(zip(pred_img_ids, pred_boxes, strict=False)):
        img_id = str(img_id)
        if img_id not in gt_by_img or gt_by_img[img_id]["boxes"].size == 0:
            fp[i] = 1.0
            continue
        gt_boxes = gt_by_img[img_id]["boxes"]
        matched = gt_by_img[img_id]["matched_flags"]
        ious = compute_iou_matrix(pbox[None, :], gt_boxes)[0]
        jmax = int(np.argmax(ious))
        if ious[jmax] >= thr and not matched[jmax]:
            tp[i] = 1.0
            matched[jmax] = True
        else:
            fp[i] = 1.0

    return tp, fp


def _tp_fp_iou_prior(
    pred_boxes: npt.NDArray[np.float32],
    pred_img_ids: npt.NDArray[np.str_],
    gt_by_img: _GtByImg,
    thr: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """IoU-prior TP/FP assignment (Ultralytics non-scipy style).

    Collects all pred-GT pairs with IoU >= thr, sorts them by IoU descending,
    then assigns greedily — each prediction and each GT matched at most once,
    highest-IoU pair wins.  Labels already match because compute_map works
    per class.

    The result is in confidence-sorted index order (same as the input arrays)
    so the cumulative P-R curve is correctly traced.
    """
    n = len(pred_boxes)
    tp = np.zeros(n, dtype=np.float32)
    fp = np.zeros(n, dtype=np.float32)

    pairs: list[tuple[float, int, str, int]] = []
    for i, (img_id, pbox) in enumerate(zip(pred_img_ids, pred_boxes, strict=False)):
        img_id = str(img_id)
        if img_id not in gt_by_img or gt_by_img[img_id]["boxes"].size == 0:
            fp[i] = 1.0
            continue
        gt_boxes = gt_by_img[img_id]["boxes"]
        ious = compute_iou_matrix(pbox[None, :], gt_boxes)[0]
        for j, iou_val in enumerate(ious):
            if float(iou_val) >= thr:
                pairs.append((float(iou_val), i, img_id, j))

    pairs.sort(key=lambda x: -x[0])
    matched_preds: set[int] = set()
    matched_gts: dict[str, set[int]] = {}

    for iou_val, pred_i, img_id, gt_j in pairs:
        if pred_i in matched_preds:
            continue
        img_gts = matched_gts.setdefault(img_id, set())
        if gt_j in img_gts:
            continue
        tp[pred_i] = 1.0
        matched_preds.add(pred_i)
        img_gts.add(gt_j)

    for i in range(n):
        if tp[i] == 0.0 and fp[i] == 0.0:
            fp[i] = 1.0

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
            ``"greedy"`` (default, confidence-sorted, YOLO-style) or
            ``"iou_prior"`` (IoU-sorted, Ultralytics non-scipy style).
            ``"hungarian"`` is not supported here (use ``"greedy"`` or
            ``"iou_prior"``).
    """
    if strategy == "hungarian":
        raise ValueError(
            "strategy='hungarian' is not supported in compute_map. "
            "Use 'greedy' or 'iou_prior'."
        )

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

        gt_by_img: _GtByImg = {}
        for img_id, df_img in gt_c.groupby("image_name"):
            boxes = df_img[[x1, y1, x2, y2]].values.astype(np.float32)
            gt_by_img[str(img_id)] = {
                "boxes": boxes,
                "matched_flags": np.zeros(boxes.shape[0], dtype=bool),
            }

        pred_c = pred_c.sort_values(by="confidence", ascending=False)
        pred_boxes = pred_c[[x1, y1, x2, y2]].values.astype(np.float32)
        pred_img_ids = pred_c["image_name"].values

        assign = _tp_fp_iou_prior if strategy == "iou_prior" else _tp_fp_greedy

        for thr in _IOU_THRESHOLDS:
            npos = npos_by_class.get(c, 0)
            if npos == 0:
                ap_per_class[c][thr] = float("nan")
                continue

            tp, fp = assign(pred_boxes, pred_img_ids, gt_by_img, thr)

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
