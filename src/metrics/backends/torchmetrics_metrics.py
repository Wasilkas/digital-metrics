"""General-purpose detection metrics via torchmetrics' ``MeanAveragePrecision``.

This is the "open-API" backend: P/R/AP are produced by torchmetrics' real
``torchmetrics.detection.MeanAveragePrecision`` (a pycocotools-backed COCO mAP),
not re-implemented here. We only assemble the per-image ``preds``/``target``
dicts it expects and read its output.

The dependency is heavy (``torchmetrics[detection]`` pulls in ``torch``,
``torchvision`` and ``pycocotools``) so it is not part of the core install.
Enable it with::

    pip install digital-metrics[torchmetrics]

and ``torch`` / ``torchmetrics`` are imported lazily inside
:func:`compute_torchmetrics_metrics`.

AP vs. P/R/F1
-------------
torchmetrics is AP-native: it reports COCO mAP and per-class AP but exposes no
single operating-point precision/recall/f1. We derive those the way YOLO reads
its headline numbers — off the IoU 0.50 precision-recall curve at the recall
point that maximises F1, per class (see :func:`_prf1_at_iou50`). The curve and
the 101 recall thresholds come from torchmetrics' ``extended_summary`` precision
tensor, so the AP values equal ``MeanAveragePrecision`` 's own
``map_50``/``map_75``/``map`` per class.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..grouping import image_row_indices
from ..types import DetectionMetrics

_BBOX_COLS = ["bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]
# Ten IoU thresholds, exactly as COCO / Ultralytics val (linspace(0.5, 0.95, 10)).
_IOU50_IDX = 0  # iou_thresholds[0] == 0.50
_IOU75_IDX = 5  # iou_thresholds[5] == 0.75
# COCO recall thresholds: linspace(0, 1, 101). The precision tensor's R axis is
# indexed by these recall values, so they *are* the recall for each curve point.
_REC_THRESHOLDS: npt.NDArray[np.float64] = np.linspace(0.0, 1.0, 101)
_AREA_ALL = 0  # area range index: 0 == 'all'
_MAXDET_100 = 2  # max-detections index: 2 == 100

_INSTALL_HINT = (
    "compute_torchmetrics_metrics requires the optional 'torchmetrics' dependency "
    "(which pulls in torch, torchvision and pycocotools). Install it with:\n"
    "    pip install digital-metrics[torchmetrics]\n"
    "or, with uv:\n"
    "    uv pip install 'torchmetrics[detection]'"
)


def _ap(prec_slice: npt.NDArray[np.float64]) -> float:
    """COCO AP for one (IoU, class): mean precision over valid recall points.

    ``prec_slice`` is the length-101 precision curve. pycocotools marks
    unreachable recall points with ``-1``; the standard COCO summarise averages
    only the valid (``> -1``) entries. An all-``-1`` slice (the class had no
    scored detections) yields ``0.0``, matching the Ultralytics backend.
    """
    valid = prec_slice[prec_slice > -1.0]
    return float(valid.mean()) if valid.size else 0.0


def _prf1_at_iou50(prec_curve: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    """Read precision/recall/f1 off the IoU-0.50 P-R curve at the max-F1 point.

    ``prec_curve`` is the length-101 precision curve at IoU 0.50; its index is the
    recall threshold (``_REC_THRESHOLDS``). Unreachable points (``-1``) are
    treated as zero precision. Returns ``(precision, recall, f1)`` at the recall
    threshold that maximises F1, or ``(0, 0, 0)`` when no point is positive.
    """
    prec = np.where(prec_curve > -1.0, prec_curve, 0.0)
    rec = _REC_THRESHOLDS
    denom = prec + rec
    f1 = np.divide(2.0 * prec * rec, denom, out=np.zeros_like(prec), where=denom > 0.0)
    best = int(np.argmax(f1))
    return float(prec[best]), float(rec[best]), float(f1[best])


def compute_torchmetrics_metrics(
    gt_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    classes: list[str] | None = None,
    split_image_names: list[str] | None = None,
) -> dict[str, DetectionMetrics]:
    """Compute per-class detection metrics using torchmetrics' COCO mAP.

    Args:
        gt_df: Ground-truth DataFrame (standard schema), already scoped to the
            split you want to score.
        preds_df: Predictions DataFrame (standard schema). Predictions on images
            outside the split are dropped.
        classes: Class vocabulary fixing the label→index mapping. Defaults to the
            sorted union of GT and prediction labels.
        split_image_names: Complete list of image names in the split, including
            empty images (no GT). Predictions on those images are counted as
            false positives. Defaults to the images present in ``gt_df``.

    Returns:
        ``{class_name: DetectionMetrics}`` for every class that has at least one
        GT box in the split (classes absent from the GT are not scored, matching
        the Ultralytics backend).

    Raises:
        ImportError: If the optional ``torchmetrics`` dependency is not installed.
    """
    try:
        import torch
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(_INSTALL_HINT) from exc

    if classes is None:
        classes = sorted(set(gt_df["instance_label"]) | set(preds_df["instance_label"]))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    images: set[str] = set(gt_df["image_name"].unique())
    if split_image_names is not None:
        images |= set(split_image_names)
    scoped_preds = preds_df[preds_df["image_name"].isin(images)]

    # Map labels and extract boxes/conf to numpy once over the whole frame,
    # then slice each image's rows by positional index (avoids per-image
    # groupby chopping and Series.map).
    gt_cls_all = gt_df["instance_label"].map(class_to_idx).to_numpy(np.int64)
    gt_bbox_all = gt_df[_BBOX_COLS].to_numpy(np.float32)
    pred_cls_arr = scoped_preds["instance_label"].map(class_to_idx).to_numpy(np.int64)
    pred_conf_arr = scoped_preds["confidence"].to_numpy(np.float32)
    pred_bbox_all = scoped_preds[_BBOX_COLS].to_numpy(np.float32)
    gt_rows = image_row_indices(gt_df)
    pred_rows = image_row_indices(scoped_preds)

    empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
    empty_labels = torch.zeros((0,), dtype=torch.int64)

    gt_class_idxs: set[int] = set()
    preds_list: list[dict[str, Any]] = []
    target_list: list[dict[str, Any]] = []

    # preds_list[i] and target_list[i] must describe the same image, in order.
    for image_name in sorted(images):
        gi = gt_rows.get(image_name)
        if gi is not None and len(gi):
            gt_cls = gt_cls_all[gi]
            gt_class_idxs.update(int(c) for c in gt_cls)
            target_list.append(
                {
                    "boxes": torch.tensor(gt_bbox_all[gi]),
                    "labels": torch.tensor(gt_cls),
                }
            )
        else:
            target_list.append({"boxes": empty_boxes, "labels": empty_labels})

        pi = pred_rows.get(image_name)
        if pi is not None and len(pi):
            preds_list.append(
                {
                    "boxes": torch.tensor(pred_bbox_all[pi]),
                    "scores": torch.tensor(pred_conf_arr[pi]),
                    "labels": torch.tensor(pred_cls_arr[pi]),
                }
            )
        else:
            preds_list.append(
                {"boxes": empty_boxes, "scores": empty_labels.float(), "labels": empty_labels}
            )

    if not gt_class_idxs:
        return {}

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
        extended_summary=True,
    )
    metric.update(preds_list, target_list)
    result = metric.compute()

    # precision: (T iou, R recall, K class, A area, M max-det). The K axis is
    # aligned with result["classes"] (sorted unique label ids present).
    precision: npt.NDArray[np.float64] = result["precision"].cpu().numpy()
    # With a single class present, ``result["classes"]`` is a 0-d tensor whose
    # .tolist() is a bare int; atleast_1d keeps the comprehension iterable.
    res_classes = [int(c) for c in np.atleast_1d(result["classes"].cpu().numpy()).tolist()]

    out: dict[str, DetectionMetrics] = {}
    for k, cls_idx in enumerate(res_classes):
        if cls_idx not in gt_class_idxs:
            continue
        # Per-class precision curves: shape (T iou, R recall), one row per IoU.
        cls_prec = precision[:, :, k, _AREA_ALL, _MAXDET_100]
        p_val, r_val, f1_val = _prf1_at_iou50(cls_prec[_IOU50_IDX])
        out[idx_to_class[cls_idx]] = DetectionMetrics(
            precision=p_val,
            recall=r_val,
            f1=f1_val,
            ap50=_ap(cls_prec[_IOU50_IDX]),
            ap75=_ap(cls_prec[_IOU75_IDX]),
            ap50_95=float(np.mean([_ap(row) for row in cls_prec])),
        )
    return out
