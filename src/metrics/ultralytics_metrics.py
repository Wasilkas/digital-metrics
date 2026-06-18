"""YOLO-exact metrics via Ultralytics' own ``ap_per_class``.

This is an *optional* path: P/R/F1/AP are produced by Ultralytics' real metric
code (``ultralytics.utils.metrics.ap_per_class``), not re-implemented here. We
only assemble the inputs it expects and let it compute the numbers.

The dependency is heavy (``ultralytics`` pulls in ``torch``) so it is not part of
the core install. Enable it with::

    pip install digital-metrics[ultralytics]

and ``ultralytics`` / ``torch`` are imported lazily inside
:func:`compute_ultralytics_metrics`.

How the inputs are built
------------------------
* IoU matrices come from ``ultralytics.utils.metrics.box_iou``.
* Predictions are assigned to GT with a faithful re-match at every one of the
  ten IoU thresholds (``0.50 … 0.95``) — a numpy port of
  ``BaseValidator.match_predictions`` (non-scipy path, v8.4.70), which lives on
  the validator class and is not exposed standalone. This is *matching*, not a
  metric, so re-matching per threshold (rather than thresholding a single stored
  IoU) keeps the result identical to ``model.val()``.
* The resulting ``(n_pred, 10)`` correctness array, confidences, predicted
  classes and target classes are handed to ``ap_per_class``.

How Ultralytics defines the headline P/R/F1 (see ``ap_per_class``): built at IoU
0.50, read off a 1000-point interpolated P-R curve at the single global
confidence that maximises the smoothed mean per-class F1.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .types import DetectionMetrics

# Backward-compatible alias: the YOLO-exact path historically returned its own
# ``YoloMetrics`` model. It now shares the common :class:`DetectionMetrics`
# schema (same fields) so both backends return identical objects.
YoloMetrics = DetectionMetrics

_BBOX_COLS = ["bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]
# Ten IoU thresholds, exactly as Ultralytics val (torch.linspace(0.5, 0.95, 10)).
_IOUV: npt.NDArray[np.float64] = np.linspace(0.5, 0.95, 10)
_IOU75_COL = 5  # _IOUV[5] == 0.75

_INSTALL_HINT = (
    "compute_ultralytics_metrics requires the optional 'ultralytics' dependency "
    "(which pulls in torch). Install it with:\n"
    "    pip install digital-metrics[ultralytics]\n"
    "or, with uv:\n"
    "    uv pip install ultralytics"
)


def _match_predictions(
    pred_classes: npt.NDArray[np.integer[Any]],
    true_classes: npt.NDArray[np.integer[Any]],
    iou: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Numpy port of ``BaseValidator.match_predictions`` (non-scipy path, v8.4.70).

    ``iou`` is the ``(n_gt, n_pred)`` IoU matrix. Returns an ``(n_pred, 10)`` bool
    array marking, for each IoU threshold, whether each prediction is a correct
    (right-class, unique, highest-IoU) match.
    """
    correct = np.zeros((pred_classes.shape[0], _IOUV.shape[0]), dtype=bool)
    correct_class = true_classes[:, None] == pred_classes  # (n_gt, n_pred)
    iou = iou * correct_class  # zero out wrong-class pairs
    for i, threshold in enumerate(_IOUV):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def compute_ultralytics_metrics(
    gt_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    classes: list[str] | None = None,
    split_image_names: list[str] | None = None,
) -> dict[str, DetectionMetrics]:
    """Compute YOLO-exact per-class metrics using Ultralytics' ``ap_per_class``.

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
        Ultralytics).

    Raises:
        ImportError: If the optional ``ultralytics`` dependency is not installed.
    """
    try:
        import torch
        from ultralytics.utils.metrics import ap_per_class, box_iou
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

    gt_groups = dict(tuple(gt_df.groupby("image_name", sort=False)))
    pred_groups = dict(tuple(scoped_preds.groupby("image_name", sort=False)))

    tp_all: list[npt.NDArray[np.bool_]] = []
    conf_all: list[npt.NDArray[np.float32]] = []
    pred_cls_all: list[npt.NDArray[np.integer[Any]]] = []
    target_cls_all: list[npt.NDArray[np.integer[Any]]] = []

    for image_name in images:
        g = gt_groups.get(image_name)
        gt_cls = (
            g["instance_label"].map(class_to_idx).to_numpy(np.int64)
            if g is not None
            else np.empty(0, np.int64)
        )
        # Every GT box is a target (drives recall / FN), even with no preds.
        target_cls_all.append(gt_cls)

        p = pred_groups.get(image_name)
        if p is None or len(p) == 0:
            continue

        pred_cls = p["instance_label"].map(class_to_idx).to_numpy(np.int64)
        conf = p["confidence"].to_numpy(np.float32)
        pred_cls_all.append(pred_cls)
        conf_all.append(conf)

        if g is None or len(g) == 0:
            # No GT on this image → every prediction is a false positive.
            tp_all.append(np.zeros((len(p), _IOUV.shape[0]), dtype=bool))
            continue

        gt_boxes = torch.tensor(g[_BBOX_COLS].to_numpy(np.float32))
        pred_boxes = torch.tensor(p[_BBOX_COLS].to_numpy(np.float32))
        iou = box_iou(gt_boxes, pred_boxes).cpu().numpy()  # (n_gt, n_pred)
        tp_all.append(_match_predictions(pred_cls, gt_cls, iou))

    target_cls = np.concatenate(target_cls_all) if target_cls_all else np.empty(0, np.int64)
    if target_cls.size == 0:
        return {}

    if not tp_all:
        # GT exists but no predictions at all → P/R/F1/AP are zero everywhere.
        present = np.unique(target_cls)
        return {
            idx_to_class[int(c)]: DetectionMetrics.model_construct(
                precision=0.0, recall=0.0, f1=0.0, ap50=0.0, ap75=0.0, ap50_95=0.0
            )
            for c in present
        }

    tp = np.concatenate(tp_all, 0)
    conf = np.concatenate(conf_all, 0)
    pred_cls = np.concatenate(pred_cls_all, 0)

    results = ap_per_class(tp, conf, pred_cls, target_cls)
    p, r, f1, ap = results[2], results[3], results[4], results[5]
    unique_classes = results[6]  # int class indices present as GT, sorted

    return {
        idx_to_class[int(c)]: DetectionMetrics(
            precision=float(p[i]),
            recall=float(r[i]),
            f1=float(f1[i]),
            ap50=float(ap[i, 0]),
            ap75=float(ap[i, _IOU75_COL]),
            ap50_95=float(ap[i].mean()),
        )
        for i, c in enumerate(unique_classes)
    }
