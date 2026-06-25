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

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..grouping import image_row_indices
from ..types import DetectionMetrics

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


def _read_prf1_at_conf(
    p_curve: npt.NDArray[np.float64],
    r_curve: npt.NDArray[np.float64],
    f1_curve: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    conf: float,
) -> tuple[float, float, float]:
    """Interpolate one class's ``(precision, recall, f1)`` at confidence ``conf``.

    ``x`` is Ultralytics' increasing confidence axis (``np.linspace(0, 1, 1000)``)
    and the ``*_curve`` are that class's metric-vs-confidence curves from
    ``ap_per_class``. Pure / torch-free.
    """
    return (
        float(np.interp(conf, x, p_curve)),
        float(np.interp(conf, x, r_curve)),
        float(np.interp(conf, x, f1_curve)),
    )


def _conf_at_max_f1(f1_curve: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> float:
    """Confidence on ``x`` that maximises a 1-D per-class (or mean) F1 curve."""
    return float(x[int(np.argmax(f1_curve))])


def _ap_per_class_results(
    gt_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    classes: list[str] | None,
    split_image_names: list[str] | None,
) -> tuple[Any, dict[int, str], npt.NDArray[np.int64] | None]:
    """Assemble Ultralytics ``ap_per_class`` inputs and run it.

    Returns ``(results, idx_to_class, zero_classes)``:

    * normal split → ``(results_tuple, idx_to_class, None)``;
    * GT present but no predictions → ``(None, idx_to_class, <present class idxs>)``;
    * no GT in the split → ``(None, idx_to_class, None)``.

    Raises ``ImportError`` if the optional ``ultralytics`` dependency is missing.
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

    # Map labels and extract boxes/conf to numpy once over the whole frame,
    # then slice each image's rows by positional index (avoids per-image
    # groupby chopping and Series.map, the dominant cost here).
    gt_cls_all = gt_df["instance_label"].map(class_to_idx).to_numpy(np.int64)
    gt_bbox_all = gt_df[_BBOX_COLS].to_numpy(np.float32)
    pred_cls_arr = scoped_preds["instance_label"].map(class_to_idx).to_numpy(np.int64)
    pred_conf_arr = scoped_preds["confidence"].to_numpy(np.float32)
    pred_bbox_all = scoped_preds[_BBOX_COLS].to_numpy(np.float32)
    gt_rows = image_row_indices(gt_df)
    pred_rows = image_row_indices(scoped_preds)

    tp_all: list[npt.NDArray[np.bool_]] = []
    conf_all: list[npt.NDArray[np.float32]] = []
    pred_cls_all: list[npt.NDArray[np.integer[Any]]] = []
    target_cls_all: list[npt.NDArray[np.integer[Any]]] = []

    # Iterate in sorted order: the per-image arrays are concatenated and handed
    # to ap_per_class, which sorts globally by confidence with an unstable sort.
    # A deterministic image order keeps confidence-tie ordering stable, so the
    # result no longer wobbles (~1e-3) with the process hash seed.
    for image_name in sorted(images):
        gi = gt_rows.get(image_name)
        gt_cls = gt_cls_all[gi] if gi is not None else np.empty(0, np.int64)
        # Every GT box is a target (drives recall / FN), even with no preds.
        target_cls_all.append(gt_cls)

        pi = pred_rows.get(image_name)
        if pi is None or len(pi) == 0:
            continue

        pred_cls = pred_cls_arr[pi]
        conf = pred_conf_arr[pi]
        pred_cls_all.append(pred_cls)
        conf_all.append(conf)

        if gi is None or len(gi) == 0:
            # No GT on this image → every prediction is a false positive.
            tp_all.append(np.zeros((len(pi), _IOUV.shape[0]), dtype=bool))
            continue

        gt_boxes = torch.tensor(gt_bbox_all[gi])
        pred_boxes = torch.tensor(pred_bbox_all[pi])
        iou = box_iou(gt_boxes, pred_boxes).cpu().numpy()  # (n_gt, n_pred)
        tp_all.append(_match_predictions(pred_cls, gt_cls, iou))

    target_cls = np.concatenate(target_cls_all) if target_cls_all else np.empty(0, np.int64)
    if target_cls.size == 0:
        return None, idx_to_class, None
    if not tp_all:
        # GT exists but no predictions at all → caller emits all-zero metrics.
        return None, idx_to_class, np.unique(target_cls)

    tp = np.concatenate(tp_all, 0)
    conf = np.concatenate(conf_all, 0)
    pred_cls = np.concatenate(pred_cls_all, 0)
    return ap_per_class(tp, conf, pred_cls, target_cls), idx_to_class, None


def compute_ultralytics_metrics(
    gt_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    classes: list[str] | None = None,
    split_image_names: list[str] | None = None,
    conf_threshold: float | dict[str, float] | None = None,
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
        conf_threshold: Where to read P/R/F1 on the per-class P/R/F1-vs-confidence
            curves. ``None`` (default) uses Ultralytics' own operating point (the
            in-sample max-mean-F1 confidence). A ``float`` reads every class at
            that shared confidence; a ``dict`` reads each class at its own (e.g. a
            threshold calibrated on a held-out split via
            :func:`find_ultralytics_confidence`). AP is always taken over the full
            curve and is unaffected by ``conf_threshold``.

    Returns:
        ``{class_name: DetectionMetrics}`` for every class that has at least one
        GT box in the split (classes absent from the GT are not scored, matching
        Ultralytics).

    Raises:
        ImportError: If the optional ``ultralytics`` dependency is not installed.
    """
    results, idx_to_class, zero_classes = _ap_per_class_results(
        gt_df, preds_df, classes, split_image_names
    )
    if results is None:
        if zero_classes is None:
            return {}
        # GT exists but no predictions at all → P/R/F1/AP are zero everywhere.
        return {
            idx_to_class[int(c)]: DetectionMetrics.model_construct(
                precision=0.0, recall=0.0, f1=0.0, ap50=0.0, ap75=0.0, ap50_95=0.0
            )
            for c in zero_classes
        }

    p, r, f1, ap = results[2], results[3], results[4], results[5]
    unique_classes = results[6]  # int class indices present as GT, sorted
    p_curve, r_curve, f1_curve, x = results[7], results[8], results[9], results[10]

    out: dict[str, DetectionMetrics] = {}
    for i, c in enumerate(unique_classes):
        name = idx_to_class[int(c)]
        if conf_threshold is None:
            p_i, r_i, f1_i = float(p[i]), float(r[i]), float(f1[i])
        else:
            cval = (
                conf_threshold.get(name, 0.0)
                if isinstance(conf_threshold, dict)
                else conf_threshold
            )
            p_i, r_i, f1_i = _read_prf1_at_conf(p_curve[i], r_curve[i], f1_curve[i], x, cval)
        out[name] = DetectionMetrics(
            precision=p_i,
            recall=r_i,
            f1=f1_i,
            ap50=float(ap[i, 0]),
            ap75=float(ap[i, _IOU75_COL]),
            ap50_95=float(ap[i].mean()),
        )
    return out


def find_ultralytics_confidence(
    gt_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    classes: list[str] | None = None,
    split_image_names: list[str] | None = None,
    mode: Literal["global", "per_class"] = "global",
) -> float | dict[str, float]:
    """Confidence threshold(s) maximising F1 on this split (calibration helper).

    Reads Ultralytics' F1-vs-confidence curves from ``ap_per_class``:
    ``"global"`` returns one threshold (max mean per-class F1, YOLO-style);
    ``"per_class"`` returns ``{class_name: threshold}``. Feed the result to
    :func:`compute_ultralytics_metrics` as ``conf_threshold=`` on another split to
    read P/R/F1 at the calibrated operating point ("calibrate on val, report on
    test"). Returns ``0.0`` / ``{}`` when the split has no scorable detections.

    Raises:
        ImportError: If the optional ``ultralytics`` dependency is not installed.
    """
    results, idx_to_class, _ = _ap_per_class_results(gt_df, preds_df, classes, split_image_names)
    if results is None:
        return 0.0 if mode == "global" else {}

    unique_classes = results[6]
    f1_curve, x = results[9], results[10]
    if mode == "global":
        return _conf_at_max_f1(np.asarray(f1_curve).mean(0), x)
    return {
        idx_to_class[int(c)]: _conf_at_max_f1(f1_curve[i], x) for i, c in enumerate(unique_classes)
    }


# Ultralytics' plotted confusion matrix is built at these fixed operating points,
# independent of the val IoU sweep (ConfusionMatrix.__init__ defaults).
_CM_CONF = 0.25
_CM_IOU = 0.45


def _confusion_process_batch(
    matrix: npt.NDArray[np.int64],
    det_classes: npt.NDArray[np.integer[Any]],
    gt_classes: npt.NDArray[np.integer[Any]],
    iou: npt.NDArray[np.float64],
    iou_thres: float,
    nc: int,
) -> None:
    """Numpy port of ``ConfusionMatrix.process_batch`` (detect task, v8.3).

    Updates ``matrix`` in place in Ultralytics' own orientation
    (``matrix[pred, gt]``; row/col ``nc`` is the background bucket). ``det_classes``
    are already confidence-filtered; ``iou`` is the ``(n_gt, n_det)`` IoU matrix for
    those detections. The greedy one-to-one dedup (sort by IoU, unique per det then
    per GT) is copied verbatim from Ultralytics.
    """
    if gt_classes.shape[0] == 0:  # no GT on this image → every prediction is a FP
        for dc in det_classes:
            matrix[dc, nc] += 1
        return
    if det_classes.shape[0] == 0:  # GT but no predictions → every GT is a FN
        for gc in gt_classes:
            matrix[nc, gc] += 1
        return

    x = np.where(iou > iou_thres)
    if x[0].shape[0]:
        matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    else:
        matches = np.zeros((0, 3))

    n = matches.shape[0] > 0
    m0, m1, _ = matches.transpose().astype(int)
    for i, gc in enumerate(gt_classes):
        j = m0 == i
        if n and j.sum() == 1:
            matrix[int(det_classes[m1[j][0]]), int(gc)] += 1  # correct (pred, gt)
        else:
            matrix[nc, int(gc)] += 1  # missed GT → background row (FN)

    if n:
        for i, dc in enumerate(det_classes):
            if not (m1 == i).any():
                matrix[int(dc), nc] += 1  # spurious prediction → background col (FP)


def compute_ultralytics_confusion_matrix(
    gt_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    classes: list[str] | None = None,
    split_image_names: list[str] | None = None,
    conf: float = _CM_CONF,
    iou_thres: float = _CM_IOU,
) -> tuple[npt.NDArray[np.int64], list[str]]:
    """Confusion matrix via Ultralytics' ``ConfusionMatrix.process_batch`` logic.

    Mirrors the matrix ``model.val()`` plots: detections are kept above ``conf``
    (default 0.25) and matched to GT at IoU ``iou_thres`` (default 0.45), with
    Ultralytics' greedy one-to-one assignment and a ``background`` bucket for
    unmatched predictions (FP) and unmatched GT (FN).

    Args:
        gt_df: Ground-truth DataFrame (standard schema), scoped to the split.
        preds_df: Predictions DataFrame (standard schema). Predictions on images
            outside the split are dropped.
        classes: Class vocabulary fixing the label→index mapping. Defaults to the
            sorted union of GT and prediction labels.
        split_image_names: Complete list of image names in the split, including
            empty images (no GT). Predictions on those images become false
            positives. Defaults to the images present in ``gt_df``.
        conf: Confidence threshold for the matrix (Ultralytics treats the YOLO-val
            default ``0.001`` as ``0.25``; that quirk is reproduced here).
        iou_thres: IoU threshold for matching.

    Returns:
        ``(matrix, labels)`` where ``labels`` is ``classes + ["background"]`` and
        ``matrix`` has shape ``(nc + 1, nc + 1)``. The matrix is transposed from
        Ultralytics' native orientation to ``matrix[true, pred]`` so it matches
        this library's :func:`metrics.confusion.get_confusion_matrix` (sklearn
        convention: row = ground truth, column = prediction).

    Raises:
        ImportError: If the optional ``ultralytics`` dependency is not installed.
    """
    try:
        import torch
        from ultralytics.utils.metrics import box_iou
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(_INSTALL_HINT) from exc

    if classes is None:
        classes = sorted(set(gt_df["instance_label"]) | set(preds_df["instance_label"]))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    nc = len(classes)
    # Ultralytics bumps the YOLO-val sentinel 0.001 up to 0.25 for the matrix.
    conf = _CM_CONF if conf in (None, 0.001) else conf

    images: set[str] = set(gt_df["image_name"].unique())
    if split_image_names is not None:
        images |= set(split_image_names)
    scoped_preds = preds_df[preds_df["image_name"].isin(images)]

    # Whole-frame numpy extraction once; per image slice by positional index.
    gt_cls_all = gt_df["instance_label"].map(class_to_idx).to_numpy(np.int64)
    gt_bbox_all = gt_df[_BBOX_COLS].to_numpy(np.float32)
    pred_cls_arr = scoped_preds["instance_label"].map(class_to_idx).to_numpy(np.int64)
    pred_conf_arr = scoped_preds["confidence"].to_numpy(np.float64)
    pred_bbox_all = scoped_preds[_BBOX_COLS].to_numpy(np.float32)
    gt_rows = image_row_indices(gt_df)
    pred_rows = image_row_indices(scoped_preds)

    matrix = np.zeros((nc + 1, nc + 1), dtype=np.int64)
    for image_name in images:
        gi = gt_rows.get(image_name)
        gt_cls = gt_cls_all[gi] if gi is not None else np.empty(0, np.int64)

        pi = pred_rows.get(image_name)
        if pi is not None and len(pi):
            pi = pi[pred_conf_arr[pi] > conf]
        if pi is not None and len(pi):
            det_cls = pred_cls_arr[pi]
            if gi is not None and gt_cls.shape[0]:
                gt_boxes = torch.tensor(gt_bbox_all[gi])
                det_boxes = torch.tensor(pred_bbox_all[pi])
                iou = box_iou(gt_boxes, det_boxes).cpu().numpy()
            else:
                iou = np.zeros((0, det_cls.shape[0]))
        else:
            det_cls = np.empty(0, np.int64)
            iou = np.zeros((gt_cls.shape[0], 0))

        _confusion_process_batch(matrix, det_cls, gt_cls, iou, iou_thres, nc)

    labels = list(classes) + ["background"]
    return matrix.T.copy(), labels  # transpose to [true, pred] (library convention)
