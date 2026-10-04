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

Calibration
-----------
Calibration searches complete observed-confidence tie groups using retained
COCO detections, including empty-image FPs. Explicit thresholds report exact
TP/FP/FN and realizable P/R/F1 independently of the AP envelope. Cached IoUs and
per-image updates avoid re-evaluating every image at every global score.

"""

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..grouping import image_row_indices
from ..types import DetectionMetrics
from ..validation import drop_na_labels
from .operating_points import calibrate_raw, complete_counts, prepare_inputs, raw_counts, raw_prf1

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
_NO_SCORES_HINT = (
    "torchmetrics did not return a 'scores' array, which calibration needs to map "
    "confidence to operating point. Upgrade torchmetrics (extended_summary must "
    "provide 'scores') or run without a calibration_split."
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


def _read_prf1_at_conf(
    prec_curve: npt.NDArray[np.float64],
    score_curve: npt.NDArray[np.float64],
    conf: float,
) -> tuple[float, float, float]:
    """Read ``(precision, recall, f1)`` at confidence ``conf`` off an IoU-0.50 curve.

    ``prec_curve`` and ``score_curve`` are the length-101 precision and score
    curves (both indexed by ``_REC_THRESHOLDS``) for one class. ``score_curve`` is
    the confidence threshold at each recall point and decreases as recall rises;
    unreachable points are ``-1``. Keeping detections with ``score >= conf`` yields
    the largest recall whose score still clears ``conf``, so we read precision/recall
    at that index. Pure / torch-free.
    """
    prec = np.where(prec_curve > -1.0, prec_curve, 0.0)
    reachable = (score_curve >= conf) & (score_curve > -1.0)
    idx = np.flatnonzero(reachable)
    if idx.size == 0:
        return 0.0, 0.0, 0.0
    i = int(idx[-1])  # max-recall point still clearing the confidence threshold
    p, r = float(prec[i]), float(_REC_THRESHOLDS[i])
    f1 = 2.0 * p * r / (p + r) if (p + r) > 0.0 else 0.0
    return p, r, f1


def _conf_at_max_f1(
    prec_curve: npt.NDArray[np.float64], score_curve: npt.NDArray[np.float64]
) -> float:
    """Confidence at the max-F1 point of one class's IoU-0.50 curve (calibration).

    Mirrors :func:`_prf1_at_iou50`'s max-F1 recall index, then returns the score
    (confidence) at that index. Unreachable points (``-1`` score) are excluded.
    Returns ``0.0`` when the class has no reachable point. Pure / torch-free.
    """
    prec = np.where(prec_curve > -1.0, prec_curve, 0.0)
    rec = _REC_THRESHOLDS
    denom = prec + rec
    f1 = np.divide(2.0 * prec * rec, denom, out=np.zeros_like(prec), where=denom > 0.0)
    f1 = np.where(score_curve > -1.0, f1, -1.0)
    if not (score_curve > -1.0).any():
        return 0.0
    return float(score_curve[int(np.argmax(f1))])


def _global_conf_at_max_mean_f1(
    curves: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
) -> float:
    """Single confidence maximising the mean per-class F1 (YOLO-style global).

    ``curves`` is one ``(prec_curve, score_curve)`` per class at IoU 0.50. Sweeps
    every observed score and returns the one whose mean F1 across classes is
    highest. Returns ``0.0`` when there are no reachable points. Pure / torch-free.
    """
    candidates = sorted({float(s) for _, sc in curves for s in sc if s > -1.0})
    if not candidates:
        return 0.0
    best_t, best_mean_f1 = 0.0, -1.0
    for t in candidates:
        f1s = [_read_prf1_at_conf(prec, sc, t)[2] for prec, sc in curves]
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        if mean_f1 > best_mean_f1:
            best_mean_f1, best_t = mean_f1, t
    return best_t


def _torchmetrics_eval(
    gt_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    classes: list[str] | None,
    split_image_names: list[str] | None,
) -> dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]]:
    """Run torchmetrics and return per-class IoU-curve data.

    Returns an insertion-ordered ``{class_name: (cls_prec, cls_scores50)}`` where
    ``cls_prec`` is the ``(T iou, R recall)`` precision rows and ``cls_scores50``
    is the length-R score curve at IoU 0.50 (``None`` if torchmetrics did not
    return a ``scores`` array). Only classes with at least one GT box appear;
    an empty dict means no GT class was present.

    Raises:
        ImportError: If the optional ``torchmetrics`` dependency is not installed.
    """
    try:
        import torch
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(_INSTALL_HINT) from exc

    gt_df, preds_df, classes = prepare_inputs(gt_df, preds_df, classes, split_image_names)
    if split_image_names is not None:
        split_image_names = [str(name) for name in split_image_names]
    # Scope the images first, then drop the NA-label placeholder rows empty
    # images carry: the image stays in scope, but the row names no class.
    images: set[str] = set(gt_df["image_name"].unique())
    if split_image_names is not None:
        images |= set(split_image_names)
    gt_df = drop_na_labels(gt_df)
    preds_df = drop_na_labels(preds_df)

    if classes is None:
        classes = sorted(set(gt_df["instance_label"]) | set(preds_df["instance_label"]))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

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
    # aligned with result["classes"] (sorted unique label ids present). scores has
    # the same shape and holds the confidence threshold at each recall point.
    precision: npt.NDArray[np.float64] = result["precision"].cpu().numpy()
    scores_t = result.get("scores")
    scores: npt.NDArray[np.float64] | None = (
        scores_t.cpu().numpy() if scores_t is not None else None
    )
    # With a single class present, ``result["classes"]`` is a 0-d tensor whose
    # .tolist() is a bare int; atleast_1d keeps the comprehension iterable.
    res_classes = [int(c) for c in np.atleast_1d(result["classes"].cpu().numpy()).tolist()]

    out: dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]] = {}
    for k, cls_idx in enumerate(res_classes):
        if cls_idx not in gt_class_idxs:
            continue
        cls_prec = precision[:, :, k, _AREA_ALL, _MAXDET_100]  # (T iou, R recall)
        cls_scores50 = (
            scores[_IOU50_IDX, :, k, _AREA_ALL, _MAXDET_100] if scores is not None else None
        )
        out[idx_to_class[cls_idx]] = (cls_prec, cls_scores50)
    return out


def compute_torchmetrics_metrics(
    gt_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    classes: list[str] | None = None,
    split_image_names: list[str] | None = None,
    conf_threshold: float | dict[str, float] | None = None,
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
        conf_threshold: ``None`` uses the backend's unthresholded summary.
            A float or per-class dict retains original confidence >= threshold
            and reports exact IoU-0.50 counts and P/R/F1 using torchmetrics matching.
            AP always uses the full prediction set.

    Returns:
        Per-class DetectionMetrics. Unthresholded output includes GT classes;
        explicit thresholds also include vocabulary classes absent from GT,
        with observed counts and NaN AP.

    Raises:
        ImportError: If the optional ``torchmetrics`` dependency is not installed.
        ValueError: If inputs or thresholds are invalid.
    """
    per_class = _torchmetrics_eval(gt_df, preds_df, classes, split_image_names)

    operating = (
        raw_prf1(gt_df, preds_df, classes, split_image_names, conf_threshold, "torchmetrics")
        if conf_threshold is not None
        else {}
    )
    counts = (
        raw_counts(gt_df, preds_df, classes, split_image_names, conf_threshold, "torchmetrics")
        if conf_threshold is not None
        else {}
    )
    out: dict[str, DetectionMetrics] = {}
    for name, (cls_prec, _) in per_class.items():
        p_val, r_val, f1_val = (
            operating[name] if conf_threshold is not None else _prf1_at_iou50(cls_prec[_IOU50_IDX])
        )
        out[name] = DetectionMetrics(
            tp=counts[name][0] if name in counts else None,
            fp=counts[name][1] if name in counts else None,
            fn=counts[name][2] if name in counts else None,
            precision=p_val,
            recall=r_val,
            f1=f1_val,
            ap50=_ap(cls_prec[_IOU50_IDX]),
            ap75=_ap(cls_prec[_IOU75_IDX]),
            ap50_95=float(np.mean([_ap(row) for row in cls_prec])),
        )
    return complete_counts(
        out, gt_df, preds_df, classes, split_image_names, conf_threshold, "torchmetrics"
    )


def find_torchmetrics_confidence(
    gt_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    classes: list[str] | None = None,
    split_image_names: list[str] | None = None,
    mode: Literal["global", "per_class"] = "global",
) -> float | dict[str, float]:
    """Confidence threshold(s) maximising F1 on this split (calibration helper).

    Searches complete observed-confidence tie groups using COCO matching: ``"global"`` returns
    one threshold (max mean per-class F1, YOLO-style); ``"per_class"`` returns
    ``{class_name: threshold}``. Feed the result to
    :func:`compute_torchmetrics_metrics` as ``conf_threshold=`` on another split to
    read P/R/F1 at the calibrated operating point ("calibrate on val, report on
    test"). With no predictions, returns zero thresholds for present GT classes;
    with no GT classes, returns ``0.0`` / ``{}``.

    Raises:
        ImportError: If the optional ``torchmetrics`` dependency is not installed.
        ValueError: If inputs or calibration mode are invalid.
    """
    try:
        from torchmetrics.detection import MeanAveragePrecision  # noqa: F401
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return calibrate_raw(gt_df, preds_df, classes, split_image_names, mode, "torchmetrics")
