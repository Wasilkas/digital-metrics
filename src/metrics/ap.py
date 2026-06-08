import numpy as np
import numpy.typing as npt
import pandas as pd

from .iou import compute_iou_matrix
from .types import Metrics

_IOU_THRESHOLDS = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]


def compute_ap(recall: npt.NDArray[np.float64], precision: npt.NDArray[np.float64]) -> float:
    """VOC 2010+ AP interpolation — byte-equivalent to ultralytics compute_ap.

    Prepends (0, 0) and appends (1, 0) sentinel points, computes the
    right-to-left precision envelope, then integrates with rectangle areas
    at recall change points.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Precision envelope (right-to-left maximum)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Recall change points
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def compute_map(
    gt_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    metrics: dict[str, Metrics],
) -> None:
    """Compute mAP50/75/50-95 per class and write into metrics in place.

    Runs its own greedy matching loop independently from match_boxes —
    mirroring the Ultralytics two-path design where mAP uses all predictions
    (not confidence-filtered) sorted by score.

    Args:
        gt_df: Ground-truth DataFrame (current split).
        preds_df: Full predictions DataFrame (not confidence-filtered).
        metrics: Dict of Metrics objects; ap50/ap75/ap50_95 are set in place.
    """
    x1, y1, x2, y2 = "bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"
    classes_in_split = set(gt_df["instance_label"].unique().tolist())
    classes = sorted(classes_in_split)
    npos_by_class = gt_df.groupby("instance_label").size().to_dict()

    ap_per_class: dict[str, dict[float, float]] = {
        c: {t: float("nan") for t in _IOU_THRESHOLDS} for c in classes
    }

    for c in classes:
        gt_c = gt_df[gt_df["instance_label"] == c]
        pred_c = preds_df[preds_df["instance_label"] == c]

        gt_by_img: dict[str, dict[str, npt.NDArray[np.float32]]] = {}
        for img_id, df_img in gt_c.groupby("image_name"):
            boxes = df_img[[x1, y1, x2, y2]].values.astype(np.float32)
            gt_by_img[str(img_id)] = {
                "boxes": boxes,
                "matched_flags": np.zeros(boxes.shape[0], dtype=bool),
            }

        pred_c = pred_c.sort_values(by="confidence", ascending=False)
        pred_boxes = pred_c[[x1, y1, x2, y2]].values.astype(np.float32)
        pred_img_ids = pred_c["image_name"].values

        for thr in _IOU_THRESHOLDS:
            npos = npos_by_class.get(c, 0)
            if npos == 0:
                ap_per_class[c][thr] = float("nan")
                continue

            for v in gt_by_img.values():
                v["matched_flags"].fill(False)

            tp = np.zeros(len(pred_c), dtype=np.float32)
            fp = np.zeros(len(pred_c), dtype=np.float32)

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

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            rec = tp_cum / max(npos, 1)
            prec = np.divide(tp_cum, (tp_cum + fp_cum + 1e-12))
            ap_per_class[c][thr] = compute_ap(rec, prec)

    for c in classes:
        if c in metrics:
            metrics[c].ap50 = ap_per_class[c][0.5]
            metrics[c].ap75 = ap_per_class[c][0.75]
            metrics[c].ap50_95 = float(np.nanmean(list(ap_per_class[c].values())))

    # Classes with no GT instances in this split were never scored for AP.
    # Report NaN (not the 0.0 default) so a class-averaged mAP correctly
    # excludes them via nanmean — matching how YOLO reports mAP50.
    for c, m in metrics.items():
        if c not in classes_in_split:
            m.ap50 = float("nan")
            m.ap75 = float("nan")
            m.ap50_95 = float("nan")
