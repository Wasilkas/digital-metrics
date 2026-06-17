import numpy as np
import numpy.typing as npt
import pandas as pd

from .iou import compute_iou_matrix

_BBOX_COLS = ["bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]


def _compute_containment_matrix(
    boxes: npt.NDArray[np.float32],
) -> npt.NDArray[np.float64]:
    """Compute pairwise max-containment for a set of boxes.

    containment(i, j) = intersection_area(i, j) / min(area_i, area_j)

    Returns 1.0 when the smaller box is fully inside the larger one.
    The matrix is symmetric.
    """
    boxes = np.atleast_2d(boxes).astype(np.float64)
    n = boxes.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    a = boxes[:, np.newaxis, :]  # (N, 1, 4)
    b = boxes[np.newaxis, :, :]  # (1, N, 4)

    inter_x2 = np.minimum(a[:, :, 2], b[:, :, 2])
    inter_y2 = np.minimum(a[:, :, 3], b[:, :, 3])
    inter_x1 = np.maximum(a[:, :, 0], b[:, :, 0])
    inter_y1 = np.maximum(a[:, :, 1], b[:, :, 1])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h  # (N, N)

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # (N,)
    min_area = np.minimum(areas[:, np.newaxis], areas[np.newaxis, :])   # (N, N)

    containment = np.zeros((n, n), dtype=np.float64)
    np.divide(inter_area, min_area, out=containment, where=min_area > 0)
    return containment


def filter_by_confidence(
    preds_df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Return predictions with confidence >= threshold.

    Args:
        preds_df: Predictions DataFrame with a "confidence" column.
        threshold: Minimum confidence to keep.

    Returns:
        Filtered DataFrame (same index, same column order).
    """
    return preds_df[preds_df["confidence"] >= threshold]


def apply_nms(
    preds_df: pd.DataFrame,
    same_class_containment_threshold: float,
    cross_class_iou_threshold: float,
) -> pd.DataFrame:
    """Apply custom NMS to predictions, processed per image.

    For each image, predictions are sorted by confidence descending and then
    greedily processed:
    - Same-class pair: suppress the lower-confidence box if one lies largely
      inside the other (max-containment >= same_class_containment_threshold).
    - Cross-class pair: suppress the lower-confidence box if IoU >=
      cross_class_iou_threshold (classic NMS across classes).

    Setting a threshold above 1.0 effectively disables that suppression type.

    Args:
        preds_df: Predictions DataFrame (standard schema).
        same_class_containment_threshold: Containment ratio [0, 1] above which
            a same-class box is considered "inside" another and suppressed.
        cross_class_iou_threshold: IoU threshold for cross-class suppression.

    Returns:
        DataFrame with suppressed rows removed (original index preserved).
    """
    if len(preds_df) == 0:
        return preds_df

    # Extract columns to numpy once; per-image sort_values + column selection
    # otherwise dominates the loop when there are tens of thousands of images.
    boxes_all = preds_df[_BBOX_COLS].to_numpy(np.float32)
    labels_all = preds_df["instance_label"].to_numpy(dtype=object)
    conf_all = preds_df["confidence"].to_numpy()
    index_all = preds_df.index.to_numpy()

    keep_indices: list[int] = []

    for raw_positions in preds_df.groupby("image_name", sort=False).indices.values():
        # Sort this image's rows by confidence descending (stable, matching the
        # original per-group sort_values ordering for tied confidences).
        positions = np.asarray(raw_positions)
        order = positions[np.argsort(-conf_all[positions], kind="stable")]
        boxes = boxes_all[order]
        labels = labels_all[order]
        n = len(order)

        if n == 0:
            continue

        iou_mat = compute_iou_matrix(boxes, boxes)
        cont_mat = _compute_containment_matrix(boxes)
        suppressed = np.zeros(n, dtype=bool)

        for i in range(n):
            if suppressed[i]:
                continue
            for j in range(i + 1, n):
                if suppressed[j]:
                    continue
                if labels[i] == labels[j]:
                    if cont_mat[i, j] >= same_class_containment_threshold:
                        suppressed[j] = True
                else:
                    if iou_mat[i, j] >= cross_class_iou_threshold:
                        suppressed[j] = True

        keep_indices.extend(index_all[order[~suppressed]].tolist())

    return preds_df.loc[keep_indices]
