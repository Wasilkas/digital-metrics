import numpy as np
import numpy.typing as npt


def compute_iou_matrix(
    boxes_a: npt.NDArray[np.float32],
    boxes_b: npt.NDArray[np.float32],
) -> npt.NDArray[np.float64]:
    """Compute pairwise IoU between two sets of bounding boxes.

    Args:
        boxes_a: Array of shape (N, 4) with columns [x1, y1, x2, y2].
        boxes_b: Array of shape (M, 4) with columns [x1, y1, x2, y2].

    Returns:
        IoU matrix of shape (N, M).
    """
    boxes_a = np.atleast_2d(boxes_a).astype(np.float64)
    boxes_b = np.atleast_2d(boxes_b).astype(np.float64)

    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)

    # Intersection coordinates via broadcasting: (N, 1, 4) vs (1, M, 4)
    a = boxes_a[:, np.newaxis, :]  # (N, 1, 4)
    b = boxes_b[np.newaxis, :, :]  # (1, M, 4)

    inter_x1 = np.maximum(a[:, :, 0], b[:, :, 0])
    inter_y1 = np.maximum(a[:, :, 1], b[:, :, 1])
    inter_x2 = np.minimum(a[:, :, 2], b[:, :, 2])
    inter_y2 = np.minimum(a[:, :, 3], b[:, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h  # (N, M)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])  # (N,)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])  # (M,)

    union_area = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter_area  # (N, M)
    iou = np.zeros_like(inter_area, dtype=np.float64)
    np.divide(inter_area, union_area, out=iou, where=union_area > 0)
    return iou


def find_duplicates_bboxes(
    iou_matrix: npt.NDArray[np.float64],
    threshold: float = 0.99,
) -> tuple[list[int], list[int]]:
    """Return positional indices of duplicate boxes (IoU > threshold).

    For each pair (i, j) with i < j and IoU > threshold, j is marked for
    removal (keeping the first occurrence).

    Returns:
        (to_remove, []) — positional indices to drop, empty second element
        for API compatibility.
    """
    n = iou_matrix.shape[0]
    to_remove: set[int] = set()
    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if iou_matrix[i, j] > threshold:
                to_remove.add(j)
    return list(to_remove), []
