"""Low-level box-assignment kernels shared by matching.py and ap.py.

Each kernel operates purely on an IoU matrix of shape ``(n_preds, n_gts)`` and
returns the matched ``(pred_index, gt_index)`` positional pairs.  Labels,
confidences, DataFrame bookkeeping and TP/FP/FN accounting are the caller's
responsibility — these functions only decide *which* prediction box is matched
to *which* ground-truth box under a given IoU threshold.

Callers that need confidence-ordered behaviour (greedy) must pass the IoU
matrix with predictions already sorted by confidence descending, so matrix row
order is the processing order.
"""

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

# Matched (pred_index, gt_index) positional pairs.
MatchedPairs = list[tuple[int, int]]


def assign_greedy(
    iou_matrix: npt.NDArray[np.float64],
    iou_threshold: float,
) -> MatchedPairs:
    """Greedy, prediction-order assignment (YOLO-style).

    Walks predictions in row order (caller sorts by confidence descending).
    Each prediction claims its single highest-IoU ground truth; the match is
    kept only if that GT is still free and the IoU is at least
    ``iou_threshold``.  There is no fallback to a second-best GT — if the
    argmax GT is already taken, the prediction goes unmatched.
    """
    n_preds, n_gts = iou_matrix.shape
    if n_gts == 0:
        return []

    matched_gt = np.zeros(n_gts, dtype=bool)
    pairs: MatchedPairs = []
    for i in range(n_preds):
        j = int(np.argmax(iou_matrix[i]))
        if iou_matrix[i, j] >= iou_threshold and not matched_gt[j]:
            matched_gt[j] = True
            pairs.append((i, j))
    return pairs


def assign_iou_prior(
    iou_matrix: npt.NDArray[np.float64],
    iou_threshold: float,
    valid_mask: npt.NDArray[np.bool_] | None = None,
) -> MatchedPairs:
    """IoU-prior assignment (Ultralytics non-scipy style).

    Considers every pred-GT pair whose IoU is at least ``iou_threshold`` and
    — when ``valid_mask`` is given — is flagged valid (e.g. labels match).
    Pairs are assigned in descending-IoU order, each prediction and each GT
    used at most once.  Confidence plays no role in the ordering.

    A stable sort is used so that ties in IoU resolve deterministically in
    favour of the lower (prediction, GT) index pair.
    """
    n_preds, n_gts = iou_matrix.shape
    if n_preds == 0 or n_gts == 0:
        return []

    valid = iou_matrix >= iou_threshold
    if valid_mask is not None:
        valid = valid & valid_mask

    pred_idxs, gt_idxs = np.nonzero(valid)
    if len(pred_idxs) == 0:
        return []

    order = np.argsort(-iou_matrix[pred_idxs, gt_idxs], kind="stable")
    pred_idxs = pred_idxs[order]
    gt_idxs = gt_idxs[order]

    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    pairs: MatchedPairs = []
    for pred_i, gt_j in zip(pred_idxs.tolist(), gt_idxs.tolist(), strict=True):
        if pred_i in matched_preds or gt_j in matched_gts:
            continue
        matched_preds.add(pred_i)
        matched_gts.add(gt_j)
        pairs.append((pred_i, gt_j))
    return pairs


def assign_hungarian(
    iou_matrix: npt.NDArray[np.float64],
    iou_threshold: float,
) -> MatchedPairs:
    """Globally optimal assignment via the Hungarian algorithm.

    Runs ``scipy.optimize.linear_sum_assignment`` on the negative IoU matrix,
    then keeps only the assigned pairs whose IoU is at least ``iou_threshold``.
    Geometry-first; confidence plays no role.
    """
    n_preds, n_gts = iou_matrix.shape
    if n_preds == 0 or n_gts == 0:
        return []

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    return [
        (int(i), int(j))
        for i, j in zip(row_ind.tolist(), col_ind.tolist(), strict=True)
        if iou_matrix[i, j] >= iou_threshold
    ]
