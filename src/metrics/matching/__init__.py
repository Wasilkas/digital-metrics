"""Box matching: IoU geometry, assignment kernels, and match-record building."""

from .assignment import assign_greedy, assign_hungarian, assign_iou_prior
from .iou import compute_iou_matrix, find_duplicates_bboxes
from .matching import MatchingStrategy, match_boxes

__all__ = [
    "MatchingStrategy",
    "assign_greedy",
    "assign_hungarian",
    "assign_iou_prior",
    "compute_iou_matrix",
    "find_duplicates_bboxes",
    "match_boxes",
]
