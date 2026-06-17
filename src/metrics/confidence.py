from typing import Literal

import numpy as np
import numpy.typing as npt

from .types import PredictMatch

ConfidenceOptimization = Literal["per_class", "global"]


def find_best_confidences(
    matches: dict[str, list[PredictMatch]],
    classes: list[str],
) -> dict[str, float]:
    """Find the confidence threshold that maximises F1 for each class.

    Args:
        matches: Dict mapping class name → list of PredictMatch objects.
        classes: Classes to search over.

    Returns:
        Dict mapping class name → best confidence threshold.
    """
    best_confidences: dict[str, float] = {}

    for c in classes:
        cls_matches = matches.get(c, [])
        if not cls_matches:
            best_confidences[c] = 0.0
            continue

        cls_matches_sorted = sorted(cls_matches, key=lambda x: x.confidence, reverse=True)

        no_fn = [m for m in cls_matches_sorted if m.type != "FN"]
        tp_flags = np.array([m.type == "TP" for m in no_fn], dtype=bool)
        fp_flags = ~tp_flags

        cum_tp = np.cumsum(tp_flags)
        cum_fp = np.cumsum(fp_flags)

        n_positives = sum(1 for m in cls_matches if m.type != "FP")
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-6)
        recalls = cum_tp / max(n_positives, 1e-6)

        f1_scores = 2.0 * precisions * recalls / np.maximum(precisions + recalls, 1e-6)

        if len(f1_scores) == 0:
            best_confidences[c] = 0.0
        else:
            best_index = int(np.argmax(f1_scores))
            best_confidences[c] = cls_matches_sorted[best_index].confidence

    return best_confidences


def find_best_global_confidence(
    matches: dict[str, list[PredictMatch]],
    classes: list[str],
) -> float:
    """Find a single confidence threshold, shared by all classes, that
    maximises the mean per-class F1 — YOLO-style global thresholding.

    Unlike :func:`find_best_confidences`, which tunes a separate threshold per
    class, this mirrors Ultralytics YOLO: one confidence threshold is applied to
    every class. The function sweeps every observed confidence value and returns
    the one whose mean F1 across classes is highest.

    Classes with no ground-truth positives and no true positives contribute a
    constant F1 of 0 at every threshold, so they do not affect which threshold
    wins.

    Args:
        matches: Dict mapping class name → list of PredictMatch objects.
        classes: Classes to average F1 over.

    Returns:
        A single confidence threshold to apply to all classes. Returns 0.0 when
        there are no detections to threshold.
    """
    # Per class: detection confidences (ascending) with suffix-cumulative
    # TP/FP counts, so the counts for "confidence >= t" are a single lookup.
    FloatArray = npt.NDArray[np.float64]
    per_class: list[tuple[FloatArray, FloatArray, FloatArray, int]] = []
    candidate_thresholds: set[float] = set()

    for c in classes:
        cls_matches = matches.get(c, [])
        detections = sorted(
            (m for m in cls_matches if m.type != "FN"),
            key=lambda m: m.confidence,
        )
        n_positives = sum(1 for m in cls_matches if m.type != "FP")
        if not detections:
            continue

        confidences = np.array([m.confidence for m in detections])
        is_tp = np.array([m.type == "TP" for m in detections], dtype=np.float64)
        # Suffix sums: element i holds the count over detections i..end, i.e.
        # those with confidence >= confidences[i].
        cum_tp = np.cumsum(is_tp[::-1])[::-1]
        cum_fp = np.cumsum((1.0 - is_tp)[::-1])[::-1]

        per_class.append((confidences, cum_tp, cum_fp, n_positives))
        candidate_thresholds.update(confidences.tolist())

    if not per_class:
        return 0.0

    best_threshold = 0.0
    best_mean_f1 = -1.0
    for t in sorted(candidate_thresholds):
        f1_sum = 0.0
        for confidences, cum_tp, cum_fp, n_positives in per_class:
            idx = int(np.searchsorted(confidences, t, side="left"))
            if idx >= len(confidences):
                continue  # no detection clears the threshold → F1 = 0
            tp = cum_tp[idx]
            fp = cum_fp[idx]
            precision = tp / max(tp + fp, 1e-6)
            recall = tp / max(n_positives, 1e-6)
            f1_sum += 2.0 * precision * recall / max(precision + recall, 1e-6)

        mean_f1 = f1_sum / len(per_class)
        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_threshold = t

    return best_threshold


def slice_by_conf(
    matches: dict[str, list[PredictMatch]],
    classes: list[str],
    confidences: dict[str, float],
) -> dict[str, list[PredictMatch]]:
    """Return a new matches dict filtered by per-class confidence thresholds.

    Predictions above their class threshold are kept as-is.
    FN entries are always kept.
    TP entries below threshold are converted to FN (the GT goes undetected).
    FP entries below threshold are discarded.

    Args:
        matches: Original matches dict.
        classes: Classes to filter.
        confidences: Per-class confidence thresholds.

    Returns:
        New dict — original is not mutated.
    """
    result: dict[str, list[PredictMatch]] = {}

    for c in classes:
        threshold = confidences.get(c, 0.0)
        filtered: list[PredictMatch] = []

        for match in matches.get(c, []):
            if match.confidence >= threshold or match.type == "FN":
                filtered.append(match)
            elif match.type == "TP":
                fn_match = PredictMatch(
                    pred_label="background",
                    gt_label=match.gt_label,
                    pred_index=-1,
                    gt_index=match.gt_index,
                    confidence=0.0,
                )
                filtered.append(fn_match)
            # FP below threshold is silently dropped

        result[c] = filtered

    return result
