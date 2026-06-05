import numpy as np

from .types import PredictMatch


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
